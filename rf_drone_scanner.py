#!/usr/bin/env python3
"""
RF Drone Scanner — Jetson Nano / Linux
Standalone scanner that uses a USB WiFi adapter to detect drone-related
radio signals on 2.4 GHz and 5 GHz bands via passive AP scanning (iwlist).

Live terminal dashboard updates in-place every scan cycle.
All detections are logged to a timestamped file.

Usage:
    python3 rf_drone_scanner.py --iface wlan1
    python3 rf_drone_scanner.py --iface wlan1 --interval 2 --log-dir ./logs
    python3 rf_drone_scanner.py --iface wlan1 --alert-only   # suppress non-alerts

Requirements:
    sudo apt install wireless-tools
    # Run with sudo, or grant passwordless sudo for iwlist:
    #   echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/iwlist" | sudo tee /etc/sudoers.d/iwlist

Detection heuristics (each adds to a threat score):
    1. SSID matches known drone vendor patterns (DJI, Mavic, Phantom, Autel…)
    2. Operating on a 5 GHz drone video-link channel (36–165)
    3. Strong 2.4 GHz signal  ≥ STRONG_SIGNAL_DBM  (nearby RC transmitter)
    4. Open (unencrypted) WiFi in drone bands  (DJI hotspot default)
    5. Exact OUI match against known DJI / drone-vendor MAC prefixes
"""

import argparse
import curses
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration / tuning constants
# ---------------------------------------------------------------------------

# Minimum RSSI (dBm) on 2.4 GHz to flag as a possible nearby RC link
STRONG_SIGNAL_DBM = -65

# 5 GHz channels commonly used by drone video links
DRONE_5GHZ_CHANNELS = {
    36, 40, 44, 48,                          # UNII-1
    100, 104, 108, 112, 116, 120,            # UNII-2e
    124, 128, 132, 136, 140,                 # UNII-2e cont.
    149, 153, 157, 161, 165,                 # UNII-3 — DJI O3/O4 primary
}

# SSID substrings/patterns (matched case-insensitively)
DRONE_SSID_PATTERNS = [
    (r'DJI[-_\s]',          "DJI device SSID"),
    (r'\bMAVIC\b',          "DJI Mavic SSID"),
    (r'\bPHANTOM\b',        "DJI Phantom SSID"),
    (r'MINI[-_\s]?\d',      "DJI Mini SSID"),
    (r'\bAUTEL\b',          "Autel drone SSID"),
    (r'\bSKYDIO\b',         "Skydio drone SSID"),
    (r'\bPARROT\b',         "Parrot drone SSID"),
    (r'\bBEBOP\b',          "Parrot Bebop SSID"),
    (r'\bTELLO\b',          "Ryze Tello SSID"),
    (r'\bFPV[-_\s]',        "FPV system SSID"),
    (r'\bDRONE\b',          "Generic drone SSID"),
    (r'\bUAV\b',            "UAV SSID"),
    (r'\bQUADCOPTER\b',     "Quadcopter SSID"),
    (r'SKY[-_]VIPER',       "Sky Viper SSID"),
    (r'\bHOVER\b',          "Hover drone SSID"),
]

# Known drone / RC-gear vendor OUI prefixes (first 3 octets, uppercase)
# Sources: Wireshark OUI database, DJI FCC filings
DRONE_OUIS = {
    "60:60:1F",   # DJI
    "48:1C:B9",   # DJI
    "34:D2:62",   # DJI
    "A0:14:3D",   # DJI
    "90:3A:E6",   # DJI
    "18:60:24",   # DJI
    "BC:8B:B2",   # SZ DJI Technology
    "C4:7C:8D",   # DJI
    "D0:9C:7A",   # DJI
    "AC:B0:AB",   # DJI
    "80:2A:A8",   # Autel Robotics
}

THREAT_COLORS = {          # curses color-pair indices
    "CRITICAL": 1,
    "HIGH":     2,
    "MEDIUM":   3,
    "LOW":      4,
    "CLEAR":    5,
    "INFO":     6,
}

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Threat scoring
# ---------------------------------------------------------------------------

def infer_source(net: dict) -> str:
    """Return a human-readable likely source for the RF signal."""
    ssid_upper = net["ssid"].upper()
    freq = net["frequency_ghz"]
    ch = net["channel"]
    dbm = net["signal_dbm"]
    oui = net["bssid"][:8].upper() if len(net["bssid"]) >= 8 else ""

    for pattern, _ in DRONE_SSID_PATTERNS:
        if re.search(pattern, ssid_upper):
            return "Drone hotspot / control link" if freq >= 5.0 else "Drone control hotspot"

    if oui in DRONE_OUIS:
        return "DJI drone radio" if freq < 5.0 else "DJI 5 GHz video link"

    if freq >= 5.1 and ch in DRONE_5GHZ_CHANNELS:
        return "5 GHz drone video uplink"

    if 2.3 <= freq < 5.0 and dbm >= STRONG_SIGNAL_DBM:
        return "Nearby RC transmitter"

    if not net["encryption"] and freq >= 2.3:
        return "Open WiFi hotspot"

    if freq >= 5.0:
        return "5 GHz WiFi / video source"

    if 2.3 <= freq < 5.0:
        return "2.4 GHz WiFi / RC source"

    return "Generic RF source"


def score_network(net: dict) -> tuple[int, list[str], str]:
    """
    Return (score, reasons, source).  Score thresholds:
        0        → not interesting
        1–2      → LOW
        3–4      → MEDIUM
        5–6      → HIGH
        7+       → CRITICAL
    """
    score = 0
    reasons: list[str] = []

    ssid_upper = net["ssid"].upper()
    freq = net["frequency_ghz"]
    ch   = net["channel"]
    dbm  = net["signal_dbm"]
    oui  = net["bssid"][:8].upper() if len(net["bssid"]) >= 8 else ""

    # 1. SSID pattern (+3 points — strongest single indicator)
    for pattern, label in DRONE_SSID_PATTERNS:
        if re.search(pattern, ssid_upper):
            score += 3
            reasons.append(f"SSID: {label}")
            break

    # 2. Known drone OUI (+3 points)
    if oui in DRONE_OUIS:
        score += 3
        reasons.append(f"OUI {oui} is a known drone vendor MAC prefix")

    # 3. 5 GHz drone channel (+2 points)
    if freq >= 5.1 and ch in DRONE_5GHZ_CHANNELS:
        score += 2
        reasons.append(f"5 GHz drone video-link channel {ch} ({freq:.3f} GHz)")

    # 4. Strong nearby 2.4 GHz (+1 point)
    if 2.3 <= freq < 5.0 and dbm >= STRONG_SIGNAL_DBM:
        score += 1
        reasons.append(f"Strong 2.4 GHz signal ({dbm} dBm) — possible nearby RC link")

    # 5. Open (unencrypted) WiFi in drone bands (+1 point)
    if not net["encryption"] and freq >= 2.3:
        score += 1
        reasons.append("Open (unencrypted) WiFi — consistent with drone hotspot")

    source = infer_source(net)
    return score, reasons, source


def threat_level(score: int) -> str:
    if score >= 7:  return "CRITICAL"
    if score >= 5:  return "HIGH"
    if score >= 3:  return "MEDIUM"
    if score >= 1:  return "LOW"
    return "CLEAR"


# ---------------------------------------------------------------------------
# iwlist parser
# ---------------------------------------------------------------------------

def parse_iwlist(raw: str) -> list[dict]:
    """Parse full iwlist scan output into a list of network dicts."""
    networks = []
    for cell in re.split(r"(?=Cell \d+ - Address:)", raw):
        if "Address:" not in cell:
            continue

        net: dict = {
            "ssid":          "",
            "bssid":         "",
            "channel":       0,
            "frequency_ghz": 0.0,
            "signal_dbm":    -100,
            "encryption":    False,
            "score":         0,
            "threat":        "CLEAR",
            "reasons":       [],
            "source":        "Generic RF source",
        }

        m = re.search(r"Address:\s*([0-9A-Fa-f:]{17})", cell)
        if m:
            net["bssid"] = m.group(1).upper()

        m = re.search(r'ESSID:"([^"]*)"', cell)
        if m:
            net["ssid"] = m.group(1)

        m = re.search(r"Channel[=:](\d+)", cell)
        if m:
            net["channel"] = int(m.group(1))

        m = re.search(r"Frequency[=:](\d+\.\d+)", cell)
        if m:
            net["frequency_ghz"] = float(m.group(1))
        elif net["channel"]:
            net["frequency_ghz"] = 2.4 if net["channel"] <= 14 else 5.0

        m = re.search(r"Signal level[=:](-?\d+)\s*dBm", cell)
        if m:
            net["signal_dbm"] = int(m.group(1))
        else:
            m = re.search(r"Signal level[=:](\d+)/100", cell)
            if m:
                net["signal_dbm"] = int(m.group(1)) // 2 - 100

        net["encryption"] = "Encryption key:on" in cell

        net["score"], net["reasons"], net["source"] = score_network(net)
        net["threat"] = threat_level(net["score"])

        networks.append(net)

    networks.sort(key=lambda n: n["score"], reverse=True)
    return networks


# ---------------------------------------------------------------------------
# Scanner (background thread)
# ---------------------------------------------------------------------------

class RFScanner:
    def __init__(self, iface: str, interval: float):
        self.iface    = iface
        self.interval = interval

        self._lock          = threading.Lock()
        self._networks:  list[dict]  = []
        self._alerts:    list[dict]  = []
        self._scan_count:  int   = 0
        self._last_scan:   float = 0.0
        self._last_error:  str   = ""
        self._stopped      = False
        self._scan_ms:     float = 0.0
        self._reconnecting: bool = False

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # -- public ------------------------------------------------------------

    def get_state(self) -> dict:
        """Return a thread-safe snapshot of current scanner state."""
        with self._lock:
            return {
                "networks":      list(self._networks),
                "alerts":        list(self._alerts),
                "scan_count":    self._scan_count,
                "last_scan":     self._last_scan,
                "last_error":    self._last_error,
                "scan_ms":       self._scan_ms,
                "reconnecting":  self._reconnecting,
            }

    def stop(self):
        self._stopped = True

    # -- internal ----------------------------------------------------------

    def _run(self, cmd: list[str], timeout: int = 15) -> str:
        """Run a shell command as sudo, return combined stdout+stderr."""
        result = subprocess.run(
            ["sudo"] + cmd,
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout + result.stderr

    def _save_connection(self) -> str | None:
        """
        Return the SSID we are currently associated with, or None.
        Uses wpa_cli — works with wpa_supplicant which JetPack uses by default.
        """
        try:
            out = self._run(["wpa_cli", "-i", self.iface, "status"])
            m = re.search(r"^ssid=(.+)$", out, re.MULTILINE)
            if m and "wpa_state=COMPLETED" in out:
                return m.group(1).strip()
        except Exception:
            pass
        return None

    def _disconnect(self):
        """Tell wpa_supplicant to drop the current association."""
        try:
            self._run(["wpa_cli", "-i", self.iface, "disconnect"])
        except Exception:
            pass

    def _reconnect(self, ssid: str | None):
        """
        Reconnect to the network we were on before scanning.
        1. Tell wpa_supplicant to reassociate.
        2. Run dhclient to renew the DHCP lease.
        """
        try:
            self._run(["wpa_cli", "-i", self.iface, "reassociate"])
            # Give wpa_supplicant a moment to complete the 4-way handshake
            time.sleep(3)
            # Renew DHCP lease in the background (don't block the scan loop)
            subprocess.Popen(
                ["sudo", "dhclient", "-1", "-v", self.iface],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as exc:
            with self._lock:
                self._last_error = f"Reconnect failed: {exc}"

    def _loop(self):
        while not self._stopped:
            t0 = time.time()
            try:
                # 1. Remember what we're connected to
                current_ssid = self._save_connection()

                with self._lock:
                    self._reconnecting = False

                # 2. Briefly drop association so iwlist gets a clean scan
                if current_ssid:
                    self._disconnect()
                    time.sleep(0.5)   # let the interface settle

                # 3. Scan
                raw = self._run(["iwlist", self.iface, "scan"], timeout=20)
                networks = parse_iwlist(raw)
                alerts   = [n for n in networks if n["score"] > 0]
                elapsed_ms = (time.time() - t0) * 1000

                with self._lock:
                    self._networks    = networks
                    self._alerts      = alerts
                    self._scan_count += 1
                    self._last_scan   = time.time()
                    self._last_error  = ""
                    self._scan_ms     = elapsed_ms
                    self._reconnecting = current_ssid is not None

                # Emit a compact JSON summary line for external dashboards
                try:
                    def _bssid_to_deg(bssid: str) -> int:
                        # Stable-ish conversion from BSSID hex to 0-359 degrees
                        s = ''.join(ch for ch in (bssid or '') if ch in '0123456789ABCDEFabcdef')
                        if not s:
                            return -1
                        total = sum(int(c, 16) for c in s)
                        return int(total) % 360

                    top = alerts[0] if len(alerts) > 0 else None
                    summary = {
                        'type': 'RF',
                        'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                        'scan_count': self._scan_count,
                        'scan_ms': round(elapsed_ms, 1),
                        'top_alert': None,
                        'direction_deg': -1,
                    }
                    if top:
                        summary['top_alert'] = {
                            'ssid': top.get('ssid',''),
                            'bssid': top.get('bssid',''),
                            'freq': top.get('frequency_ghz',0.0),
                            'dbm': top.get('signal_dbm',-100),
                            'threat': top.get('threat','CLEAR'),
                        }
                        summary['direction_deg'] = _bssid_to_deg(top.get('bssid',''))
                    print(json.dumps(summary), flush=True)
                except Exception:
                    pass

                # 4. Reconnect if we were associated before
                if current_ssid:
                    self._reconnect(current_ssid)
                    with self._lock:
                        self._reconnecting = False

            except subprocess.TimeoutExpired:
                with self._lock:
                    self._last_error  = "iwlist timed out"
                    self._reconnecting = False
            except Exception as exc:
                with self._lock:
                    self._last_error  = str(exc)
                    self._reconnecting = False

            # Sleep for the remainder of the interval
            sleep_for = max(0.0, self.interval - (time.time() - t0))
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# File logger
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str, iface: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(log_dir) / f"rf_drone_{iface}_{ts}.log"

    logger = logging.getLogger("rf_drone")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(fh)

    return logger, str(path)


# ---------------------------------------------------------------------------
# Curses dashboard
# ---------------------------------------------------------------------------

BAR_CHARS = " ▏▎▍▌▋▊▉█"


def dbm_bar(dbm: int, width: int = 20) -> str:
    """Convert dBm (-100…0) to a Unicode block bar."""
    clamped = max(-100, min(0, dbm))
    filled  = int((clamped + 100) / 100 * width)
    return ("█" * filled).ljust(width)


def run_dashboard(stdscr, scanner: RFScanner, logger: logging.Logger,
                  log_path: str, iface: str, alert_only: bool):
    """Main curses loop."""
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()

    # colour pairs
    curses.init_pair(THREAT_COLORS["CRITICAL"], curses.COLOR_WHITE,   curses.COLOR_RED)
    curses.init_pair(THREAT_COLORS["HIGH"],     curses.COLOR_BLACK,   curses.COLOR_YELLOW)
    curses.init_pair(THREAT_COLORS["MEDIUM"],   curses.COLOR_YELLOW,  -1)
    curses.init_pair(THREAT_COLORS["LOW"],      curses.COLOR_CYAN,    -1)
    curses.init_pair(THREAT_COLORS["CLEAR"],    curses.COLOR_WHITE,   -1)
    curses.init_pair(THREAT_COLORS["INFO"],     curses.COLOR_GREEN,   -1)

    # alert ring buffer for the log pane
    alert_log: deque[str] = deque(maxlen=200)
    seen_bssids: set[str] = set()   # track new-alert logging

    stdscr.nodelay(True)
    stdscr.timeout(250)  # refresh every 250 ms

    while True:
        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):   # q / ESC
            break

        state = scanner.get_state()
        networks = state["networks"]
        alerts   = state["alerts"]

        # ── log new alerts ──────────────────────────────────────────────
        for a in alerts:
            bid = a["bssid"]
            if bid not in seen_bssids:
                seen_bssids.add(bid)
                ssid = a["ssid"] if a["ssid"] else "<hidden>"
                msg  = (f"NEW ALERT [{a['threat']}] score={a['score']}  "
                        f"SSID={ssid}  BSSID={bid}  "
                        f"source={a['source']}  "
                        f"{a['frequency_ghz']:.3f}GHz  ch{a['channel']}  "
                        f"{a['signal_dbm']}dBm  "
                        f"reasons={'; '.join(a['reasons'])}")
                logger.warning(msg)
                alert_log.appendleft(
                    f"{datetime.now().strftime('%H:%M:%S')}  "
                    f"[{a['threat']:8s}]  {ssid[:22]:<22}  "
                    f"{a['frequency_ghz']:.1f}GHz  {a['signal_dbm']:4d}dBm"
                )
        # clear seen set for bssids no longer visible (they may reappear)
        visible = {n["bssid"] for n in networks}
        seen_bssids &= visible

        # ── draw ────────────────────────────────────────────────────────
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        row = 0

        # header bar
        header = f" RF DRONE SCANNER  |  iface: {iface}  |  q = quit "
        attr   = curses.color_pair(THREAT_COLORS["INFO"]) | curses.A_BOLD
        stdscr.addstr(row, 0, header[:max_x - 1], attr)
        row += 1

        # status line
        age   = time.time() - state["last_scan"] if state["last_scan"] else -1
        err   = f"  ⚠ {state['last_error']}" if state["last_error"] else ""
        recon = "  📶 reconnecting..." if state.get("reconnecting") else ""
        status = (f"  Scans: {state['scan_count']}   "
                  f"Last scan: {age:.1f}s ago  ({state['scan_ms']:.0f}ms)  "
                  f"Networks: {len(networks)}   Alerts: {len(alerts)}{recon}{err}")
        stdscr.addstr(row, 0, status[:max_x - 1], curses.A_DIM)
        row += 1

        # divider
        stdscr.addstr(row, 0, "─" * (max_x - 1), curses.A_DIM)
        row += 1

        # column headers
        col_hdr = (f"  {'THREAT':<10} {'SSID':<24} {'SRC':<18} {'BSSID':<19} "
                   f"{'GHz':>6} {'CH':>4} {'dBm':>5}  SIGNAL")
        stdscr.addstr(row, 0, col_hdr[:max_x - 1], curses.A_UNDERLINE)
        row += 1

        # network rows
        display_nets = alerts if alert_only else networks
        net_rows_available = max(4, max_y - row - 12)  # leave room for log pane

        for net in display_nets[:net_rows_available]:
            if row >= max_y - 1:
                break

            threat = net["threat"]
            cp     = curses.color_pair(THREAT_COLORS.get(threat, THREAT_COLORS["CLEAR"]))
            ssid   = (net["ssid"] if net["ssid"] else "<hidden>")[:23]
            source = (net.get("source", "Generic RF source") or "Generic RF source")[:17]
            bar    = dbm_bar(net["signal_dbm"], width=16)
            enc    = "" if net["encryption"] else " 🔓"

            line = (f"  {threat:<10} {ssid:<24} {source:<18} {net['bssid']:<19} "
                    f"{net['frequency_ghz']:>6.3f} {net['channel']:>4} "
                    f"{net['signal_dbm']:>5}  {bar}{enc}")
            stdscr.addstr(row, 0, line[:max_x - 1], cp)

            # reasons on next line (indented), only for alerted networks
            if net["reasons"] and row + 1 < max_y - 1:
                row += 1
                reason_str = "    ↳ " + "  |  ".join(net["reasons"])
                stdscr.addstr(row, 0, reason_str[:max_x - 1],
                              cp | curses.A_DIM)
            row += 1

        # log pane
        log_start = max_y - 9
        if log_start > row + 1:
            stdscr.addstr(log_start - 1, 0, "─" * (max_x - 1), curses.A_DIM)
            stdscr.addstr(log_start - 1, 2,
                          " Recent Alerts (newest first) ",
                          curses.A_DIM | curses.A_ITALIC)
            for i, entry in enumerate(list(alert_log)[:8]):
                if log_start + i >= max_y - 1:
                    break
                stdscr.addstr(log_start + i, 0,
                              ("  " + entry)[:max_x - 1],
                              curses.color_pair(THREAT_COLORS["MEDIUM"]))

        # footer
        footer = (f" Log: {log_path}  |  "
                  f"{'ALERT-ONLY MODE' if alert_only else 'ALL NETWORKS'}  |  q = quit ")
        try:
            stdscr.addstr(max_y - 1, 0, footer[:max_x - 1], curses.A_REVERSE)
        except curses.error:
            pass

        stdscr.refresh()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def check_prerequisites(iface: str) -> bool:
    ok = True

    # Check interface exists
    result = subprocess.run(["ip", "link", "show", iface],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌  Interface '{iface}' not found.")
        print(f"    Run `ip link show` to list available interfaces.")
        ok = False

    # Check iwlist is available
    result = subprocess.run(["which", "iwlist"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌  iwlist not found.  Install with:  sudo apt install wireless-tools")
        ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="RF Drone Scanner — live dashboard for USB WiFi adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 rf_drone_scanner.py --iface wlan1
  python3 rf_drone_scanner.py --iface wlan1 --interval 2
  python3 rf_drone_scanner.py --iface wlan1 --alert-only
  python3 rf_drone_scanner.py --iface wlan1 --log-dir /var/log/drone

Setup (run once):
  sudo apt install wireless-tools
  # Optional: passwordless sudo for iwlist only
  echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/iwlist" | sudo tee /etc/sudoers.d/iwlist
        """
    )
    parser.add_argument("--iface", "-i", default="wlan1",
                        help="WiFi interface name (default: wlan1)")
    parser.add_argument("--interval", "-t", type=float, default=3.0,
                        help="Seconds between scans (default: 3.0)")
    parser.add_argument("--log-dir", default="./rf_logs",
                        help="Directory for log files (default: ./rf_logs)")
    parser.add_argument("--alert-only", action="store_true",
                        help="Show only networks with a non-zero threat score")
    parser.add_argument("--no-prereq-check", action="store_true",
                        help="Skip interface / iwlist availability checks")
    args = parser.parse_args()

    if not args.no_prereq_check:
        if not check_prerequisites(args.iface):
            sys.exit(1)

    print(f"📡  Starting RF drone scanner on {args.iface} ...")
    print(f"📝  Logs → {args.log_dir}/")
    print(f"⏱   Scan interval: {args.interval}s")
    print("    Press q inside the dashboard to quit.\n")
    time.sleep(1)

    logger, log_path = setup_logger(args.log_dir, args.iface)
    logger.info(f"Scanner started  iface={args.iface}  interval={args.interval}s")

    scanner = RFScanner(iface=args.iface, interval=args.interval)

    try:
        curses.wrapper(
            run_dashboard,
            scanner, logger, log_path,
            args.iface, args.alert_only
        )
    except KeyboardInterrupt:
        pass
    finally:
        scanner.stop()
        logger.info("Scanner stopped.")
        print(f"\n✅  Stopped.  Log saved to: {log_path}")


if __name__ == "__main__":
    main()