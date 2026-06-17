#!/usr/bin/env python3
"""
Safe RF Drone Detection using USB WiFi Adapter (Monitor Mode + Scapy)
Optimized for NVIDIA Jetson Nano with TP-Link (wlxb8fbb3c4e916).
WiFi connection preserving version.
"""

import argparse
import time
import threading
import queue
import subprocess
from datetime import datetime
from collections import deque
import sys

try:
    from scapy.all import sniff, Dot11, Dot11Beacon, Dot11ProbeReq, Dot11ProbeResp
except ImportError:
    print("❌ Scapy not installed.")
    print("Run: pip3 install scapy")
    print("Also: sudo apt update && sudo apt install aircrack-ng")
    sys.exit(1)


class RFDroneDetector:
    def __init__(self, monitor_iface=None, channel_hop=True):
        self.channel_hop = channel_hop
        self.running = False
        self.thread = None
        self.detections = queue.Queue(maxsize=50)
        
        self.drone_ssids = ["DJI_", "Phantom", "Mavic", "Air", "Mini", "Parrot", 
                           "Bebop", "Anafi", "FPV_", "drone", "RC_", "Sky", "Tello"]
        self.drone_ouis = ["60:60", "00:0C", "90:3A", "E4:83", "00:26", "18:64"]
        
        self.recent_drones = deque(maxlen=30)
        self.stats = {"beacons": 0, "probes": 0, "drones": 0}
        
        self.monitor_iface = monitor_iface or self._find_usb_wifi_adapter()

    def _find_usb_wifi_adapter(self):
        """Detect wlxb8fbb3c4e916 or any USB WiFi adapter."""
        try:
            lsusb_out = subprocess.check_output(["lsusb"], text=True, stderr=subprocess.DEVNULL)
            if "2357:0138" in lsusb_out or "TP-Link" in lsusb_out:
                print("✅ TP-Link USB WiFi adapter detected via lsusb")
            
            out = subprocess.check_output(["iwconfig"], text=True, stderr=subprocess.DEVNULL)
            wireless_ifaces = []
            for line in out.splitlines():
                if line.strip() and not line.startswith(" "):
                    iface = line.split()[0].strip()
                    if iface in ["lo", "can0", "l4tbr0", "usb0", "usb1", "enP8p1s0", "br-", "docker0"]:
                        continue
                    if "no wireless extensions" not in line:
                        wireless_ifaces.append(iface)
                        if "ESSID" in line:
                            try:
                                essid = line.split("ESSID:")[1].strip().split()[0].strip('"')
                                print(f"   {iface} → Network: {essid}")
                            except:
                                pass
            # Prefer USB adapter
            for iface in wireless_ifaces:
                if iface.startswith("wlx") or (iface.startswith("wlan") and iface != "wlan0"):
                    print(f"✅ Using USB WiFi adapter: {iface}")
                    return f"{iface}mon"
            if wireless_ifaces:
                print(f"⚠️ Using first available: {wireless_ifaces[0]}mon")
                return f"{wireless_ifaces[0]}mon"
            print("⚠️ No wireless adapter found. Using wlan0mon.")
            return "wlan0mon"
        except Exception as e:
            print(f"⚠️ Detection issue: {e}. Using wlan0mon.")
            return "wlan0mon"

    def setup_monitor_mode(self):
        print(f"🔧 Setting up monitor mode on {self.monitor_iface} (safe mode)...")
        try:
            # Compatible calls for Jetson Python version
            subprocess.run(["sudo", "airmon-ng", "check", "kill"], 
                         capture_output=True, stderr=subprocess.DEVNULL)
            
            base_iface = self.monitor_iface.replace("mon", "")
            # Use simple run for compatibility
            result = subprocess.run(["sudo", "airmon-ng", "start", base_iface], 
                                  capture_output=True, text=True)
            
            # Verify
            iw_out = subprocess.check_output(["iwconfig"], text=True, stderr=subprocess.DEVNULL)
            if "Mode:Monitor" in iw_out or self.monitor_iface in iw_out:
                print(f"✅ Monitor mode ready on {self.monitor_iface}")
                return True
            else:
                print("⚠️ Monitor mode may not be active yet.")
                return False
        except Exception as e:
            print(f"❌ Monitor mode setup failed: {e}")
            print("💡 Try this first:")
            print(f"   sudo airmon-ng start {self.monitor_iface.replace('mon', '')}")
            return False

    def _packet_handler(self, pkt):
        if not pkt.haslayer(Dot11):
            return

        try:
            mac = None
            ssid = None
            is_drone = False

            if pkt.haslayer(Dot11Beacon):
                self.stats["beacons"] += 1
                ssid = pkt[Dot11Beacon].info.decode(errors='ignore').strip()
                mac = pkt.addr2
            elif pkt.haslayer(Dot11ProbeReq) or pkt.haslayer(Dot11ProbeResp):
                self.stats["probes"] += 1
                if pkt.haslayer(Dot11ProbeReq):
                    ssid = pkt[Dot11ProbeReq].info.decode(errors='ignore').strip()
                mac = pkt.addr2 or pkt.addr1

            if mac:
                mac_upper = mac.upper()
                oui = mac_upper[:8]
                if any(oui.startswith(o) for o in self.drone_ouis):
                    is_drone = True
                if ssid and any(d in ssid.upper() for d in self.drone_ssids):
                    is_drone = True

                if is_drone:
                    detection = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "mac": mac,
                        "ssid": ssid or "Unknown",
                        "type": "Beacon" if pkt.haslayer(Dot11Beacon) else "Probe",
                        "signal": getattr(pkt, 'dBm_AntSignal', 'N/A')
                    }
                    key = f"{mac}_{ssid}"
                    if key not in self.recent_drones:
                        self.recent_drones.append(key)
                        self.detections.put(detection)
                        self.stats["drones"] += 1
                        print(f"🚁 DRONE DETECTED! {mac} | {ssid} | Signal: {detection['signal']}")
        except:
            pass

    def _channel_hopper(self):
        channels = list(range(1, 14))
        idx = 0
        while self.running:
            try:
                subprocess.run(["sudo", "iwconfig", self.monitor_iface, "channel", str(channels[idx])],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                idx = (idx + 1) % len(channels)
                time.sleep(1.2)
            except:
                time.sleep(2)

    def start(self):
        if not self.setup_monitor_mode():
            return False

        self.running = True
        if self.channel_hop:
            threading.Thread(target=self._channel_hopper, daemon=True).start()

        self.thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.thread.start()
        print("🚀 Safe RF Drone Detector running... (Ctrl+C to stop)")
        print(f"   Using interface: {self.monitor_iface}")
        return True

    def _sniff_loop(self):
        try:
            sniff(iface=self.monitor_iface, prn=self._packet_handler, store=False, monitor=True)
        except Exception as e:
            if self.running:
                print(f"Sniff error: {e}")

    def stop(self):
        self.running = False
        try:
            base = self.monitor_iface.replace("mon", "")
            subprocess.run(["sudo", "airmon-ng", "stop", self.monitor_iface], 
                         capture_output=True, stderr=subprocess.DEVNULL)
            print(f"✅ Restored {base} to managed mode.")
        except:
            pass

    def print_stats(self):
        print(f"📊 Beacons: {self.stats['beacons']} | Probes: {self.stats['probes']} | Drones: {self.stats['drones']}")


def main():
    parser = argparse.ArgumentParser(description="Safe RF Drone Detector (WiFi-preserving)")
    parser.add_argument("--iface", help="Monitor interface (e.g. wlxb8fbb3c4e916mon)")
    parser.add_argument("--no-hop", action="store_true", help="Disable channel hopping")
    args = parser.parse_args()

    detector = RFDroneDetector(monitor_iface=args.iface, channel_hop=not args.no_hop)
    
    if not detector.start():
        sys.exit(1)

    try:
        while True:
            time.sleep(4)
            detector.print_stats()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping...")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()