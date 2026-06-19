#!/usr/bin/env python3
"""
Acoustic + RF Drone Detector
Optimized for NVIDIA Jetson Nano with TP-Link (wlxb8fbb3c4e916).

Combines two independent detection methods:
  1. ACOUSTIC — microphone FFT analysis for drone motor/propeller signatures
  2. RF       — WiFi monitor mode (Scapy) for drone control/video SSIDs & OUIs

Either method alone can trigger a detection. Both run in parallel threads.

Acoustic detection works by identifying the harmonic frequency pattern that
multi-rotor drones produce. Each motor spins a propeller, generating a
fundamental tone plus harmonics. Consumer drones typically fall in these bands:
  - Propeller fundamental:  80–300 Hz  (blade-pass frequency)
  - Motor/ESC harmonics:   300–800 Hz
  - Frame resonance:       800–2000 Hz
A drone is flagged when >= MIN_HARMONIC_PEAKS evenly-spaced peaks appear in
the FFT with consistent spacing (harmonic series) and sufficient energy ratio.

Requirements:
    pip3 install scapy pyaudio numpy scipy
    sudo apt update && sudo apt install aircrack-ng portaudio19-dev

Usage:
    sudo python3 acoustic_drone_detector.py
    sudo python3 acoustic_drone_detector.py --iface wlxb8fbb3c4e916mon --audio-device 0
    sudo python3 acoustic_drone_detector.py --list-audio   # list microphone devices
    sudo python3 acoustic_drone_detector.py --no-rf        # acoustic only
    sudo python3 acoustic_drone_detector.py --no-acoustic  # RF only
"""

import argparse
import time
import threading
import queue
import subprocess
from datetime import datetime
from collections import deque
import sys
import math

# ── RF dependencies ──────────────────────────────────────────────────────────
try:
    from scapy.all import sniff, Dot11, Dot11Beacon, Dot11ProbeReq, Dot11ProbeResp
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False
    print("⚠️  Scapy not installed — RF detection disabled.")
    print("   Run: pip3 install scapy && sudo apt install aircrack-ng")

# ── Acoustic dependencies ────────────────────────────────────────────────────
try:
    import numpy as np
    from scipy.signal import find_peaks
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️  numpy/scipy not installed — acoustic detection disabled.")
    print("   Run: pip3 install numpy scipy")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  PyAudio not installed — acoustic detection disabled.")
    print("   Run: pip3 install pyaudio  (needs: sudo apt install portaudio19-dev)")


# ═══════════════════════════════════════════════════════════════════════════════
# Acoustic constants  — tune these for your environment
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE       = 44100   # Hz — standard mic sample rate
CHUNK_SIZE        = 4096    # samples per FFT frame (~93 ms at 44100 Hz)
AUDIO_CHANNELS    = 1       # mono

# Frequency window where drone propeller fundamentals live (Hz)
DRONE_FREQ_LOW    = 80
DRONE_FREQ_HIGH   = 800     # narrowed: drone fundamentals only, avoids USB/electrical HF noise

# A peak must be this many dB above the local noise floor to count
PEAK_PROMINENCE   = 20      # dB  (raised from 12 — peaks must really stand out)

# How many harmonic peaks we need to classify as a drone
MIN_HARMONIC_PEAKS = 5      # (raised from 3 — requires a fuller harmonic series)

# Maximum allowed deviation between measured harmonic spacing and ideal (%)
HARMONIC_TOLERANCE = 0.10   # 10 %  (tightened from 15 — less tolerance for irregular spacing)

# The total energy in the drone band must be this fraction of full-spectrum
# energy (filters out silence / very quiet background)
MIN_ENERGY_RATIO  = 0.35    # (raised from 0.15 — drone must dominate the spectrum)

# How many consecutive positive frames before we fire an alert
CONFIRM_FRAMES    = 4       # (raised from 2 — signal must persist ~370 ms)

# Ring buffer length for acoustic detections displayed in the summary
ACOUSTIC_HISTORY  = 30


# ═══════════════════════════════════════════════════════════════════════════════
# Acoustic detector
# ═══════════════════════════════════════════════════════════════════════════════

class AcousticDroneDetector:
    """
    Listens to a microphone and analyses each audio frame with an FFT.

    Detection algorithm:
      1. Compute magnitude spectrum (dB) of the incoming chunk.
      2. Find all prominent peaks in the drone frequency band (80–2000 Hz).
      3. For each candidate fundamental frequency F, count how many of its
         harmonics (2F, 3F, 4F, …) also appear as peaks within HARMONIC_TOLERANCE.
      4. If >= MIN_HARMONIC_PEAKS harmonics are found AND the band energy ratio
         exceeds MIN_ENERGY_RATIO, increment a confirmation counter.
      5. Alert fires when CONFIRM_FRAMES consecutive frames all pass.
    """

    def __init__(self, device_index=None, detections_queue=None):
        self.device_index   = device_index
        self.detections     = detections_queue or queue.Queue(maxsize=50)
        self.running        = False
        self._thread        = None
        self._confirm_count = 0          # consecutive positive frames
        self.recent_detections = deque(maxlen=ACOUSTIC_HISTORY)
        self.stats = {
            "frames_analysed": 0,
            "positive_frames": 0,
            "alerts":          0,
        }

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        if not (NUMPY_AVAILABLE and PYAUDIO_AVAILABLE):
            print("❌ Acoustic detection unavailable (missing numpy/scipy/pyaudio).")
            return False
        self.running = True
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        print("🎙️  Acoustic drone detector running...")
        return True

    def stop(self):
        self.running = False

    # ── internal ──────────────────────────────────────────────────────────────

    def _audio_loop(self):
        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream = pa.open(
                format            = pyaudio.paInt16,
                channels          = AUDIO_CHANNELS,
                rate              = SAMPLE_RATE,
                input             = True,
                input_device_index= self.device_index,
                frames_per_buffer = CHUNK_SIZE,
            )
            print(f"   Microphone open (device {self.device_index or 'default'}, "
                  f"{SAMPLE_RATE} Hz, chunk {CHUNK_SIZE})")
            while self.running:
                try:
                    raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    self._analyse_frame(raw)
                except OSError as e:
                    if self.running:
                        print(f"⚠️  Audio read error: {e}")
                    time.sleep(0.1)
        except Exception as e:
            print(f"❌ Could not open microphone: {e}")
            print("   Check device index with --list-audio")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            pa.terminate()

    def _analyse_frame(self, raw_bytes: bytes):
        """Core DSP: FFT → peak-finding → harmonic-series check."""
        self.stats["frames_analysed"] += 1

        # Convert int16 PCM → float32 normalised to [-1, 1]
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        samples /= 32768.0

        # Apply a Hann window to reduce spectral leakage
        window  = np.hanning(len(samples))
        samples = samples * window

        # FFT magnitude in dB
        spectrum  = np.fft.rfft(samples)
        magnitude = np.abs(spectrum)
        # Avoid log(0)
        magnitude = np.where(magnitude < 1e-10, 1e-10, magnitude)
        db        = 20.0 * np.log10(magnitude)

        # Frequency axis
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / SAMPLE_RATE)

        # ── Energy ratio check ──────────────────────────────────────────────
        total_energy = np.sum(magnitude ** 2)
        band_mask    = (freqs >= DRONE_FREQ_LOW) & (freqs <= DRONE_FREQ_HIGH)
        band_energy  = np.sum(magnitude[band_mask] ** 2)
        energy_ratio = band_energy / (total_energy + 1e-10)

        if energy_ratio < MIN_ENERGY_RATIO:
            self._confirm_count = 0
            return

        # ── Peak finding in drone band ───────────────────────────────────────
        band_db    = db[band_mask]
        band_freqs = freqs[band_mask]

        # find_peaks works on the band-limited spectrum
        peaks_idx, props = find_peaks(
            band_db,
            prominence = PEAK_PROMINENCE,
            distance   = int(len(band_freqs) * 0.02),  # ≥2% band apart
        )

        if len(peaks_idx) < MIN_HARMONIC_PEAKS:
            self._confirm_count = 0
            return

        peak_freqs = band_freqs[peaks_idx]

        # ── Harmonic series check ────────────────────────────────────────────
        best_count    = 0
        best_fund     = 0.0

        for candidate_fund in peak_freqs:
            if candidate_fund < DRONE_FREQ_LOW:
                continue
            count = 1   # the fundamental itself counts
            for harmonic_n in range(2, 10):
                expected = candidate_fund * harmonic_n
                if expected > DRONE_FREQ_HIGH:
                    break
                # Is there a peak within tolerance of this harmonic?
                tolerance_hz = expected * HARMONIC_TOLERANCE
                close = np.any(np.abs(peak_freqs - expected) <= tolerance_hz)
                if close:
                    count += 1
            if count > best_count:
                best_count = count
                best_fund  = candidate_fund

        is_drone = best_count >= MIN_HARMONIC_PEAKS

        if is_drone:
            self.stats["positive_frames"] += 1
            self._confirm_count += 1
        else:
            self._confirm_count = 0

        # ── Fire alert after CONFIRM_FRAMES consecutive positives ────────────
        if self._confirm_count >= CONFIRM_FRAMES:
            self._confirm_count = 0   # reset so we don't spam
            self.stats["alerts"] += 1

            detection = {
                "source":      "ACOUSTIC",
                "timestamp":   datetime.now().strftime("%H:%M:%S"),
                "fundamental": f"{best_fund:.1f} Hz",
                "harmonics":   best_count,
                "energy_pct":  f"{energy_ratio * 100:.1f}%",
                "detail":      (f"Harmonic series at {best_fund:.0f} Hz "
                                f"({best_count} harmonics, "
                                f"band energy {energy_ratio*100:.1f}%)"),
            }

            key = f"acoustic_{int(best_fund)}"
            if key not in self.recent_detections:
                self.recent_detections.append(key)
                try:
                    self.detections.put_nowait(detection)
                except queue.Full:
                    pass
                print(f"🔊 DRONE DETECTED (ACOUSTIC)! "
                      f"Fund: {best_fund:.0f} Hz | "
                      f"Harmonics: {best_count} | "
                      f"Band energy: {energy_ratio*100:.1f}%")

    def print_stats(self):
        s = self.stats
        if s["frames_analysed"] > 0:
            hit_rate = 100.0 * s["positive_frames"] / s["frames_analysed"]
        else:
            hit_rate = 0.0
        print(f"   🎙️  Frames: {s['frames_analysed']} | "
              f"Positive: {s['positive_frames']} ({hit_rate:.1f}%) | "
              f"Alerts: {s['alerts']}")


# ═══════════════════════════════════════════════════════════════════════════════
# RF detector  (unchanged logic, merged into shared detection queue)
# ═══════════════════════════════════════════════════════════════════════════════

class RFDroneDetector:
    def __init__(self, monitor_iface=None, channel_hop=True, detections_queue=None):
        self.channel_hop = channel_hop
        self.running     = False
        self._thread     = None
        self.detections  = detections_queue or queue.Queue(maxsize=50)

        self.drone_ssids = ["DJI_", "PHANTOM", "MAVIC", "AIR", "MINI", "PARROT",
                            "BEBOP", "ANAFI", "FPV_", "DRONE", "RC_", "SKY", "TELLO"]
        self.drone_ouis  = ["60:60:1F", "48:1C:B9", "34:D2:62", "A0:14:3D",
                            "90:3A:E6", "18:60:24", "BC:8B:B2", "C4:7C:8D"]

        self.recent_drones = deque(maxlen=30)
        self.stats = {"beacons": 0, "probes": 0, "drones": 0}

        self.monitor_iface = monitor_iface or self._find_usb_wifi_adapter()

    def _find_usb_wifi_adapter(self):
        try:
            lsusb_out = subprocess.check_output(["lsusb"], text=True,
                                                 stderr=subprocess.DEVNULL)
            if "2357:0138" in lsusb_out or "TP-Link" in lsusb_out:
                print("✅ TP-Link USB WiFi adapter detected")

            out = subprocess.check_output(["iwconfig"], text=True,
                                           stderr=subprocess.DEVNULL)
            wireless_ifaces = []
            for line in out.splitlines():
                if line.strip() and not line.startswith(" "):
                    iface = line.split()[0].strip()
                    if iface in ["lo", "can0", "l4tbr0", "usb0", "usb1",
                                 "enP8p1s0", "br-", "docker0"]:
                        continue
                    if "no wireless extensions" not in line:
                        wireless_ifaces.append(iface)

            for iface in wireless_ifaces:
                if iface.startswith("wlx") or (iface.startswith("wlan")
                                                and iface != "wlan0"):
                    print(f"✅ Using USB WiFi adapter: {iface}")
                    return f"{iface}mon"
            if wireless_ifaces:
                return f"{wireless_ifaces[0]}mon"
            return "wlan0mon"
        except Exception as e:
            print(f"⚠️  Adapter detection issue: {e}")
            return "wlan0mon"

    def setup_monitor_mode(self):
        print(f"🔧 Setting up monitor mode on {self.monitor_iface}...")
        try:
            subprocess.run(["sudo", "airmon-ng", "check", "kill"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            base = self.monitor_iface.replace("mon", "")
            subprocess.run(["sudo", "airmon-ng", "start", base],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            iw_out = subprocess.check_output(["iwconfig"], text=True,
                                              stderr=subprocess.DEVNULL)
            if "Mode:Monitor" in iw_out or self.monitor_iface in iw_out:
                print(f"✅ Monitor mode ready on {self.monitor_iface}")
                return True
            print("⚠️  Monitor mode may not be active.")
            return False
        except Exception as e:
            print(f"❌ Monitor mode setup failed: {e}")
            return False

    def _packet_handler(self, pkt):
        if not pkt.haslayer(Dot11):
            return
        try:
            mac, ssid = None, None

            if pkt.haslayer(Dot11Beacon):
                self.stats["beacons"] += 1
                ssid = pkt[Dot11Beacon].info.decode(errors='ignore').strip()
                mac  = pkt.addr2
            elif pkt.haslayer(Dot11ProbeReq) or pkt.haslayer(Dot11ProbeResp):
                self.stats["probes"] += 1
                if pkt.haslayer(Dot11ProbeReq):
                    ssid = pkt[Dot11ProbeReq].info.decode(errors='ignore').strip()
                mac = pkt.addr2 or pkt.addr1

            if not mac:
                return

            mac_upper = mac.upper()
            oui       = mac_upper[:8]
            is_drone  = (
                any(mac_upper.startswith(o) for o in self.drone_ouis) or
                (ssid and any(d in ssid.upper() for d in self.drone_ssids))
            )

            if is_drone:
                self.stats["drones"] += 1
                key = f"{mac}_{ssid}"
                if key not in self.recent_drones:
                    self.recent_drones.append(key)
                    detection = {
                        "source":    "RF",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "mac":       mac,
                        "ssid":      ssid or "Unknown",
                        "type":      "Beacon" if pkt.haslayer(Dot11Beacon) else "Probe",
                        "signal":    getattr(pkt, 'dBm_AntSignal', 'N/A'),
                        "detail":    (f"MAC {mac} | SSID '{ssid or 'Unknown'}' | "
                                      f"Signal {getattr(pkt, 'dBm_AntSignal', 'N/A')}"),
                    }
                    try:
                        self.detections.put_nowait(detection)
                    except queue.Full:
                        pass
                    print(f"📡 DRONE DETECTED (RF)! "
                          f"{mac} | {ssid} | "
                          f"Signal: {detection['signal']}")
        except Exception:
            pass

    def _channel_hopper(self):
        channels = list(range(1, 14))
        idx = 0
        while self.running:
            try:
                subprocess.run(
                    ["sudo", "iwconfig", self.monitor_iface,
                     "channel", str(channels[idx])],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                idx = (idx + 1) % len(channels)
                time.sleep(1.2)
            except Exception:
                time.sleep(2)

    def _sniff_loop(self):
        try:
            sniff(iface=self.monitor_iface, prn=self._packet_handler,
                  store=False, monitor=True)
        except Exception as e:
            if self.running:
                print(f"Sniff error: {e}")

    def start(self):
        if not RF_AVAILABLE:
            print("❌ RF detection unavailable (Scapy not installed).")
            return False
        if not self.setup_monitor_mode():
            return False
        self.running = True
        if self.channel_hop:
            threading.Thread(target=self._channel_hopper, daemon=True).start()
        self._thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self._thread.start()
        print(f"📡 RF drone detector running on {self.monitor_iface}...")
        return True

    def stop(self):
        self.running = False
        try:
            subprocess.run(["sudo", "airmon-ng", "stop", self.monitor_iface],
                           capture_output=True, stderr=subprocess.DEVNULL)
            base = self.monitor_iface.replace("mon", "")
            print(f"✅ Restored {base} to managed mode.")
        except Exception:
            pass

    def print_stats(self):
        s = self.stats
        print(f"   📡 Beacons: {s['beacons']} | "
              f"Probes: {s['probes']} | RF drones: {s['drones']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def list_audio_devices():
    """Print available PyAudio input devices and exit."""
    if not PYAUDIO_AVAILABLE:
        print("❌ PyAudio not installed.")
        sys.exit(1)
    import pyaudio
    pa = pyaudio.PyAudio()
    print("\nAvailable audio INPUT devices:")
    print(f"  {'Index':<6} {'Name':<40} {'Channels':<10} {'Sample Rate'}")
    print("  " + "-" * 70)
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  {i:<6} {info['name'][:39]:<40} "
                  f"{int(info['maxInputChannels']):<10} "
                  f"{int(info['defaultSampleRate'])} Hz")
    pa.terminate()
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global DRONE_FREQ_LOW, DRONE_FREQ_HIGH, PEAK_PROMINENCE
    global MIN_HARMONIC_PEAKS, CONFIRM_FRAMES

    parser = argparse.ArgumentParser(
        description="Acoustic + RF Drone Detector (Jetson Nano)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sudo python3 acoustic_drone_detector.py
  sudo python3 acoustic_drone_detector.py --list-audio
  sudo python3 acoustic_drone_detector.py --audio-device 1
  sudo python3 acoustic_drone_detector.py --no-rf
  sudo python3 acoustic_drone_detector.py --no-acoustic
  sudo python3 acoustic_drone_detector.py --iface wlxb8fbb3c4e916mon

Acoustic tuning flags (advanced):
  --freq-low / --freq-high   : drone band Hz  (default 80–2000)
  --prominence               : peak prominence threshold dB  (default 12)
  --min-harmonics            : harmonics required  (default 3)
  --confirm-frames           : consecutive positive frames to alert  (default 2)
        """
    )
    parser.add_argument("--iface",           help="Monitor interface (RF)")
    parser.add_argument("--no-hop",          action="store_true",
                        help="Disable WiFi channel hopping (RF)")
    parser.add_argument("--no-rf",           action="store_true",
                        help="Disable RF detection entirely")
    parser.add_argument("--no-acoustic",     action="store_true",
                        help="Disable acoustic detection entirely")
    parser.add_argument("--list-audio",      action="store_true",
                        help="List microphone devices and exit")
    parser.add_argument("--audio-device",    type=int, default=None,
                        help="PyAudio device index for microphone (default: system default)")
    parser.add_argument("--freq-low",        type=int, default=DRONE_FREQ_LOW,
                        help=f"Low end of drone frequency band Hz (default {DRONE_FREQ_LOW})")
    parser.add_argument("--freq-high",       type=int, default=DRONE_FREQ_HIGH,
                        help=f"High end of drone frequency band Hz (default {DRONE_FREQ_HIGH})")
    parser.add_argument("--prominence",      type=float, default=PEAK_PROMINENCE,
                        help=f"FFT peak prominence dB (default {PEAK_PROMINENCE})")
    parser.add_argument("--min-harmonics",   type=int, default=MIN_HARMONIC_PEAKS,
                        help=f"Min harmonic peaks to classify as drone (default {MIN_HARMONIC_PEAKS})")
    parser.add_argument("--confirm-frames",  type=int, default=CONFIRM_FRAMES,
                        help=f"Consecutive positive frames before alert (default {CONFIRM_FRAMES})")
    args = parser.parse_args()

    if args.list_audio:
        list_audio_devices()
        sys.exit(0)

    # Apply tunable overrides
    DRONE_FREQ_LOW    = args.freq_low
    DRONE_FREQ_HIGH   = args.freq_high
    PEAK_PROMINENCE   = args.prominence
    MIN_HARMONIC_PEAKS = args.min_harmonics
    CONFIRM_FRAMES    = args.confirm_frames

    if args.no_rf and args.no_acoustic:
        print("❌ Both --no-rf and --no-acoustic specified — nothing to do.")
        sys.exit(1)

    print("=" * 60)
    print("  Acoustic + RF Drone Detector")
    print("=" * 60)

    # Shared detection queue — both detectors write here
    shared_queue = queue.Queue(maxsize=100)

    rf_detector       = None
    acoustic_detector = None

    if not args.no_rf:
        rf_detector = RFDroneDetector(
            monitor_iface   = args.iface,
            channel_hop     = not args.no_hop,
            detections_queue= shared_queue,
        )
        rf_started = rf_detector.start()
    else:
        print("ℹ️  RF detection disabled.")
        rf_started = False

    if not args.no_acoustic:
        acoustic_detector = AcousticDroneDetector(
            device_index     = args.audio_device,
            detections_queue = shared_queue,
        )
        acoustic_started = acoustic_detector.start()
    else:
        print("ℹ️  Acoustic detection disabled.")
        acoustic_started = False

    if not rf_started and not acoustic_started:
        print("❌ No detectors started. Exiting.")
        sys.exit(1)

    print("\n🚀 Detector running — press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(5)
            print(f"\n── Stats @ {datetime.now().strftime('%H:%M:%S')} ──")
            if rf_detector and rf_started:
                rf_detector.print_stats()
            if acoustic_detector and acoustic_started:
                acoustic_detector.print_stats()

            # Drain and display any new detections
            while not shared_queue.empty():
                det = shared_queue.get_nowait()
                src = det.get("source", "?")
                ts  = det.get("timestamp", "")
                print(f"  [{ts}] [{src}] {det.get('detail', '')}")

    except KeyboardInterrupt:
        print("\n⏹️  Stopping...")
    finally:
        if rf_detector:
            rf_detector.stop()
        if acoustic_detector:
            acoustic_detector.stop()
        print("✅ Done.")


if __name__ == "__main__":
    main()
