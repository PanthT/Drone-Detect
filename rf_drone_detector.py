#!/usr/bin/env python3
"""
RF Drone Detector using RTL-SDR (Software Defined Radio)
Scans actual drone RF bands: 433 MHz, 868/915 MHz, 1.2 GHz, 2.4 GHz, 5.8 GHz
Detects by FFT power threshold — catches ANY signal on known drone frequencies,
including DJI OcuSync/Lightbridge, analog FPV, and WiFi-based drones.

Requirements:
    - RTL-SDR dongle (RTL2832U based, e.g. RTL-SDR v3/v4, Nooelec NESDR, etc.)
    - pip install pyrtlsdr numpy scipy
    - For 5.8 GHz: an upconverter (e.g. Ham It Up, SpyVerter) OR an SDR that
      tunes that high (HackRF, LimeSDR, PlutoSDR, etc.). The RTL-SDR native
      range is only ~24 MHz – 1.7 GHz.

Usage:
    # Scan all drone bands
    python rf_drone_detector.py --gain 40

    # Scan only 2.4 GHz and 5.8 GHz (requires upconverter for 5.8)
    python rf_drone_detector.py --bands 2400,5800 --gain 40

    # Continuous monitoring with CSV logging
    python rf_drone_detector.py --gain 40 --log detections.csv

    # Quick 10-second scan for testing
    python rf_drone_detector.py --gain 40 --duration 10
"""

import argparse
import time
import threading
import queue
import csv
import numpy as np
from datetime import datetime
from collections import deque

# ---------------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------------
try:
    from rtlsdr import RtlSdr
except ImportError:
    print("❌ pyrtlsdr not installed. Run: pip install pyrtlsdr")
    print("   Also: sudo apt install librtlsdr-dev")
    exit(1)

try:
    import scipy.signal as signal
except ImportError:
    signal = None

# ---------------------------------------------------------------------------
# Drone frequency bands of interest (in MHz)
# ---------------------------------------------------------------------------
# Each entry: (name, center_freq_mhz, bandwidth_mhz, description)
DRONE_BANDS = [
    ("433_RC",       433.0,   2.0,   "L-band RC control / telemetry"),
    ("868_SR",       868.0,   2.0,   "868 MHz SRD band (EU RC)"),
    ("915_ISM",      915.0,   2.0,   "915 MHz ISM band (US RC)"),
    ("1200_FPV",    1200.0,   8.0,   "1.2 GHz analog FPV video"),
    ("2400_WiFi",   2400.0,  80.0,   "2.4 GHz WiFi / RC / OcuSync 2.4"),
    ("2450_DJI",    2450.0,  20.0,   "2.45 GHz DJI OcuSync control"),
    ("5800_FPV",    5800.0,  80.0,   "5.8 GHz analog / DJI FPV"),
]

# ---- Upconverter config (for bands above RTL-SDR limit ~1.7 GHz) ----
# Standard Ham It Up uses 125 MHz offset; SpyVerter uses 120 MHz.
# Set UPCONVERTER_OFFSET = 0 to disable.
UPCONVERTER_OFFSET = 0.0   # MHz — set to 125.0 or 120.0 if using one

# ---- Detection thresholds (tunable) ----
DEFAULT_THRESHOLD_DB    = 12.0   # dB above noise floor to flag a signal
DEFAULT_BANDWIDTH_MIN   = 0.3    # MHz — ignore spikes narrower than this (likely noise)
DEFAULT_BANDWIDTH_MAX   = 60.0   # MHz — ignore signals wider than this (likely broadband noise)

# ---- Average noise floor estimation params ----
NOISE_PERCENTILE = 30    # percentile used to estimate noise floor
PEAK_PROMINENCE  = 4.0   # relative prominence for scipy peak detection


class SDRDroneDetector:
    """RTL-SDR based RF spectrum scanner for drone frequency bands."""

    def __init__(self, gain: float = 40.0, sample_rate: float = 2.4,
                 threshold_db: float = DEFAULT_THRESHOLD_DB,
                 band_bw_min: float = DEFAULT_BANDWIDTH_MIN,
                 band_bw_max: float = DEFAULT_BANDWIDTH_MAX,
                 bands: list[str] | None = None,
                 log_path: str | None = None):
        """
        Args:
            gain:          RTL-SDR gain in dB (auto = 'auto'). Typical: 20–50.
            sample_rate:   Sample rate in MS/s. 2.4 MS/s is safe for RTL-SDR.
            threshold_db:  dB above estimated noise floor to flag a detection.
            band_bw_min:   Minimum occupied bandwidth (MHz) for a valid signal.
            band_bw_max:   Maximum occupied bandwidth (MHz) to ignore.
            bands:         List of band centre frequencies in MHz. If None, all.
            log_path:      If set, append detections to this CSV file.
        """
        self.gain = gain
        self.sample_rate = int(sample_rate * 1e6)
        self.threshold_db = threshold_db
        self.band_bw_min = band_bw_min
        self.band_bw_max = band_bw_max
        self.log_path = log_path

        # Select bands
        if bands is not None:
            self.bands = [b for b in DRONE_BANDS if str(int(b[1])) in bands]
        else:
            self.bands = list(DRONE_BANDS)

        # Validate against RTL-SDR frequency limit
        self.sdr_max_mhz = 1700.0  # Native RTL-SDR limit
        for band in self.bands:
            fc = band[1]
            if fc > self.sdr_max_mhz and UPCONVERTER_OFFSET == 0.0:
                print(f"  ⚠️  {band[0]} @ {fc} MHz exceeds RTL-SDR limit "
                      f"({self.sdr_max_mhz} MHz). Use an upconverter or "
                      f"a higher-frequency SDR (HackRF, LimeSDR, etc.). "
                      f"Skipping this band.")
        self.bands = [b for b in self.bands
                      if b[1] <= self.sdr_max_mhz or UPCONVERTER_OFFSET > 0.0]

        self.running = False
        self.sdr = None
        self.detections = queue.Queue(maxsize=200)
        self.recent_logged = deque(maxlen=100)  # deduplication key cache
        self._lock = threading.Lock()
        self.stats = {"scans": 0, "detections": 0, "bands_covered": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Initialise SDR and begin scanning on background thread."""
        if not self._init_sdr():
            return False
        self.running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        print(f"\n🚀 SDR Drone Detector Started!")
        print(f"   Scanning {len(self.bands)} band(s): "
              f"{', '.join(b[0] for b in self.bands)}")
        print(f"   Sample rate: {self.sample_rate / 1e6:.1f} MS/s  |  "
              f"Gain: {self.gain} dB")
        print(f"   Threshold: {self.threshold_db:.1f} dB above noise floor\n")
        print("   Press Ctrl+C to stop.\n")
        return True

    def stop(self):
        """Stop scanning and release SDR resources."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._close_sdr()

    def get_detections(self) -> list[dict]:
        """Drain all pending detections from the internal queue."""
        dets = []
        while not self.detections.empty():
            dets.append(self.detections.get_nowait())
        return dets

    def print_stats(self):
        d = self.stats
        print(f"\n📊 Stats — Scans: {d['scans']} | Bands/scan: {d['bands_covered']} "
              f"| Detections: {d['detections']}")

    # ------------------------------------------------------------------
    # SDR initialisation / teardown
    # ------------------------------------------------------------------
    def _init_sdr(self) -> bool:
        print("🔧 Initialising RTL-SDR...", end=" ", flush=True)
        try:
            self.sdr = RtlSdr()
            # device_index 0 = first dongle found
            self.sdr.sample_rate = self.sample_rate
            if self.gain != 'auto':
                self.sdr.gain = float(self.gain)
            else:
                self.sdr.gain = 'auto'
            print("✅")
            return True
        except Exception as e:
            print(f"\n❌ RTL-SDR init failed: {e}")
            print("   Check: dongle plugged in? Permissions? (sudo?)  "
                  "Other apps using it?")
            return False

    def _close_sdr(self):
        if self.sdr:
            try:
                self.sdr.close()
            except Exception:
                pass
            self.sdr = None

    # ------------------------------------------------------------------
    # Main scan loop
    # ------------------------------------------------------------------
    def _scan_loop(self):
        while self.running:
            for band in self.bands:
                if not self.running:
                    break
                self._scan_band(band)
                self.stats["bands_covered"] += 1
            self.stats["scans"] += 1

    def _scan_band(self, band: tuple):
        """Single band scan: tune, capture, FFT, detect peaks."""
        name, fc_mhz, bw_mhz, desc = band

        # Compute tune frequency (apply upconverter offset)
        tune_mhz = fc_mhz - UPCONVERTER_OFFSET if UPCONVERTER_OFFSET > 0 else fc_mhz
        tune_hz = int(tune_mhz * 1e6)

        # Number of FFT points — determines frequency resolution
        n_fft = 1024

        # Half the bandwidth we want to capture
        half_bw = bw_mhz / 2.0
        fc_min = fc_mhz - half_bw
        fc_max = fc_mhz + half_bw

        try:
            self.sdr.center_freq = tune_hz
            # Capture raw IQ samples
            samples = self.sdr.read_samples(256 * 1024)

            # Compute power spectral density via Welch's method
            if signal is not None:
                freqs, psd = signal.welch(
                    samples,
                    fs=self.sample_rate,
                    nperseg=n_fft,
                    scaling='density',
                    return_onesided=False
                )
                # Shift zero-frequency component to centre
                freqs = np.fft.fftshift(freqs)
                psd = np.fft.fftshift(psd)
                # Convert to MHz relative to centre
                freqs_mhz = tune_mhz + (freqs - self.sample_rate / 2) / 1e6
            else:
                # Fallback: simple FFT
                spectrum = np.fft.fftshift(np.fft.fft(samples, n=n_fft))
                psd = np.abs(spectrum) ** 2
                freqs_mhz = tune_mhz + \
                    (np.arange(-n_fft // 2, n_fft // 2) * self.sample_rate / n_fft) / 1e6

            # Convert PSD to dB
            psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))

            # Estimate noise floor (percentile of all PSD values)
            noise_floor = float(np.percentile(psd_db, NOISE_PERCENTILE))
            threshold = noise_floor + self.threshold_db

            # Find peaks above threshold
            if signal is not None:
                peaks, properties = signal.find_peaks(
                    psd_db,
                    height=threshold,
                    prominence=PEAK_PROMINENCE,
                    width=None   # we'll compute bandwidth manually
                )
            else:
                # Manual peak detection fallback
                peaks = []
                for i in range(1, len(psd_db) - 1):
                    if psd_db[i] > threshold and \
                       psd_db[i] > psd_db[i-1] and \
                       psd_db[i] > psd_db[i+1]:
                        peaks.append(i)

            # Process each detected peak
            for p_idx in peaks:
                peak_db = psd_db[p_idx]
                peak_freq_mhz = freqs_mhz[p_idx]

                # Only consider signals within the intended band
                if not (fc_min <= peak_freq_mhz <= fc_max):
                    continue

                # Estimate occupied bandwidth around the peak
                half_max = peak_db - 3.0  # 3 dB down from peak
                left = p_idx
                while left > 0 and psd_db[left] > half_max:
                    left -= 1
                right = p_idx
                while right < len(psd_db) - 1 and psd_db[right] > half_max:
                    right += 1
                occ_bw_mhz = (freqs_mhz[right] - freqs_mhz[left]) if right > left else 0.0

                # Filter by plausible bandwidth
                if occ_bw_mhz < self.band_bw_min or occ_bw_mhz > self.band_bw_max:
                    continue

                # Build detection record
                det = {
                    "timestamp":  datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "band":       name,
                    "band_desc":  desc,
                    "frequency":  f"{peak_freq_mhz:.3f} MHz",
                    "freq_hz":    int(peak_freq_mhz * 1e6),
                    "power_db":   f"{peak_db:.1f} dB",
                    "noise_floor": f"{noise_floor:.1f} dB",
                    "snr_db":     f"{peak_db - noise_floor:.1f} dB",
                    "bw_mhz":     f"{occ_bw_mhz:.2f} MHz",
                }

                # Deduplicate (same band + similar freq within ~200 kHz)
                dedup_key = f"{name}_{int(peak_freq_mhz * 10)}"
                if dedup_key not in self.recent_logged:
                    self.recent_logged.append(dedup_key)
                    self.detections.put(det)
                    self.stats["detections"] += 1
                    self._print_detection(det)

            # Log if path set
            if self.log_path and not self.detections.empty():
                self._flush_log()

        except Exception as e:
            if self.running:
                print(f"⚠️  Scan error on {name} @ {fc_mhz} MHz: {e}")

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _print_detection(self, d: dict):
        print(f"🚁 {d['timestamp']} | "
              f"{d['band']:>10s} | "
              f"Freq: {d['frequency']:>10s} | "
              f"SNR: {d['snr_db']:>6s} | "
              f"BW: {d['bw_mhz']:>6s} | "
              f"{d['band_desc']}")

    def _flush_log(self):
        dets = self.get_detections()
        if not dets:
            return
        try:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=dets[0].keys())
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(dets)
        except Exception as e:
            print(f"⚠️  Log write error: {e}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SDR-based RF Drone Detector (RTL-SDR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Full auto-gain scan across all bands
  python {__file__}

  # Fixed gain, quick 10-second sweep
  python {__file__} --gain 40 --duration 10

  # Scan only 2.4 GHz + 5.8 GHz bands (needs upconverter for 5.8)
  python {__file__} --bands 2400,5800 --upconverter 125

  # Log detections to CSV for later analysis
  python {__file__} --gain 40 --log drone_log.csv

  # Lower threshold for more sensitivity (more false positives)
  python {__file__} --gain 40 --threshold 8

Available bands:
{chr(10).join(f'  {b[1]:>6.0f} MHz  {b[0]:<10s}  {b[3]}' for b in DRONE_BANDS)}
        """
    )

    parser.add_argument("--gain", type=str, default="40",
                        help="SDR gain in dB, or 'auto' (default: 40)")
    parser.add_argument("--sample-rate", type=float, default=2.4,
                        help="Sample rate in MS/s (default: 2.4)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_DB,
                        help=f"Detection threshold in dB above noise floor "
                             f"(default: {DEFAULT_THRESHOLD_DB})")
    parser.add_argument("--bandwidth-min", type=float, default=DEFAULT_BANDWIDTH_MIN,
                        help=f"Minimum signal bandwidth in MHz (default: "
                             f"{DEFAULT_BANDWIDTH_MIN})")
    parser.add_argument("--bandwidth-max", type=float, default=DEFAULT_BANDWIDTH_MAX,
                        help=f"Maximum signal bandwidth in MHz (default: "
                             f"{DEFAULT_BANDWIDTH_MAX})")
    parser.add_argument("--bands", type=str, default=None,
                        help="Comma-separated band centre freqs in MHz, "
                             "e.g. '433,2400,5800'")
    parser.add_argument("--upconverter", type=float, default=0.0,
                        help="Upconverter LO offset in MHz (e.g. 125 for Ham It Up)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Run for N seconds (0 = infinite)")
    parser.add_argument("--log", type=str, default=None,
                        help="CSV file path to log detections")
    parser.add_argument("--stats-interval", type=float, default=10.0,
                        help="Seconds between stats printouts (default: 10)")

    args = parser.parse_args()

    # Parse gain
    gain: float | str = args.gain
    if gain.lower() == 'auto':
        gain = 'auto'
    else:
        gain = float(gain)

    # Parse band selection
    selected_bands = None
    if args.bands:
        selected_bands = [b.strip() for b in args.bands.split(",")]

    # Allow upconverter to be set via CLI
    global UPCONVERTER_OFFSET
    if args.upconverter > 0:
        UPCONVERTER_OFFSET = args.upconverter
        print(f"🔧 Upconverter enabled: {UPCONVERTER_OFFSET} MHz offset")
        print(f"   Bands from {args.upconverter + 24} MHz are now reachable\n")

    detector = SDRDroneDetector(
        gain=gain,
        sample_rate=args.sample_rate,
        threshold_db=args.threshold,
        band_bw_min=args.bandwidth_min,
        band_bw_max=args.bandwidth_max,
        bands=selected_bands,
        log_path=args.log,
    )

    if not detector.start():
        exit(1)

    try:
        start_time = time.time()
        last_stats = time.time()
        while True:
            time.sleep(1)

            # Print stats every N seconds
            if time.time() - last_stats > args.stats_interval:
                detector.print_stats()
                last_stats = time.time()

            # Duration limit
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                print(f"\n⏱️  Reached --duration={args.duration}s limit.")
                break

    except KeyboardInterrupt:
        print("\n⏹️  Stopping...")
    finally:
        detector.stop()
        detector.print_stats()
        print("✅ Detector stopped.")

        # Final flush of any remaining detections to log
        remaining = detector.get_detections()
        if remaining and args.log:
            print(f"📝 Writing {len(remaining)} remaining detection(s) to log...")
            try:
                with open(args.log, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=remaining[0].keys())
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerows(remaining)
            except Exception as e:
                print(f"⚠️  Log write error: {e}")
            print(f"   Logged to: {args.log}")


if __name__ == "__main__":
    main()