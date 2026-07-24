#!/usr/bin/env python3
"""
Acoustic Drone Detector
Optimized for NVIDIA Jetson Nano with microphone FFT analysis.

Acoustic detection works by identifying the harmonic frequency pattern that
multi-rotor drones produce. Each motor spins a propeller, generating a
fundamental tone plus harmonics. Consumer drones typically fall in these bands:
  - Propeller fundamental:  80–300 Hz  (blade-pass frequency)
  - Motor/ESC harmonics:   300–800 Hz
  - Frame resonance:       800–2000 Hz
A drone is flagged when >= MIN_HARMONIC_PEAKS evenly-spaced peaks appear in
the FFT with consistent spacing (harmonic series) and sufficient energy ratio.

Requirements:
    pip3 install pyaudio numpy scipy
    sudo apt update && sudo apt install portaudio19-dev

Usage:
    sudo python3 acoustic_drone_detector.py
    sudo python3 acoustic_drone_detector.py --audio-device 0
    sudo python3 acoustic_drone_detector.py --list-audio   # list microphone devices
"""

import argparse
import time
import threading
import queue
from datetime import datetime
from collections import deque
import sys
import warnings
import json

# ── Acoustic dependencies ────────────────────────────────────────────────────
NUMPY_AVAILABLE = False
SCIPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️  NumPy not installed — acoustic detection disabled.")
    print("   Run: pip3 install numpy")
except Exception as e:
    print("⚠️  NumPy import failed — acoustic detection disabled.")
    print(f"   Error: {e}")
    print("   Run: pip3 install numpy")

def _numpy_find_peaks(x, prominence=1, distance=1):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.array([], dtype=int), {}

    distance = max(1, int(distance))
    peaks = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            peaks.append(i)

    filtered = []
    last = -distance - 1
    for idx in peaks:
        if idx - last < distance:
            continue
        left = max(0, idx - distance)
        right = min(len(x), idx + distance + 1)
        left_min = np.min(x[left:idx + 1]) if idx > 0 else x[idx]
        right_min = np.min(x[idx:right]) if idx < len(x) - 1 else x[idx]
        baseline = max(left_min, right_min)
        if x[idx] - baseline >= prominence:
            filtered.append(idx)
            last = idx

    return np.array(filtered, dtype=int), {}

if NUMPY_AVAILABLE:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from scipy.signal import find_peaks
        SCIPY_AVAILABLE = True
    except Exception as e:
        print("⚠️  SciPy unavailable — falling back to NumPy peak detector.")
        print(f"   Error: {e}")
        print("   If you want the higher-quality peak finder, install a compatible SciPy:")
        print("   pip3 install scipy")
        SCIPY_AVAILABLE = False
        find_peaks = _numpy_find_peaks

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  PyAudio not installed — acoustic detection disabled.")
    print("   Run: pip3 install pyaudio  (needs: sudo apt install portaudio19-dev)")
except Exception as e:
    PYAUDIO_AVAILABLE = False
    print("⚠️  PyAudio import failed — acoustic detection disabled.")
    print(f"   Error: {e}")
    print("   Run: pip3 install pyaudio  (needs: sudo apt install portaudio19-dev)")


# ═══════════════════════════════════════════════════════════════════════════════
# Acoustic constants  — tune these for your environment
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE       = 44100   # Hz — standard mic sample rate
CHUNK_SIZE        = 4096    # samples per FFT frame (~93 ms at 44100 Hz)
AUDIO_CHANNELS    = 1       # mono

# Notes:
# - CHUNK_SIZE controls time vs frequency resolution for the FFT. Larger
#   chunks give finer frequency resolution but more latency; 4096 is a
#   reasonable balance for short drone blade tones in-room.
# Frequency window where drone propeller fundamentals and motor harmonics live (Hz)
DRONE_FREQ_LOW    = 80
DRONE_FREQ_HIGH   = 2000    # broader band to capture fundamentals, harmonics, and frame resonance

# A peak must be this many dB above the local noise floor to count
PEAK_PROMINENCE   = 12      # dB  (default sensitivity for tonal drone peaks)

# How many harmonic peaks we need to classify as a drone
MIN_HARMONIC_PEAKS = 3      # requires a fuller harmonic series, but allows quieter drones

# Maximum allowed deviation between measured harmonic spacing and ideal (%)
HARMONIC_TOLERANCE = 0.15   # 15 %  (allows more real-world harmonic drift)

# The total energy in the drone band must be this fraction of full-spectrum
# energy (filters out silence / very quiet background)
MIN_ENERGY_RATIO  = 0.15    # (lowered to accept quieter drone tones)

# ── Decibel-level thresholds specific to drone acoustic signatures ──────────
# Drones produce a characteristic loudness range; these filters help reject
# both quiet noise and non-drone loud sounds. Values are in dBFS (decibels
# relative to full scale), which varies with mic gain — adjust for your setup.
MIN_BAND_DB_LEVEL     = -45   # dBFS — minimum average level in drone band
MAX_BAND_DB_LEVEL     = -5    # dBFS — maximum average level (clipping guard)

# Harmonic dB rolloff ranges for drone verification. Real drone harmonics
# decay at roughly 3–10 dB per step; inconsistent rolloff (e.g., a high
# harmonic being louder than a lower one) may indicate non-drone RFI or
# mechanical noise rather than a genuine multi-rotor harmonic series.
MIN_HARMONIC_DB_ROLLOFF = 2.0   # dB minimum drop per harmonic step
MAX_HARMONIC_DB_ROLLOFF = 14.0  # dB maximum drop per harmonic step

# Minimum dB prominence of the strongest harmonic peak above the noise floor.
# A genuine drone will have at least one strong tonal peak standing clearly
# above the background — this filters out diffuse broadband noise.
MIN_PEAK_DB_ABOVE_NOISE = 8.0   # dB

# Acoustic debug prints extra FFT/peak information to help tune detection
ACOUSTIC_DEBUG        = False
ACOUSTIC_SPECTROGRAM  = False

# How many consecutive positive frames before we fire an alert
CONFIRM_FRAMES    = 2       # shorter confirmation window for faster detection

# Ring buffer length for acoustic detections displayed in the summary
ACOUSTIC_HISTORY  = 30


# ═══════════════════════════════════════════════════════════════════════════════
# Acoustic detector
# ═══════════════════════════════════════════════════════════════════════════════

def classify_acoustic_source(fundamental_hz: float) -> str:
    """Map a detected tonal fundamental to the likely drone sound source."""
    if fundamental_hz <= 0:
        return "unknown acoustic source"
    if fundamental_hz < 140:
        return "propeller/rotor blade-pass tone"
    if fundamental_hz < 400:
        return "motor/ESC tone"
    if fundamental_hz < 900:
        return "frame resonance / vibration"
    return "high-frequency structural vibration"


def classify_acoustic_area(energy_ratio: float, avg_band_db: float,
                           peak_db_level: float) -> str:
    """Provide a broad area hint from the signal strength.

    No longer uses device label keywords — direction is configured by the user
    via the --mic-direction argument, and the numeric direction_deg is set
    independently based on signal strength proximity.
    """
    if avg_band_db > -18 and energy_ratio > 0.25 and peak_db_level > -20:
        return "near the microphone side"
    if avg_band_db > -32 and energy_ratio > 0.15:
        return "same room, nearby area"
    return "uncertain / distant area"


def get_direction_for_strength(mic_direction: int, area_hint: str) -> int:
    """Map signal strength to a direction relative to the chosen microphone.

    The mic's configured direction (in degrees, 0..360 where 0 = front/North)
    defines where the microphone is physically pointing. The direction_deg
    output tells the user which way the sound source is relative to that mic:

      - Strong signal ("near the microphone side") → sound is coming from the
        direction the mic is pointing. Use the exact configured direction.
      - Medium signal ("same room, nearby area") → sound is nearby but at a
        wider angle. Return the mic direction ± some uncertainty; we use the
        direction as-is since it still indicates the general quadrant.
      - Weak/uncertain ("uncertain / distant area") → cannot determine
        direction. Return -1 to indicate unknown.
    """
    hint_l = (area_hint or '').lower()
    if 'uncertain' in hint_l or 'distant' in hint_l:
        return -1
    # For any detectable signal, the source direction is relative to
    # where the selected microphone is oriented/positioned.
    return mic_direction


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

    def __init__(self, device_index=None, detections_queue=None, mic_direction=0):
        # If no device specified, auto-detect a camera microphone.
        # Preserve explicit 0 device index as a valid selection.
        self.explicit_device_index = device_index is not None
        self.device_index = device_index if self.explicit_device_index else find_camera_microphone()
        self.detections     = detections_queue or queue.Queue(maxsize=50)
        self.running        = False
        self._thread        = None
        self._confirm_count = 0          # consecutive positive frames
        self.audio_source_label = "unknown microphone"
        self.recent_detections = deque(maxlen=ACOUSTIC_HISTORY)
        self.stats = {
            "frames_analysed": 0,
            "positive_frames": 0,
            "alerts":          0,
        }
        # Direction (in degrees) that the selected microphone is oriented/pointing.
        # 0 = front/North, increasing clockwise. User-configurable so that the
        # direction_deg in detection output is relative to this mic's position.
        self.mic_direction = mic_direction

        # `detections` is a shared queue used by RF + acoustic detectors to
        # publish human-readable detection dictionaries to the main loop.
        # Keeping it bounded prevents unbounded memory growth under flood.
    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        if not (NUMPY_AVAILABLE and PYAUDIO_AVAILABLE):
            print("❌ Acoustic detection unavailable (missing or incompatible numpy/pyaudio).")
            return False
        self.running = True
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        if not SCIPY_AVAILABLE:
            print("⚠️  SciPy unavailable; using NumPy peak detector fallback.")
        print(f"🎙️  Acoustic drone detector running (mic direction: {self.mic_direction}°)...")
        return True

    def stop(self):
        self.running = False

    def _render_spectrogram(self, band_freqs, band_db, peaks_idx):
        if len(band_freqs) == 0:
            return

        bins = 40
        edges = np.linspace(band_freqs[0], band_freqs[-1], bins + 1)
        bin_db = np.full(bins, -100.0, dtype=np.float32)
        for i in range(bins):
            mask = (band_freqs >= edges[i]) & (band_freqs < edges[i + 1])
            if np.any(mask):
                bin_db[i] = np.mean(band_db[mask])

        db_max = np.max(bin_db)
        db_min = max(np.min(bin_db), db_max - 40.0)
        chars = " .:-=+*#@"
        line = []
        for val in bin_db:
            level = int(np.clip((val - db_min) / (db_max - db_min + 1e-6) * (len(chars) - 1),
                                 0, len(chars) - 1))
            line.append(chars[level])

        if len(peaks_idx) > 0:
            peak_bins = np.digitize(band_freqs[peaks_idx], edges) - 1
            peak_bins = np.clip(peak_bins, 0, bins - 1)
            for b in np.unique(peak_bins):
                line[b] = "|"

        print(f"[spec] {''.join(line)}  {band_freqs[0]:.0f}-{band_freqs[-1]:.0f} Hz")

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
            if self.explicit_device_index:
                device_label = f"device {self.device_index} (explicit)"
            else:
                device_label = f"device {self.device_index}" if self.device_index is not None else "system default"
            self.audio_source_label = device_label
            print(f"   Microphone open ({device_label}, "
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

        # Window the samples to reduce spectral leakage before the FFT. A
        # Hann window is a good default for tonal analysis.
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

        # If the band doesn't carry enough energy compared with the full
        # spectrum, it's unlikely a drone is present (helps avoid false
        # positives in quiet rooms or when other broadband noise dominates).
        if energy_ratio < MIN_ENERGY_RATIO:
            if ACOUSTIC_DEBUG:
                print(f"[debug] energy_ratio={energy_ratio:.3f} below MIN_ENERGY_RATIO={MIN_ENERGY_RATIO}")
            self._confirm_count = 0
            return

        # ── Decibel-level checks ────────────────────────────────────────────
        # Drones produce sound within a characteristic loudness range. If the
        # average dB level in the drone band is too quiet (likely just noise)
        # or too loud (likely clipping or a non-drone source), reject the frame.
        band_db    = db[band_mask]
        band_freqs = freqs[band_mask]
        avg_band_db = np.mean(band_db)

        if avg_band_db < MIN_BAND_DB_LEVEL:
            if ACOUSTIC_DEBUG:
                print(f"[debug] avg_band_db={avg_band_db:.1f} dBFS below MIN_BAND_DB_LEVEL={MIN_BAND_DB_LEVEL} dBFS")
            self._confirm_count = 0
            return

        if avg_band_db > MAX_BAND_DB_LEVEL:
            if ACOUSTIC_DEBUG:
                print(f"[debug] avg_band_db={avg_band_db:.1f} dBFS above MAX_BAND_DB_LEVEL={MAX_BAND_DB_LEVEL} dBFS — possible clipping")
            self._confirm_count = 0
            return

        # Check that the strongest peak in the drone band stands clearly above
        # the noise floor. A genuine drone produces at least one strong tonal
        # peak; diffuse broadband noise won't pass this check.
        noise_floor_db = np.percentile(band_db, 10)  # 10th percentile as noise floor estimate
        peak_db_max = np.max(band_db)
        peak_db_above_noise = peak_db_max - noise_floor_db

        if peak_db_above_noise < MIN_PEAK_DB_ABOVE_NOISE:
            if ACOUSTIC_DEBUG:
                print(f"[debug] peak_db_above_noise={peak_db_above_noise:.1f} dB below MIN_PEAK_DB_ABOVE_NOISE={MIN_PEAK_DB_ABOVE_NOISE} dB")
            self._confirm_count = 0
            return

        # ── Peak finding in drone band ───────────────────────────────────────
        # (band_db and band_freqs are already assigned from the dB-level checks above)
        # find_peaks works on the band-limited spectrum. We look for
        # prominent narrowband peaks (tonal components) that could be the
        # propeller fundamental and its harmonics.
        peaks_idx, props = find_peaks(
            band_db,
            prominence = PEAK_PROMINENCE,
            distance   = int(len(band_freqs) * 0.02),  # ≥2% band apart
        )

        if ACOUSTIC_DEBUG:
            peak_freqs_debug = band_freqs[peaks_idx] if len(peaks_idx) > 0 else []
            print(f"[debug] frame={self.stats['frames_analysed']} energy_ratio={energy_ratio:.3f} peaks={len(peaks_idx)} "
                  f"prom={PEAK_PROMINENCE} fund_range={DRONE_FREQ_LOW}-{DRONE_FREQ_HIGH} "
                  f"peak_freqs={peak_freqs_debug}")

        if len(peaks_idx) < MIN_HARMONIC_PEAKS:
            self._confirm_count = 0
            if ACOUSTIC_SPECTROGRAM:
                self._render_spectrogram(band_freqs, band_db, peaks_idx)
            return

        peak_freqs = band_freqs[peaks_idx]

        if ACOUSTIC_SPECTROGRAM:
            self._render_spectrogram(band_freqs, band_db, peaks_idx)

        # ── Harmonic series check ────────────────────────────────────────────
        # For each candidate fundamental we check whether integer multiples of
        # that frequency (2x, 3x, …) also appear as peaks. A real multi-rotor
        # drone produces a harmonic series; random noises typically do not.
        best_count    = 0
        best_fund     = 0.0
        best_db_rolloffs = []  # dB rolloff values for the best candidate's harmonics

        for candidate_fund in peak_freqs:
            if candidate_fund < DRONE_FREQ_LOW:
                continue
            count = 1   # the fundamental itself counts
            harmonic_dbs = []  # store dB levels of each matched harmonic
            for harmonic_n in range(2, 10):
                expected = candidate_fund * harmonic_n
                if expected > DRONE_FREQ_HIGH:
                    break
                # Is there a peak within tolerance of this harmonic?
                tolerance_hz = expected * HARMONIC_TOLERANCE
                close_mask = np.abs(peak_freqs - expected) <= tolerance_hz
                if np.any(close_mask):
                    count += 1
                    # Find the closest peak and record its dB level
                    close_idx = np.argmin(np.abs(peak_freqs - expected))
                    peak_db_val = band_db[peaks_idx[close_idx]]
                    harmonic_dbs.append(peak_db_val)

            if count > best_count:
                best_count = count
                best_fund  = candidate_fund
                best_db_rolloffs = harmonic_dbs

        # ── Decibel rolloff check for the best candidate ────────────────────
        # Real drone harmonics decay gradually at roughly 2–14 dB per step.
        # If the rolloff between consecutive matched harmonics is outside this
        # range, the signal is less likely to be a genuine multi-rotor drone.
        db_rolloff_valid = True
        if len(best_db_rolloffs) >= 2:
            for i in range(len(best_db_rolloffs) - 1):
                step_db = best_db_rolloffs[i] - best_db_rolloffs[i + 1]
                if step_db < MIN_HARMONIC_DB_ROLLOFF or step_db > MAX_HARMONIC_DB_ROLLOFF:
                    db_rolloff_valid = False
                    if ACOUSTIC_DEBUG:
                        print(f"[debug] harmonic dB rolloff {step_db:.1f} dB outside "
                              f"[{MIN_HARMONIC_DB_ROLLOFF}, {MAX_HARMONIC_DB_ROLLOFF}] range")

        # Decide whether we have enough harmonics to consider this a drone.
        # `MIN_HARMONIC_PEAKS` is the tunable threshold for how many harmonics
        # must be present (including the fundamental).
        is_drone = best_count >= MIN_HARMONIC_PEAKS and db_rolloff_valid

        if is_drone:
            self.stats["positive_frames"] += 1
            self._confirm_count += 1
        else:
            self._confirm_count = 0

        # ── Fire alert after CONFIRM_FRAMES consecutive positives ────────────
        # The confirmation window reduces spurious alerts from single-frame
        # glitches — we require a short persistence before publishing.
        if self._confirm_count >= CONFIRM_FRAMES:
            self._confirm_count = 0   # reset so we don't spam
            self.stats["alerts"] += 1

            # Compute the dB level of the strongest peak for the alert message
            peak_db_level = np.max(band_db[peaks_idx]) if len(peaks_idx) > 0 else -100

            sound_origin = classify_acoustic_source(best_fund)
            area_hint = classify_acoustic_area(
                energy_ratio,
                avg_band_db,
                peak_db_level,
            )
            # Direction is relative to the selected microphone's configured
            # orientation (--mic-direction). Strong = exact direction,
            # weak/uncertain = -1 (unknown).
            direction_deg = get_direction_for_strength(self.mic_direction, area_hint)

            detection = {
                "source":      "ACOUSTIC",
                "timestamp":   datetime.now().strftime("%H:%M:%S"),
                "fundamental": f"{best_fund:.1f} Hz",
                "harmonics":   best_count,
                "energy_pct":  f"{energy_ratio * 100:.1f}%",
                "peak_db":     f"{peak_db_level:.1f} dBFS",
                "mic_source":  self.audio_source_label,
                "sound_origin": sound_origin,
                "area_hint":   area_hint,
                "detail":      (f"Harmonic series at {best_fund:.0f} Hz "
                                f"({best_count} harmonics, "
                                f"band energy {energy_ratio*100:.1f}%, "
                                f"peak {peak_db_level:.1f} dBFS). "
                                f"Likely source: {sound_origin}; area: {area_hint}"),
            }

            key = f"acoustic_{int(best_fund)}"
            if key not in self.recent_detections:
                self.recent_detections.append(key)
                try:
                    self.detections.put_nowait(detection)
                except queue.Full:
                    pass
                # Human-readable console message
                dir_str = f" (direction: {direction_deg}° relative to mic)" if direction_deg >= 0 else " (direction: unknown)"
                print(f"🔊 DRONE DETECTED (ACOUSTIC)! "
                      f"Fund: {best_fund:.0f} Hz | "
                      f"Harmonics: {best_count} | "
                      f"Band energy: {energy_ratio*100:.1f}% | "
                      f"Likely source: {sound_origin} | "
                      f"Area: {area_hint} | "
                      f"Dir: {direction_deg}° via {self.audio_source_label}")

                # Emit a machine-readable JSON line for external dashboards
                try:
                    out = {
                        'type': 'ACOUSTIC',
                        'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                        'fundamental_hz': round(best_fund, 1),
                        'harmonics': best_count,
                        'energy_pct': round(energy_ratio*100, 1),
                        'peak_db': round(peak_db_level, 1),
                        'mic_source': self.audio_source_label,
                        'sound_origin': sound_origin,
                        'area_hint': area_hint,
                        # direction_deg is relative to the selected microphone's
                        # configured orientation (--mic-direction).
                        'direction_deg': direction_deg,
                    }
                    print(json.dumps(out), flush=True)
                except Exception:
                    pass

    def print_stats(self):
        s = self.stats
        if s["frames_analysed"] > 0:
            hit_rate = 100.0 * s["positive_frames"] / s["frames_analysed"]
        else:
            hit_rate = 0.0
        print(f"   🎙️  Frames: {s['frames_analysed']} | "
              f"Positive: {s['positive_frames']} ({hit_rate:.1f}%) | "
              f"Alerts: {s['alerts']} | "
              f"Mic dir: {self.mic_direction}°")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_camera_microphone():
    """
    Auto-detect a camera microphone, preferring built-in/internal audio over
    USB dongles (like the TP-Link WiFi adapter). Returns device index or None
    if no suitable camera mic found.
    """
    if not PYAUDIO_AVAILABLE:
        return None
    import pyaudio
    pa = pyaudio.PyAudio()
    camera_candidates = []  # (index, name) of likely built-in mics
    usb_candidates = []     # (index, name) of likely USB audio

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] == 0:
            continue
        name_lower = info["name"].lower()
        # Prefer built-in / camera keywords
        if any(kw in name_lower for kw in ["camera", "built-in", "builtin",
                                             "internal", "headset", "headphone"]):
            camera_candidates.append((i, info["name"]))
        # Deprioritize USB
        elif any(kw in name_lower for kw in ["usb", "adapter"]):
            usb_candidates.append((i, info["name"]))

    pa.terminate()

    # Return the first camera candidate if found
    if camera_candidates:
        idx, name = camera_candidates[0]
        print(f"✅ Auto-detected camera microphone: {name} (device {idx})")
        return idx
    # Otherwise return None (will use system default)
    return None


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


def get_audio_devices():
    """Return a list of available audio input devices as JSON-serializable dicts."""
    if not PYAUDIO_AVAILABLE:
        return []
    import pyaudio
    pa = pyaudio.PyAudio()
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append({
                'index': i,
                'name': info['name'],
                'channels': int(info['maxInputChannels']),
                'sample_rate': int(info['defaultSampleRate']),
            })
    pa.terminate()
    return devices


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global DRONE_FREQ_LOW, DRONE_FREQ_HIGH, PEAK_PROMINENCE
    global MIN_HARMONIC_PEAKS, CONFIRM_FRAMES
    global HARMONIC_TOLERANCE, MIN_ENERGY_RATIO, ACOUSTIC_DEBUG
    global MIN_BAND_DB_LEVEL, MAX_BAND_DB_LEVEL
    global MIN_HARMONIC_DB_ROLLOFF, MAX_HARMONIC_DB_ROLLOFF
    global MIN_PEAK_DB_ABOVE_NOISE

    parser = argparse.ArgumentParser(
        description="Acoustic Drone Detector (Jetson Nano)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sudo python3 acoustic_drone_detector.py
  sudo python3 acoustic_drone_detector.py --list-audio
  sudo python3 acoustic_drone_detector.py --audio-device 1
  sudo python3 acoustic_drone_detector.py --audio-device 1 --mic-direction 0

Acoustic tuning flags (advanced):
  --freq-low / --freq-high   : drone band Hz  (default 80–2000)
  --prominence               : peak prominence threshold dB  (default 12)
  --min-harmonics            : harmonics required  (default 3)
  --confirm-frames           : consecutive positive frames to alert  (default 2)
        """
    )
    parser.add_argument("--list-audio",      action="store_true",
                        help="List microphone devices and exit")
    parser.add_argument("--audio-device",    type=int, default=None,
                        help="PyAudio device index for microphone (default: auto-detect camera mic, or system default)")
    parser.add_argument("--mic-direction",   type=int, default=0,
                        help="Direction (in degrees) the selected microphone is oriented/pointing. "
                             "0=front, 90=right, 180=back, 270=left (default: 0)")
    parser.add_argument("--freq-low",        type=int, default=DRONE_FREQ_LOW,
                        help=f"Low end of drone frequency band Hz (default {DRONE_FREQ_LOW})")
    parser.add_argument("--freq-high",       type=int, default=DRONE_FREQ_HIGH,
                        help=f"High end of drone frequency band Hz (default {DRONE_FREQ_HIGH})")
    parser.add_argument("--prominence",      type=float, default=PEAK_PROMINENCE,
                        help=f"FFT peak prominence dB (default {PEAK_PROMINENCE})")
    parser.add_argument("--min-harmonics",   type=int, default=MIN_HARMONIC_PEAKS,
                        help=f"Min harmonic peaks to classify as drone (default {MIN_HARMONIC_PEAKS})")
    parser.add_argument("--harmonic-tolerance", type=float,
                        default=HARMONIC_TOLERANCE,
                        help=f"Harmonic frequency tolerance fraction (default {HARMONIC_TOLERANCE})")
    parser.add_argument("--energy-ratio",    type=float, default=MIN_ENERGY_RATIO,
                        help=f"Minimum drone band energy ratio (default {MIN_ENERGY_RATIO})")
    parser.add_argument("--confirm-frames",  type=int, default=CONFIRM_FRAMES,
                        help=f"Consecutive positive frames before alert (default {CONFIRM_FRAMES})")
    parser.add_argument("--debug-acoustic",  action="store_true",
                        help="Print detailed acoustic FFT/peak diagnostics to help tune detection")
    parser.add_argument("--spectrogram",    action="store_true",
                        help="Print an ASCII spectrum visualization for each audio frame")
    parser.add_argument("--min-band-db",    type=float, default=MIN_BAND_DB_LEVEL,
                        help=f"Minimum average dBFS level in drone band (default {MIN_BAND_DB_LEVEL})")
    parser.add_argument("--max-band-db",    type=float, default=MAX_BAND_DB_LEVEL,
                        help=f"Maximum average dBFS level in drone band (default {MAX_BAND_DB_LEVEL})")
    parser.add_argument("--min-db-rolloff", type=float, default=MIN_HARMONIC_DB_ROLLOFF,
                        help=f"Minimum dB rolloff between consecutive harmonics (default {MIN_HARMONIC_DB_ROLLOFF})")
    parser.add_argument("--max-db-rolloff", type=float, default=MAX_HARMONIC_DB_ROLLOFF,
                        help=f"Maximum dB rolloff between consecutive harmonics (default {MAX_HARMONIC_DB_ROLLOFF})")
    parser.add_argument("--min-peak-db-above-noise", type=float, default=MIN_PEAK_DB_ABOVE_NOISE,
                        help=f"Minimum dB of strongest peak above noise floor (default {MIN_PEAK_DB_ABOVE_NOISE})")
    args = parser.parse_args()

    if args.list_audio:
        list_audio_devices()
        sys.exit(0)

    # Apply tunable overrides
    DRONE_FREQ_LOW      = args.freq_low
    DRONE_FREQ_HIGH     = args.freq_high
    PEAK_PROMINENCE     = args.prominence
    MIN_HARMONIC_PEAKS  = args.min_harmonics
    HARMONIC_TOLERANCE  = args.harmonic_tolerance
    MIN_ENERGY_RATIO    = args.energy_ratio
    CONFIRM_FRAMES      = args.confirm_frames
    ACOUSTIC_DEBUG      = args.debug_acoustic
    ACOUSTIC_SPECTROGRAM = args.spectrogram
    MIN_BAND_DB_LEVEL   = args.min_band_db
    MAX_BAND_DB_LEVEL   = args.max_band_db
    MIN_HARMONIC_DB_ROLLOFF = args.min_db_rolloff
    MAX_HARMONIC_DB_ROLLOFF = args.max_db_rolloff
    MIN_PEAK_DB_ABOVE_NOISE = args.min_peak_db_above_noise

    print("=" * 60)
    print("  Acoustic Drone Detector")
    print("=" * 60)

    shared_queue = queue.Queue(maxsize=100)

    acoustic_detector = AcousticDroneDetector(
        device_index     = args.audio_device,
        detections_queue = shared_queue,
        mic_direction    = args.mic_direction,
    )
    acoustic_started = acoustic_detector.start()

    if not acoustic_started:
        print("❌ Acoustic detector failed to start. Exiting.")
        sys.exit(1)

    print("\n🚀 Detector running — press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(0.1)
            print(f"\n── Stats @ {datetime.now().strftime('%H:%M:%S')} ──")
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
        acoustic_detector.stop()
        print("✅ Done.")


if __name__ == "__main__":
    main()