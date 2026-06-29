# Drone Detection System — Sensory Methods

This document describes the three sensory detection methods integrated into the Drone Detection System. Each method operates on a different physical principle and can be used independently or combined for multi-sensor fusion.

---

## Table of Contents

1. [Camera Detection (Computer Vision)](#1-camera-detection-computer-vision)
2. [Acoustic Detection (Audio / FFT)](#2-acoustic-detection-audio--fft)
3. [RF Detection (Radio Frequency Scanning)](#3-rf-detection-radio-frequency-scanning)
4. [Integration with the Dashboard](#4-integration-with-the-dashboard)
5. [Multi-Sensor Fusion Notes](#5-multi-sensor-fusion-notes)

---

## 1. Camera Detection (Computer Vision)

**Source file:** `detect_drones_jetson_working.py`  
**Dashboard label:** **Camera · YOLO v26**

### Description

Real-time drone detection using a deep learning object detection model (YOLO — You Only Look Once), optimized for the NVIDIA Jetson Orin Nano platform. The system captures video frames from a USB or CSI camera, runs them through a TensorRT-accelerated YOLO model, and draws bounding boxes around detected drones.

### How It Works

1. **Video Capture:** Frames are captured from a camera source (USB webcam, CSI camera module, or RTSP network stream) using an `AsyncVideoCapture` wrapper that decouples I/O from inference in a background thread.
2. **Object Detection:** Each frame is passed through a YOLO model (trained on drone imagery). The model outputs bounding boxes, class labels, and confidence scores.
3. **Frame Skipping:** To boost effective display FPS, inference runs every *N* frames (default: every 3rd frame). The CSRT tracker fills the gaps.
4. **CSRT Tracking:** Between YOLO inferences, an OpenCV CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) tracker follows the detected drone. If YOLO misses a frame, the tracker "coasts" for up to 10 frames (≈0.33 s @ 30 fps) before dropping back to IDLE.
5. **Annotated Output:** Bounding boxes, confidence percentages, FPS, inference time, and tracker state are drawn onto the frame. The result is written to `/tmp/drone_detect_frame.jpg` for the dashboard server to serve as an MJPEG stream.

**Tracker states:**
| State | Meaning |
|---|---|
| `IDLE` | No drone detected and no active tracker |
| `TRACKING` | YOLO detected a drone; tracker is locked and following it |
| `COASTING` | YOLO missed a frame; tracker is still predicting location (countdown: N/10) |

### Hardware Requirements

- **NVIDIA Jetson Orin Nano** (8 GB recommended) with JetPack 5.0+
- **Camera:** USB webcam, CSI camera module (e.g., IMX219), or RTSP network camera
- **GPU memory:** At least 2 GB free for TensorRT inference

### Software Requirements

```bash
pip3 install ultralytics opencv-python torch numpy
# For Jetson with TensorRT:
# JetPack 5.0+ includes CUDA, cuDNN, TensorRT
```

### Usage

#### Basic — USB camera, CUDA inference
```bash
python3 detect_drones_jetson_working.py
```

#### TensorRT acceleration (fastest on Jetson)
```bash
python3 detect_drones_jetson_working.py --tensorrt --fp16
```

#### CSI camera with GStreamer pipeline
```bash
python3 detect_drones_jetson_working.py --gstreamer --source 0
```

#### RTSP network stream
```bash
python3 detect_drones_jetson_working.py --source "rtsp://192.168.1.100/stream"
```

#### Maximum performance mode
```bash
sudo nvpmodel -m 0   # Set MAXN 15W power mode
python3 detect_drones_jetson_working.py --tensorrt --fp16 --imgsz 640
```

#### Record detected footage
```bash
python3 detect_drones_jetson_working.py --record --output output.mp4
```

### Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Video source (camera index, path, or URL) |
| `--tensorrt` | off | Enable TensorRT engine (caches `.engine` for reuse) |
| `--fp16` | off | Use FP16 half-precision inference |
| `--imgsz` | `640` | Inference resolution in pixels |
| `--confidence` | `0.3` | Minimum detection confidence threshold |
| `--frame-skip` | `2` | Run inference every N+1 frames (0 = every frame) |
| `--no-tracker` | off | Disable CSRT tracker |
| `--tracker-coast` | `10` | Frames tracker coasts after YOLO misses |
| `--debug` | off | Print raw model detections to console |
| `--gstreamer` | off | Use GStreamer pipeline for CSI camera |
| `--no-display` | off | Run headless (no GUI window) |
| `--power-mode` | — | Set Jetson power mode (MAXN/10W/7W) |
| `--width` / `--height` | `640`/`480` | Camera capture resolution |

### Output

- **On-screen:** OpenCV window showing annotated video with FPS overlay
- **File:** `/tmp/drone_detect_frame.jpg` — latest annotated frame (served by dashboard)
- **Recording:** MP4 video file if `--record` is used
- **Console:** Periodic stats (frame count, FPS, inference time, detections)

---

## 2. Acoustic Detection (Audio / FFT)

**Source file:** `acoustic_drone_detector.py`  
**Dashboard label:** **Acoustic Detector**

### Description

Detects drones by analysing microphone audio with a Fast Fourier Transform (FFT). Multi-rotor drones produce a distinctive harmonic frequency pattern from their spinning propellers and motors — the detector identifies this signature.

### How It Works

1. **Audio Capture:** Raw PCM audio is captured from a microphone at 44.1 kHz in 4096-sample chunks (~93 ms per frame).
2. **FFT Analysis:** Each chunk is windowed (Hann window) and transformed to the frequency domain via `numpy.rfft`. The magnitude spectrum is converted to dB.
3. **Energy Ratio Check:** The total energy in the drone band (80–2000 Hz) is compared to the full-spectrum energy. If it falls below `MIN_ENERGY_RATIO` (0.15), the frame is discarded as quiet noise.
4. **Peak Finding:** Prominent spectral peaks are identified in the drone band using `scipy.signal.find_peaks` (or a NumPy fallback). A peak must be at least 12 dB above the local noise floor (`PEAK_PROMINENCE`).
5. **Harmonic Series Check:** For each candidate fundamental frequency *F*, the algorithm checks if integer multiples (2*F*, 3*F*, … up to 9*F*) also appear as peaks within a 15% frequency tolerance. If ≥3 harmonics are found, the frame is considered positive.
6. **Confirmation & Alerting:** After `CONFIRM_FRAMES` (2) consecutive positive frames, an alert is fired. This reduces false positives from transient noise.

**Why it works:** A multi-rotor drone with *N* propellers spinning at *R* revolutions per second produces a blade-pass frequency of *N × R* Hz (typically 80–300 Hz), plus strong harmonics at 2×, 3×, 4×, etc. Random environmental noise rarely exhibits this harmonic structure.

### Hardware Requirements

- **Microphone:** Any USB microphone, built-in webcam mic, or line-in audio device
- **Platform:** Works on any Linux system (Jetson, Raspberry Pi, desktop)
- **Recommended:** A directional microphone or shotgun mic for longer-range detection

### Software Requirements

```bash
# System dependencies
sudo apt update && sudo apt install portaudio19-dev

# Python packages
pip3 install pyaudio numpy scipy
```

### Usage

#### Basic — auto-detect camera microphone
```bash
python3 acoustic_drone_detector.py
```
*Note: `sudo` may be required for microphone access.*

#### List available microphones
```bash
python3 acoustic_drone_detector.py --list-audio
```

#### Specify a microphone device
```bash
python3 acoustic_drone_detector.py --audio-device 1
```

#### Tuning for sensitivity
```bash
# Lower prominence threshold to detect quieter drones
sudo python3 acoustic_drone_detector.py --prominence 8

# Narrower frequency band (less false positives)
sudo python3 acoustic_drone_detector.py --freq-low 100 --freq-high 1500

# Faster alerting (fewer confirmation frames)
sudo python3 acoustic_drone_detector.py --confirm-frames 1

# Debug mode (see FFT peaks per frame)
sudo python3 acoustic_drone_detector.py --debug-acoustic

# ASCII spectrogram visualization
sudo python3 acoustic_drone_detector.py --spectrogram
```

### Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--audio-device` | auto | PyAudio device index for microphone |
| `--list-audio` | off | List available microphone devices and exit |
| `--freq-low` | `80` | Low end of drone frequency band (Hz) |
| `--freq-high` | `2000` | High end of drone frequency band (Hz) |
| `--prominence` | `12` | FFT peak prominence threshold (dB) |
| `--min-harmonics` | `3` | Minimum harmonic peaks required for classification |
| `--harmonic-tolerance` | `0.15` | Allowable harmonic frequency deviation (fraction) |
| `--energy-ratio` | `0.15` | Minimum drone-band energy ratio |
| `--confirm-frames` | `2` | Consecutive positive frames before alert |
| `--debug-acoustic` | off | Print detailed FFT / peak diagnostics |
| `--spectrogram` | off | Print ASCII spectrum per frame |

### Detection Output

When a drone is detected, the console prints:
```
🔊 DRONE DETECTED (ACOUSTIC)! Fund: 240.0 Hz | Harmonics: 5 | Band energy: 34.2%
```

Every 5 seconds, a stats line is printed:
```
── Stats @ 14:23:05 ──
   🎙️  Frames: 1075 | Positive: 42 (3.9%) | Alerts: 8
```

The detection dictionary published to the shared queue includes:
- `source`: Always `"ACOUSTIC"`
- `timestamp`: Time of detection
- `fundamental`: Fundamental frequency in Hz
- `harmonics`: Number of harmonic peaks found
- `energy_pct`: Band energy percentage
- `detail`: Human-readable summary

---

## 3. RF Detection (Radio Frequency Scanning)

**Source file:** `rf_drone_scanner.py`  
**Dashboard label:** **RF Scanner**

### Description

Scans 2.4 GHz and 5 GHz WiFi bands using a USB wireless adapter to detect drone-related radio signals. Many drones broadcast WiFi hotspots for control/video (DJI, Autel, Parrot, Tello, etc.) — this scanner identifies them by SSID patterns, MAC OUI prefixes, channel usage, and signal characteristics.

### How It Works

1. **Association Drop:** The scanner briefly disconnects from any associated WiFi network so the adapter can perform a passive scan (this prevents the card from only listening on its associated channel).
2. **Scan Execution:** `iwlist scan` is invoked (via `sudo`) to gather all visible access points and their parameters (SSID, BSSID, frequency, signal level, encryption).
3. **Threat Scoring:** Each detected network is scored based on five heuristics:

| Heuristic | Points | Description |
|---|---|---|
| SSID pattern match | +3 | SSID matches known drone vendor patterns (DJI, Mavic, Phantom, Autel, Tello, FPV, etc.) |
| Known drone OUI | +3 | MAC address prefix matches DJI, Autel, or other drone-vendor OUI |
| 5 GHz drone channel | +2 | Operating on a 5 GHz channel commonly used for drone video links (UNII-1/2e/3 bands) |
| Strong 2.4 GHz signal | +1 | RSSI ≥ -65 dBm on 2.4 GHz — suggests a nearby RC transmitter |
| Open (unencrypted) WiFi | +1 | No encryption — consistent with drone ad-hoc hotspots |

4. **Threat Level Assignment:**
   - **CRITICAL** (≥7 points)
   - **HIGH** (5–6 points)
   - **MEDIUM** (3–4 points)
   - **LOW** (1–2 points)
   - **CLEAR** (0 points)

5. **Reconnection:** After scanning, the adapter reconnects to any previously-associated network and renews its DHCP lease.
6. **Curses Dashboard:** A real-time terminal UI shows all detected networks with threat levels, signal bars, and a scrolling alert log.

### Hardware Requirements

- **USB WiFi adapter** that supports monitor mode and `iwlist` scanning. Recommended chipsets: RTL8812AU, RTL8814AU, RTL8821AU, or any adapter supported by `wireless-tools`.
- A secondary WiFi adapter (not the one used for internet access), since scanning requires briefly disconnecting.

### Software Requirements

```bash
sudo apt install wireless-tools
```

For passwordless sudo (optional but recommended):
```bash
echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/iwlist" | sudo tee /etc/sudoers.d/iwlist
```

### Usage

#### Basic scan on wlan1
```bash
python3 rf_drone_scanner.py --iface wlan1
```

#### Faster scan interval (2 seconds)
```bash
python3 rf_drone_scanner.py --iface wlan1 --interval 2
```

#### Alert-only mode (hide CLEAR networks)
```bash
python3 rf_drone_scanner.py --iface wlan1 --alert-only
```

#### Custom log directory
```bash
python3 rf_drone_scanner.py --iface wlan1 --log-dir /var/log/drone
```

### Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--iface` / `-i` | `wlan1` | WiFi interface name |
| `--interval` / `-t` | `3.0` | Seconds between scans |
| `--log-dir` | `./rf_logs` | Directory for log files |
| `--alert-only` | off | Show only networks with non-zero threat score |
| `--no-prereq-check` | off | Skip interface/iwlist availability checks |

### Curses Dashboard Controls

- **q** or **ESC** — Quit

### Logs

All alerts are timestamped and written to a log file at `./rf_logs/rf_drone_{iface}_{timestamp}.log` in the format:

```
2026-06-29 11:48:00  WARNING   NEW ALERT [CRITICAL] score=8  SSID=DJI_Mavic3_8F2A  BSSID=60:60:1F:XX:XX:XX  5.745GHz  ch149  -52dBm  reasons=SSID: DJI device SSID; OUI 60:60:1F is a known drone vendor MAC prefix; 5 GHz drone video-link channel 149 (5.745 GHz)
```

### Drone SSID Patterns Detected

The scanner checks against these patterns (case-insensitive):
- `DJI`, `MAVIC`, `PHANTOM`, `MINI` — DJI product lines
- `AUTEL`, `SKYDIO`, `PARROT`, `BEBOP` — Other consumer drone brands
- `TELLO` — Ryze/Robomaster Tello
- `FPV`, `DRONE`, `UAV`, `QUADCOPTER` — Generic identifiers
- `SKY_VIPER`, `HOVER` — Budget consumer drones

### Known Drone MAC OUIs

The scanner flags these MAC address prefixes:
- `60:60:1F`, `48:1C:B9`, `34:D2:62`, `A0:14:3D`, `90:3A:E6`, `18:60:24`, `BC:8B:B2`, `C4:7C:8D`, `D0:9C:7A`, `AC:B0:AB` — SZ DJI Technology
- `80:2A:A8` — Autel Robotics

---

## 4. Integration with the Dashboard

All three sensory methods feed into `drone_dashboard.html` through the `dashboard_server.py` backend (running on port 9000).

### Starting the Dashboard Server

```bash
python3 dashboard_server.py
```

### Toggling Sensors in the UI

The dashboard top-bar provides toggle switches for each sensor:

```
[Acoustic] | [RF] | [Camera]
```

When toggled **ON**, the dashboard sends a request to:

- `GET /start/acoustic` — Starts the acoustic detector process
- `GET /start/rf` — Starts the RF scanner process
- `GET /start/camera` — Starts the camera detector process

When toggled **OFF**:

- `GET /stop/acoustic` — Stops the acoustic detector
- `GET /stop/rf` — Stops the RF scanner
- `GET /stop/camera` — Stops the camera detector

### Camera Stream

The camera video feed is served as an MJPEG stream at:

```
http://localhost:9000/camera/stream
```

This is sourced from `/tmp/drone_detect_frame.jpg`, which the detector writes every frame.

### Sensor Displays

| Sensor | Dashboard Panel Shows |
|---|---|
| **Acoustic** | Frames analysed, positive frames, alerts fired, FFT spectrum bars, harmonic series bars |
| **RF** | Scan count, visible networks, drone SSID alerts, table with SSID/freq/dBm/signal/threat |
| **Camera** | FPS, inference time, drone count, video stream with bounding boxes, tracker state, confidence gauge |

### Detection Log

All sensors write to a unified detection log panel, with colour-coded source labels:
- **`ACOUSTIC`** — Purple (#a855f7)
- **`RF`** — Cyan (#06b6d4)
- **`CAMERA`** — Green (#22c55e)

---

## 5. Multi-Sensor Fusion Notes

Each method has strengths and weaknesses. For the most reliable drone detection, use all three together:

| Method | Range | Pros | Cons |
|---|---|---|---|
| **Camera** | Visual line-of-sight | Precise location, visual confirmation | Requires lighting, limited range, no detection behind obstacles |
| **Acoustic** | 100–500 m (wind-dependent) | Works around obstacles, passive (no emissions), omni-directional | Wind noise, cannot distinguish drone type, lower precision |
| **RF** | Up to 1+ km (depends on drone TX power) | Longest range, identifies drone model/vendor, passive | Only detects Wi-Fi-based drones, requires compatible radio hardware |

**Fusion strategy:** The dashboard computes a global threat level by taking the maximum across all three sensors:
- RF networks with CRITICAL/HIGH threat = elevated
- Acoustic confirmation count ≥ 4 = HIGH
- Camera actively tracking a drone = HIGH

The highest threat level among all sensors drives the global threat badge (CLEAR → LOW → MEDIUM → HIGH → CRITICAL).