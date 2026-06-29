#!/usr/bin/env python3
"""
Jetson Orin Nano Optimized Drone Detection
Real-time drone detection optimized for NVIDIA Jetson Orin Nano.

Optimizations:
- TensorRT engine export and inference
- Efficient video I/O with GStreamer
- Power mode management
- Thermal monitoring
- Memory optimization for embedded systems
- CSI camera support

Requirements:
- NVIDIA Jetson Orin Nano (8GB recommended)
- JetPack 5.0+ with CUDA, cuDNN, TensorRT
- Python 3.8+
"""

import argparse
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import sys
import time
import torch
from collections import deque
import subprocess
import os
import threading


class AsyncVideoCapture:
    """
    Non-blocking video capture that reads frames in a background thread.
    Eliminates camera I/O latency from the inference loop, improving FPS.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._ret = False
        self._frame = None
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        # Wait until the reader thread has grabbed at least one frame before
        # returning — avoids a race where the main loop calls read() before
        # _frame is populated, causing immediate "no frames" failure.
        timeout = 5.0
        waited = 0.0
        while waited < timeout:
            with self._lock:
                if self._frame is not None:
                    break
            time.sleep(0.05)
            waited += 0.05
        else:
            raise RuntimeError(
                "Camera did not produce a frame within 5 seconds. "
                "Check the device index (--source) and that no other process is using the camera."
            )

    def _reader(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
            # Yield CPU briefly so the reader thread doesn't spin at 100%.
            # 1 ms is imperceptible for video but prevents starving the inference thread.
            time.sleep(0.001)

    def read(self):
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    def get(self, prop):
        return self._cap.get(prop)

    def isOpened(self):
        return self._cap.isOpened()

    def release(self):
        self._stopped = True
        self._thread.join(timeout=1.0)
        self._cap.release()


class DroneTracker:
    """
    CSRT-based tracker that holds a bounding box between YOLO detections.

    Lifecycle:
      - IDLE      : No drone seen yet, or tracker was lost too long.
      - TRACKING  : YOLO fired recently; tracker is following the drone.
      - COASTING  : YOLO missed this frame but tracker is still active.

    The tracker is initialised / re-initialised every time YOLO produces a
    confident detection.  If YOLO goes quiet for more than `max_coast_frames`
    consecutive frames the tracker is discarded and we go back to IDLE.
    """

    # Colour used when drawing tracker-only boxes (distinct from YOLO green)
    TRACKER_COLOR = (0, 165, 255)   # Orange

    def __init__(self, max_coast_frames: int = 10):
        """
        Args:
            max_coast_frames: How many consecutive frames without a YOLO hit
                              before the tracker is abandoned.  At 30 fps,
                              10 frames ≈ 0.33 s.
        """
        self.max_coast_frames = max_coast_frames
        self._tracker = None
        self._coast_count = 0          # frames since last YOLO detection
        self._tracked_det = None       # last detection dict (for label/conf)
        self.state = 'IDLE'            # 'IDLE' | 'TRACKING' | 'COASTING'

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, frame, yolo_detections: list) -> list:
        """
        Given the raw YOLO detections for this frame, return the best
        combined detection list (YOLO result OR tracker fallback).

        Args:
            frame:            Current BGR frame (numpy array).
            yolo_detections:  List of detection dicts from detect_frame.

        Returns:
            List of detection dicts to display.  Each dict has the same
            keys as a normal YOLO detection dict, plus 'tracked': bool.
        """
        if yolo_detections:
            # ── YOLO fired ──────────────────────────────────────────────
            # Pick the highest-confidence detection to anchor the tracker.
            best = max(yolo_detections, key=lambda d: d['confidence'])
            self._reinit(frame, best)
            self._coast_count = 0
            self.state = 'TRACKING'
            # Tag every detection so callers know the source
            for d in yolo_detections:
                d['tracked'] = False
            return yolo_detections

        else:
            # ── YOLO missed ─────────────────────────────────────────────
            if self._tracker is None or self.state == 'IDLE':
                return []

            self._coast_count += 1
            if self._coast_count > self.max_coast_frames:
                self._reset()
                return []

            # Ask the tracker where the drone went
            success, bbox = self._tracker.update(frame)
            if not success:
                self._reset()
                return []

            x, y, w, h = [int(v) for v in bbox]
            x2, y2 = x + w, y + h
            # Keep bbox inside frame
            fh, fw = frame.shape[:2]
            x  = max(0, x);  y  = max(0, y)
            x2 = min(fw, x2); y2 = min(fh, y2)

            self.state = 'COASTING'
            tracked_det = {
                'class':      self._tracked_det['class'],
                'confidence': self._tracked_det['confidence'],
                'bbox':       np.array([x, y, x2, y2], dtype=np.float32),
                'center':     ((x + x2) / 2, (y + y2) / 2),
                'tracked':    True,   # drawn differently in _draw_detections
            }
            return [tracked_det]

    def reset(self):
        """Externally force the tracker back to IDLE (e.g. on scene cut)."""
        self._reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reinit(self, frame, detection: dict):
        """Create a fresh CSRT tracker seeded from a YOLO bbox."""
        x1, y1, x2, y2 = detection['bbox'].astype(int)
        bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
        self._tracker = cv2.TrackerCSRT_create()
        self._tracker.init(frame, bbox_xywh)
        self._tracked_det = detection

    def _reset(self):
        self._tracker = None
        self._coast_count = 0
        self._tracked_det = None
        self.state = 'IDLE'


class JetsonDroneDetector:
    """Jetson Orin Nano optimized drone detector."""
    
    def __init__(self, model_size='n', confidence_threshold=0.3, iou_threshold=0.45,
                 use_tensorrt=True, fp16=True, monitor_thermal=True,
                 frame_skip=2, imgsz=640, use_tracker=True, tracker_coast_frames=10,
                 debug=False):
        """
        Initialize the Jetson-optimized drone detector.
        
        Args:
            model_size: YOLO model size ('n', 's' recommended for Jetson)
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
            use_tensorrt: Convert and use TensorRT engine for faster inference
            fp16: Use FP16 precision (highly recommended for Jetson)
            monitor_thermal: Monitor temperature and throttle if needed
            frame_skip: Run inference every (frame_skip+1) frames, reusing last result.
                        Default raised from 0→2: inference runs every 3rd frame while
                        the CSRT tracker fills the gaps, giving ~3× display FPS with
                        no perceptible detection lag.  Set to 0 to run every frame.
            imgsz: Inference image size. Defaults to 832 to match the trained
                   model's training resolution — mismatching this causes internal
                   rescaling and hurts both speed and accuracy.
            use_tracker: Enable CSRT tracker to hold detections between YOLO hits
            tracker_coast_frames: Frames the tracker keeps running after YOLO last
                                  fired before it gives up (default 10 ≈ 0.33 s @ 30fps)
        """
        print("="*70)
        print("🚀 Jetson Orin Nano Drone Detection System")
        print("="*70)
        
        self.is_jetson = self._check_jetson_platform()
        self.monitor_thermal = monitor_thermal and self.is_jetson
        
        if self.is_jetson:
            print("\n✓ Running on Jetson Platform")
            self._display_jetson_info()
        else:
            print("\n⚠️  Not running on Jetson - Some optimizations disabled")
        
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_tensorrt = use_tensorrt and self.is_jetson
        self.fp16 = fp16
        self.frame_skip = frame_skip
        self.imgsz = imgsz
        self._skip_counter = 0
        self._last_detections = []
        self.debug = debug

        # Tracker setup
        self.use_tracker = use_tracker
        if use_tracker:
            self.tracker = DroneTracker(max_coast_frames=tracker_coast_frames)
            print(f"🎯 CSRT tracker enabled (coast={tracker_coast_frames} frames)")
        else:
            self.tracker = None
        
        # Model setup - check for trained model first
        print(f"\n📦 Loading model...")
        
        # First, check for trained drone model from train_drone_model.py
        trained_model_pattern = Path('runs/drone_training_*/train/weights/best.pt')
        trained_models = sorted(Path('.').glob('runs/drone_training_*/train/weights/best.pt'))
        
        if trained_models:
            # Use the most recent trained model
            model_path = str(trained_models[-1])
            print(f"✓ Found trained drone model: {model_path}")
        else:
            # Fall back to base YOLO26 model
            model_path = f'best.pt'
            if not Path(model_path).exists():
                print(f"❌ No trained model found and base model not found: {model_path}")
                print("Options:")
                print(f"  1. Train a model: python3 train_drone_model.py")
                print(f"  2. Let ultralytics auto-download: it will fetch yolo26{model_size}.pt on first run")
                sys.exit(1)
            print(f"Using base YOLO26 model: {model_path}")
        
        self.model = YOLO(model_path)
        
        # TensorRT optimization
        if self.use_tensorrt:
            print("\n⚡ Optimizing with TensorRT...")
            self._setup_tensorrt(model_size)
        else:
            # Standard CUDA setup
            if torch.cuda.is_available():
                self.model.to('cuda')
                if self.fp16:
                    self.model.model.half()
                    print("✓ Using FP16 precision")
        
        # Statistics
        self.frame_count = 0
        self.start_time = None
        self.total_detections = 0
        self.inference_times = deque(maxlen=100)
        
        # Drone classes — read directly from the loaded model so this always
        # matches regardless of whether it's a custom drone model or base COCO.
        # A custom best.pt will typically expose just {0: 'drone'}.
        self.drone_classes = list(self.model.names.values())
        print(f"\n{'='*70}")
        print(f"🎯 Model class names: {self.model.names}")
        print(f"🎯 Filtering for ALL {len(self.drone_classes)} class(es): {self.drone_classes}")
        print(f"   Confidence threshold: {self.conf_threshold}")
        print(f"   (Use --confidence to lower this if nothing is detected)")
        print(f"{'='*70}\n")
        
        # Thermal monitoring setup
        if self.monitor_thermal:
            self.temp_history = deque(maxlen=60)  # Last 60 readings
            self.thermal_throttle = False
            self._thermal_check_counter = 0
            self._thermal_check_interval = 30  # Check temp every 30 frames
        
        # Warm-up
        if torch.cuda.is_available():
            print("\n🔥 Warming up GPU...")
            self._warm_up()
            print("✓ Warm-up complete")
        
        print("\n" + "="*70)
        print("✓ Initialization Complete - Ready for Detection!")
        print("="*70 + "\n")
    
    def _check_jetson_platform(self):
        """Check if running on Jetson platform."""
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                return True
        except:
            return False
    
    def _display_jetson_info(self):
        """Display Jetson system information."""
        try:
            # Get Jetson model
            result = subprocess.run(['cat', '/proc/device-tree/model'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                model = result.stdout.strip().replace('\x00', '')
                print(f"  Model: {model}")
            
            # Get JetPack version
            result = subprocess.run(['dpkg', '-l', 'nvidia-jetpack'], 
                                  capture_output=True, text=True)
            if 'nvidia-jetpack' in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.startswith('ii'):
                        parts = line.split()
                        if len(parts) >= 3:
                            print(f"  JetPack: {parts[2]}")
                            break
            
            # Get CUDA version
            if torch.cuda.is_available():
                print(f"  CUDA: {torch.version.cuda}")
                print(f"  cuDNN: {torch.backends.cudnn.version()}")
            
            # Get power mode
            try:
                result = subprocess.run(['nvpmodel', '-q'], 
                                      capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'NV Power Mode' in line:
                        print(f"  {line.strip()}")
            except:
                pass
            
            # Get current temperature
            temp = self._read_temperature()
            if temp:
                print(f"  Temperature: {temp}°C")
            
        except Exception as e:
            print(f"  Info unavailable: {e}")
    
    def _setup_tensorrt(self, model_size):
        """Setup TensorRT engine for faster inference.

        The engine is keyed on both model_size AND imgsz so that a 320-px
        engine is not accidentally reused for 640-px inference (or vice-versa).
        """
        trt_path = f'best_{self.imgsz}.engine'
        
        try:
            if Path(trt_path).exists():
                print(f"✓ Found existing TensorRT engine: {trt_path}")
                self.model = YOLO(trt_path, task='detect')
            else:
                print(f"Building TensorRT engine at imgsz={self.imgsz} (this may take a few minutes)...")
                print("⏳ Please wait...")
                
                # Export to TensorRT — use self.imgsz so the engine matches
                # the inference size used at runtime.  A 320-px engine is
                # ~2–3× faster than a 640-px one with minimal accuracy loss.
                self.model.export(
                    format='engine',
                    half=self.fp16,
                    imgsz=self.imgsz,   # was hardcoded to 640
                    workspace=4,        # 4 GB workspace for Jetson Orin Nano
                    device=0
                )
                
                # Load TensorRT model
                self.model = YOLO(trt_path, task='detect')
                print(f"✓ TensorRT engine created: {trt_path}")
                print("  (This engine is cached for future use)")
        
        except Exception as e:
            print(f"⚠️  TensorRT optimization failed: {e}")
            print("  Falling back to standard CUDA inference")
            self.use_tensorrt = False
            if torch.cuda.is_available():
                self.model.to('cuda')
                if self.fp16:
                    self.model.model.half()
    
    def _warm_up(self, iterations=10):
        """Warm up the model with dummy inference.

        10 iterations (up from 5) ensures TensorRT engines fully JIT-compile
        all CUDA kernels before real frames arrive, which eliminates the
        'cold-start' latency spike on the first few real frames.
        """
        dummy_img = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        
        for _ in range(iterations):
            with torch.no_grad():
                self.model(dummy_img, verbose=False, imgsz=self.imgsz)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    def _read_temperature(self):
        """Read GPU temperature on Jetson."""
        try:
            # Try reading from thermal zone
            thermal_paths = [
                '/sys/devices/virtual/thermal/thermal_zone0/temp',
                '/sys/devices/virtual/thermal/thermal_zone1/temp',
                '/sys/class/thermal/thermal_zone0/temp'
            ]
            
            for path in thermal_paths:
                if Path(path).exists():
                    with open(path, 'r') as f:
                        temp = int(f.read().strip()) / 1000.0
                        if 20 < temp < 120:  # Sanity check
                            return temp
            
            # Fallback: try tegrastats
            result = subprocess.run(['tegrastats', '--interval', '100'], 
                                  capture_output=True, text=True, timeout=0.5)
            if 'temp' in result.stdout.lower():
                # Parse temperature from output
                pass
                
        except:
            pass
        
        return None
    
    def _check_thermal_throttle(self):
        """Check if thermal throttling is needed (rate-limited to every N frames)."""
        if not self.monitor_thermal:
            return False

        self._thermal_check_counter += 1
        if self._thermal_check_counter < self._thermal_check_interval:
            return self.thermal_throttle  # Return cached result between checks
        self._thermal_check_counter = 0

        temp = self._read_temperature()
        if temp:
            self.temp_history.append(temp)
            
            # Throttle if temperature exceeds 80°C
            if temp > 80:
                if not self.thermal_throttle:
                    print(f"\n⚠️  High temperature detected: {temp:.1f}°C")
                    print("  Enabling thermal throttling...")
                    self.thermal_throttle = True
                return True
            elif temp < 70 and self.thermal_throttle:
                print(f"\n✓ Temperature normalized: {temp:.1f}°C")
                print("  Disabling thermal throttling...")
                self.thermal_throttle = False
        
        return self.thermal_throttle
    
    def get_system_stats(self):
        """Get Jetson system statistics."""
        stats = {}
        
        # GPU stats
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**2
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved(0) / 1024**2
        
        # Temperature
        temp = self._read_temperature()
        if temp:
            stats['temperature'] = temp
        
        # CPU usage
        try:
            result = subprocess.run(['cat', '/proc/stat'], 
                                  capture_output=True, text=True, timeout=0.1)
            # Parse CPU stats (simplified)
        except:
            pass
        
        return stats
    
    def create_gstreamer_pipeline(self, capture_width=1280, capture_height=720, 
                                 framerate=30, flip_method=0, camera_id=0):
        """
        Create GStreamer pipeline for CSI camera (more efficient than V4L2).
        
        Args:
            capture_width: Camera capture width
            capture_height: Camera capture height
            framerate: Camera framerate
            flip_method: Image flip method (0-7)
            camera_id: Camera ID (0 or 1 for dual cameras)
        """
        return (
            f"nvarguscamerasrc sensor-id={camera_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){capture_width}, "
            f"height=(int){capture_height}, format=(string)NV12, "
            f"framerate=(fraction){framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){capture_width}, height=(int){capture_height}, "
            f"format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink"
        )
    
    def detect_frame(self, frame):
        """Detect drones in a single frame, with optional frame-skip for higher FPS."""

        # --- Frame skipping: reuse last detections when skipping ---
        if self.frame_skip > 0:
            self._skip_counter += 1
            if self._skip_counter <= self.frame_skip:
                skip_start = time.time()
                # Still update tracker on skipped frames so it doesn't drift
                if self.tracker is not None:
                    skipped_dets = self.tracker.update(frame, [])
                else:
                    skipped_dets = self._last_detections
                annotated_frame = self._draw_detections(frame, skipped_dets, inplace=True)
                inference_time = (time.time() - skip_start) * 1000
                return annotated_frame, skipped_dets, inference_time
            self._skip_counter = 0  # Reset: this frame runs inference

        # Check thermal throttling
        if self._check_thermal_throttle():
            time.sleep(0.05)  # Add small delay if throttling
        
        # Run detection.
        # stream=True returns a lazy generator — inference doesn't execute until
        # next() is called, so the timer wraps both calls.
        # cuda.synchronize() is omitted from the hot path — it blocks the CPU
        # until the GPU finishes every frame, costing 5-15 ms per frame.
        inference_start = time.time()
        with torch.no_grad():
            results = self.model(
                frame,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                verbose=False,
                stream=True
            )
            result = next(iter(results))  # inference actually executes here
        inference_time = (time.time() - inference_start) * 1000
        self.inference_times.append(inference_time)

        # Debug: print every raw detection before filtering so you can see
        # exactly what the model is producing and why things may be dropped.
        if self.debug and result.boxes is not None and len(result.boxes) > 0:
            print(f"\n[DEBUG] Frame raw detections ({len(result.boxes)} total):")
            for box in result.boxes:
                raw_cls_id = int(box.cls[0])
                raw_cls_name = result.names[raw_cls_id]
                raw_conf = float(box.conf[0])
                print(f"  class_id={raw_cls_id}  name='{raw_cls_name}'  conf={raw_conf:.3f}"
                      f"  in_filter={raw_cls_name in self.drone_classes}")
        elif self.debug:
            print(f"[DEBUG] Frame: no detections above conf={self.conf_threshold}")

        # Filter for target classes
        drone_detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                if cls_name in self.drone_classes:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    drone_detections.append({
                        'class': cls_name,
                        'confidence': conf,
                        'bbox': xyxy,
                        'center': ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                    })
        
        self._last_detections = drone_detections  # Cache for frame-skip reuse

        # ── Tracker: supplement YOLO with CSRT when it misses ────────────
        if self.tracker is not None:
            drone_detections = self.tracker.update(frame, drone_detections)

        # Create annotated frame
        annotated_frame = self._draw_detections(frame, drone_detections, inplace=True)
        
        return annotated_frame, drone_detections, inference_time
    
    def _draw_detections(self, frame, detections, inplace=False):
        """Draw detection boxes and labels on frame.
        
        Args:
            inplace: If True, draw directly on frame (avoids copy; safe when frame
                     won't be written to disk separately from annotated output).
        """
        annotated_frame = frame if inplace else frame.copy()
        
        color_map = {
            'bird': (255, 165, 0),
            'airplane': (0, 0, 255),
            'kite': (255, 0, 255),
            'frisbee': (0, 255, 255),
            'sports ball': (255, 255, 0)
        }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cls_name = det['class']
            conf = det['confidence']
            is_tracked = det.get('tracked', False)

            # Tracked-only boxes are drawn in orange with a dotted border style
            # to distinguish them from solid YOLO detections
            if is_tracked:
                color = DroneTracker.TRACKER_COLOR
                # Draw dashed rectangle by drawing corner segments
                seg = 20   # segment length in pixels
                thickness = 2
                for (px, py) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    dx = 1 if px == x1 else -1
                    dy = 1 if py == y1 else -1
                    cv2.line(annotated_frame, (px, py), (px + dx * seg, py), color, thickness)
                    cv2.line(annotated_frame, (px, py), (px, py + dy * seg), color, thickness)
                label = f"TRACKING {conf:.2f}"
            else:
                color = color_map.get(cls_name, (0, 255, 0))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name.upper()} {conf:.2f}"
            
            # Draw center dot
            center = (int(det['center'][0]), int(det['center'][1]))
            cv2.circle(annotated_frame, center, 4, color, -1)
            
            # Draw label background + text
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1-h-8), (x1+w+8, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1+4, y1-4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_frame
    
    def detect_stream(self, source=0, display=True, record=False,
                     output_path=None, max_frames=None, use_gstreamer=False,
                     capture_width=1280, capture_height=720):
        """
        Detect drones in live video stream.
        
        Args:
            source: Video source (0 for camera, URL for stream)
            display: Show video feed
            record: Record output
            output_path: Output video path
            max_frames: Maximum frames to process
            use_gstreamer: Use GStreamer for CSI camera (Jetson only)
            capture_width: Camera capture width
            capture_height: Camera capture height
        """
        print("📹 Opening video stream...")
        
        # Setup video capture
        if use_gstreamer and self.is_jetson and isinstance(source, int):
            pipeline = self.create_gstreamer_pipeline(
                capture_width, capture_height, camera_id=source
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            print(f"✓ Using GStreamer pipeline for CSI camera {source}")
        else:
            # Use V4L2 explicitly on Jetson for the fastest USB camera path.
            backend = cv2.CAP_V4L2 if self.is_jetson else cv2.CAP_ANY
            cap = cv2.VideoCapture(source, backend)
            # Set capture resolution — lower res means less USB bus and decode work.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
            # Buffer of 1 prevents OpenCV accumulating stale frames.
            # Default is 10 frames which adds ~300 ms of latency.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        # Wrap capture in async reader to decouple I/O from inference
        cap = AsyncVideoCapture(cap)
        print("✓ Async frame capture enabled")
        
        # Get stream properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0 or fps > 120:
            fps = 30
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Stream: {width}x{height} @ {fps}fps")
        print(f"\n🎮 Controls: 'q'=Quit, 's'=Snapshot, 'r'=Record, 't'=Thermal info")
        print("▶️  Starting detection...\n")

        # ── Write a "starting up" placeholder frame so the dashboard's MJPEG
        # stream gets *something* immediately instead of waiting 10-40+ seconds
        # for the model to load and the first real detection frame to appear. ──
        try:
            placeholder = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Starting detection...",
                        (width // 2 - 140, height // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(placeholder, "(initializing camera & model)",
                        (width // 2 - 120, height // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            _startup_tmp = '/tmp/drone_detect_frame_tmp.jpg'
            cv2.imwrite(_startup_tmp, placeholder)
            os.rename(_startup_tmp, '/tmp/drone_detect_frame.jpg')
        except Exception as e:
            print(f"⚠️  Could not write startup placeholder frame: {e}")
        
        # Setup video writer
        writer = None
        recording = record
        if recording and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"🔴 Recording to: {output_path}")
        
        # Statistics
        self.start_time = time.time()
        self.frame_count = 0
        self.total_detections = 0
        frames_with_detections = 0
        
        consecutive_failures = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        print("❌ Failed to grab frame (30 consecutive failures). "
                              "Check device index and camera permissions.")
                        break
                    time.sleep(0.033)
                    continue
                consecutive_failures = 0

                self.frame_count += 1
                
                if max_frames and self.frame_count >= max_frames:
                    print(f"✓ Reached max frames: {max_frames}")
                    break
                
                # Detect
                annotated_frame, detections, inference_time = self.detect_frame(frame)
                
                # Update stats
                num_detections = len(detections)
                self.total_detections += num_detections
                if num_detections > 0:
                    frames_with_detections += 1
                
                # Calculate metrics
                elapsed = time.time() - self.start_time
                current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                avg_inference = np.mean(self.inference_times) if self.inference_times else 0
                
                # Add overlay
                self._add_overlay(annotated_frame, current_fps, inference_time, 
                                num_detections, recording)
                
                # Save latest annotated frame so dashboard_server.py can serve it
                # via MJPEG over HTTP (written atomically via temp file + rename)
                _frame_path = '/tmp/drone_detect_frame.jpg'
                _frame_tmp   = '/tmp/drone_detect_frame_tmp.jpg'
                try:
                    ok = cv2.imwrite(_frame_tmp, annotated_frame)
                    if not ok:
                        print(f"⚠️  cv2.imwrite returned False for {_frame_tmp}")
                    else:
                        os.rename(_frame_tmp, _frame_path)
                except Exception as e:
                    print(f"⚠️  Could not save frame: {e}")
                
                # Record
                if recording and writer:
                    writer.write(annotated_frame)
                
                # Display
                if display:
                    cv2.imshow('Jetson Drone Detection', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("\n⏹️  Stopping...")
                        break
                    elif key == ord('s'):
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        snap_path = f"jetson_snapshot_{ts}.jpg"
                        cv2.imwrite(snap_path, annotated_frame)
                        print(f"📸 Saved: {snap_path}")
                    elif key == ord('r'):
                        if not recording:
                            if not writer:
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                output_path = f"jetson_recording_{ts}.mp4"
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                            recording = True
                            print(f"🔴 Recording: {output_path}")
                        else:
                            recording = False
                            print("⏸️  Stopped recording")
                    elif key == ord('t'):
                        stats = self.get_system_stats()
                        print(f"\n📊 System Stats:")
                        for k, v in stats.items():
                            print(f"  {k}: {v}")
                
                # Periodic updates
                if self.frame_count % 100 == 0:
                    stats = self.get_system_stats()
                    temp_str = f" | Temp: {stats.get('temperature', 'N/A'):.1f}°C" if 'temperature' in stats else ""
                    print(f"Frame {self.frame_count} | FPS: {current_fps:.1f} | "
                          f"Inference: {avg_inference:.1f}ms | Detections: {num_detections}{temp_str}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final stats
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        avg_inference = np.mean(self.inference_times) if self.inference_times else 0
        
        print("\n" + "="*70)
        print("🏁 Detection Complete")
        print("="*70)
        print(f"Total frames: {self.frame_count}")
        print(f"Frames with detections: {frames_with_detections}")
        print(f"Total detections: {self.total_detections}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average inference: {avg_inference:.1f}ms")
        print("="*70 + "\n")
        
        return {
            'total_frames': self.frame_count,
            'frames_with_detections': frames_with_detections,
            'total_detections': self.total_detections,
            'average_fps': avg_fps,
            'average_inference_ms': avg_inference
        }
    
    def _add_overlay(self, frame, fps, inference_time, detections, recording):
        """Add information overlay to frame."""
        y = 30
        spacing = 35
        
        # FPS
        fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        y += spacing
        
        # Inference time
        cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += spacing
        
        # Detections
        det_color = (0, 255, 0) if detections > 0 else (255, 255, 255)
        cv2.putText(frame, f"Drones: {detections}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, det_color, 2)
        y += spacing

        # Tracker state
        if self.tracker is not None:
            state = self.tracker.state
            if state == 'TRACKING':
                t_color, t_text = (0, 255, 0), 'TRACKER: LOCK'
            elif state == 'COASTING':
                t_color, t_text = DroneTracker.TRACKER_COLOR, f'TRACKER: COAST ({self.tracker._coast_count}/{self.tracker.max_coast_frames})'
            else:
                t_color, t_text = (180, 180, 180), 'TRACKER: IDLE'
            cv2.putText(frame, t_text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, t_color, 2)
        
        # Temperature warning
        if self.monitor_thermal and self.thermal_throttle:
            cv2.putText(frame, "THERMAL THROTTLE", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Device indicator
        device_text = "TensorRT" if self.use_tensorrt else "CUDA"
        if self.fp16:
            device_text += " FP16"
        cv2.putText(frame, device_text, (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Recording indicator
        if recording:
            cv2.circle(frame, (frame.shape[1] - 30, 60), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 70, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def set_power_mode(mode='MAXN'):
    """
    Set Jetson power mode.
    
    Modes for Orin Nano:
    - MAXN (15W): Maximum performance
    - 10W: Balanced
    - 7W: Power saving
    """
    try:
        modes = {'MAXN': 0, '10W': 1, '7W': 2}
        if mode in modes:
            subprocess.run(['sudo', 'nvpmodel', '-m', str(modes[mode])], check=True)
            print(f"✓ Power mode set to: {mode}")
        else:
            print(f"⚠️  Unknown power mode: {mode}")
    except Exception as e:
        print(f"⚠️  Could not set power mode: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Jetson Orin Nano Optimized Drone Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with USB camera
  python detect_drones_jetson.py
  
  # CSI camera with GStreamer (faster)
  python detect_drones_jetson.py --gstreamer --source 0
  
  # TensorRT with FP16 (fastest)
  python detect_drones_jetson.py --tensorrt --fp16
  
  # Set power mode to maximum performance
  sudo nvpmodel -m 0  # Run before starting detection
  python detect_drones_jetson.py --tensorrt --fp16
  
  # RTSP stream
  python detect_drones_jetson.py --source "rtsp://192.168.1.100/stream"
        """
    )
    
    parser.add_argument('-s', '--source', type=str, default='0',
                       help='Video source (0 for camera, URL for stream)')
    parser.add_argument('-m', '--model', type=str, default='n',
                       choices=['n', 's'],
                       help='Model size (n=nano/fastest, s=small) (default: n)')
    parser.add_argument('-c', '--confidence', type=float, default=0.3,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--tensorrt', action='store_true',
                       help='Use TensorRT optimization (recommended)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 precision (recommended)')
    parser.add_argument('--gstreamer', action='store_true',
                       help='Use GStreamer for CSI camera (Jetson only)')
    parser.add_argument('-r', '--record', action='store_true',
                       help='Record output video')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--no-display', action='store_true',
                       help='No display (headless)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--no-thermal', action='store_true',
                       help='Disable thermal monitoring')
    parser.add_argument('--width', type=int, default=640,
                       help='Capture width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Capture height (default: 480)')
    parser.add_argument('--power-mode', type=str, choices=['MAXN', '10W', '7W'],
                       help='Set power mode (requires sudo)')
    parser.add_argument('--frame-skip', type=int, default=2,
                       help='Skip N frames between inferences (0=every frame, '
                            '1=every other frame, 2=every 3rd frame). Boosts display '
                            'FPS; CSRT tracker fills the gaps. (default: 2)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Inference image size in pixels (default: 640)')
    parser.add_argument('--debug', action='store_true',
                       help='Print every raw model detection to console — useful for '
                            'diagnosing why drones are not being detected (class names, '
                            'confidence scores, filter matches)')
    parser.add_argument('--no-tracker', action='store_true',
                       help='Disable CSRT tracker (tracker is on by default)')
    parser.add_argument('--tracker-coast', type=int, default=10,
                       help='Frames the tracker keeps running after YOLO last fired '
                            'before giving up. Higher = longer coasting. (default: 10)')
    
    args = parser.parse_args()
    
    # Set power mode if requested
    if args.power_mode:
        set_power_mode(args.power_mode)
    
    # Parse source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Generate output path
    output_path = args.output
    if args.record and not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"jetson_detection_{ts}.mp4"
    
    print("\n" + "="*70)
    print("Jetson Orin Nano Drone Detection")
    print("="*70)
    print(f"Source: {source}")
    print(f"Model: best.pt")
    print(f"TensorRT: {args.tensorrt}")
    print(f"FP16: {args.fp16}")
    print(f"GStreamer: {args.gstreamer}")
    print(f"Frame skip: {args.frame_skip} (inference every {args.frame_skip+1} frames)")
    print(f"Inference size: {args.imgsz}px")
    print(f"Tracker: {'disabled' if args.no_tracker else f'enabled (coast={args.tracker_coast})'}")
    print("="*70 + "\n")
    
    try:
        # Initialize detector
        detector = JetsonDroneDetector(
            model_size=args.model,
            confidence_threshold=args.confidence,
            use_tensorrt=args.tensorrt,
            fp16=args.fp16,
            monitor_thermal=not args.no_thermal,
            frame_skip=args.frame_skip,
            imgsz=args.imgsz,
            use_tracker=not args.no_tracker,
            tracker_coast_frames=args.tracker_coast,
            debug=args.debug
        )
        
        # Run detection
        detector.detect_stream(
            source=source,
            display=not args.no_display,
            record=args.record,
            output_path=output_path,
            max_frames=args.max_frames,
            use_gstreamer=args.gstreamer,
            capture_width=args.width,
            capture_height=args.height
        )
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()