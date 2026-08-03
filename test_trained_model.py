#!/usr/bin/env python3
"""
Test Trained Drone Model
Tests a custom trained model on videos with:
- Accurate drone-specific detection rate reporting
- FP16, imgsz, and performance options
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import torch
import numpy as np
from datetime import datetime
from collections import deque
import time


# ── Drone classes the detector looks for ──────────────────────────────────────
DRONE_CLASSES = ['drone', 'bird', 'airplane', 'kite', 'frisbee', 'sports ball']


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_yolo_results(result, drone_classes):
    """
    Extract drone-specific detections from a YOLO result object.
    Only counts classes in drone_classes — fixes the detection-rate bug
    where any object (car, person, etc.) would count as a 'detection'.
    """
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        cls_id   = int(box.cls[0])
        cls_name = result.names[cls_id]
        if cls_name not in drone_classes:
            continue
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        detections.append({
            'class':      cls_name,
            'confidence': conf,
            'bbox':       xyxy,
            'center':     ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2),
        })
    return detections


def draw_detections(frame, detections):
    """Draw all detections onto frame in-place."""
    color_map = {
        'drone':       (0, 255, 0),
        'bird':        (255, 165, 0),
        'airplane':    (0, 0, 255),
        'kite':        (255, 0, 255),
        'frisbee':     (0, 255, 255),
        'sports ball': (255, 255, 0),
    }
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cls_name = det['class']
        conf     = det['confidence']
        color    = color_map.get(cls_name, (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cx, cy = int(det['center'][0]), int(det['center'][1])
        cv2.circle(frame, (cx, cy), 4, color, -1)

        label = f"{cls_name.upper()} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def add_overlay(frame, frame_count, total_frames, fps, num_detections,
                recording=False):
    """Burn stats onto the frame."""
    y, spacing = 30, 32

    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    y += spacing

    fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)
    y += spacing

    det_color = (0, 255, 0) if num_detections > 0 else (255, 255, 255)
    cv2.putText(frame, f"Drones: {num_detections}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, det_color, 2)

    if recording:
        cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (frame.shape[1] - 70, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ── Main test function ────────────────────────────────────────────────────────

def test_model(model_path, video_path, output_path=None, display=True,
               confidence=0.5, imgsz=640, fp16=False):
    """Test trained model on a video."""

    print("=" * 70)
    print("🧪 Testing Trained Drone Model")
    print("=" * 70)

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return

    print(f"\n📦 Model:      {model_path}")
    print(f"🎬 Video:      {video_path}")
    print(f"🎯 Confidence: {confidence}")
    print(f"📐 Imgsz:      {imgsz}px")
    print(f"⚡ FP16:       {fp16}")

    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"🔧 Device:     {device}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print("\n📥 Loading model...")
    model = YOLO(model_path)
    if cuda_available:
        model.to('cuda')
        if fp16:
            model.model.half()
            print("✓ FP16 enabled")
    print("✓ Model loaded")

    print("\n📹 Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    fps_src      = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✓ {width}x{height} @ {fps_src}fps  ({total_frames} frames)")

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (width, height))
        print(f"💾 Output: {output_path}")

    print(f"\n🎯 Counting detections for classes: {DRONE_CLASSES}")
    print("▶️  Processing...\n")

    frame_count            = 0
    total_drone_detections = 0
    detected_frames        = 0
    inference_times        = deque(maxlen=200)
    start_time             = time.time()
    display_fps            = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # YOLO inference
            t0 = time.time()
            with torch.no_grad():
                results = model(frame, conf=confidence, imgsz=imgsz, verbose=False)
            if cuda_available:
                torch.cuda.synchronize()
            inference_times.append((time.time() - t0) * 1000)

            # Drone-only detections
            dets = parse_yolo_results(results[0], DRONE_CLASSES)

            if len(dets) > 0:
                detected_frames        += 1
                total_drone_detections += len(dets)

            # Draw & overlay
            draw_detections(frame, dets)
            elapsed     = time.time() - start_time
            display_fps = frame_count / elapsed if elapsed > 0 else 0.0
            add_overlay(frame, frame_count, total_frames, display_fps,
                        len(dets), recording=writer is not None)

            if writer:
                writer.write(frame)

            if display:
                cv2.imshow('Trained Model Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Stopped by user")
                    break

            # Progress every 30 frames
            if frame_count % 30 == 0:
                avg_inf     = np.mean(inference_times)
                detect_rate = (detected_frames / frame_count) * 100
                print(f"  {frame_count}/{total_frames} "
                      f"| FPS: {display_fps:.1f} "
                      f"| Inference: {avg_inf:.1f}ms "
                      f"| Detection rate: {detect_rate:.1f}%")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # Final report
    avg_inference  = np.mean(inference_times) if inference_times else 0
    detection_rate = (detected_frames / frame_count * 100) if frame_count else 0

    print("\n" + "=" * 70)
    print("✅ Results")
    print("=" * 70)
    print(f"Total frames processed :  {frame_count}")
    print(f"Avg inference time     :  {avg_inference:.1f} ms")
    print(f"Avg display FPS        :  {display_fps:.1f}")
    print()
    print("── Detection breakdown (drone classes only) ──────────────────────")
    print(f"  Detected frames      :  {detected_frames} / {frame_count}  ({detection_rate:.1f}%)")
    print(f"  Total drone boxes    :  {total_drone_detections}")
    if output_path:
        print(f"\n💾 Saved to: {output_path}")
    print("=" * 70 + "\n")

    return {
        'total_frames':     frame_count,
        'detected_frames':  detected_frames,
        'detection_rate':   detection_rate,
        'avg_inference_ms': avg_inference,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Test Trained Drone Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python test_trained_model.py best.pt drone_video.mp4

  # Save output, no display window
  python test_trained_model.py best.pt drone_video.mp4 -o results/out.mp4 --no-display

  # Higher resolution — better on small/distant drones
  python test_trained_model.py best.pt drone_video.mp4 --imgsz 1280

  # FP16 for faster GPU inference
  python test_trained_model.py best.pt drone_video.mp4 --fp16

  # Lower confidence to catch more detections
  python test_trained_model.py best.pt drone_video.mp4 --confidence 0.2
        """
    )

    parser.add_argument('model', type=str, help='Path to trained model (.pt file)')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output video path (auto-generated if omitted)')
    parser.add_argument('-c', '--confidence', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference image size (default: 640)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision (faster on GPU, requires CUDA)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display window')

    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        video_name  = Path(args.video).stem
        ts          = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"test_results/{video_name}_detected_{ts}.mp4"
        Path("test_results").mkdir(exist_ok=True)

    test_model(
        model_path  = args.model,
        video_path  = args.video,
        output_path = output_path,
        display     = not args.no_display,
        confidence  = args.confidence,
        imgsz       = args.imgsz,
        fp16        = args.fp16,
    )


if __name__ == '__main__':
    main()