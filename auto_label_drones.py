#!/usr/bin/env python3
"""
Auto-label drone images using YOLO for training dataset.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import shutil

def auto_label_images(image_dir: str, output_labels_dir: str, model_path="yolo11n.pt", conf=0.25, imgsz=640):
    model = YOLO(model_path)
    image_dir = Path(image_dir)
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Auto-labeling images in {image_dir}...")
    labeled_count = 0
    
    for img_path in image_dir.glob("*.jpg"):
        results = model(img_path, conf=conf, imgsz=imgsz, verbose=False)
        
        label_lines = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == 0:  # assuming class 0 = drone
                x_center, y_center, width, height = box.xywhn[0].tolist()
                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        if label_lines:
            label_path = output_labels_dir / f"{img_path.stem}.txt"
            label_path.write_text("\n".join(label_lines))
            labeled_count += 1
            print(f"✅ Labeled: {img_path.name} ({len(label_lines)} drones)")
        else:
            # Create empty label for negative images
            (output_labels_dir / f"{img_path.stem}.txt").touch()
    
    print(f"\nAuto-labeling complete! {labeled_count} images labeled.")
    return output_labels_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Path to folder with drone images")
    parser.add_argument("--output", default="auto_labels", help="Output labels directory")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model to use for labeling")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    
    auto_label_images(args.images, args.output, args.model, args.conf)