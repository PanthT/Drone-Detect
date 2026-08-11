#!/usr/bin/env python3
"""
Visualize YOLO-format label files overlaid on their corresponding images.
Usage:
    python visualize_labels.py --images <image_folder> --labels <label_folder>
    python visualize_labels.py --images drone_dataset/data/train/images --labels auto_labels
"""

import argparse
from pathlib import Path
import cv2
import random

# Consolidated dataset built by merging every source folder (see all_images/
# all_label). Filenames here are already globally unique (source-prefixed).
ALL_IMAGES_DIR = "/Users/panththakkar/Documents/Drone Detection/all_images"
ALL_LABEL_DIR  = "/Users/panththakkar/Documents/Drone Detection/all_label"

def draw_yolo_labels(image_path: Path, label_path: Path, img_size=None):
    """Draw YOLO-format labels on an image and return the annotated image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    if not label_path.exists():
        print(f"⚠️  No label file found for {image_path.name}")
        return img
    
    lines = label_path.read_text().strip().splitlines()
    if not lines or lines[0] == '':
        print(f"ℹ️  Empty label file (no detections): {image_path.name}")
        return img
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x_center, y_center, width, height = parts
        x_center, y_center, width, height = map(float, [x_center, y_center, width, height])
        
        # Convert normalized coords to pixel coords
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label text
        label_text = f"drone ({float(cls):.0f})"
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on images")
    parser.add_argument("--images", help="Folder containing images")
    parser.add_argument("--labels", help="Folder containing YOLO .txt label files")
    parser.add_argument("--all-images", action="store_true",
                         help="Shortcut for --images all_images --labels all_label "
                              "(the consolidated dataset folders).")
    parser.add_argument("--save", default="labeled_preview", help="Folder to save annotated images")
    parser.add_argument("--show", action="store_true", help="Display images one by one (press any key to advance)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images to process (0 = all)")
    args = parser.parse_args()

    if args.all_images:
        images_dir = Path(args.images or ALL_IMAGES_DIR)
        labels_dir = Path(args.labels or ALL_LABEL_DIR)
    else:
        if not args.images or not args.labels:
            parser.error("--images and --labels are required unless --all-images is passed")
        images_dir = Path(args.images)
        labels_dir = Path(args.labels)

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect image files
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_files = []
    for ext in extensions:
        image_files.extend(images_dir.rglob(ext))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    if args.limit > 0:
        image_files = image_files[:args.limit]
        print(f"Processing first {args.limit} images...")
    
    for i, img_path in enumerate(image_files):
        label_path = labels_dir / f"{img_path.stem}.txt"
        annotated = draw_yolo_labels(img_path, label_path)
        
        if annotated is None:
            continue
        
        # Save annotated image
        save_path = save_dir / f"labeled_{img_path.name}"
        cv2.imwrite(str(save_path), annotated)
        
        if args.show:
            cv2.imshow(f"Labeled: {img_path.name}", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images...")
    
    print(f"\n✅ Done! Annotated images saved to: {save_dir.resolve()}")
    print(f"   Total: {len(image_files)} images processed")


if __name__ == "__main__":
    main()