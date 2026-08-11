#!/usr/bin/env python3
"""
Auto-label drone images using YOLO for training dataset.
Uses specified folders for drone images:
- /Users/panththakkar/Documents/Drone Detection/drone_dataset/data
- /Users/panththakkar/Documents/Drone Detection/negative_images
- /Users/panththakkar/Documents/Drone Detection/drone data
- /Users/panththakkar/Documents/Drone Detection/usc drone-001
"""

import argparse
import os

# Must be set before ultralytics/torch import — avoids hard crashes on any op
# MPS doesn't support by falling back to CPU for that op only (same flag the
# training scripts use).
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path
from ultralytics import YOLO
import cv2
import shutil

# Consolidated dataset built by merging every source folder (see all_images/
# all_label). Filenames here are already globally unique (source-prefixed),
# so auto-labeling can write flat output with no collision risk.
ALL_IMAGES_DIR = "/Users/panththakkar/Documents/Drone Detection/all_images"
ALL_LABEL_DIR  = "/Users/panththakkar/Documents/Drone Detection/all_label"

# Default image directories to process
DEFAULT_IMAGE_DIRS = [
    "/Users/panththakkar/Documents/Drone Detection/drone_dataset/data/train/images",
    "/Users/panththakkar/Documents/Drone Detection/drone_dataset/data/valid/images",
    "/Users/panththakkar/Documents/Drone Detection/negative_images/airplane",
    "/Users/panththakkar/Documents/Drone Detection/negative_images/bird",
    "/Users/panththakkar/Documents/Drone Detection/negative_images/sky",
    "/Users/panththakkar/Documents/Drone Detection/negative_images/static",
    "/Users/panththakkar/Documents/Drone Detection/drone data/thermal data/images&label/thermal1",
    "/Users/panththakkar/Documents/Drone Detection/drone data/thermal data/images&label/thermal2",
    "/Users/panththakkar/Documents/Drone Detection/drone data/youtube/image&label/1",
    "/Users/panththakkar/Documents/Drone Detection/usc drone-001/Drone_frames/img",
    "/Users/panththakkar/Documents/Drone Detection/usc drone-001/Drone_frames_1",
    "/Users/panththakkar/Documents/Drone Detection/usc drone-001/Drone_garden",
    "/Users/panththakkar/Documents/Drone Detection/usc drone-001/Drone_garden1",
    "/Users/panththakkar/Documents/Drone Detection/usc drone-001/Drone_garden2",
    "/Users/panththakkar/Documents/Drone Detection/usc drone-001/Drone_test",
    "/Users/panththakkar/Documents/Drone Detection/usc drone-001/Drone_test1",
    "/Users/panththakkar/Documents/Drone Detection/usc drone-001/Drone8",
    "/Users/panththakkar/Documents/Drone Detection/usc drone-001/Drone9",
]


def collect_images_from_dir(image_dir: Path, extensions=("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")):
    """Collect all image files from a directory recursively."""
    images = []
    for ext in extensions:
        images.extend(image_dir.rglob(ext))
    return images


def fix_empty_labels(images_dir: str, labels_dir: str, model_path="yolo11n.pt", conf=0.25, imgsz=640, batch: int = 16,
                      max_images: int = 0, device: str = "mps"):
    """
    Re-runs detection ONLY on images whose existing label file in labels_dir
    is empty. Non-empty labels are never touched. An empty file is only
    overwritten if a drone is actually found; otherwise it's left empty,
    which is the correct outcome for genuinely drone-free frames.
    """
    import time
    import gc
    import torch
    model = YOLO(model_path)
    print(f"Model loaded. Running on device={device!r}")
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    empty_labels = [p for p in labels_dir.glob("*.txt") if p.stat().st_size == 0]
    print(f"Found {len(empty_labels)} empty label file(s) in {labels_dir}")

    pairs = []
    missing_image = 0
    for lbl in empty_labels:
        img = images_dir / f"{lbl.stem}.jpg"
        if img.exists():
            pairs.append((lbl, img))
        else:
            missing_image += 1
    print(f"{len(pairs)} have a matching image ({missing_image} skipped — no image found)")

    if not pairs:
        return

    if max_images > 0 and len(pairs) > max_images:
        pairs = pairs[:max_images]
        print(f"Limiting this run to the first {max_images} image(s) (chunked run).")

    filled = 0
    still_empty = 0
    i = 0
    start_time = time.time()

    # Handing thousands of paths to one model() call — even with stream=True —
    # blew MPS memory during setup before a single result was yielded (a
    # separate failure mode from the mid-run climb the periodic flush below
    # handles). Sub-batching keeps every individual model() call to a size
    # that's been confirmed to work (500), while flushing between sub-batches
    # keeps the whole run in one process — no repeated model-reload cost.
    SUBBATCH = 500
    for start in range(0, len(pairs), SUBBATCH):
        sub_pairs = pairs[start:start + SUBBATCH]

        results_iter = model(
            [str(img) for _, img in sub_pairs],
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            stream=True,
            batch=batch,
            device=device,
        )

        for (lbl_path, img_path), result in zip(sub_pairs, results_iter):
            i += 1
            label_lines = []
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # assuming class 0 = drone
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            if label_lines:
                lbl_path.write_text("\n".join(label_lines) + "\n")
                filled += 1
                status = f"✅ {len(label_lines)} drone(s)"
            else:
                still_empty += 1
                status = "⬜ empty"

            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta_s = (len(pairs) - i) / rate if rate > 0 else 0
            print(f"[{i}/{len(pairs)}] {img_path.name} -> {status}   "
                  f"(filled={filled} empty={still_empty} | {rate:.1f} img/s | ETA {eta_s/60:.1f} min)")

        # Flush MPS memory after every sub-batch (same torch.mps.synchronize()
        # + empty_cache() pattern train_thesis_model.py uses between epochs).
        if device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
            gc.collect()

    print(f"\n✅ Done. Filled {filled} previously-empty label(s). {still_empty} remained empty. "
          f"Total time: {(time.time()-start_time)/60:.1f} min")


def auto_label_images(image_dirs: list, output_labels_dir: str, model_path="yolo11n.pt", conf=0.25, imgsz=640,
                       existing_labels_dir: str = None, batch: int = 16, device: str = "mps"):
    import time
    model = YOLO(model_path)
    print(f"Model loaded. Running on device={device!r}")
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Skip images that already have a real (manual or merged) label so this
    # never re-labels — or risks being confused with — ground-truth data.
    existing_stems = set()
    if existing_labels_dir:
        existing_dir = Path(existing_labels_dir)
        if existing_dir.exists():
            existing_stems = {p.stem for p in existing_dir.glob("*.txt")}
            print(f"Found {len(existing_stems)} existing label(s) in {existing_dir} — those images will be skipped.")

    total_labeled = 0

    for image_dir_str in image_dirs:
        image_dir = Path(image_dir_str)
        if not image_dir.exists():
            print(f"⚠️  Directory not found, skipping: {image_dir}")
            continue

        print(f"\n🔍 Auto-labeling images in {image_dir}...")
        labeled_count = 0

        images = collect_images_from_dir(image_dir)
        if existing_stems:
            before = len(images)
            images = [p for p in images if p.stem not in existing_stems]
            print(f"  Skipping {before - len(images)} image(s) that already have a label")

        if not images:
            print("  Nothing left to label.")
            continue

        # Batched + streamed inference. all_images/ can hold tens of thousands
        # of files — model(single_path) in a loop pays per-call dispatch
        # overhead per image; stream=True keeps memory flat across the whole run.
        results_iter = model(
            [str(p) for p in images],
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            stream=True,
            batch=batch,
            device=device,
        )

        start_time = time.time()
        for i, (img_path, result) in enumerate(zip(images, results_iter), start=1):
            label_lines = []
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # assuming class 0 = drone
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            label_path = output_labels_dir / f"{img_path.stem}.txt"
            if label_lines:
                label_path.write_text("\n".join(label_lines))
                labeled_count += 1
                status = f"✅ {len(label_lines)} drone(s)"
            else:
                # Create empty label for negative images
                label_path.touch()
                status = "⬜ empty"

            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta_s = (len(images) - i) / rate if rate > 0 else 0
            print(f"[{i}/{len(images)}] {img_path.name} -> {status}   "
                  f"(labeled={labeled_count} | {rate:.1f} img/s | ETA {eta_s/60:.1f} min)")

        print(f"  Labeled {labeled_count} images from {image_dir.name}")
        total_labeled += labeled_count

    print(f"\n✅ Auto-labeling complete! Total images labeled: {total_labeled}")
    return output_labels_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", help="Path(s) to folder(s) with drone images", default=DEFAULT_IMAGE_DIRS)
    parser.add_argument("--all-images", action="store_true",
                         help="Auto-label the consolidated all_images/ folder instead of --images. "
                              "Images that already have a real label in all_label/ are skipped, and "
                              "output goes to a separate folder so ground-truth labels are never overwritten.")
    parser.add_argument("--output", default=None, help="Output labels directory")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model to use for labeling")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size")
    parser.add_argument("--fix-empty", action="store_true",
                         help="Instead of a normal labeling run, only re-check images whose "
                              "existing label in --labels is empty, using --images as the image "
                              "source. Non-empty labels are never touched.")
    parser.add_argument("--labels", help="Existing labels directory (required with --fix-empty)")
    parser.add_argument("--max-images", type=int, default=0,
                         help="With --fix-empty, only process this many empty labels in this run "
                              "(0 = no limit). Re-running picks up wherever the previous run left off, "
                              "since it re-scans for whatever's still empty each time.")
    parser.add_argument("--device", default="mps", help="Inference device: mps, cpu, or cuda:0 (default: mps)")
    args = parser.parse_args()

    if args.fix_empty:
        if not args.images or len(args.images) != 1 or not args.labels:
            parser.error("--fix-empty requires exactly one --images <dir> and --labels <dir>")
        fix_empty_labels(args.images[0], args.labels, args.model, args.conf, args.imgsz,
                          batch=args.batch, max_images=args.max_images, device=args.device)
    elif args.all_images:
        image_dirs = [ALL_IMAGES_DIR]
        output = args.output or "all_images_auto_labels"
        existing_labels_dir = ALL_LABEL_DIR
        auto_label_images(image_dirs, output, args.model, args.conf, args.imgsz,
                           existing_labels_dir=existing_labels_dir, batch=args.batch, device=args.device)
    else:
        image_dirs = args.images
        output = args.output or "auto_labels"
        auto_label_images(image_dirs, output, args.model, args.conf, args.imgsz,
                           existing_labels_dir=None, batch=args.batch, device=args.device)