"""
train_thesis_model.py
=====================
YOLO drone-detection training using manual + auto labels,
with subprocess isolation for Apple Silicon MPS memory safety.

WHY SUBPROCESS ISOLATION?
─────────────────────────
Ultralytics callbacks cannot free MPS memory because:
  1. empty_cache() only frees cached blocks — live tensors (weights, gradients,
     optimizer state) stay allocated for the entire lifetime of the trainer.
  2. The YOLO trainer accumulates Python objects that GC cannot reach.
  3. MPS Metal command buffers are ref-counted by the driver, not Python.

The only guaranteed fix is to let the process that owns the trainer EXIT.
Each block of epochs runs in a fresh subprocess. When it exits the OS
reclaims every byte of GPU memory before the next block starts.
"""

import csv
import gc
import os
import sys
import random
import shutil
import subprocess
import yaml
from pathlib import Path

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

MANUAL_LABELS = "/Users/panththakkar/Documents/Drone Detection/project-4-at-2026-04-09-02-39-f4d2a32b/labels"
AUTO_LABELS   = "/Users/panththakkar/Documents/Drone Detection/drone_dataset/data/train/labels"
IMAGES_DIR    = "/Users/panththakkar/Documents/Drone Detection/drone_dataset/data/train/images"
THESIS_DIR    = "/Users/panththakkar/Documents/Drone Detection/thesis_dataset"
MERGED_LABELS = os.path.join(THESIS_DIR, "merged_labels")

# Negative images live in subfolders — no label files needed, we generate
# empty .txt files for them automatically during dataset creation.
NEGATIVE_IMAGES_DIR = "/Users/panththakkar/Documents/Drone Detection/negative_images"
NEGATIVE_SUBFOLDERS = ["airplanes", "bird", "sky", "static"]

# Additional positive image directories (drone present).
# Subfolders are discovered automatically at runtime — no manual list needed.
USC_DRONE_DIR  = "/Users/panththakkar/Documents/Drone Detection/usc drone-001"
DRONE_DATA_DIR = "/Users/panththakkar/Documents/Drone Detection/drone data"


def _collect_images_from_dir(root: Path) -> tuple[list[Path], dict[str, int]]:
    """
    Walk every immediate subfolder of *root* and collect all image files.
    Returns:
        images        — flat list of Path objects
        folder_counts — {subfolder_name: image_count} for reporting
    """
    images: list[Path] = []
    folder_counts: dict[str, int] = {}

    if not root.exists():
        print(f"Warning: directory not found: {root}")
        return images, folder_counts

    subfolders = sorted([p for p in root.iterdir() if p.is_dir()])
    if not subfolders:
        # No subfolders — treat the root itself as the source
        subfolders = [root]

    for subfolder in subfolders:
        found = [
            p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
            for p in subfolder.glob(ext)
        ]
        folder_counts[subfolder.name] = len(found)
        images.extend(found)

    return images, folder_counts


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def merge_labels():
    """
    Merge manual (Label Studio) and auto labels into one folder.
    Manual labels always take priority over auto-labels.
    """
    os.makedirs(THESIS_DIR, exist_ok=True)
    os.makedirs(MERGED_LABELS, exist_ok=True)

    manual_files = set(os.listdir(MANUAL_LABELS))
    auto_files   = set(os.listdir(AUTO_LABELS)) if os.path.exists(AUTO_LABELS) else set()

    manual_used, auto_used, skipped = 0, 0, 0

    for fname in manual_files | auto_files:
        if not fname.endswith(".txt"):
            continue

        base = os.path.splitext(fname)[0]
        img_path = os.path.join(IMAGES_DIR, base + ".jpg")

        if not os.path.exists(img_path):
            skipped += 1
            continue

        # Manual label takes priority over auto-label
        if fname in manual_files:
            shutil.copy(os.path.join(MANUAL_LABELS, fname), os.path.join(MERGED_LABELS, fname))
            manual_used += 1
        else:
            shutil.copy(os.path.join(AUTO_LABELS, fname), os.path.join(MERGED_LABELS, fname))
            auto_used += 1

    print(f"Manual labels used:  {manual_used}")
    print(f"Auto labels used:    {auto_used}")
    print(f"Skipped (no image):  {skipped}")
    print(f"Total merged:        {manual_used + auto_used}")


def create_dataset():
    """
    Build train/val split from MANUAL LABELS ONLY (plus additional positive
    image directories) using symlinked image folders that Ultralytics reads
    natively.  Returns path to data.yaml.

    Positive sources
    ────────────────
    1. Manual labels (MANUAL_LABELS / IMAGES_DIR) — annotations present
    2. USC_DRONE_DIR  — subfolders auto-discovered, all images treated as positives
    3. DRONE_DATA_DIR — subfolders auto-discovered, all images treated as positives

    NOTE: Images from USC_DRONE_DIR and DRONE_DATA_DIR have no label files.
    They are assigned the class-0 ("drone") bounding box covering the full
    image (YOLO format: "0 0.5 0.5 1.0 1.0") so the trainer sees them as
    positives.  If you have real annotations for these directories, replace
    the full-image fallback with the correct label files.

    Negative sources
    ────────────────
    1. Empty manual labels (MANUAL_LABELS with blank .txt)
    2. NEGATIVE_IMAGES_DIR / NEGATIVE_SUBFOLDERS — no label files → empty .txt
    """
    import shutil as _shutil

    # Use manual labels directly — skip merge_labels() which pulls in
    # noisy auto-labels that caused overfitting in the previous run.
    label_source  = Path(MANUAL_LABELS)

    train_img_dir = Path(THESIS_DIR) / "images" / "train"
    val_img_dir   = Path(THESIS_DIR) / "images" / "val"
    train_lbl_dir = Path(THESIS_DIR) / "labels" / "train"
    val_lbl_dir   = Path(THESIS_DIR) / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Clear any stale Ultralytics dataset caches — if these exist from a
    # previous run with fewer images, Ultralytics will reuse them and
    # silently load only the old subset of images every time.
    for cache_file in (Path(THESIS_DIR) / "labels").rglob("*.cache"):
        cache_file.unlink()
        print(f"Deleted stale cache: {cache_file}")

    # Always write data.yaml regardless of whether we rebuild folders
    data_yaml = {
        "path":  str(Path(THESIS_DIR).resolve()),
        "train": str((Path(THESIS_DIR) / "images" / "train").resolve()),
        "val":   str((Path(THESIS_DIR) / "images" / "val").resolve()),
        "nc":    1,
        "names": ["drone"],
    }
    yaml_path = os.path.join(THESIS_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
    print(f"data.yaml created: {yaml_path}")

    # ── Collect additional positives from the two new directories ──────────
    usc_images,  usc_counts  = _collect_images_from_dir(Path(USC_DRONE_DIR))
    data_images, data_counts = _collect_images_from_dir(Path(DRONE_DATA_DIR))
    extra_positive_images    = usc_images + data_images
    extra_positive_set       = set(extra_positive_images)

    # Count ALL labels (positives + negatives) to detect stale dataset
    available_labels   = len([f for f in os.listdir(MANUAL_LABELS) if f.endswith(".txt")])
    available_total    = available_labels + len(extra_positive_images)
    expected_train     = int(0.8 * available_total)

    existing_train = sum(1 for _ in train_img_dir.iterdir()) if train_img_dir.exists() else 0
    existing_val   = sum(1 for _ in val_img_dir.iterdir())   if val_img_dir.exists()   else 0

    # Rebuild if empty OR if count is significantly off from expected
    needs_rebuild = existing_train == 0 or abs(existing_train - expected_train) > 10

    if not needs_rebuild:
        print(f"Dataset valid: {existing_train} train / {existing_val} val. Skipping rebuild.")
        return yaml_path
    elif existing_train > 0:
        print(f"Dataset mismatch: found {existing_train} train, expected ~{expected_train}.")
        print("Clearing stale dataset and rebuilding from manual labels only...")
        import shutil as _shutil2
        for folder in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            if folder.exists():
                _shutil2.rmtree(str(folder))
                folder.mkdir(parents=True, exist_ok=True)
        for cache_file in (Path(THESIS_DIR) / "labels").rglob("*.cache"):
            cache_file.unlink()

    if True:
        # ── Positives: manual labels with annotations ─────────────────────
        label_files = [f for f in os.listdir(MANUAL_LABELS) if f.endswith(".txt")]

        positive_paths = []
        negative_paths = []   # from manual labels (empty .txt files)
        missing_images = 0

        for label_file in label_files:
            base       = os.path.splitext(label_file)[0]
            image_path = Path(IMAGES_DIR) / (base + ".jpg")

            if not image_path.exists():
                missing_images += 1
                continue

            with open(os.path.join(MANUAL_LABELS, label_file)) as f:
                content = f.read().strip()

            if content:
                positive_paths.append(image_path)
            else:
                negative_paths.append(image_path)

        if missing_images:
            print(f"Warning: {missing_images} label file(s) skipped — image not found")

        # Merge additional positives (USC + drone data) into positive_paths
        positive_paths.extend(extra_positive_images)

        # ── Negatives: images from the dedicated negative_images folder ───
        # These have no label files at all. We collect them here and will
        # write empty .txt files for them when building the dataset folders.
        folder_negatives = []
        neg_dir = Path(NEGATIVE_IMAGES_DIR)
        subfolder_counts = {}

        for subfolder in NEGATIVE_SUBFOLDERS:
            subfolder_path = neg_dir / subfolder
            if not subfolder_path.exists():
                print(f"Warning: negative subfolder not found: {subfolder_path}")
                continue
            images = [
                p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
                for p in subfolder_path.glob(ext)
            ]
            subfolder_counts[subfolder] = len(images)
            folder_negatives.extend(images)

        # Combine negatives from both sources
        all_negatives = negative_paths + folder_negatives

        # Print dataset composition before splitting
        total   = len(positive_paths) + len(all_negatives)
        neg_pct = (len(all_negatives) / total * 100) if total else 0
        manual_pos = len(positive_paths) - len(extra_positive_images)

        print(f"\n── Dataset composition ──────────────────────────────")
        print(f"  Positives (drone present)    : {len(positive_paths):>5}  ({100 - neg_pct:.1f}%)")
        print(f"  ├─ manual labels             : {manual_pos:>5}")
        for subfolder, count in usc_counts.items():
            print(f"  ├─ usc drone-001/{subfolder:<12}: {count:>5}")
        for subfolder, count in data_counts.items():
            print(f"  ├─ drone data/{subfolder:<15}: {count:>5}")
        print(f"  Negatives — labelled empty   : {len(negative_paths):>5}")
        for subfolder, count in subfolder_counts.items():
            print(f"  Negatives — {subfolder:<16}  : {count:>5}")
        print(f"  Negatives total              : {len(all_negatives):>5}  ({neg_pct:.1f}%)")
        print(f"  Total                        : {total:>5}")
        print(f"────────────────────────────────────────────────────\n")

        # Stratified 80/20 split — shuffle each group independently
        random.seed(42)
        random.shuffle(positive_paths)
        random.shuffle(all_negatives)

        pos_split = int(0.8 * len(positive_paths))
        neg_split = int(0.8 * len(all_negatives))

        train_images = positive_paths[:pos_split] + all_negatives[:neg_split]
        val_images   = positive_paths[pos_split:]  + all_negatives[neg_split:]

        # Shuffle again so negatives aren't bunched at the end of each split
        random.shuffle(train_images)
        random.shuffle(val_images)

        neg_set      = set(all_negatives)
        train_neg    = sum(1 for p in train_images if p in neg_set)
        val_neg      = sum(1 for p in val_images   if p in neg_set)

        print(f"Train : {len(train_images):>5} images  ({len(train_images)-train_neg} pos / {train_neg} neg)")
        print(f"Val   : {len(val_images):>5} images  ({len(val_images)-val_neg} pos / {val_neg} neg)")

        # ── Collect images that came from the folder negatives ────────────
        # These have no existing label file — we write empty .txt for them.
        folder_neg_set = set(folder_negatives)

        # Full-image positive label for extra directories with no annotation files.
        # YOLO format: <class> <cx> <cy> <w> <h> (all normalised to 1.0)
        FULL_IMAGE_LABEL = "0 0.5 0.5 1.0 1.0\n"

        # Create symlinks for images and copy/create labels.
        # Positives (manual)         → copy label from MANUAL_LABELS
        # Positives (USC / data dir) → write full-image label (class 0)
        # Empty-labelled negatives   → copy the empty .txt from MANUAL_LABELS
        # Folder negatives           → write a new empty .txt file
        for img_path in train_images:
            lnk = train_img_dir / img_path.name
            if not lnk.exists():
                lnk.symlink_to(img_path.resolve())
            lbl_dst = train_lbl_dir / (img_path.stem + ".txt")
            if not lbl_dst.exists():
                if img_path in folder_neg_set:
                    # Folder negative — create an empty label file
                    lbl_dst.touch()
                elif img_path in extra_positive_set:
                    # Extra positive (USC / drone data) — full-image bounding box
                    lbl_dst.write_text(FULL_IMAGE_LABEL)
                else:
                    lbl_src = Path(MANUAL_LABELS) / (img_path.stem + ".txt")
                    if lbl_src.exists():
                        _shutil.copy(lbl_src, lbl_dst)

        for img_path in val_images:
            lnk = val_img_dir / img_path.name
            if not lnk.exists():
                lnk.symlink_to(img_path.resolve())
            lbl_dst = val_lbl_dir / (img_path.stem + ".txt")
            if not lbl_dst.exists():
                if img_path in folder_neg_set:
                    lbl_dst.touch()
                elif img_path in extra_positive_set:
                    lbl_dst.write_text(FULL_IMAGE_LABEL)
                else:
                    lbl_src = Path(MANUAL_LABELS) / (img_path.stem + ".txt")
                    if lbl_src.exists():
                        _shutil.copy(lbl_src, lbl_dst)

        print("Image symlinks and label folders created.")

    return yaml_path


# =============================================================================
# Single-block worker (runs inside a subprocess — owns all GPU memory)
# =============================================================================

def _train_one_block(
    model_path,
    yaml_path,
    start_epoch,
    end_epoch,
    total_epochs,
    batch_size,
    imgsz,
    results_dir,
    crash_log_path,
):
    """
    Train epochs [start_epoch, end_epoch) then EXIT.
    All MPS memory is released when this process exits.
    Writes crash_log_path with traceback if an exception occurs.
    """
    import gc, os, traceback, torch, matplotlib
    matplotlib.use("Agg")
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["ULTRALYTICS_AMP"] = "0"
    from ultralytics import YOLO

    try:
        print(f"\n{'='*60}")
        print(f"SCRIPT VERSION: fraction=1.0 fix active")
        print(f"Block: epochs {start_epoch+1} to {end_epoch} of {total_epochs}")
        print(f"Model: {model_path}  |  imgsz={imgsz}  batch={batch_size}")
        print(f"{'='*60}")

        model = YOLO(model_path)
        model.overrides["amp"] = False

        # No-op scaler to avoid MPS stride bug in backward()
        class _NoOpScaler:
            def scale(self, loss):           return loss
            def unscale_(self, optimizer):   pass
            def step(self, optimizer):       optimizer.step()
            def update(self):                pass
            def state_dict(self):            return {}
            def load_state_dict(self, d):    pass

        def _patch_scaler(trainer):
            trainer.scaler = _NoOpScaler()
            # Fix the epoch counter so it shows the true cumulative epoch
            # (e.g. 8/100) instead of resetting to 1/100 each block.
            trainer.start_epoch = start_epoch
            trainer.epochs      = total_epochs

        model.add_callback("on_train_start", _patch_scaler)

        # Flush MPS memory at the end of every epoch to prevent the
        # progressive slowdown (epoch 1: 7:43, epoch 2: 8:49, epoch 3: 10:32).
        # The leak is caused by Metal command buffers and cached allocations
        # accumulating across epochs within the same process. Synchronizing
        # the MPS stream and emptying the cache after each epoch reclaims
        # these allocations before the next epoch's dataloader spins up.
        def _flush_mps(trainer):
            if torch.backends.mps.is_available():
                torch.mps.synchronize()   # wait for all Metal ops to finish
                torch.mps.empty_cache()   # release cached (but unused) MPS memory
            gc.collect()                  # collect any Python objects referencing tensors

        model.add_callback("on_train_epoch_end", _flush_mps)
        model.add_callback("on_val_end", _flush_mps)

        # Write a results row to the master CSV after every epoch.
        # true_epoch is computed from start_epoch + internal counter so it
        # always reflects the cumulative epoch number, not the per-block one.
        master_csv_path = Path(results_dir) / "results_full.csv"
        def _log_epoch(trainer):
            metrics    = trainer.metrics or {}
            true_epoch = start_epoch + trainer.epoch + 1

            # Guard: never log an epoch beyond what this block is allowed to run.
            # This prevents a crashed/restarted subprocess from logging extra rows.
            if true_epoch > end_epoch:
                return

            # Read losses by name from trainer.label_loss_items + trainer.loss_items
            # to ensure correct column alignment regardless of model architecture.
            loss_names = list(trainer.label_loss_items(prefix="train")) \
                if hasattr(trainer, "label_loss_items") else []
            loss_items = trainer.loss_items \
                if hasattr(trainer, "loss_items") and trainer.loss_items is not None else []

            loss_dict = {}
            for i, name in enumerate(loss_names):
                if i < len(loss_items):
                    try:
                        loss_dict[name] = float(loss_items[i])
                    except Exception:
                        loss_dict[name] = ""

            def _safe(metrics, key):
                try:
                    return float(metrics.get(key, ""))
                except Exception:
                    return ""

            row = {
                "epoch":             true_epoch,
                "time":              round(getattr(trainer, "epoch_time", 0) or 0, 3),
                "train/box_loss":    loss_dict.get("train/box_loss",  loss_dict.get("train/box_om_loss",  "")),
                "train/cls_loss":    loss_dict.get("train/cls_loss",  ""),
                "train/dfl_loss":    loss_dict.get("train/dfl_loss",  loss_dict.get("train/dfl_om_loss",  "")),
                "metrics/precision": _safe(metrics, "metrics/precision(B)"),
                "metrics/recall":    _safe(metrics, "metrics/recall(B)"),
                "metrics/mAP50":     _safe(metrics, "metrics/mAP50(B)"),
                "metrics/mAP50-95":  _safe(metrics, "metrics/mAP50-95(B)"),
                "val/box_loss":      _safe(metrics, "val/box_loss"),
                "val/cls_loss":      _safe(metrics, "val/cls_loss"),
                "val/dfl_loss":      _safe(metrics, "val/dfl_loss"),
            }

            # Deduplicate: rewrite file keeping only one row per epoch
            header   = ",".join(row.keys())
            new_line = ",".join(str(v) for v in row.values())

            if master_csv_path.exists():
                lines = master_csv_path.read_text().splitlines()
                kept  = [l for l in lines[1:]
                         if l and l.split(",")[0] != str(true_epoch)]
                with open(master_csv_path, "w") as f:
                    f.write(header + "\n")
                    for l in kept:
                        f.write(l + "\n")
                    f.write(new_line + "\n")
            else:
                with open(master_csv_path, "w") as f:
                    f.write(header + "\n")
                    f.write(new_line + "\n")

        model.add_callback("on_fit_epoch_end", _log_epoch)

        # Stop the trainer at the block boundary using the TRUE cumulative epoch.
        # Previously this used trainer.epoch (internal, resets to 0 each subprocess)
        # which meant a crashed/restarted block never hit its end_epoch and ran
        # all the way to total_epochs — producing the 151–299 runaway rows.
        def _stop_at_block_end(trainer):
            true_epoch = start_epoch + trainer.epoch + 1
            if true_epoch >= end_epoch:
                trainer.stop = True

        model.add_callback("on_train_epoch_end", _stop_at_block_end)

        # DO NOT use resume=True — it restores all args from the checkpoint
        # including fraction, which was cached as ~0.12 from a previous run
        # and causes only 307/2571 batches to be used per epoch.
        #
        # Instead we load last.pt as weights only (no config restoration)
        # and pass all training args explicitly every block. The LR schedule
        # is preserved because we pass epochs=total_epochs and the trainer
        # computes the correct LR position from start_epoch in the checkpoint.
        print(f"Resume: weights-only from {model_path}  |  target_epochs={total_epochs}")

        model.train(
            data=yaml_path,
            epochs=total_epochs,        # full schedule so LR is correct
            batch=batch_size,
            imgsz=imgsz,
            resume=False,               # never use resume — we control epoch continuity
            fraction=1.0,               # CRITICAL: always use full dataset
            rect=False,                 # disable rect sorting — irect truncates dataset
            # ── Memory ───────────────────────────────────────────
            cache=False,
            workers=2,
            amp=False,
            # ── Augmentation — tuned for aerial drone detection ──
            mosaic=1.0,         # re-enabled — critical for small datasets
            close_mosaic=20,    # disable mosaic in final 20 epochs — forces model to
                                # learn from clean single images, reduces false positives
            mixup=0.15,         # light mixup for variety
            copy_paste=0.1,     # paste drone instances onto new backgrounds
            hsv_h=0.02,
            hsv_s=0.8,
            hsv_v=0.5,
            fliplr=0.5,
            flipud=0.5,         # drones appear at all orientations
            scale=0.9,          # wider scale range — drones vary greatly in size
            translate=0.2,
            degrees=90.0,       # full rotation — drones appear at any angle
            shear=5.0,
            erasing=0.3,        # simulate occlusion
            # ── Localisation ─────────────────────────────────────
            box=9.0,            # up from default 7.5 — penalises imprecise boxes more,
                                # directly improves mAP50-95
            # ── Optimizer ────────────────────────────────────────
            optimizer="SGD",
            lr0=0.0005,         # lower lr — more careful learning with small dataset
            lrf=0.005,          # cosine decay to very low lr for fine-tuning at end
            weight_decay=0.0005,
            nbs=32,
            warmup_epochs=3.0,  # longer warmup for stability
            # ── NMS ──────────────────────────────────────────────
            conf=0.1,           # up from 0.01 — more honest precision signal during
                                # training validation, pushes model to be more selective
            max_det=50,
            iou=0.5,
            # ── Training control ─────────────────────────────────
            patience=40,        # more patience — small dataset has noisy mAP
            pretrained=True,
            device="mps",
            save=True,
            save_period=-1,
            plots=False,
            project=str(Path(results_dir).parent),
            name="thesis_drone_training",
            exist_ok=True,
            verbose=True,
        )

    except Exception:
        tb = traceback.format_exc()
        print(f"\nBLOCK CRASHED:\n{tb}")
        Path(crash_log_path).write_text(tb)
        sys.exit(1)

    finally:
        try:
            del model
        except Exception:
            pass
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        gc.collect()
        print("Block done. Process exiting — all MPS memory released.")


# =============================================================================
# CSV UTILITIES
# =============================================================================

def _append_block_csv(results_dir: Path, master_csv: Path, block_start: int,
                      block_end: int, total_epochs: int):
    """
    After each block completes, merge that block's results.csv rows into
    the master results_full.csv, deduplicating by epoch number.

    Only rows whose epoch number falls within [block_start+1, block_end]
    AND is <= total_epochs are written.  This is the second line of defence
    against crashed subprocesses logging extra rows beyond total_epochs —
    the first being the guard in _log_epoch inside the subprocess itself.
    """
    block_csv = results_dir / "results.csv"
    if not block_csv.exists():
        print("Warning: no results.csv found for this block, skipping append.")
        return

    rows = block_csv.read_text().splitlines()
    if len(rows) < 2:
        return

    header = rows[0]
    n_cols = len(header.split(","))

    # Only keep rows with correct column count AND epoch within valid range
    valid_rows = []
    for r in rows[1:]:
        parts = r.split(",")
        if len(parts) != n_cols:
            continue
        try:
            epoch_num = int(float(parts[0]))
        except ValueError:
            continue
        if block_start < epoch_num <= block_end and epoch_num <= total_epochs:
            valid_rows.append((epoch_num, r))

    if not valid_rows:
        print("Warning: no valid in-range rows found in block results.csv")
        return

    if not master_csv.exists():
        with open(master_csv, "w") as f:
            f.write(header + "\n")
            for _, row in valid_rows:
                f.write(row + "\n")
    else:
        existing       = master_csv.read_text().splitlines()
        existing_header = existing[0] if existing else header
        existing_rows  = {}
        for line in existing[1:]:
            parts = line.split(",")
            if parts:
                try:
                    existing_rows[int(float(parts[0]))] = line
                except ValueError:
                    pass

        # Add/overwrite with new valid block rows
        for epoch_num, row in valid_rows:
            existing_rows[epoch_num] = row

        # Rewrite master CSV sorted by epoch, capped at total_epochs
        with open(master_csv, "w") as f:
            f.write(existing_header + "\n")
            for epoch_num in sorted(existing_rows.keys()):
                if epoch_num <= total_epochs:
                    f.write(existing_rows[epoch_num] + "\n")

    print(f"Appended {len(valid_rows)} epoch(s) (ep {block_start+1}–{block_end}) to {master_csv}")


def _save_clean_weights(last_pt: Path, clean_pt: Path):
    """
    Copy last.pt to clean_weights.pt with all cached training args stripped.
    This lets the next block load the learned weights without inheriting
    any stale config (fraction, epochs, YOLO26-specific args, etc.)
    that would override our explicit training arguments.
    """
    import torch
    try:
        ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
        # Strip all cached training args — next block will use its own
        ckpt.pop("train_args", None)
        ckpt.pop("training_results", None)
        torch.save(ckpt, str(clean_pt))
        print(f"Clean weights saved: {clean_pt}")
        del ckpt
    except Exception as e:
        print(f"Warning: could not save clean weights: {e}. Will use last.pt directly.")
        import shutil
        shutil.copy(str(last_pt), str(clean_pt))


# =============================================================================
# Orchestrator (main process — never touches the GPU)
# =============================================================================

def train_thesis_model(
    yaml_path,
    total_epochs=100,
    epochs_per_block=5,
    batch_size=4,
    imgsz=640,
    max_retries=2,
):
    """
    Train using subprocess isolation so GPU memory is fully reset every
    epochs_per_block epochs. The main process never touches the GPU.

    Args:
        yaml_path (str):         Path to data.yaml.
        total_epochs (int):      Total training epochs (default: 100).
        epochs_per_block (int):  Epochs per subprocess block (default: 5).
        batch_size (int):        Batch size (default: 4).
        imgsz (int):             Input image size (default: 640).
        max_retries (int):       Retries per crashed block (default: 2).
    """
    model_path   = "yolo11n.pt"   # upgraded from nano — more capacity, ~2x slower
    script_dir   = Path(__file__).parent
    results_dir  = script_dir / "runs" / "detect" / "runs" / "detect" / "thesis_drone_training"
    weights_dir  = results_dir / "weights"
    master_csv   = results_dir / "results_full.csv"
    crash_log    = results_dir / "crash.log"
    clean_pt     = weights_dir / "clean_weights.pt"
    last_pt      = weights_dir / "last.pt"
    best_pt      = weights_dir / "best.pt"
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Thesis Drone Detection Training (subprocess-isolated, MPS-safe)")
    print(f"  imgsz={imgsz}  batch={batch_size}  epochs={total_epochs}  block={epochs_per_block}")
    print("=" * 60)

    # --fresh flag: wipe all weights and CSV for a completely clean start
    if "--fresh" in sys.argv:
        print("\n--fresh flag detected. Deleting all old weights and CSV...")
        for pt in weights_dir.glob("*.pt"):
            pt.unlink()
            print(f"  Deleted: {pt.name}")
        if master_csv.exists():
            master_csv.unlink()
            print(f"  Deleted: {master_csv.name}")
        print("Fresh start ready.\n")
    else:
        # Always remove stale per-epoch checkpoints (epoch1.pt, epoch2.pt etc.)
        for pt in weights_dir.glob("epoch*.pt"):
            pt.unlink()
            print(f"Removed stale checkpoint: {pt.name}")

    # Detect how many epochs have already been completed so we can
    # skip them on restart. Read from clean_weights.pt if it exists,
    # falling back to last.pt. The epoch field is 0-based so add 1.
    completed_epochs = 0
    for ckpt_path in [clean_pt, last_pt]:
        if ckpt_path.exists():
            try:
                import torch as _torch
                ckpt = _torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                saved_epoch = ckpt.get("epoch", -1)
                del ckpt
                if saved_epoch >= 0:
                    completed_epochs = saved_epoch + 1
                    print(f"Resuming from epoch {completed_epochs} (read from {ckpt_path.name})")
                    break
            except Exception as e:
                print(f"Could not read epoch from {ckpt_path.name}: {e}")

    if completed_epochs >= total_epochs:
        print(f"Training already complete ({completed_epochs}/{total_epochs} epochs).")
        return

    if completed_epochs > 0:
        print(f"Skipping epochs 1–{completed_epochs}, starting from epoch {completed_epochs + 1}.")

    for block_start in range(completed_epochs, total_epochs, epochs_per_block):
        block_end    = min(block_start + epochs_per_block, total_epochs)
        # Block 1: start from yolo26n.pt pretrained weights
        # Block 2+: start from clean_weights.pt (last.pt with all cached
        #           args stripped so fraction/config don't get restored)
        resume_model = str(clean_pt) if block_start > 0 and clean_pt.exists() else model_path

        print(f"\n{'─'*60}")
        print(f"Launching block: epochs {block_start+1}–{block_end}  ({block_end - block_start} epochs)")
        print(f"Resuming from:   {resume_model}")
        print(f"{'─'*60}")

        worker_code = f"""
import sys
sys.path.insert(0, {repr(str(Path(__file__).parent))})
from train_thesis_model import _train_one_block
_train_one_block(
    model_path     = {repr(resume_model)},
    yaml_path      = {repr(str(yaml_path))},
    start_epoch    = {block_start},
    end_epoch      = {block_end},
    total_epochs   = {total_epochs},
    batch_size     = {batch_size},
    imgsz          = {imgsz},
    results_dir    = {repr(str(results_dir))},
    crash_log_path = {repr(str(crash_log))},
)
"""
        success = False
        for attempt in range(1, max_retries + 2):
            if attempt > 1:
                print(f"\nRetry {attempt-1}/{max_retries} for block {block_start+1}–{block_end}...")

            result = subprocess.run([sys.executable, "-c", worker_code])

            if result.returncode == 0:
                success = True
                break
            else:
                if crash_log.exists():
                    print(f"\nCrash log:\n{'─'*40}")
                    print(crash_log.read_text())
                    print('─' * 40)
                    crash_log.unlink()

                if attempt <= max_retries:
                    print(f"Block failed (exit {result.returncode}). Retrying...")
                else:
                    print(f"Block failed after {max_retries} retries. Stopping.")
                    print(f"Last successful epoch was ~{block_start}.")
                    print(f"Best model saved at: {best_pt}")
                    return

        if success:
            # Strip cached args from last.pt before next block loads it
            if last_pt.exists():
                _save_clean_weights(last_pt, clean_pt)
            _append_block_csv(results_dir, master_csv,
                              block_start, block_end, total_epochs)
            print(f"Block complete. All GPU memory released.")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results:      runs/thesis_drone_training")
    print(f"Full history: {master_csv}")
    print(f"Best model:   {best_pt}")
    print("=" * 60)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # --fresh: wipe dataset folders, caches, and weights for a clean start
    if "--fresh" in sys.argv:
        print("\n--fresh: clearing dataset folders and caches...")
        thesis_dir = Path(THESIS_DIR)
        for folder in ["images", "labels"]:
            target = thesis_dir / folder
            if target.exists():
                import shutil as _shutil
                _shutil.rmtree(str(target))
                print(f"  Deleted: {target}")
        data_yaml = thesis_dir / "data.yaml"
        if data_yaml.exists():
            data_yaml.unlink()
            print(f"  Deleted: data.yaml")
        print("Dataset cleared.\n")

    yaml_path = create_dataset()

    train_thesis_model(
        yaml_path        = yaml_path,
        total_epochs     = 150,     # longer run — yolo26s needs more epochs to converge
        epochs_per_block = 1,       # 1 epoch per block — only reliable MPS memory fix
        batch_size       = 8,       # smaller batch — yolo26s uses more memory than nano
        imgsz            = 832,     # up from 640 — better localisation on small drones
        max_retries      = 2,
    )