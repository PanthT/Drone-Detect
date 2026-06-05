"""
train_few_shot.py
=================
Few-shot drone detection training using ONLY manually labelled images (~1,500).

WHY FEW-SHOT WITH MANUAL LABELS ONLY?
──────────────────────────────────────
The previous run peaked at mAP50 ~0.236 then declined because ~48k noisy
auto-labels drowned out the 1,500 quality manual labels. This script uses
only the clean manual labels, which typically yields better results:

  - No label noise to overfit on
  - Stronger signal per image
  - Faster training (5–10 min/epoch vs 5 min on subset of 50k)
  - Expected mAP50: 0.4–0.6 at epoch 100 with clean data

WHY SUBPROCESS ISOLATION?
──────────────────────────
Apple Silicon MPS leaks memory across epochs within the same process.
The only guaranteed fix is to EXIT the subprocess after each block.
Each block runs in a fresh subprocess; when it exits the OS reclaims
every byte of GPU memory before the next block starts.

USAGE:
    python train_few_shot.py

RESUME:
    Just re-run the same command. The script detects completed epochs
    from clean_weights.pt and skips them automatically.
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
# PATH CONFIGURATION — update these to match your setup
# =============================================================================

MANUAL_LABELS = "/Users/panththakkar/Documents/Drone Detection/project-4-at-2026-03-01-03-09-2db5a2eb/labels"
IMAGES_DIR    = "/Users/panththakkar/Documents/Drone Detection/drone_dataset/data/train/images"
DATASET_DIR   = "/Users/panththakkar/Documents/Drone Detection/few_shot_dataset"


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def create_dataset():
    """
    Build a clean dataset from manual labels only.
    Creates symlinked image folders + copied label folders that
    Ultralytics can read natively without fraction issues.
    Returns path to data.yaml.
    """
    dataset_dir   = Path(DATASET_DIR)
    train_img_dir = dataset_dir / "images" / "train"
    val_img_dir   = dataset_dir / "images" / "val"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_lbl_dir   = dataset_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Clear any stale Ultralytics dataset caches
    for cache_file in (dataset_dir / "labels").rglob("*.cache"):
        cache_file.unlink()
        print(f"Deleted stale cache: {cache_file}")

    # Always write data.yaml with absolute paths
    data_yaml = {
        "path":  str(dataset_dir.resolve()),
        "train": str(train_img_dir.resolve()),
        "val":   str(val_img_dir.resolve()),
        "nc":    1,
        "names": ["drone"],
    }
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    # Skip rebuild if already populated
    existing_train = list(train_img_dir.iterdir())
    if existing_train:
        train_count = sum(1 for _ in train_img_dir.iterdir())
        val_count   = sum(1 for _ in val_img_dir.iterdir())
        print(f"Dataset exists: {train_count} train / {val_count} val. Skipping rebuild.")
        print(f"data.yaml: {yaml_path}")
        return str(yaml_path)

    # Scan manual labels and find matching images
    manual_label_dir = Path(MANUAL_LABELS)
    images_dir       = Path(IMAGES_DIR)

    label_files = sorted(manual_label_dir.glob("*.txt"))
    print(f"Found {len(label_files)} manual label files.")

    paired = []
    missing_images = 0
    empty_labels   = 0

    for lbl_path in label_files:
        # Try .jpg and .png
        img_path = images_dir / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            img_path = images_dir / (lbl_path.stem + ".png")
        if not img_path.exists():
            missing_images += 1
            continue

        # Skip empty label files
        content = lbl_path.read_text().strip()
        if not content:
            empty_labels += 1
            continue

        paired.append((img_path, lbl_path))

    print(f"Valid pairs:     {len(paired)}")
    print(f"Missing images:  {missing_images}")
    print(f"Empty labels:    {empty_labels}")

    if len(paired) == 0:
        raise RuntimeError(
            f"No valid image-label pairs found!\n"
            f"  Labels dir: {MANUAL_LABELS}\n"
            f"  Images dir: {IMAGES_DIR}\n"
            f"Check that image filenames match label filenames."
        )

    # 80/20 train/val split — stratified by random seed for reproducibility
    random.seed(42)
    random.shuffle(paired)
    split      = int(0.8 * len(paired))
    train_pairs = paired[:split]
    val_pairs   = paired[split:]

    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    # Create symlinks for images, copy labels
    def _populate(pairs, img_dir, lbl_dir):
        for img_path, lbl_path in pairs:
            lnk = img_dir / img_path.name
            if not lnk.exists():
                lnk.symlink_to(img_path.resolve())
            dst = lbl_dir / lbl_path.name
            if not dst.exists():
                shutil.copy(str(lbl_path), str(dst))

    _populate(train_pairs, train_img_dir, train_lbl_dir)
    _populate(val_pairs,   val_img_dir,   val_lbl_dir)

    print(f"Dataset created: {dataset_dir}")
    print(f"data.yaml:       {yaml_path}")
    return str(yaml_path)


# =============================================================================
# SINGLE BLOCK TRAINER  (runs in subprocess — never called directly)
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
    """
    import gc, os, traceback, torch, matplotlib
    matplotlib.use("Agg")
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]      = "1"
    os.environ["ULTRALYTICS_AMP"]                  = "0"
    from ultralytics import YOLO

    try:
        print(f"\n{'='*60}")
        print(f"FEW-SHOT BLOCK: epochs {start_epoch+1} to {end_epoch} of {total_epochs}")
        print(f"Model: {model_path}  |  imgsz={imgsz}  batch={batch_size}")
        print(f"{'='*60}")

        model = YOLO(model_path)
        model.overrides["amp"] = False

        # No-op scaler — avoids MPS stride bug in backward()
        class _NoOpScaler:
            def scale(self, loss):           return loss
            def unscale_(self, optimizer):   pass
            def step(self, optimizer):       optimizer.step()
            def update(self):                pass
            def state_dict(self):            return {}
            def load_state_dict(self, d):    pass

        def _patch_scaler(trainer):
            trainer.scaler      = _NoOpScaler()
            trainer.start_epoch = start_epoch
            trainer.epochs      = total_epochs

        model.add_callback("on_train_start", _patch_scaler)

        # Flush MPS cache after every epoch to prevent progressive slowdown
        def _flush_mps(trainer):
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
            gc.collect()

        model.add_callback("on_train_epoch_end", _flush_mps)
        model.add_callback("on_val_end",         _flush_mps)

        # Stop at block boundary
        def _stop_at_block_end(trainer):
            if (trainer.epoch + 1) >= end_epoch:
                trainer.stop = True

        model.add_callback("on_train_epoch_end", _stop_at_block_end)

        model.train(
            data=yaml_path,
            epochs=total_epochs,
            batch=batch_size,
            imgsz=imgsz,
            resume=False,
            fraction=1.0,
            rect=False,
            # ── Memory ───────────────────────────────────────────
            cache=False,
            workers=2,
            amp=False,
            # ── Augmentation — aggressive for small dataset ──────
            # With only ~1,200 train images, heavy augmentation is
            # essential to prevent overfitting and improve generalisation.
            mosaic=1.0,         # re-enabled — critical for small datasets
            mixup=0.15,         # light mixup for variety
            copy_paste=0.1,     # paste drone instances onto new backgrounds
            hsv_h=0.02,
            hsv_s=0.8,
            hsv_v=0.5,
            fliplr=0.5,
            flipud=0.5,
            scale=0.8,          # wider scale range for size invariance
            translate=0.2,
            degrees=45.0,
            shear=5.0,
            perspective=0.0005,
            erasing=0.3,        # random erasing simulates occlusion
            # ── Optimizer ────────────────────────────────────────
            optimizer="SGD",
            lr0=0.005,          # slightly higher lr for small dataset
            lrf=0.01,
            weight_decay=0.0005,
            nbs=32,
            warmup_epochs=3.0,  # longer warmup for stability
            # ── NMS ──────────────────────────────────────────────
            conf=0.01,
            max_det=50,
            iou=0.5,
            # ── Training control ─────────────────────────────────
            patience=20,        # stop early if no improvement for 20 epochs
            pretrained=True,
            device="mps",
            save=True,
            save_period=-1,
            plots=False,
            project=str(Path(results_dir).parent),
            name="few_shot_drone",
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
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
            gc.collect()
        except Exception:
            pass
        print("\nBlock done. Process exiting — all MPS memory released.")


# =============================================================================
# CLEAN WEIGHTS HANDOFF
# =============================================================================

def _save_clean_weights(last_pt: Path, clean_pt: Path):
    """
    Strip all cached training args from last.pt and save as clean_weights.pt.
    This prevents stale fraction/config from being restored by the next block.
    """
    try:
        ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
        ckpt.pop("train_args",        None)
        ckpt.pop("training_results",  None)
        torch.save(ckpt, str(clean_pt))
        epoch = ckpt.get("epoch", "?")
        print(f"Clean weights saved (epoch {epoch}): {clean_pt.name}")
        del ckpt
    except Exception as e:
        print(f"Warning: could not save clean weights: {e}. Copying last.pt directly.")
        shutil.copy(str(last_pt), str(clean_pt))


# =============================================================================
# CSV UTILITIES
# =============================================================================

def _append_block_csv(results_dir: Path, master_csv: Path):
    """
    Append completed block's results.csv into master results_full.csv.
    Deduplicates by epoch number and filters out partial rows.
    """
    block_csv = results_dir / "results.csv"
    if not block_csv.exists():
        print("Warning: no results.csv found, skipping append.")
        return

    lines  = block_csv.read_text().splitlines()
    if len(lines) < 2:
        return

    header = lines[0]
    n_cols = len(header.split(","))
    valid  = [l for l in lines[1:] if len(l.split(",")) == n_cols]

    if not valid:
        print("Warning: no complete rows in block results.csv")
        return

    if not master_csv.exists():
        with open(master_csv, "w") as f:
            f.write(header + "\n")
            for row in valid:
                f.write(row + "\n")
    else:
        existing    = master_csv.read_text().splitlines()
        epoch_map   = {}
        for line in existing[1:]:
            parts = line.split(",")
            try:
                epoch_map[int(float(parts[0]))] = line
            except ValueError:
                pass
        for row in valid:
            parts = row.split(",")
            try:
                epoch_map[int(float(parts[0]))] = row
            except ValueError:
                pass
        with open(master_csv, "w") as f:
            f.write(existing[0] + "\n")
            for ep in sorted(epoch_map.keys()):
                f.write(epoch_map[ep] + "\n")

    print(f"Appended {len(valid)} epoch(s) to {master_csv.name}")


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def train_few_shot(
    yaml_path,
    total_epochs    = 150,      # more epochs — small dataset needs longer training
    epochs_per_block = 1,       # 1 epoch per block = guaranteed MPS memory release
    batch_size      = 8,        # smaller batch — only ~1,200 train images
    imgsz           = 640,
    max_retries     = 2,
):
    """
    Orchestrate few-shot training with subprocess isolation.
    The main process never touches the GPU.
    """
    model_path  = "yolo26s.pt"
    script_dir  = Path(__file__).parent
    results_dir = script_dir / "runs" / "few_shot_drone"
    master_csv  = results_dir / "results_full.csv"
    crash_log   = results_dir / "crash.log"
    clean_pt    = results_dir / "weights" / "clean_weights.pt"
    last_pt     = results_dir / "weights" / "last.pt"
    best_pt     = results_dir / "weights" / "best.pt"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Few-Shot Drone Detection Training")
    print(f"  Manual labels only  |  imgsz={imgsz}  batch={batch_size}")
    print(f"  epochs={total_epochs}  block={epochs_per_block}")
    print("=" * 60)

    # Detect completed epochs from checkpoint to enable clean resume
    completed_epochs = 0
    for ckpt_path in [clean_pt, last_pt]:
        if ckpt_path.exists():
            try:
                ckpt        = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                saved_epoch = ckpt.get("epoch", -1)
                del ckpt
                if saved_epoch >= 0:
                    completed_epochs = saved_epoch + 1
                    print(f"Resuming from epoch {completed_epochs} ({ckpt_path.name})")
                    break
            except Exception as e:
                print(f"Could not read epoch from {ckpt_path.name}: {e}")

    if completed_epochs >= total_epochs:
        print(f"Training already complete ({completed_epochs}/{total_epochs} epochs).")
        print(f"Best model: {best_pt}")
        return

    if completed_epochs > 0:
        print(f"Skipping epochs 1–{completed_epochs}, starting from {completed_epochs+1}.")

    for block_start in range(completed_epochs, total_epochs, epochs_per_block):
        block_end    = min(block_start + epochs_per_block, total_epochs)
        resume_model = str(clean_pt) if block_start > 0 and clean_pt.exists() else model_path

        print(f"\n{'─'*60}")
        print(f"Block: epochs {block_start+1}–{block_end}  |  weights: {Path(resume_model).name}")
        print(f"{'─'*60}")

        worker_code = f"""
import sys
sys.path.insert(0, {repr(str(Path(__file__).parent))})
from train_few_shot import _train_one_block
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
                print(f"Retry {attempt-1}/{max_retries}...")

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
                    print(f"Block failed. Retrying...")
                else:
                    print(f"Block failed after {max_retries} retries. Stopping.")
                    print(f"Best model saved at: {best_pt}")
                    return

        if success:
            if last_pt.exists():
                _save_clean_weights(last_pt, clean_pt)
            _append_block_csv(results_dir, master_csv)
            print(f"Epoch {block_end} complete.")
        else:
            print("Training stopped due to repeated failures.")
            return

    print("\n" + "=" * 60)
    print(f"Training complete! ({total_epochs} epochs)")
    print(f"Best model:  {best_pt}")
    print(f"Last model:  {last_pt}")
    print(f"Full CSV:    {master_csv}")
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    yaml_path = create_dataset()

    train_few_shot(
        yaml_path        = yaml_path,
        total_epochs     = 150,   # longer run — small dataset needs more epochs
        epochs_per_block = 1,
        batch_size       = 8,     # smaller batch for ~1,200 train images
        imgsz            = 640,
        max_retries      = 2,
    )