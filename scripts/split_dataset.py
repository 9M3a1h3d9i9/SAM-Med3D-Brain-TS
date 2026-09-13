"""
Split preprocessed brain data into train/val/test OR K-fold splits.

Modes:
  - split  : single split with configurable ratios (default 70/15/15)
  - kfold  : K-fold cross validation, each fold has train/val/test

Each fold in K-Fold mode:
  - test_idx = fold
  - val_idx  = (fold + 1) % K
  - train_idx = all others

So each case appears exactly:
  - once in test across all folds
  - once in val across all folds
  - (K-2) times in train across all folds

Output:
  - split mode: data/brain_train/, data/brain_val/, data/brain_test/
  - kfold mode: data/brain_fold_1/{train,val,test}/, ..., data/brain_fold_K/...

Uses hard links to save disk space (fallback to copy if cross-filesystem).
"""
import os
import sys
import shutil
import random
import argparse
from tqdm import tqdm

# ---------------- CONFIG ----------------
SRC_ROOT = "./data/brain_pre_sam"
DST_ROOT = "./data"
DATASET_NAME = "Task01_BrainTumour"
SEED = 42
# ----------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Split brain dataset into train/val/test or K folds.")
    p.add_argument("--mode", choices=["split", "kfold"], default="split",
                   help="Split mode: 'split' for single split, 'kfold' for K-fold CV.")
    p.add_argument("--k", type=int, default=5,
                   help="Number of folds for K-Fold mode (default: 5).")
    p.add_argument("--ratios", type=str, default="0.70,0.15,0.15",
                   help="Train,val,test ratios for split mode (sum must be 1.0).")
    p.add_argument("--src", type=str, default=SRC_ROOT,
                   help=f"Source root dir (default: {SRC_ROOT}).")
    p.add_argument("--dst", type=str, default=DST_ROOT,
                   help=f"Destination root dir (default: {DST_ROOT}).")
    p.add_argument("--seed", type=int, default=SEED,
                   help=f"Random seed (default: {SEED}).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing destination directories.")
    return p.parse_args()


def discover_rois(src_root):
    """Return sorted list of ROI folder names."""
    if not os.path.isdir(src_root):
        print(f"❌ Source dir not found: {src_root}")
        sys.exit(1)
    rois = sorted([
        d for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d)) and not d.startswith(".")
    ])
    if not rois:
        print(f"❌ No ROI dirs in {src_root}")
        sys.exit(1)
    return rois


def collect_cases(src_root, rois):
    """Collect common cases across all ROIs (intersection)."""
    roi_cases = {}
    for roi in rois:
        img_dir = os.path.join(src_root, roi, DATASET_NAME, "imagesTr")
        if not os.path.isdir(img_dir):
            print(f"⚠️  {roi}: no imagesTr, skipping")
            continue
        cases = set(
            f for f in os.listdir(img_dir)
            if f.endswith(".nii.gz") and not f.startswith("._")
        )
        if cases:
            roi_cases[roi] = cases

    if not roi_cases:
        print("❌ No valid cases found.")
        sys.exit(1)

    common = sorted(set.intersection(*roi_cases.values()))
    if not common:
        print("❌ No common cases across ROIs.")
        sys.exit(1)
    return roi_cases, common


def link_or_copy(src, dst):
    """Hard link src→dst, fallback to copy.
    On Windows, os.link may need admin privileges — fallback to copy always."""
    if os.path.exists(dst):
        return
    if not os.path.exists(src):
        print(f"  ⚠️  missing source: {src}")
        return
    # On Windows, use copy to avoid admin requirement
    if sys.platform.startswith("win"):
        shutil.copy2(src, dst)
        return
    try:
        os.link(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)

def copy_cases_for_split(src_root, roi, cases, dst_root, split_name):
    """Copy/link cases for a given (roi, split_name) into dst_root."""
    img_out = os.path.join(dst_root, roi, DATASET_NAME, "imagesTr")
    lbl_out = os.path.join(dst_root, roi, DATASET_NAME, "labelsTr")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for case in cases:
        src_img = os.path.join(src_root, roi, DATASET_NAME, "imagesTr", case)
        src_lbl = os.path.join(src_root, roi, DATASET_NAME, "labelsTr", case)
        link_or_copy(src_img, os.path.join(img_out, case))
        link_or_copy(src_lbl, os.path.join(lbl_out, case))


def prepare_dst_dir(path, overwrite):
    """Create dst dir; abort if exists and not overwrite."""
    if os.path.exists(path):
        if not overwrite:
            print(f"❌ Destination exists: {path}")
            print("   Use --overwrite to replace it.")
            sys.exit(1)
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def run_split(args, rois, common_cases):
    """Single train/val/test split."""
    try:
        ratios = [float(x) for x in args.ratios.split(",")]
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError
    except ValueError:
        print(f"❌ Invalid ratios: {args.ratios} (must be 3 floats summing to 1.0)")
        sys.exit(1)

    r_train, r_val, r_test = ratios
    n_total = len(common_cases)
    n_train = max(1, int(r_train * n_total))
    n_val = max(1, int(r_val * n_total))
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n_total - n_train - n_test)

    random.Random(args.seed).shuffle(common_cases)
    splits = {
        "train": common_cases[:n_train],
        "val": common_cases[n_train:n_train + n_val],
        "test": common_cases[n_train + n_val:],
    }
    for k, v in splits.items():
        print(f"  {k}: {len(v)} cases")

    for split_name, cases in splits.items():
        dst = os.path.join(args.dst, f"brain_{split_name}")
        prepare_dst_dir(dst, args.overwrite)
        for roi in rois:
            print(f"  Linking {split_name}/{roi} ...")
            copy_cases_for_split(args.src, roi, cases, dst, split_name)

    print(f"\n✅ Split done.")
    for split_name in splits:
        print(f"   {args.dst}/brain_{split_name}/")


def run_kfold(args, rois, common_cases):
    """K-Fold cross validation."""
    k = args.k
    if k < 2:
        print(f"❌ --k must be >= 2, got {k}")
        sys.exit(1)
    if k > len(common_cases):
        print(f"❌ --k ({k}) > number of cases ({len(common_cases)})")
        sys.exit(1)

    shuffled = list(common_cases)
    random.Random(args.seed).shuffle(shuffled)

    # Distribute into k folds as evenly as possible
    folds = [[] for _ in range(k)]
    for i, case in enumerate(shuffled):
        folds[i % k].append(case)

    for i, f in enumerate(folds):
        print(f"  Fold {i+1}: {len(f)} cases")

    for fold_idx in range(k):
        test_cases = folds[fold_idx]
        val_cases = folds[(fold_idx + 1) % k]
        train_cases = []
        for j in range(k):
            if j != fold_idx and j != (fold_idx + 1) % k:
                train_cases.extend(folds[j])

        fold_root = os.path.join(args.dst, f"brain_fold_{fold_idx + 1}")
        print(f"\n📂 Fold {fold_idx + 1}/{k} → {fold_root}")
        print(f"   train: {len(train_cases)}, val: {len(val_cases)}, test: {len(test_cases)}")

        for split_name, cases in [("train", train_cases),
                                   ("val", val_cases),
                                   ("test", test_cases)]:
            dst = os.path.join(fold_root, split_name)
            prepare_dst_dir(dst, args.overwrite)
            for roi in rois:
                copy_cases_for_split(args.src, roi, cases, dst, split_name)

    print(f"\n✅ K-Fold done (K={k}).")
    for i in range(k):
        print(f"   {args.dst}/brain_fold_{i+1}/{{train,val,test}}/")


def main():
    args = parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print(f"Mode     : {args.mode}")
    print(f"Source   : {args.src}")
    print(f"Dest     : {args.dst}")
    print(f"Seed     : {args.seed}")
    if args.mode == "kfold":
        print(f"K        : {args.k}")
    else:
        print(f"Ratios   : {args.ratios}")
    print("=" * 60)

    rois = discover_rois(args.src)
    print(f"Found ROIs: {rois}")
    roi_cases, common_cases = collect_cases(args.src, rois)
    print(f"Common cases across ROIs: {len(common_cases)}")

    if len(common_cases) < 5:
        print("⚠️  Very few cases — results may be unstable.")

    if args.mode == "split":
        run_split(args, rois, common_cases)
    else:
        run_kfold(args, rois, common_cases)


if __name__ == "__main__":
    main()


# حالت ۱ : تقسیم عادی ( پیش فرض)
# cd ~/SAM-Med3D
# python scripts/split_dataset.py --mode split --ratios 0.70,0.15,0.15 --overwrite


# حالت ۲: K-Fold (مثلاً ۵ فولد)
# python scripts/split_dataset.py --mode kfold --k 5 --overwrite

# حالت ۳: K-Fold با ۱۰ فولد (برای دیتاست بزرگ‌تر)
# python scripts/split_dataset.py --mode kfold --k 10 --overwrite


# ---

#  تنظیمات کامل آرگومان‌ها
# Argoman   Default     Explain

# --mode	split	     split یا kfold
# --k	    5	        تعداد فولدها (فقط kfold)
# --ratios	0.70,0.15,0.15	نسبت‌های train/val/test (فقط split)
# --src	    ./data/brain_pre_sam	مسیر مبدأ
# --dst	    ./data	     مسیر مقصد
# --seed	 42	         seed برای reproducibility
# --overwrite	False	بازنویسی پوشه‌های موجود

