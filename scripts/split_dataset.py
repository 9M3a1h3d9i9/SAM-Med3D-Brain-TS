"""
Split preprocessed brain data into train/val/test.
Reads from data/brain_pre_sam/ (with edema/enhancing/non_enhancing subfolders).
Writes to data/brain_train/, data/brain_val/, data/brain_test/.
Uses hard links to save disk space (same filesystem).
"""
import os
import sys
import random
from tqdm import tqdm

# ---------------- CONFIG ----------------
SRC_ROOT = "./data/brain_pre_sam"
DST_ROOT = "./data"
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
DATASET_NAME = "Task01_BrainTumour"
# ----------------------------------------

def main():
    random.seed(SEED)

    if not os.path.isdir(SRC_ROOT):
        print(f"❌ Source dir not found: {SRC_ROOT}")
        sys.exit(1)

    # Discover ROIs
    rois = sorted([d for d in os.listdir(SRC_ROOT)
                   if os.path.isdir(os.path.join(SRC_ROOT, d))
                   and not d.startswith(".")])
    if not rois:
        print(f"❌ No ROI dirs in {SRC_ROOT}")
        sys.exit(1)
    print(f"Found ROIs: {rois}")

    # Collect cases per ROI
    roi_cases = {}
    for roi in rois:
        img_dir = os.path.join(SRC_ROOT, roi, DATASET_NAME, "imagesTr")
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

    # Common cases across all ROIs
    common_cases = sorted(set.intersection(*roi_cases.values()))
    print(f"Common cases across all ROIs: {len(common_cases)}")

    if len(common_cases) < 5:
        print("⚠️  Very few cases. Split ratios may be off.")

    # Shuffle
    random.shuffle(common_cases)
    n_total = len(common_cases)
    n_train = max(1, int(RATIOS["train"] * n_total))
    n_val = max(1, int(RATIOS["val"] * n_total))
    # ensure test has at least 1
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n_total - n_train - n_test)

    splits = {
        "train": common_cases[:n_train],
        "val": common_cases[n_train:n_train + n_val],
        "test": common_cases[n_train + n_val:],
    }
    for k, v in splits.items():
        print(f"  {k}: {len(v)} cases")

    # Build destination structure
    for split_name, cases in splits.items():
        for roi in roi_cases:
            img_out = os.path.join(DST_ROOT, f"brain_{split_name}", roi,
                                   DATASET_NAME, "imagesTr")
            lbl_out = os.path.join(DST_ROOT, f"brain_{split_name}", roi,
                                   DATASET_NAME, "labelsTr")
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)

            for case in tqdm(cases, desc=f"{split_name}/{roi}", leave=False):
                src_img = os.path.join(SRC_ROOT, roi, DATASET_NAME, "imagesTr", case)
                src_lbl = os.path.join(SRC_ROOT, roi, DATASET_NAME, "labelsTr", case)
                dst_img = os.path.join(img_out, case)
                dst_lbl = os.path.join(lbl_out, case)

                for src, dst in [(src_img, dst_img), (src_lbl, dst_lbl)]:
                    if not os.path.exists(src):
                        print(f"  ⚠️  missing source: {src}")
                        continue
                    if os.path.exists(dst):
                        continue
                    try:
                        os.link(os.path.abspath(src), dst)  # hard link
                    except OSError:
                        import shutil
                        shutil.copy2(src, dst)

    print("\n✅ Split done.")
    print(f"   {DST_ROOT}/brain_train/")
    print(f"   {DST_ROOT}/brain_val/")
    print(f"   {DST_ROOT}/brain_test/")

if __name__ == "__main__":
    main()
