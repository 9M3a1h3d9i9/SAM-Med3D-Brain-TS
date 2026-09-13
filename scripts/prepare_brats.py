"""
Custom preprocessor for MSD Task01 (Brain Tumour / BraTS) -> SAM-Med3D format.
- Extracts FLAIR modality (channel 0) as image
- Creates binary masks for each ROI: edema, non-enhancing, enhancing
- Resamples to 1.5mm spacing
- Saves in data/brain_pre_sam/<cls>/Task01_BrainTumour/{imagesTr,labelsTr}/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import nibabel as nib
import torchio as tio
from tqdm import tqdm

# --------- CONFIG ---------
SRC_DIR = "./data/raw/Task01_BrainTumour"
DST_DIR = "./data/brain_pre_sam"
DATASET_NAME = "Task01_BrainTumour"
TARGET_SPACING = (1.5, 1.5, 1.5)
FLAIR_CHANNEL = 0
ROIS = {"edema": 1, "non_enhancing": 2, "enhancing": 3}

MAX_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
# --------------------------

def resample_tensor(tensor, affine, is_label=False, ref_shape=None):
    """
    Resample a 4D tensor (1, H, W, D) using torchio.
    Returns (resampled_tensor, new_affine).
    """
    mode = "nearest" if is_label else "linear"
    subject = tio.Subject(
        img=tio.ScalarImage(tensor=tensor, affine=affine)
    )
    resampler = tio.Resample(target=TARGET_SPACING, image_interpolation=mode)
    resampled = resampler(subject)

    if ref_shape is not None:
        cropper = tio.CropOrPad(ref_shape)
        resampled = cropper(resampled)

    out_tensor = resampled.img.data.numpy()
    out_affine = resampled.img.affine
    return out_tensor, out_affine


def main():
    img_dir = os.path.join(SRC_DIR, "imagesTr")
    lbl_dir = os.path.join(SRC_DIR, "labelsTr")

    all_cases = sorted([f for f in os.listdir(lbl_dir)
                        if f.endswith(".nii.gz") and not f.startswith("._")])
    if MAX_CASES is not None:
        all_cases = all_cases[:MAX_CASES]
    print(f"Processing {len(all_cases)} case(s)...")

    for roi_name in ROIS:
        for sub in ["imagesTr", "labelsTr"]:
            os.makedirs(os.path.join(DST_DIR, roi_name, DATASET_NAME, sub), exist_ok=True)

    for case in tqdm(all_cases, desc="Cases"):
        img_path = os.path.join(img_dir, case)
        lbl_path = os.path.join(lbl_dir, case)

        img = nib.load(img_path)
        lbl = nib.load(lbl_path)

        img_data = img.get_fdata()                      # (240, 240, 155, 4)
        lbl_data = lbl.get_fdata().astype(np.int16)     # (240, 240, 155)

        # Extract FLAIR and add channel dim -> (1, H, W, D)
        flair_tensor = np.expand_dims(img_data[..., FLAIR_CHANNEL].astype(np.float32), axis=0)

        # Resample image once (shared across all ROIs)
        flair_resampled, flair_affine = resample_tensor(
            flair_tensor, img.affine, is_label=False
        )
        ref_shape = flair_resampled.shape[1:]  # (H, W, D) after resample

        for roi_name, roi_id in ROIS.items():
            binary_mask = (lbl_data == roi_id).astype(np.float32)
            if binary_mask.sum() < 10:
                continue
            mask_tensor = np.expand_dims(binary_mask, axis=0)

            # Resample mask to match image shape
            mask_resampled, mask_affine = resample_tensor(
                mask_tensor, lbl.affine, is_label=True, ref_shape=ref_shape
            )

            img_out = os.path.join(DST_DIR, roi_name, DATASET_NAME, "imagesTr", case)
            lbl_out = os.path.join(DST_DIR, roi_name, DATASET_NAME, "labelsTr", case)

            nib.save(nib.Nifti1Image(flair_resampled[0], flair_affine), img_out)
            nib.save(nib.Nifti1Image(mask_resampled[0], mask_affine), lbl_out)

    print(f"\n✅ Preprocessing done. Output in: {DST_DIR}")

if __name__ == "__main__":
    main()