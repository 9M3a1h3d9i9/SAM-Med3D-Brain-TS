"""
Evaluate SAM-Med3D on brain ROI test set with prompt point from GT.
"""
import sys
import os
# Add project root to path so we can import segment_anything and utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from glob import glob
from tqdm import tqdm

from segment_anything.build_sam3D import sam_model_registry3D
from utils.metric_utils import compute_dice_coefficient

# ---------------- CONFIG ----------------
CKPT_PATH = sys.argv[1] if len(sys.argv) > 1 else "SAM_Med3D/ckpt/sam_med3d_turbo.pth"
TEST_ROOT = sys.argv[2] if len(sys.argv) > 2 else "./data/brain_pre_sam"
OUT_CSV = sys.argv[3] if len(sys.argv) > 3 else "results_pretrained.csv"
MODEL_TYPE = "vit_b_ori"
IMG_SIZE = 128
DEVICE = torch.device("cpu")
# ----------------------------------------

def load_model(ckpt_path):
    print(f"Loading: {ckpt_path}")
    model = sam_model_registry3D[MODEL_TYPE](checkpoint=None).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def make_prompt(gt, device):
    """Sample centroid of GT as prompt point."""
    coords = np.argwhere(gt > 0)
    if len(coords) == 0:
        return None
    centroid = coords.mean(axis=0)
    # points shape: (B=1, N=1, 3)
    points_co = torch.tensor([[[float(centroid[0]), float(centroid[1]), float(centroid[2])]]],
                             dtype=torch.float32, device=device)
    # labels shape: (B=1, N=1)
    labels = torch.ones((1, 1), dtype=torch.int64, device=device)
    return (points_co, labels)

def infer_one(model, img_tensor, gt):
    with torch.no_grad():
        img_tensor = img_tensor.to(DEVICE)
        emb = model.image_encoder(img_tensor)

        # Initial zero low-res mask (mimic training)
        low_res_masks = F.interpolate(
            torch.zeros_like(img_tensor.float()),
            size=(IMG_SIZE // 4, IMG_SIZE // 4, IMG_SIZE // 4),
            mode="trilinear", align_corners=False,
        )

        prompt = make_prompt(gt, DEVICE)

        if prompt is None:
            sparse, dense = model.prompt_encoder(
                points=None, boxes=None, masks=low_res_masks
            )
        else:
            sparse, dense = model.prompt_encoder(
                points=prompt, boxes=None, masks=low_res_masks
            )

        low_res, _ = model.mask_decoder(
            image_embeddings=emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        up = F.interpolate(low_res, size=img_tensor.shape[-3:],
                           mode="trilinear", align_corners=False)
        return (up[0, 0].cpu().numpy() > 0.0).astype(np.uint8)

def main():
    model = load_model(CKPT_PATH)
    rows = []
    roi_dirs = sorted([d for d in glob(os.path.join(TEST_ROOT, "*"))
                       if os.path.isdir(d)])
    print(f"ROI dirs: {len(roi_dirs)}")

    for roi_dir in roi_dirs:
        roi = os.path.basename(roi_dir)
        img_dir = os.path.join(roi_dir, "Task01_BrainTumour", "imagesTr")
        lbl_dir = os.path.join(roi_dir, "Task01_BrainTumour", "labelsTr")
        if not os.path.isdir(img_dir):
            continue
        cases = sorted([f for f in os.listdir(img_dir)
                        if f.endswith(".nii.gz") and not f.startswith("._")])
        print(f"  {roi}: {len(cases)} cases")

        for case in tqdm(cases, desc=roi, leave=False):
            img_path = os.path.join(img_dir, case)
            lbl_path = os.path.join(lbl_dir, case)
            try:
                crop = tio.CropOrPad((IMG_SIZE, IMG_SIZE, IMG_SIZE))

                # ---- Load GT FIRST ----
                lbl = tio.ScalarImage(lbl_path)
                gt_cropped = crop(lbl)
                gt = (gt_cropped.data[0].numpy() > 0.5).astype(np.uint8)

                # ---- Load Image ----
                img = tio.ScalarImage(img_path)
                img_cropped = crop(img)
                data = img_cropped.data.float()
                m = data > 0
                mean = data[m].mean() if m.any() else data.mean()
                std = data[m].std() if m.any() else data.std()
                data = (data - mean) / (std + 1e-8)
                data = data.unsqueeze(0)

                # ---- Inference ----
                pred = infer_one(model, data, gt)

                dice = compute_dice_coefficient(gt, pred)
                rows.append({
                    "case": case.replace(".nii.gz", ""),
                    "roi": roi,
                    "dice": float(dice) if not np.isnan(dice) else 0.0,
                    "gt_voxels": int(gt.sum()),
                    "pred_voxels": int(pred.sum()),
                })
            except Exception as e:
                print(f"  [SKIP] {case}: {e}")
                continue

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "roi", "dice", "gt_voxels", "pred_voxels"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n✅ Saved {len(rows)} rows → {OUT_CSV}")

if __name__ == "__main__":
    main()