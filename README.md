# SAM-Med3D Brain Segmentation

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#license)

Fine-tuning **SAM-Med3D** for **Brain Tumour Segmentation** on the Medical Segmentation Decathlon (MSD) Task01 dataset (BraTS 2016/2017).

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Preprocessing](#preprocessing)
7. [Train / Val / Test Split](#train--val--test-split)
8. [Training](#training)
9. [Evaluation](#evaluation)
10. [Comparison & Visualization](#comparison--visualization)
11. [Project Structure](#project-structure)
12. [Windows Support](#windows-support)
13. [Server Deployment](#server-deployment)
14. [Troubleshooting](#troubleshooting)
15. [Documentation](#documentation)
16. [Citation](#citation)
17. [License](#license)
18. [Authors](#authors)

---

## Overview

**SAM-Med3D** is a fully 3D adaptation of the Segment Anything Model (SAM) designed for volumetric medical images. This project fine-tunes SAM-Med3D on the **MSD Task01 (Brain Tumour)** dataset to segment three regions of interest:

- **Edema (ED)**
- **Non-enhancing Tumour (NCR)**
- **Enhancing Tumour (ET)**

The pipeline covers the complete workflow: dataset download, preprocessing to SAM-Med3D format, train/val/test splitting (with optional K-Fold Cross-Validation), training, evaluation, and comparison with the pretrained model.

**Model:** `sam_med3d_turbo.pth` (~91M parameters)  
**Dataset:** MSD Task01 — Brain Tumour (BraTS 2016/2017)  
**Input size:** 128 × 128 × 128  
**Modalities used:** FLAIR (channel 0)

---

## Features

- ✅ Complete preprocessing pipeline from raw MSD NIfTI files to SAM-Med3D format
- ✅ Support for Train/Val/Test split and **K-Fold Cross-Validation**
- ✅ Prompt-based inference (centroid of ground truth as prompt point)
- ✅ Per-case Dice, IoU, and HD95 computation
- ✅ Automated comparison table with Wilcoxon signed-rank test
- ✅ Publication-quality bar plots and box plots
- ✅ Cross-platform: Linux, WSL, and Windows
- ✅ 10 critical bug fixes for the upstream SAM-Med3D code
- ✅ Ready-to-use scripts for server deployment

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 12 GB VRAM | 16 GB VRAM (RTX 4060 Ti / A100) |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB free | 100 GB free |

### Software

- **OS:** Linux (Ubuntu 20.04+ / Pop!_OS), WSL2, or Windows 10+
- **Python:** 3.10
- **CUDA:** 11.8 or 12.1
- **PyTorch:** 2.x
- **Conda:** Miniforge or Miniconda

---

## Installation

### Step 1: Install Miniforge (Linux / WSL)

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc
conda --version
```

### Step 2: Create Environment

```bash
conda create --name sammed3d_gpu python=3.10 -y
conda activate sammed3d_gpu
```

### Step 3: Install PyTorch with CUDA

Choose the command that matches your CUDA version:

```bash
# For CUDA 12.1 (recommended for RTX 40xx series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU access:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Ti
```

### Step 4: Install Dependencies

```bash
pip install uv
uv pip install torchio opencv-python-headless matplotlib \
    prefetch_generator monai edt surface-distance medim \
    nibabel pandas scipy tqdm
```

### Step 5: Clone Repository

```bash
git clone https://github.com/9M3a1h3d9i9/SAM-Med3D-Brain-TS.git
cd SAM-Med3D-Brain-TS
```

---

## Dataset Preparation

### Download MSD Task01 (Brain Tumour)

```bash
mkdir -p data/raw
cd data/raw
wget -c https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
tar -xvf Task01_BrainTumour.tar
rm Task01_BrainTumour.tar
find Task01_BrainTumour -name "._*" -delete
find Task01_BrainTumour -name ".DS_Store" -delete
cd ../..
```

**File size:** ~7.6 GB  
**Cases:** 484 training + 266 test  
**Modalities:** FLAIR, T1w, t1gd, T2w (stored in a 4D file)

Expected directory structure:

```
data/raw/Task01_BrainTumour/
├── imagesTr/          # 484 images (240, 240, 155, 4)
├── labelsTr/          # 484 labels (240, 240, 155)
├── imagesTs/          # 266 test images
└── dataset.json
```

### Download SAM-Med3D Checkpoint

Download `SAM-Med3D-turbo.pth` from the [official repository](https://github.com/uni-medical/SAM-Med3D) and place it in:

```bash
mkdir -p SAM_Med3D/ckpt
# Move the downloaded file here:
# SAM_Med3D/ckpt/sam_med3d_turbo.pth
```

---

## Preprocessing

The `prepare_brats.py` script converts MSD Task01 into SAM-Med3D format:

- Extracts **FLAIR** (channel 0) as the input image
- Creates **binary masks** for each ROI (edema, enhancing, non_enhancing)
- Resamples to **1.5 mm isotropic** spacing
- Saves as **3D tensors** (not 4D)

Run on all 484 cases:

```bash
python scripts/prepare_brats.py
```

Run on a small subset (e.g., 3 cases) for quick testing:

```bash
python scripts/prepare_brats.py 3
```

**Output structure:**

```
data/brain_pre_sam/
├── edema/Task01_BrainTumour/
│   ├── imagesTr/BRATS_001.nii.gz    # (1, 160, 160, 103)
│   └── labelsTr/BRATS_001.nii.gz    # (1, 160, 160, 103) binary
├── enhancing/Task01_BrainTumour/...
└── non_enhancing/Task01_BrainTumour/...
```

---

## Train / Val / Test Split

### Mode 1: Standard Split (70 / 15 / 15)

```bash
python scripts/split_dataset.py --mode split --ratios 0.70,0.15,0.15 --overwrite
```

Output:
```
data/brain_train/
data/brain_val/
data/brain_test/
```

### Mode 2: K-Fold Cross-Validation

```bash
python scripts/split_dataset.py --mode kfold --k 5 --overwrite
```

Each fold uses a different subset for test and validation:

- `test_idx` = fold
- `val_idx` = (fold + 1) % K
- `train_idx` = all remaining folds

Output:
```
data/brain_fold_1/{train,val,test}/
data/brain_fold_2/{train,val,test}/
...
data/brain_fold_5/{train,val,test}/
```

### Verify the Split

```bash
# Standard split
for s in train val test; do
    n=$(find data/brain_${s} -name "*.nii.gz" | wc -l)
    echo "brain_${s}: ${n} files"
done

# K-Fold
for i in 1 2 3 4 5; do
    echo "=== Fold ${i} ==="
    for s in train val test; do
        n=$(find data/brain_fold_${i}/${s} -name "*.nii.gz" | wc -l)
        echo "  ${s}: ${n}"
    done
done
```

---

## Training

### Configure `train.sh`

Edit the training script:

```bash
nano train.sh
```

Recommended configuration for **16 GB VRAM GPU**:

```bash
python train.py \
 --batch_size 2 \
 --num_workers 4 \
 --task_name "ft_brats_full" \
 --checkpoint "SAM_Med3D/ckpt/sam_med3d_turbo.pth" \
 --device "cuda" \
 --num_epochs 50 \
 --accumulation_steps 8 \
 --img_size 128 \
 --lr 8e-5
```

### Training Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--batch_size` | Number of samples per batch | 2 (16GB) / 4 (24GB) / 8 (40GB) |
| `--num_workers` | Data loader workers | 4-8 |
| `--num_epochs` | Total training epochs | 50-200 |
| `--accumulation_steps` | Gradient accumulation | 8 (16GB) / 4 (24GB) / 2 (40GB) |
| `--img_size` | Input volume size | 128 (must match pretrained) |
| `--lr` | Learning rate | 8e-5 |

### Launch Training

**Option 1: Direct run**

```bash
cd ~/SAM-Med3D
bash train.sh
```

**Option 2: Using tmux (recommended for long runs)**

```bash
tmux new -s training
conda activate sammed3d_gpu
cd ~/SAM-Med3D
bash train.sh

# Detach without stopping: Ctrl+B then D
# Re-attach later: tmux attach -t training
```

**Option 3: K-Fold training loop**

```bash
for fold in 1 2 3 4 5; do
    echo "=== Training Fold ${fold} ==="
    SAM_DATA_ROOT=data/brain_fold_${fold}/train \
        python train.py \
            --batch_size 2 \
            --num_workers 4 \
            --task_name "fold_${fold}" \
            --checkpoint "SAM_Med3D/ckpt/sam_med3d_turbo.pth" \
            --device "cuda" \
            --num_epochs 50 \
            --accumulation_steps 8 \
            --img_size 128 \
            --lr 8e-5
done
```

### Output

Training saves results in `work_dir/<task_name>/`:

```
work_dir/ft_brats_full/
├── sam_model_latest.pth      # Latest checkpoint
├── sam_model_loss_best.pth   # Best loss
├── sam_model_dice_best.pth   # Best Dice
├── Loss.png                  # Loss curve
├── Dice.png                  # Dice curve
└── output_*.log              # Training log
```

### Monitor Training

In a second terminal:

```bash
# GPU usage (live)
watch -n 2 nvidia-smi

# Training log
tail -f ~/SAM-Med3D/work_dir/ft_brats_full/output_*.log

# RAM and CPU
htop
```

---

## Evaluation

### Evaluate Pretrained Model (Zero-Shot Baseline)

```bash
python scripts/evaluate.py \
    "SAM_Med3D/ckpt/sam_med3d_turbo.pth" \
    "./data/brain_test" \
    "results/results_pretrained.csv"
```

### Evaluate Fine-Tuned Model

```bash
python scripts/evaluate.py \
    "work_dir/ft_brats_full/sam_model_dice_best.pth" \
    "./data/brain_test" \
    "results/results_finetuned.csv"
```

### Evaluation Output (CSV)

```
case,roi,dice,gt_voxels,pred_voxels
BRATS_001,edema,0.6123,15712,14089
BRATS_001,enhancing,0.2856,9354,20250
...
```

### Expected Dice Ranges

| ROI | Pretrained (Zero-Shot) | Fine-Tuned (Full) |
|-----|------------------------|-------------------|
| Edema | 0.55 - 0.65 | 0.75 - 0.85 |
| Enhancing | 0.20 - 0.30 | 0.70 - 0.80 |
| Non-enhancing | 0.10 - 0.20 | 0.50 - 0.65 |

---

## Comparison & Visualization

Compare two evaluation CSVs and generate plots:

```bash
python scripts/compare_results.py \
    results/results_pretrained.csv \
    results/results_finetuned.csv \
    "Pretrained" \
    "Fine-tuned" \
    "results/comparison_final"
```

### Outputs

- `results/comparison_final_table.csv` — Comparison table with mean/std/delta
- `results/comparison_final_bar.png` — Bar plot of Dice per ROI
- `results/comparison_final_box.png` — Box plot of Dice distribution
- Console output with **Wilcoxon signed-rank test** per ROI

### Sample Output

```
=== Comparison Table ===
          ROI  Pretrained_mean  Fine-tuned_mean  Delta
        edema           0.607            0.745  +0.138
    enhancing           0.226            0.720  +0.494
non_enhancing           0.142            0.560  +0.418

=== Wilcoxon signed-rank test (per ROI) ===
  edema: stat=123.000, p=0.0001  ✅ significant
  enhancing: stat=156.000, p=0.0001  ✅ significant
  non_enhancing: stat=98.000, p=0.0002  ✅ significant
```

---

## Project Structure

```
SAM-Med3D-Brain-TS/
├── scripts/
│   ├── prepare_brats.py        # MSD Task01 → SAM-Med3D format
│   ├── split_dataset.py        # Train/Val/Test or K-Fold split
│   ├── evaluate.py             # Evaluation on test set
│   ├── compare_results.py      # Comparison + plots + Wilcoxon
│   └── deploy_to_server.sh     # Automated server deployment
├── utils/
│   ├── data_loader.py          # Dataset class
│   ├── data_paths.py           # Path globbing with SAM_DATA_ROOT
│   └── metric_utils.py         # Dice, NSD, surface distances
├── segment_anything/           # SAM-Med3D model architecture
│   ├── modeling/
│   │   ├── image_encoder3D.py
│   │   ├── prompt_encoder3D.py
│   │   └── mask_decoder3D.py
│   └── build_sam3D.py
├── data/                       # (gitignored)
│   ├── raw/
│   └── brain_pre_sam/
├── work_dir/                   # (gitignored) Checkpoints & logs
├── results/                    # Reports, CSVs, plots
│   ├── TECHNICAL_REPORT.md
│   ├── FULL_GUIDE_0_to_100.md
│   └── comparison_*.png
├── train.py                    # Main training script
├── train.sh / train.bat        # Launch scripts
├── evaluate.bat                # Windows evaluation
├── prepare_brats.bat           # Windows preprocessing
├── README.md
└── LICENSE
```

---

## Windows Support

For Windows 10/11 users (without WSL), use the provided `.bat` scripts.

### Prerequisites

1. Install **Miniconda for Windows** from [here](https://docs.conda.io/en/latest/miniconda.html)
2. Open **Anaconda Prompt** (no admin required)

### Setup

```cmd
conda create --name sammed3d_gpu python=3.10 -y
conda activate sammed3d_gpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchio monai nibabel pandas scipy tqdm
```

### Preprocess

```cmd
prepare_brats.bat
```

### Split

```cmd
python scripts\split_dataset.py --mode split --ratios 0.70,0.15,0.15 --overwrite
```

### Train

```cmd
train.bat
```

### Evaluate

```cmd
evaluate.bat work_dir\ft_brats_full\sam_model_dice_best.pth .\data\brain_test results\results_finetuned.csv
```

**Note:** On Windows, `os.link` requires admin privileges. The `split_dataset.py` script automatically falls back to `shutil.copy2` on Windows.

---

## Server Deployment

For training on a remote GPU server (e.g., university A100):

### From Your Local Machine

```bash
./scripts/deploy_to_server.sh USERNAME@SERVER_ADDRESS PORT --with-test-data
```

### Manual Transfer

```bash
# Package code (excludes data, work_dir, checkpoints)
tar -czf /tmp/sam_code.tar.gz \
    --exclude='./data' \
    --exclude='./work_dir' \
    --exclude='./results' \
    --exclude='./.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pth' \
    -C ~/SAM-Med3D .

# Upload
scp -P PORT /tmp/sam_code.tar.gz USERNAME@SERVER:~/
```

### On the Server

```bash
# Connect
ssh USERNAME@SERVER -p PORT

# Extract
mkdir -p ~/SAM-Med3D
tar -xzf ~/sam_code.tar.gz -C ~/SAM-Med3D
rm ~/sam_code.tar.gz
cd ~/SAM-Med3D

# Setup environment
conda create --name sammed3d_gpu python=3.10 -y
conda activate sammed3d_gpu
nvidia-smi  # Check CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install uv
uv pip install torchio monai nibabel pandas scipy tqdm opencv-python-headless matplotlib

# Download dataset
mkdir -p data/raw && cd data/raw
wget -c https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
tar -xvf Task01_BrainTumour.tar && rm Task01_BrainTumour.tar
cd ../..

# Preprocess and split
python scripts/prepare_brats.py
python scripts/split_dataset.py --mode kfold --k 5 --overwrite

# Train with tmux
tmux new -s training
bash train.sh
# Ctrl+B then D to detach
```

---

## Troubleshooting

### `CUDA out of memory`

- Reduce `--batch_size` (try 1)
- Increase `--accumulation_steps` (e.g., 16)
- Reduce `--img_size` (but must remain 128 for pretrained)

### `Torch not compiled with CUDA enabled`

You installed the CPU version of PyTorch. Reinstall with:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### `No module named 'segment_anything'`

Add the project root to `sys.path` at the top of your script:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### `Tensors must have same number of dimensions`

4D image vs 3D label mismatch in `data_loader.py`. Ensure images are saved as 3D (use `flair_resampled[0]` in `prepare_brats.py`).

### `AttributeError: 'BaseTrainer' object has no attribute 'seq_loss'`

Typo in `train.py`. Replace `seq_loss` with `seg_loss`.

### `DiceCELoss: Expected floating point tensor`

Cast the target to float in `train.py`:

```python
loss = self.seg_loss(prev_masks, gt3D.float())
```

### `pos_embed` size mismatch

Ensure `--img_size 128` (the pretrained model expects 128³ input).

### Windows: `wmic is not recognized`

`wmic` is deprecated in Windows 10/11. Use PowerShell instead:

```powershell
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores
[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
nvidia-smi
```

### Windows: `The requested operation requires elevation`

Run PowerShell or Command Prompt **as Administrator**. Right-click → **Run as administrator**.

---

## Documentation

- [**TECHNICAL_REPORT.md**](results/TECHNICAL_REPORT.md) — Detailed technical report (in Persian)
- [**FULL_GUIDE_0_to_100.md**](results/FULL_GUIDE_0_to_100.md) — Comprehensive guide from DL basics to server execution (in Persian)

---

## Citation

If you use this code in your research, please cite the original SAM-Med3D paper:

```bibtex
@article{wang2023sam,
  title={SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images},
  author={Wang, Haoyu and Guo, Sizheng and Ye, Jin and Deng, Zhongying and Cheng, Junlong and Li, Tianbin and Chen, Jianpin and Su, Yanzhou and Huang, Ziyan and Shen, Yiqing and others},
  journal={arXiv preprint arXiv:2310.15161},
  year={2023}
}
```

And the MSD dataset paper:

```bibtex
@article{antonelli2022medical,
  title={The Medical Segmentation Decathlon},
  author={Antonelli, Michela and Reinke, Annika and Bakas, Spyridon and Farahani, Keyvan and Kopp-Schneider, Annette and Landman, Bennett A and Litjens, Geert and Menze, Bjoern and Ronneberger, Olaf and Summers, Ronald M and others},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={4128},
  year={2022},
  publisher={Nature Publishing Group}
}
```

---

## License

This project is intended for **academic and research use only**. Commercial use is not permitted. Please refer to the licenses of the original SAM-Med3D and MSD datasets before redistribution.

---

## Authors

- **[9M3a1h3d9i9](https://github.com/9M3a1h3d9i9)** — MSc Artificial Intelligence, Shahed University
- **Nazanin Sarabi** — MSc Biomedical Engineering, Shahed University
- **Supervisor:** Dr. Forouzan Jostoghani — Shahed University

---

## Acknowledgments

- The SAM-Med3D team (Shanghai Jiao Tong University & Shanghai AI Laboratory) for the pretrained model.
- The Medical Segmentation Decathlon organizers for the Brain Tumour dataset.
- The MONAI and TorchIO teams for medical imaging libraries.

---

**Last updated:** 1405/06/23 (2026-09-14)