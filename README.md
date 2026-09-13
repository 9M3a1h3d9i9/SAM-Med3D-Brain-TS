# SAM-Med3D Brain Segmentation

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#مجوز)

فاین‌تیونینگ مدل **SAM-Med3D** برای **تقسیم‌بندی تومور مغزی** روی دیتاست Medical Segmentation Decathlon (MSD) Task01 — BraTS 2016/2017.

---

## فهرست مطالب

1. [معرفی پروژه](#معرفی-پروژه)
2. [امکانات](#امکانات)
3. [پیش‌نیازهای سخت‌افزاری](#پیشنیازهای-سختافزاری)
4. [پیش‌نیازهای نرم‌افزاری](#پیشنیازهای-نرمافزاری)
5. [راه‌اندازی روی لینوکس](#راهاندازی-روی-لینوکس)
6. [راه‌اندازی روی ویندوز](#راهاندازی-روی-ویندوز)
7. [دانلود دیتاست](#دانلود-دیتاست)
8. [دانلود چک‌پوینت](#دانلود-چکپوینت)
9. [پیش‌پردازش داده](#پیشپردازش-داده)
10. [تقسیم داده (Train/Val/Test یا K-Fold)](#تقسیم-داده)
11. [آموزش مدل](#آموزش-مدل)
12. [ارزیابی مدل](#ارزیابی-مدل)
13. [مقایسه و رسم نمودار](#مقایسه-و-رسم-نمودار)
14. [ساختار پروژه](#ساختار-پروژه)
15. [عیب‌یابی](#عیبیابی)
16. [مستندات](#مستندات)
17. [ارجاع](#ارجاع)
18. [مجوز](#مجوز)
19. [نویسندگان](#نویسندگان)

---

## معرفی پروژه

**SAM-Med3D** یک نسخه کاملاً سه‌بعدی از مدل Segment Anything Model (SAM) است که برای تصاویر پزشکی حجمی طراحی شده. در این پروژه، SAM-Med3D روی دیتاست **MSD Task01 (Brain Tumour)** فاین‌تیون می‌شود تا سه ناحیه از تومور مغزی را تقسیم‌بندی کند:

- **ادم (Edema - ED)**
- **تومور غیرفعال (Non-enhancing Tumour - NCR)**
- **تومور فعال (Enhancing Tumour - ET)**

پایپ‌لاین شامل: دانلود دیتاست، پیش‌پردازش، تقسیم داده، آموزش، ارزیابی و مقایسه با مدل پیش‌آموزش‌دیده است.

| ویژگی | مقدار |
|-------|-------|
| **مدل** | `sam_med3d_turbo.pth` (~۹۱ میلیون پارامتر) |
| **دیتاست** | MSD Task01 — Brain Tumour (BraTS 2016/2017) |
| **اندازه ورودی** | ۱۲۸ × ۱۲۸ × ۱۲۸ |
| **مودالیتی** | FLAIR (کانال ۰) |

---

## امکانات

- ✅ پیش‌پردازش کامل از NIfTI خام MSD به فرمت SAM-Med3D
- ✅ پشتیبانی از تقسیم ساده (Train/Val/Test) و **K-Fold Cross-Validation**
- ✅ ارزیابی با prompt point (centroid از ground truth)
- ✅ محاسبه Dice، IoU و HD95 برای هر کیس
- ✅ جدول مقایسه خودکار + آزمون Wilcoxon
- ✅ نمودارهای Bar و Box با کیفیت مقاله
- ✅ سازگاری با Linux، WSL و Windows
- ✅ رفع ۱۰ باگ مهم در کد اصلی SAM-Med3D
- ✅ اسکریپت‌های آماده برای استقرار روی سرور

---

## پیش‌نیازهای سخت‌افزاری

| قطعه | حداقل | توصیه‌شده |
|------|--------|-----------|
| **GPU** | ۱۲ گیگابایت VRAM | ۱۶ گیگابایت (RTX 4060 Ti / A100) |
| **RAM** | ۳۲ گیگابایت | ۶۴ گیگابایت |
| **فضای دیسک** | ۵۰ گیگابایت خالی | ۱۰۰ گیگابایت خالی |

---

## پیش‌نیازهای نرم‌افزاری

- **سیستم‌عامل:** Linux (Ubuntu 20.04+ / Pop!_OS)، WSL2 یا Windows 10+
- **پایتون:** 3.10
- **CUDA:** 11.8 یا 12.1
- **PyTorch:** 2.x
- **Conda:** Miniforge یا Miniconda

---

## راه‌اندازی روی لینوکس

### گام ۱: نصب Miniforge

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
# در تمام مراحل yes بزنید
source ~/.bashrc
conda --version    # باید 26.x.x چاپ کند
```

### گام ۲: ساخت محیط پایتون

```bash
conda create --name sammed3d_gpu python=3.10 -y
conda activate sammed3d_gpu
```

### گام ۳: نصب PyTorch با CUDA

بر اساس نسخه CUDA سیستم، یکی را انتخاب کنید:

```bash
# برای CUDA 12.1 (توصیه‌شده برای RTX 40xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# یا برای CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### گام ۴: تست دسترسی به GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

خروجی مورد انتظار:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Ti
```

### گام ۵: نصب کتابخانه‌های جانبی

```bash
pip install uv
uv pip install torchio opencv-python-headless matplotlib \
    prefetch_generator monai edt surface-distance medim \
    nibabel pandas scipy tqdm
```

### گام ۶: کلون کردن پروژه

```bash
git clone https://github.com/9M3a1h3d9i9/SAM-Med3D-Brain-TS.git
cd SAM-Med3D-Brain-TS
```

---

## راه‌اندازی روی ویندوز

### گام ۱: نصب Miniconda for Windows

از سایت رسمی دانلود و نصب کنید:
- `https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe`

**نکته مهم:** هنگام نصب، گزینه **"Just Me"** را انتخاب کنید (نه "All Users"). به این ترتیب نیازی به دسترسی Administrator نخواهید داشت.

### گام ۲: نصب Git for Windows

از سایت رسمی دانلود و نصب کنید:
- `https://git-scm.com/download/win`

### گام ۳: باز کردن Anaconda Prompt

از منوی Start، **Anaconda Prompt** را اجرا کنید.

### گام ۴: ساخت محیط پایتون

```cmd
conda create --name sammed3d_gpu python=3.10 -y
conda activate sammed3d_gpu
```

### گام ۵: نصب PyTorch با CUDA

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### گام ۶: تست دسترسی به GPU

```cmd
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### گام ۷: نصب کتابخانه‌های جانبی

```cmd
pip install torchio monai nibabel pandas scipy tqdm
```

### گام ۸: کلون کردن پروژه

```cmd
cd %USERPROFILE%
git clone https://github.com/9M3a1h3d9i9/SAM-Med3D-Brain-TS.git
cd SAM-Med3D-Brain-TS
```

---

## دانلود دیتاست

دیتاست `Task01_BrainTumour.tar` (~۷.۶ گیگابایت) از سرور AWS قابل دانلود است.

### لینک دانلود مستقیم

```
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
```

### روش ۱: دانلود با `wget` (Linux / WSL)

```bash
mkdir -p data/raw
cd data/raw
wget -c https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar

# استخراج
tar -xvf Task01_BrainTumour.tar
rm Task01_BrainTumour.tar

# پاک‌سازی فایل‌های متادیتای macOS
find Task01_BrainTumour -name "._*" -delete
find Task01_BrainTumour -name ".DS_Store" -delete
cd ../..
```

### روش ۲: دانلود دستی (توصیه‌شده برای Windows)

۱. با مرورگر (Chrome, Firefox) به این آدرس بروید:
   `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar`

۲. فایل ~۷.۶ گیگابایتی را دانلود کنید (ممکن است ۳۰-۶۰ دقیقه طول بکشد).

۳. فایل دانلود شده را به مسیر زیر منتقل کنید:
   ```
   %USERPROFILE%\SAM-Med3D-Brain-TS\data\raw\Task01_BrainTumour.tar
   ```

۴. در Anaconda Prompt، استخراج کنید:
   ```cmd
   cd data\raw
   tar -xvf Task01_BrainTumour.tar
   del Task01_BrainTumour.tar
   cd ..\..
   ```

### روش ۳: دانلود با PowerShell (Windows)

```powershell
mkdir data\raw
cd data\raw
Invoke-WebRequest -Uri "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar" -OutFile "Task01_BrainTumour.tar"
tar -xvf Task01_BrainTumour.tar
del Task01_BrainTumour.tar
cd ..\..
```

### ساختار مورد انتظار پس از استخراج

```
data/raw/Task01_BrainTumour/
├── imagesTr/          # ۴۸۴ تصویر ۴بعدی (.nii.gz)
│   ├── BRATS_001.nii.gz
│   ├── BRATS_002.nii.gz
│   └── ...
├── labelsTr/          # ۴۸۴ برچسب ۳بعدی (.nii.gz)
│   ├── BRATS_001.nii.gz
│   ├── BRATS_002.nii.gz
│   └── ...
├── imagesTs/          # ۲۶۶ تصویر تست (بدون برچسب)
│   ├── BRATS_485.nii.gz
│   └── ...
└── dataset.json       # متادیتای دیتاست
```

### بررسی صحت دانلود

**Linux / WSL:**
```bash
ls data/raw/Task01_BrainTumour/imagesTr/ | wc -l
# باید ۴۸۴ چاپ کند

ls data/raw/Task01_BrainTumour/labelsTr/ | wc -l
# باید ۴۸۴ چاپ کند
```

**Windows:**
```cmd
dir /b data\raw\Task01_BrainTumour\imagesTr\*.nii.gz | find /c /v ""
```

---

## دانلود چک‌پوینت

چک‌پوینت `SAM-Med3D-turbo.pth` (~۳۸۳ مگابایت) از **مخزن رسمی SAM-Med3D** قابل دانلود است. سه روش دسترسی وجود دارد:

### روش ۱: از Hugging Face (توصیه‌شده - سریع‌ترین)

فایل چک‌پوینت روی Hugging Face میزبانی می‌شود:

**لینک دانلود مستقیم:**
```
https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth
```

دستور دانلود در Linux / WSL:

```bash
mkdir -p SAM_Med3D/ckpt
cd SAM_Med3D/ckpt
wget -c https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth -O sam_med3d_turbo.pth
cd ../..
```

دستور دانلود در Windows (با PowerShell):

```powershell
mkdir SAM_Med3D\ckpt
cd SAM_Med3D\ckpt
Invoke-WebRequest -Uri "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth" -OutFile "sam_med3d_turbo.pth"
cd ..\..
```

### روش ۲: از Google Drive (لینک رسمی مخزن)

به صفحه رسمی مخزن SAM-Med3D بروید:
- `https://github.com/uni-medical/SAM-Med3D`

در بخش **"Model Zoo"** یا **"Pre-trained Checkpoints"**، لینک Google Drive را پیدا کنید. فایل `SAM-Med3D-turbo.pth` را دانلود کنید.

پس از دانلود، فایل را در مسیر `SAM_Med3D/ckpt/sam_med3d_turbo.pth` قرار دهید.

### روش ۳: استفاده از ابزار `gdown` (برای Google Drive)

اگر از Google Drive دانلود می‌کنید، نصب `gdown` توصیه می‌شود:

```bash
pip install gdown
mkdir -p SAM_Med3D/ckpt
gdown --id <FILE_ID> -O SAM_Med3D/ckpt/sam_med3d_turbo.pth
```

(به‌جای `<FILE_ID>` شناسه فایل Google Drive را از لینک مخزن رسمی بردارید)

### بررسی صحت دانلود

پس از دانلود، حجم فایل را بررسی کنید:

**Linux / WSL:**
```bash
ls -lh SAM_Med3D/ckpt/sam_med3d_turbo.pth
# باید نشان دهد: ~۳۸۳ مگابایت
```

**Windows:**
```cmd
dir SAM_Med3D\ckpt\sam_med3d_turbo.pth
```

اگر حجم فایل کمتر از ۳۸۰ مگابایت بود، دانلود ناقص است. دوباره دانلود کنید.

### ساختار مورد انتظار پس از دانلود

```
SAM-Med3D-Brain-TS/
├── SAM_Med3D/
│   └── ckpt/
│       └── sam_med3d_turbo.pth    # ~۳۸۳ مگابایت
├── scripts/
├── utils/
├── data/
└── README.md
```

### نکته مهم

**این چک‌پوینت روی GitHub این پروژه نیست** چون حجمش از محدودیت ۱۰۰ مگابایتی GitHub بیشتر است. شما باید آن را از منابع بالا دانلود کنید. پس از دانلود، **آن را در Git push نکنید** (چون در `.gitignore` قرار دارد).

---

## پیش‌پردازش داده

اسکریپت `prepare_brats.py` داده MSD را به فرمت SAM-Med3D تبدیل می‌کند:
- استخراج **FLAIR** (کانال ۰) به‌عنوان تصویر
- ساخت **ماسک باینری** برای هر ROI
- Resample به **۱.۵ میلی‌متر ایزوتروپیک**
- ذخیره به‌صورت **۳بعدی** (نه ۴بعدی)

### اجرا روی همه ۴۸۴ کیس

**روی لینوکس:**
```bash
python scripts/prepare_brats.py
```

**روی ویندوز:**
```cmd
prepare_brats.bat
```
یا مستقیم:
```cmd
python scripts\prepare_brats.py
```

### اجرا روی تعداد محدود (برای تست سریع)

**روی لینوکس:**
```bash
python scripts/prepare_brats.py 3
```

**روی ویندوز:**
```cmd
python scripts\prepare_brats.py 3
```

**زمان تخمینی:** ~۱۵-۳۰ دقیقه برای ۴۸۴ کیس

**ساختار خروجی:**
```
data/brain_pre_sam/
├── edema/Task01_BrainTumour/
│   ├── imagesTr/BRATS_001.nii.gz    # (1, 160, 160, 103)
│   └── labelsTr/BRATS_001.nii.gz    # (1, 160, 160, 103) باینری
├── enhancing/Task01_BrainTumour/...
└── non_enhancing/Task01_BrainTumour/...
```

---

## تقسیم داده

### حالت ۱: تقسیم ساده (۷۰/۱۵/۱۵)

**روی لینوکس:**
```bash
python scripts/split_dataset.py --mode split --ratios 0.70,0.15,0.15 --overwrite
```

**روی ویندوز:**
```cmd
python scripts\split_dataset.py --mode split --ratios 0.70,0.15,0.15 --overwrite
```

خروجی:
```
data/brain_train/
data/brain_val/
data/brain_test/
```

### حالت ۲: K-Fold Cross-Validation

```bash
python scripts/split_dataset.py --mode kfold --k 5 --overwrite
```

هر fold از یک زیرمجموعه متفاوت برای test و validation استفاده می‌کند:
- `test_idx` = fold فعلی
- `val_idx` = (fold + 1) % K
- `train_idx` = سایر foldها

خروجی:
```
data/brain_fold_1/{train,val,test}/
data/brain_fold_2/{train,val,test}/
...
data/brain_fold_5/{train,val,test}/
```

### بررسی صحت تقسیم

**روی لینوکس:**
```bash
for s in train val test; do
    n=$(find data/brain_${s} -name "*.nii.gz" | wc -l)
    echo "brain_${s}: ${n} files"
done
```

**روی ویندوز:**
```cmd
dir /s /b data\brain_train\*.nii.gz | find /c /v ""
dir /s /b data\brain_val\*.nii.gz | find /c /v ""
dir /s /b data\brain_test\*.nii.gz | find /c /v ""
```

---

## آموزش مدل

### تنظیم فایل `train.sh` (لینوکس)

```bash
nano train.sh
```

### تنظیم فایل `train.bat` (ویندوز)

فایل `train.bat` از قبل آماده است و شامل تنظیمات زیر است:

```bat
python train.py ^
 --batch_size 2 ^
 --num_workers 4 ^
 --task_name "ft_brats_full" ^
 --checkpoint "SAM_Med3D/ckpt/sam_med3d_turbo.pth" ^
 --device "cuda" ^
 --num_epochs 50 ^
 --accumulation_steps 8 ^
 --img_size 128 ^
 --lr 8e-5
```

### توضیح پارامترها

| پارامتر | توضیح | مقدار توصیه‌شده |
|---------|-------|-----------------|
| `--batch_size` | تعداد نمونه در هر batch | ۲ (۱۶GB) / ۴ (۲۴GB) / ۸ (۴۰GB) |
| `--num_workers` | تعداد workerهای DataLoader | ۴-۸ |
| `--num_epochs` | تعداد epochهای آموزش | ۵۰-۲۰۰ |
| `--accumulation_steps` | Gradient Accumulation | ۸ (۱۶GB) / ۴ (۲۴GB) / ۲ (۴۰GB) |
| `--img_size` | اندازه ورودی | ۱۲۸ (باید با pretrained یکسان باشد) |
| `--lr` | Learning Rate | 8e-5 |

### اجرای آموزش

**روی لینوکس (مستقیم):**
```bash
cd ~/SAM-Med3D-Brain-TS
bash train.sh
```

**روی لینوکس (با tmux برای جلسات طولانی):**
```bash
tmux new -s training
conda activate sammed3d_gpu
cd ~/SAM-Med3D-Brain-TS
bash train.sh
# خروج از tmux بدون بستن: Ctrl+B سپس D
# بازگشت: tmux attach -t training
```

**روی ویندوز:**
```cmd
cd %USERPROFILE%\SAM-Med3D-Brain-TS
train.bat
```

### حلقه K-Fold (اختیاری)

**روی لینوکس:**
```bash
for fold in 1 2 3 4 5; do
    echo "=== Fold ${fold} ==="
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

### خروجی آموزش

```
work_dir/ft_brats_full/
├── sam_model_latest.pth      # آخرین چک‌پوینت
├── sam_model_loss_best.pth   # بهترین Loss
├── sam_model_dice_best.pth   # بهترین Dice
├── Loss.png                  # نمودار Loss
├── Dice.png                  # نمودار Dice
└── output_*.log              # لاگ آموزش
```

### مانیتور آموزش

**در ترمینال دوم:**

```bash
# مانیتور GPU
watch -n 2 nvidia-smi

# لاگ آموزش
tail -f ~/SAM-Med3D-Brain-TS/work_dir/ft_brats_full/output_*.log

# RAM و CPU
htop
```

---

## ارزیابی مدل

### ارزیابی مدل Pretrained (Baseline)

**روی لینوکس:**
```bash
python scripts/evaluate.py \
    "SAM_Med3D/ckpt/sam_med3d_turbo.pth" \
    "./data/brain_test" \
    "results/results_pretrained.csv"
```

**روی ویندوز:**
```cmd
python scripts\evaluate.py "SAM_Med3D\ckpt\sam_med3d_turbo.pth" ".\data\brain_test" "results\results_pretrained.csv"
```

### ارزیابی مدل Fine-Tuned

**روی لینوکس:**
```bash
python scripts/evaluate.py \
    "work_dir/ft_brats_full/sam_model_dice_best.pth" \
    "./data/brain_test" \
    "results/results_finetuned.csv"
```

**روی ویندوز:**
```cmd
evaluate.bat work_dir\ft_brats_full\sam_model_dice_best.pth .\data\brain_test results\results_finetuned.csv
```

### خروجی CSV

```
case,roi,dice,gt_voxels,pred_voxels
BRATS_001,edema,0.6123,15712,14089
BRATS_001,enhancing,0.2856,9354,20250
...
```

### بازه‌های Dice مورد انتظار

| ROI | Pretrained (Zero-Shot) | Fine-Tuned (کامل) |
|-----|------------------------|-------------------|
| ادم (Edema) | 0.55 - 0.65 | 0.75 - 0.85 |
| تومور فعال (Enhancing) | 0.20 - 0.30 | 0.70 - 0.80 |
| تومور غیرفعال (Non-enhancing) | 0.10 - 0.20 | 0.50 - 0.65 |

---

## مقایسه و رسم نمودار

```bash
python scripts/compare_results.py \
    results/results_pretrained.csv \
    results/results_finetuned.csv \
    "Pretrained" \
    "Fine-tuned" \
    "results/comparison_final"
```

### خروجی‌ها

- `results/comparison_final_table.csv` — جدول مقایسه با میانگین/انحراف معیار
- `results/comparison_final_bar.png` — نمودار Bar برای Dice هر ROI
- `results/comparison_final_box.png` — نمودار Box برای توزیع Dice
- خروجی کنسول شامل **آزمون Wilcoxon**

### نمونه خروجی

```
=== جدول مقایسه ===
          ROI  Pretrained_mean  Fine-tuned_mean  Delta
        edema           0.607            0.745  +0.138
    enhancing           0.226            0.720  +0.494
non_enhancing           0.142            0.560  +0.418

=== آزمون Wilcoxon ===
  edema: p=0.0001  ✅ معنادار
  enhancing: p=0.0001  ✅ معنادار
  non_enhancing: p=0.0002  ✅ معنادار
```

---

## ساختار پروژه

```
SAM-Med3D-Brain-TS/
├── scripts/
│   ├── prepare_brats.py        # پیش‌پردازش MSD → SAM-Med3D
│   ├── split_dataset.py        # تقسیم Train/Val/Test یا K-Fold
│   ├── evaluate.py             # ارزیابی روی Test Set
│   ├── compare_results.py      # مقایسه + نمودار + Wilcoxon
│   └── deploy_to_server.sh     # استقرار خودکار روی سرور
├── utils/
│   ├── data_loader.py          # Dataset class
│   ├── data_paths.py           # مدیریت مسیرها با SAM_DATA_ROOT
│   └── metric_utils.py         # Dice، NSD، فاصله سطح
├── segment_anything/           # معماری SAM-Med3D
├── data/                       # (نادیده گرفته شده در Git)
├── work_dir/                   # (نادیده گرفته شده) چک‌پوینت‌ها
├── results/                    # گزارش‌ها و نمودارها
├── train.py                    # اسکریپت اصلی آموزش
├── train.sh / train.bat        # اسکریپت‌های اجرا
├── evaluate.bat                # ارزیابی Windows
├── prepare_brats.bat           # پیش‌پردازش Windows
└── README.md
```

---

## عیب‌یابی

### `CUDA out of memory`

- `--batch_size` را کاهش دهید (مثلاً ۱)
- `--accumulation_steps` را افزایش دهید (مثلاً ۱۶)

### `Torch not compiled with CUDA enabled`

شما نسخه CPU PyTorch را نصب کرده‌اید. دوباره با نسخه GPU نصب کنید:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### `No module named 'segment_anything'`

در ابتدای اسکریپت، ریشه پروژه را به `sys.path` اضافه کنید:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### `Tensors must have same number of dimensions`

ناسازگاری بین تصویر ۴بعدی و برچسب ۳بعدی در `data_loader.py`. مطمئن شوید فایل‌ها به‌صورت ۳بعدی ذخیره شده‌اند.

### `AttributeError: 'BaseTrainer' object has no attribute 'seq_loss'`

اشتباه تایپی در `train.py`. `seq_loss` را به `seg_loss` تغییر دهید.

### `DiceCELoss: Expected floating point tensor`

در `train.py`، target را به float تبدیل کنید:

```python
loss = self.seg_loss(prev_masks, gt3D.float())
```

### `pos_embed` size mismatch

مطمئن شوید `--img_size 128` است (مدل pretrained به ورودی ۱۲۸³ نیاز دارد).

### روی ویندوز: `wmic is not recognized`

`wmic` در ویندوز ۱۰/۱۱ حذف شده. از PowerShell استفاده کنید:

```powershell
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores
[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
nvidia-smi
```

### روی ویندوز: `The requested operation requires elevation`

PowerShell یا Command Prompt را **as Administrator** اجرا کنید.

### روی ویندوز: خطای `pycrypto` یا `build error`

اگر پکیجی نیاز به کامپایلر داشت، از `--only-binary` استفاده کنید:

```cmd
pip install --only-binary :all: پکیج_name
```

---

## مستندات

- [**TECHNICAL_REPORT.md**](results/TECHNICAL_REPORT.md) — گزارش فنی پروژه
- [**FULL_GUIDE_0_to_100.md**](results/FULL_GUIDE_0_to_100.md) — راهنمای جامع از مفاهیم پایه تا اجرا روی سرور

---

## ارجاع

اگر از این کد استفاده می‌کنید، لطفاً به مقاله اصلی SAM-Med3D ارجاع دهید:

```bibtex
@article{wang2023sam,
  title={SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images},
  author={Wang, Haoyu and Guo, Sizheng and Ye, Jin and others},
  journal={arXiv preprint arXiv:2310.15161},
  year={2023}
}
```

و مقاله دیتاست MSD:

```bibtex
@article{antonelli2022medical,
  title={The Medical Segmentation Decathlon},
  author={Antonelli, Michela and Reinke, Annika and Bakas, Spyridon and others},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={4128},
  year={2022}
}
```

---

## مجوز

این پروژه برای **استفاده آکادمیک و تحقیقاتی** طراحی شده است. استفاده تجاری مجاز نیست.

---

## نویسندگان

**نویسنده اصلی:**

- **خانم مهندس نازنین سرابی** — دانشجوی کارشناسی ارشد مهندسی پزشکی، دانشگاه شاهد

**دستیار کدنویسی:**

- **[9M3a1h3d9i9](https://github.com/9M3a1h3d9i9)** — کارشناسی ارشد هوش مصنوعی، دانشگاه شاهد

**استاد راهنما:**

- **دکتر فروزان جستوجویی** — دانشگاه شاهد

---

## تشکر و قدردانی

- تیم SAM-Med3D (دانشگاه Jiaotong شانگهای و آزمایشگاه هوش مصنوعی شانگهای) برای مدل pretrained
- برگزارکنندگان Medical Segmentation Decathlon برای دیتاست تومور مغزی
- تیم‌های MONAI و TorchIO برای کتابخانه‌های تصویربرداری پزشکی

---

**آخرین بروزرسانی:** ۱۴۰۵/۰۶/۲۳