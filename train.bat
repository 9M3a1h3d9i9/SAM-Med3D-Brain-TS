@echo off
REM Windows batch equivalent of train.sh
REM Usage: train.bat

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

pause
