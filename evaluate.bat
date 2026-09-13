@echo off
REM Windows batch equivalent of evaluate.sh
REM Usage: evaluate.bat <checkpoint_path> <test_root> <output_csv>

if "%~1"=="" (
    set CKPT=work_dir\ft_brats_full\sam_model_dice_best.pth
) else (
    set CKPT=%~1
)

if "%~2"=="" (
    set TEST_ROOT=.\data\brain_test
) else (
    set TEST_ROOT=%~2
)

if "%~3"=="" (
    set OUT_CSV=results\results_finetuned.csv
) else (
    set OUT_CSV=%~3
)

python scripts\evaluate.py "%CKPT%" "%TEST_ROOT%" "%OUT_CSV%"
pause
