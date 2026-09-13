@echo off
REM Windows batch equivalent of prepare_brats.py
REM Usage: prepare_brats.bat [num_cases]
REM If no arg, processes all 484 cases.

if "%~1"=="" (
    python scripts\prepare_brats.py
) else (
    python scripts\prepare_brats.py %~1
)
pause
