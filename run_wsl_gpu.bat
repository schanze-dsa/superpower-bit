@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "BASH_SCRIPT=run_wsl_gpu.sh"

where wsl.exe >nul 2>nul
if errorlevel 1 (
  echo [error] WSL is not installed or not in PATH.
  echo [hint] Install WSL first: wsl --install
  pause
  exit /b 1
)

if not exist "%SCRIPT_DIR%\%BASH_SCRIPT%" (
  echo [error] Cannot find %BASH_SCRIPT% in: %SCRIPT_DIR%
  pause
  exit /b 1
)

echo [info] Launching WSL script from: %SCRIPT_DIR%
wsl.exe --cd "%SCRIPT_DIR%" -e bash "%BASH_SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo [error] %BASH_SCRIPT% failed with exit code %EXIT_CODE%.
  pause
  exit /b %EXIT_CODE%
)

echo [ok] %BASH_SCRIPT% finished successfully.
pause
exit /b 0
