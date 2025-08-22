@echo off
echo Setting up the environment for Perchance Revival (AMD GPU)...

:: Define the virtual environment directory name
set VENV_DIR=venv

:: Change directory to the script's location
cd /d "%~dp0"

:: Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python 3.8+ and ensure it's in your PATH.
    echo Download: https://www.python.org/downloads/windows/
    goto final_pause
)
echo Found Python.

:: Create virtual environment if it doesn't exist
if not exist %VENV_DIR% (
    echo Creating virtual environment "%VENV_DIR%"...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        goto final_pause
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists. Skipping creation.
)

:: Activate the virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment.
    goto final_pause
)
echo Virtual environment activated.

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements first (excluding torch)
echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install core dependencies.
    goto final_pause
) else (
    echo Core dependencies installed successfully.
)

:: Install PyTorch with DirectML for AMD GPUs
echo.
echo Installing PyTorch with DirectML for AMD GPUs...
pip install torch-directml
if %errorlevel% neq 0 (
    echo PyTorch for DirectML installation failed.
    echo Ensure you have a DirectX 12 compatible AMD GPU and up-to-date drivers.
) else (
    echo PyTorch for DirectML installed successfully.
)

:: Summary
echo.
echo ------- AMD SETUP SUMMARY -------
echo Setup for AMD (DirectML) is complete.
echo Please use the new 'run-AMD.bat' script to launch the application.
echo ---------------------------------

:: Deactivate the environment
echo Deactivating virtual environment...
call deactivate

:: Final prompt
:final_pause
echo.
echo Setup complete. Press any key to exit...
pause >nul
