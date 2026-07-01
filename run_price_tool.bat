@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PYTHON_CMD="

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
    goto :python_found
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
        goto :python_found
    )
)

for %%P in (python python3) do (
    where %%P >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%A in ('where %%P 2^>nul') do (
            echo %%A | findstr /i "\\WindowsApps\\" >nul
            if errorlevel 1 (
                %%P --version >nul 2>nul
                if not errorlevel 1 (
                    set "PYTHON_CMD=%%P"
                    goto :python_found
                )
            )
        )
    )
)

echo Python was not found.
echo Install Python 3.12 from https://www.python.org/downloads/
echo During install, check "Add python.exe to PATH".
pause
exit /b 1

:python_found
echo Using Python: %PYTHON_CMD%

%PYTHON_CMD% -c "import PySide6, mss, cv2, win32gui, pynput, yaml" >nul 2>nul
if errorlevel 1 (
    echo Required packages are missing.
    echo Installing packages from requirements.txt...
    %PYTHON_CMD% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Package installation failed.
        pause
        exit /b 1
    )
)

%PYTHON_CMD% app.py

if errorlevel 1 (
    echo.
    echo The app closed with an error.
    pause
)

endlocal
