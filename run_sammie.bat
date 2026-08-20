@echo off
setlocal EnableExtensions
cd /d "%~dp0" || goto :working_directory_error

set "UV_DIR=%~dp0.uv"
set "UV_EXE=%UV_DIR%\uvw.exe"
set "VENV_PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

rem Prefer the existing project environment.  A source checkout created by
rem uv normally has .venv but does not necessarily contain the portable
rem .uv\uvw.exe used by the release archive.
if exist "%VENV_PYTHONW%" (
    start "" /D "%~dp0" "%VENV_PYTHONW%" "%~dp0launcher.py" %*
    exit /b 0
)

if exist "%VENV_PYTHON%" (
    start "" /D "%~dp0" "%VENV_PYTHON%" "%~dp0launcher.py" %*
    exit /b 0
)

if exist "%UV_EXE%" (
    start "" /B /D "%~dp0" "%UV_EXE%" run --no-sync launcher.py %*
    exit /b 0
)

where uv.exe >nul 2>&1
if not errorlevel 1 (
    start "" /B /D "%~dp0" uv.exe run --no-sync launcher.py %*
    exit /b 0
)

echo [Sammie Roto Studio] No usable Python environment was found.
echo Expected: "%VENV_PYTHONW%"
echo Run install_dependencies.bat, then try again.
pause
exit /b 1

:working_directory_error
echo [Sammie Roto Studio] Unable to open the application directory:
echo "%~dp0"
pause
exit /b 1
