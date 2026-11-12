@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0..\网购推荐.exe" (
    start "" "%~dp0..\网购推荐.exe"
    exit /b
)
python app.py
if errorlevel 1 (
    echo.
    echo 启动失败。请查看上方错误，并确认已安装 requirements.txt 中的依赖。
    pause
)
