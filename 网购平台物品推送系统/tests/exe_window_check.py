"""Capture the actual standalone window using host-side QA tools only."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv.append('--lightweight')
from app import RecommendationApp, enable_dpi_awareness


def main():
    enable_dpi_awareness()
    executable = ROOT / 'dist' / '网购推荐.exe'
    empty = ROOT / 'build' / 'portable-check' / 'empty-working-directory'
    empty.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    windows = os.environ.get('SystemRoot', r'C:\Windows')
    env['PATH'] = str(Path(windows) / 'System32') + ';' + windows
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    process = subprocess.Popen([str(executable)], cwd=str(empty), env=env, startupinfo=startup)
    user32 = ctypes.windll.user32
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    window = None
    try:
        deadline = time.monotonic() + 30
        while not window:
            if process.poll() is not None:
                raise RuntimeError('EXE在窗口创建前退出')
            if time.monotonic() > deadline:
                raise TimeoutError('等待EXE窗口超时')
            window = user32.FindWindowW(None, '猜你喜欢 · 网购推荐实验室')
            time.sleep(.1)
        time.sleep(1.5)
        screenshot = ROOT / 'outputs' / 'screenshots' / 'portable-demo.png'
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        proxy = SimpleNamespace(root=SimpleNamespace(winfo_id=lambda: window))
        RecommendationApp._capture_windows_client(proxy, screenshot)
        assert screenshot.stat().st_size > 10000
        print('Standalone window captured successfully')
    finally:
        if window:
            user32.PostMessageW(window, 0x0010, 0, 0)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)


if __name__ == '__main__':
    main()
