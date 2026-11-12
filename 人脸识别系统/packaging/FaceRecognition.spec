# -*- mode: python ; coding: utf-8 -*-
"""Reproducible single-file Windows bundle; no developer paths or user data."""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH).parent
datas = collect_data_files('cv2', includes=['data/haarcascade_frontalface_default.xml'])
# audit.collect_environment uses importlib.metadata at run time.
datas += copy_metadata('matplotlib')
binaries = collect_dynamic_libs('cv2')
a = Analysis(
    [str(root / 'main.py')],
    pathex=[str(root)],
    datas=datas,
    binaries=binaries,
    hiddenimports=['cv2.data', 'matplotlib.backends.backend_agg', 'PIL.ImageTk',
                   'faceid.portable_check', 'faceid.runtime'],
    hookspath=[],
    hooksconfig={'matplotlib': {'backends': ['Agg']}},
    runtime_hooks=[],
    excludes=['tests', 'pytest', 'IPython', 'pandas', 'scipy', 'sympy',
              'torch', 'tensorflow', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
              'jupyter', 'notebook', 'numpy.f2py', 'numpy.testing', 'setuptools', 'pip'],
    noarchive=False,
    optimize=0,
)
# Keep NumPy's wheel-local DLLs, without storing duplicate copies at bundle root.
numpy_dlls = {Path(source).name.lower() for dest, source, kind in a.binaries
              if Path(dest).parent.as_posix() == 'numpy.libs'}
a.binaries = [(dest, source, kind) for dest, source, kind in a.binaries
              if not (Path(dest).parent == Path('.') and Path(source).name.lower() in numpy_dlls)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [],
          name='人脸识别系统', console=False, debug=False, strip=False, upx=False,
          version=str(root / 'packaging' / 'windows-version.txt'),
          bootloader_ignore_signals=False, disable_windowed_traceback=False)
