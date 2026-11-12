# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import os

root = Path(SPECPATH).parent
extra_site = os.environ.get('SHOP_BUILD_EXTRA_SITE')
a = Analysis(
    [str(root / 'app.py')],
    pathex=[str(root)] + ([extra_site] if extra_site else []),
    datas=[(str(root / 'data' / name), 'data')
           for name in ('users.csv', 'items.csv', 'behaviors.csv')],
    binaries=[],
    hiddenimports=['recsys.tk_charts', 'recsys.portable_check', 'secrets'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'PIL', 'pandas', 'scipy', 'IPython', 'pytest',
              'tests', 'numpy.f2py', 'numpy.testing', 'setuptools', 'pip',
              'recsys.reporting', 'recsys.experiments'],
    noarchive=False,
    optimize=0,
)
# NumPy's hook supplies numpy.libs; dependency scanning may also copy identical
# DLLs into the archive root. Keep the package-local copy loaded by NumPy.
numpy_dlls = {Path(source).name.lower() for dest, source, kind in a.binaries
              if str(Path(dest).parent).replace('\\', '/') == 'numpy.libs'}
a.binaries = [(dest, source, kind) for dest, source, kind in a.binaries
              if not (Path(dest).parent == Path('.') and
                      Path(source).name.lower() in numpy_dlls)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [],
          name='网购推荐', console=False, debug=False, strip=False, upx=False,
          bootloader_ignore_signals=False, disable_windowed_traceback=False)
