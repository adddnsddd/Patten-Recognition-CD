param(
    [switch]$SkipDependencyInstall,
    [string]$BuildPython = ''
)
$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
if (-not $BuildPython) {
    $BuildPython = Join-Path $projectDir '.venv-build\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $BuildPython)) {
        & python -m venv (Join-Path $projectDir '.venv-build')
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create the build environment.' }
    }
}
$savedCache = $env:PYINSTALLER_CONFIG_DIR
Push-Location -LiteralPath $projectDir
try {
    $env:PYINSTALLER_CONFIG_DIR = Join-Path $projectDir 'build\pyinstaller-cache'
    if (-not $SkipDependencyInstall) {
        & $BuildPython -m pip install --disable-pip-version-check -r requirements-build.txt
        if ($LASTEXITCODE -ne 0) { throw 'Failed to install build dependencies.' }
    }
    & $BuildPython -m PyInstaller --noconfirm --clean --distpath dist --workpath build\work packaging\FaceRecognition.spec
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
    $exe = Join-Path $projectDir 'dist\人脸识别系统.exe'
    if (-not (Test-Path -LiteralPath $exe)) { throw 'Expected EXE was not produced.' }
    $file = Get-Item -LiteralPath $exe
    $hash = Get-FileHash -LiteralPath $exe -Algorithm SHA256
    [pscustomobject]@{Executable=$exe; SizeMiB=[math]::Round($file.Length/1MB, 2); SHA256=$hash.Hash} | Format-List
} finally {
    $env:PYINSTALLER_CONFIG_DIR = $savedCache
    Pop-Location
}
