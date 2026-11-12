param(
    [string]$BuildPython = 'python',
    [string]$ExtraPackagePath = ''
)
$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$savedInstallerCache = $env:PYINSTALLER_CONFIG_DIR
$savedExtraSite = $env:SHOP_BUILD_EXTRA_SITE
$savedPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $projectDir
try {
    $env:PYINSTALLER_CONFIG_DIR = Join-Path $projectDir 'build/pyinstaller-cache'
    if ($ExtraPackagePath) {
        $env:SHOP_BUILD_EXTRA_SITE = $ExtraPackagePath
        $env:PYTHONPATH = $ExtraPackagePath
    }
    & $BuildPython -m PyInstaller --noconfirm --clean --distpath dist --workpath build/work packaging/ShoppingRecommender.spec
    if ($LASTEXITCODE -ne 0) { throw 'Build failed. Install requirements-build.txt in the build environment.' }
    $portableExe = Join-Path $projectDir 'dist/网购推荐.exe'
    $checkDir = Join-Path $projectDir 'build/portable-check/独立程序 with spaces'
    $emptyWorkingDir = Join-Path $projectDir 'build/portable-check/empty-working-directory'
    New-Item -ItemType Directory -Path $checkDir -Force | Out-Null
    New-Item -ItemType Directory -Path $emptyWorkingDir -Force | Out-Null
    $checkExe = Join-Path $checkDir '网购推荐.exe'
    Copy-Item -LiteralPath $portableExe -Destination $checkExe -Force
    $savedPath = $env:PATH
    $savedPythonHome = $env:PYTHONHOME
    try {
        $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
        $env:PYTHONPATH = $null
        $env:PYTHONHOME = $null
        $process = Start-Process -FilePath $checkExe -ArgumentList '--self-test' `
            -WorkingDirectory $emptyWorkingDir -WindowStyle Hidden -PassThru
        if (-not $process.WaitForExit(90000)) {
            Stop-Process -Id $process.Id
            throw 'Portable runtime check timed out.'
        }
        $process.Refresh()
        if ($process.ExitCode -ne 0) { throw 'Portable runtime check failed.' }
    } finally {
        $env:PATH = $savedPath
        $env:PYTHONHOME = $savedPythonHome
        $env:PYTHONPATH = $savedPythonPath
    }
    $report = Get-Content -LiteralPath (Join-Path $checkDir 'portable_self_test.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $report.passed -or -not $report.frozen) { throw 'Portable report did not pass.' }
    $hash = Get-FileHash -LiteralPath $portableExe -Algorithm SHA256
    $info = Get-Item -LiteralPath $portableExe
    $summary = [ordered]@{Executable=$portableExe; SizeMiB=[math]::Round($info.Length / 1MB, 2); SHA256=$hash.Hash; Checks=$report.checks}
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath 'build/portable-build.json' -Encoding UTF8
    $summary | ConvertTo-Json -Depth 5
} finally {
    $env:PYINSTALLER_CONFIG_DIR = $savedInstallerCache
    $env:SHOP_BUILD_EXTRA_SITE = $savedExtraSite
    $env:PYTHONPATH = $savedPythonPath
    Pop-Location
}
