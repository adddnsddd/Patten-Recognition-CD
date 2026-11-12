param([switch]$SkipDependencyInstall)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$buildPython = Join-Path $projectDir '.venv-build/Scripts/python.exe'
$savedInstallerCache = $env:PYINSTALLER_CONFIG_DIR
$env:PYINSTALLER_CONFIG_DIR = Join-Path $projectDir 'build/pyinstaller-cache'
Push-Location -LiteralPath $projectDir
try {
    if (-not (Test-Path -LiteralPath $buildPython)) {
        & python -m venv .venv-build
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create the build environment.' }
    }
    if (-not $SkipDependencyInstall) {
        & $buildPython -m pip install --disable-pip-version-check -r requirements-build.txt
        if ($LASTEXITCODE -ne 0) { throw 'Failed to install the build dependencies.' }
    }
    $versionInfoPath = Join-Path $projectDir 'packaging/windows-version.txt'
    $commonOptions = @('--noconfirm', '--clean', '--windowed',
        '--version-file', $versionInfoPath, '--specpath', 'build/specs')

    & $buildPython -m PyInstaller @commonOptions --onedir --name RobotCleaner `
        --distpath build/stage --workpath build/work-onedir app.py
    if ($LASTEXITCODE -ne 0) { throw 'Folder build failed.' }
    $folderExe = Join-Path $projectDir 'build/stage/RobotCleaner/RobotCleaner.exe'
    $folderCheck = Start-Process -FilePath $folderExe -ArgumentList '--self-test' `
        -WorkingDirectory $projectDir -WindowStyle Hidden -Wait -PassThru
    if ($folderCheck.ExitCode -ne 0) { throw 'Folder build runtime check failed.' }

    & $buildPython -m PyInstaller @commonOptions --onefile --name RobotCleaner-Portable `
        --distpath dist --workpath build/work-onefile app.py
    if ($LASTEXITCODE -ne 0) { throw 'Single-file build failed.' }

    $portableExe = Join-Path $projectDir 'dist/RobotCleaner-Portable.exe'
    $checkDir = Join-Path $projectDir 'build/portable-check/Portable app with spaces'
    $emptyWorkingDir = Join-Path $projectDir 'build/portable-check/empty-working-directory'
    New-Item -ItemType Directory -Path $checkDir -Force | Out-Null
    New-Item -ItemType Directory -Path $emptyWorkingDir -Force | Out-Null
    $checkExe = Join-Path $checkDir 'RobotCleaner-Portable.exe'
    Copy-Item -LiteralPath $portableExe -Destination $checkExe -Force
    $savedPath = $env:PATH
    $savedPythonPath = $env:PYTHONPATH
    $savedPythonHome = $env:PYTHONHOME
    try {
        $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
        $env:PYTHONPATH = $null
        $env:PYTHONHOME = $null
        $portableCheck = Start-Process -FilePath $checkExe -ArgumentList '--self-test' `
            -WorkingDirectory $emptyWorkingDir -WindowStyle Hidden -Wait -PassThru
        if ($portableCheck.ExitCode -ne 0) { throw 'Portable runtime check failed.' }
    } finally {
        $env:PATH = $savedPath
        $env:PYTHONPATH = $savedPythonPath
        $env:PYTHONHOME = $savedPythonHome
    }
    $reportPath = Join-Path $checkDir 'outputs/portable_self_test.json'
    $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $report.passed -or -not $report.frozen) { throw 'The portable report did not pass.' }
    $hash = Get-FileHash -LiteralPath $portableExe -Algorithm SHA256
    $fileInfo = Get-Item -LiteralPath $portableExe
    [pscustomobject]@{
        Output = $portableExe
        SizeMiB = [math]::Round($fileInfo.Length / 1MB, 2)
        SHA256 = $hash.Hash
        PortableChecksPassed = $report.checks.Count
    } | Format-List
} finally {
    $env:PYINSTALLER_CONFIG_DIR = $savedInstallerCache
    Pop-Location
}
