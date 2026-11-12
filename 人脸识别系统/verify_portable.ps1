param([string]$Executable = '')
$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
if (-not $Executable) { $Executable = Join-Path $projectDir 'dist\人脸识别系统.exe' }
$Executable = (Resolve-Path -LiteralPath $Executable).Path
$checkRoot = Join-Path $projectDir 'build\portable-check'
$emptyCwd = Join-Path $checkRoot 'empty working directory'
New-Item -ItemType Directory -Path $emptyCwd -Force | Out-Null
$savedPath = $env:PATH
$savedPythonHome = $env:PYTHONHOME
$savedPythonPath = $env:PYTHONPATH
$savedVirtualEnv = $env:VIRTUAL_ENV
$cases = @()
try {
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    $env:PYTHONHOME = $null
    $env:PYTHONPATH = $null
    $env:VIRTUAL_ENV = $null
    foreach ($name in @('中文目录 with spaces', '二次迁移 second location')) {
        $destination = Join-Path $checkRoot $name
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        $copiedExe = Join-Path $destination '移动后的人脸识别.exe'
        Copy-Item -LiteralPath $Executable -Destination $copiedExe -Force
        foreach ($action in @('portable-self-test', 'ui-smoke', 'environment')) {
            $resultFile = Join-Path $destination ($action + '.json')
            $arguments = '--' + $action + ' --result-file "' + $resultFile + '"'
            $process = Start-Process -FilePath $copiedExe -ArgumentList $arguments -WorkingDirectory $emptyCwd -WindowStyle Hidden -PassThru
            if (-not $process.WaitForExit(180000)) {
                Stop-Process -Id $process.Id
                throw "Timed out: $name / $action"
            }
            $process.Refresh()
            if ($process.ExitCode -ne 0) { throw "Executable check failed: $name / $action / exit=$($process.ExitCode)" }
            if (-not (Test-Path -LiteralPath $resultFile)) { throw "Missing JSON result: $name / $action" }
            $report = Get-Content -LiteralPath $resultFile -Raw -Encoding UTF8 | ConvertFrom-Json
            $expectedRoot = [IO.Path]::GetFullPath((Join-Path $destination '人脸识别数据'))
            if (-not $report.runtime.frozen) { throw 'The check did not run inside a frozen executable.' }
            if ([IO.Path]::GetFullPath($report.runtime.data_root) -ne $expectedRoot) { throw 'Data was not saved next to the relocated EXE.' }
            if ($action -eq 'portable-self-test' -and -not $report.passed) { throw 'Portable functional checks failed.' }
            $cases += [ordered]@{Location=$name; Action=$action; ExitCode=$process.ExitCode; Result=$resultFile; DataRoot=$report.runtime.data_root; Frozen=$report.runtime.frozen; Passed=$true}
        }
    }
} finally {
    $env:PATH = $savedPath
    $env:PYTHONHOME = $savedPythonHome
    $env:PYTHONPATH = $savedPythonPath
    $env:VIRTUAL_ENV = $savedVirtualEnv
}
$hash = Get-FileHash -LiteralPath $Executable -Algorithm SHA256
$summary = [ordered]@{Passed=$true; Executable=$Executable; SHA256=$hash.Hash; Cases=$cases; CameraAccessed=$false; Scope='Same Windows host, two relocated Unicode/space paths, unrelated cwd, Python environment variables removed and PATH restricted to Windows. This is not a second physical computer or camera-driver test.'}
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $projectDir 'build\portable_verification.json') -Encoding UTF8
$summary | ConvertTo-Json -Depth 8
