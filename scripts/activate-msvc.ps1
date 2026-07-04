Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$vsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vsWhere)) {
    throw "vswhere.exe was not found. Install Visual Studio Build Tools with the C++ workload."
}

$vsPath = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ([string]::IsNullOrWhiteSpace($vsPath)) {
    throw "Visual Studio C++ build tools were not found."
}

$vcVars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $vcVars)) {
    throw "vcvars64.bat was not found under $vsPath."
}

# vcvars64.bat configures the MSVC compiler, linker, Windows SDK, and library
# search paths that nvcc needs when compiling CUDA code on Windows.
$environmentLines = cmd /d /s /c "`"$vcVars`" >nul && set"
foreach ($line in $environmentLines) {
    $equalsIndex = $line.IndexOf("=")
    if ($equalsIndex -le 0) {
        continue
    }

    $name = $line.Substring(0, $equalsIndex)
    $value = $line.Substring($equalsIndex + 1)
    Set-Item -Path "Env:$name" -Value $value
}

$vsCMakeBin = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
if ((Test-Path -LiteralPath $vsCMakeBin) -and ($env:Path -notlike "*$vsCMakeBin*")) {
    $env:Path = "$vsCMakeBin;$env:Path"
}

Write-Host "MSVC environment activated: $vsPath"
