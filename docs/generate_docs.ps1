$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$Doxyfile = Join-Path $ScriptDir "Doxyfile"
$OutputDir = Join-Path $RepoRoot "build/docs"

if (-not (Get-Command doxygen -ErrorAction SilentlyContinue)) {
Write-Error "doxygen was not found in PATH. Please install Doxygen and add it to PATH."
}

if (-not (Get-Command dot -ErrorAction SilentlyContinue)) {
Write-Warning "Graphviz 'dot' was not found in PATH. Doxygen may skip graph generation."
}

Push-Location $RepoRoot
try {
if (Test-Path $OutputDir) {
Remove-Item $OutputDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

doxygen $Doxyfile

}
finally {
Pop-Location
}
