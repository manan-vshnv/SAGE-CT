# Windows (PowerShell): train and test SAGE-CT on one dataset, five seeds (paper configuration).
#   .\run.ps1 ctrg        (or indinfm)
param([Parameter(Mandatory = $true)][ValidateSet("ctrg", "indinfm")][string]$DS)
$ErrorActionPreference = "Stop"
$work = if ($env:SAGECT_WORK) { $env:SAGECT_WORK } else { "work" }
foreach ($S in 0..4) {
    python -m sagect.train --ds $DS --seed $S --out "$work/runs/sagect_${DS}_s$S"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
