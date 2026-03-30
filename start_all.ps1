<#
.SYNOPSIS
    Khoi dong toan bo he thong Photo Restoration API chi bang 1 lenh.
    Mo 5 cua so PowerShell rieng biet cho 4 Workers + 1 API Gateway.

.USAGE
    Mo PowerShell tai thu muc goc cua du an, roi chay:
        .\start_all.ps1

    Neu gap loi Execution Policy:
        powershell -ExecutionPolicy Bypass -File .\start_all.ps1
#>

# ============================================================================
# CAU HINH
# ============================================================================
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$TEMP_SCRIPTS = Join-Path $PROJECT_ROOT ".tmp_launchers"

# Conda environments (khop voi .env.example)
$ENV_ZEROSCRATCHES = "rs-clean"
$ENV_GFPGAN        = "gfpgan-clean"

# Tao thu muc tam
if (Test-Path $TEMP_SCRIPTS) { Remove-Item $TEMP_SCRIPTS -Recurse -Force }
New-Item -ItemType Directory -Path $TEMP_SCRIPTS -Force | Out-Null

# ============================================================================
# HAM HO TRO: tao file .ps1 tam roi chay trong cua so moi
# ============================================================================
function Start-Worker {
    param(
        [string]$Name,
        [string]$CondaEnv,
        [string]$PythonScript,
        [string]$WorkDir
    )

    $scriptFile = Join-Path $TEMP_SCRIPTS "$Name.ps1"

    # Ghi noi dung script ra file — khong co van de escape
    @"
`$Host.UI.RawUI.WindowTitle = '$Name'
Set-Location '$WorkDir'
Write-Host '========================================'
Write-Host '  $Name'
Write-Host '  Conda Env: $CondaEnv'
Write-Host '========================================'
Write-Host ''
conda activate $CondaEnv
python $PythonScript
Write-Host ''
Write-Host 'Worker da dung. Nhan Enter de dong...'
Read-Host
"@ | Set-Content -Path $scriptFile -Encoding UTF8

    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $scriptFile
}

# ============================================================================
# KHOI DONG HE THONG
# ============================================================================
Write-Host ""
Write-Host "============================================================"
Write-Host "   PHOTO RESTORATION API - SYSTEM LAUNCHER"
Write-Host "============================================================"
Write-Host ""
Write-Host "  Project: $PROJECT_ROOT"
Write-Host ""

# --- Worker 1: ZeroScratches (port 8001) ---
Write-Host "[1/5] Khoi dong ZeroScratches Worker (port 8001)..."
Start-Worker `
    -Name "Worker1_ZeroScratches_8001" `
    -CondaEnv $ENV_ZEROSCRATCHES `
    -PythonScript "api/workers/zeroscratches_worker.py" `
    -WorkDir $PROJECT_ROOT
Write-Host "  [OK] ZeroScratches"

Start-Sleep -Seconds 2

# --- Worker 2: Colorization (port 8002) ---
Write-Host "[2/5] Khoi dong Colorization Worker (port 8002)..."
Start-Worker `
    -Name "Worker2_Colorization_8002" `
    -CondaEnv $ENV_GFPGAN `
    -PythonScript "api/workers/colorization_worker.py" `
    -WorkDir $PROJECT_ROOT
Write-Host "  [OK] Colorization"

Start-Sleep -Seconds 2

# --- Worker 3: GFPGAN (port 8003) ---
Write-Host "[3/5] Khoi dong GFPGAN Worker (port 8003)..."
Start-Worker `
    -Name "Worker3_GFPGAN_8003" `
    -CondaEnv $ENV_GFPGAN `
    -PythonScript "api/workers/gfpgan_worker.py" `
    -WorkDir $PROJECT_ROOT
Write-Host "  [OK] GFPGAN"

Start-Sleep -Seconds 3

# --- Worker 4: CodeFormer (port 8004) ---
Write-Host "[4/5] Khoi dong CodeFormer Worker (port 8004)..."
Start-Worker `
    -Name "Worker4_CodeFormer_8004" `
    -CondaEnv $ENV_GFPGAN `
    -PythonScript "api/workers/codeformer_worker.py" `
    -WorkDir $PROJECT_ROOT
Write-Host "  [OK] CodeFormer"

Start-Sleep -Seconds 3

# --- API Gateway (port 8000) ---
Write-Host "[5/5] Khoi dong FastAPI Gateway (port 8000)..."

$apiScript = Join-Path $TEMP_SCRIPTS "API_Gateway_8000.ps1"
@"
`$Host.UI.RawUI.WindowTitle = 'API Gateway (port 8000)'
Set-Location '$PROJECT_ROOT\api'
Write-Host '========================================'
Write-Host '  API Gateway: FastAPI (port 8000)'
Write-Host '========================================'
Write-Host ''
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Write-Host ''
Write-Host 'API Gateway da dung. Nhan Enter de dong...'
Read-Host
"@ | Set-Content -Path $apiScript -Encoding UTF8

Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $apiScript
Write-Host "  [OK] API Gateway"

# ============================================================================
# HOAN TAT
# ============================================================================
Write-Host ""
Write-Host "============================================================"
Write-Host "   TAT CA 5 DICH VU DA DUOC KHOI DONG!"
Write-Host "============================================================"
Write-Host ""
Write-Host "  Doi khoang 15-30 giay de tat ca model AI load xong."
Write-Host ""
Write-Host "  Endpoints:"
Write-Host "    - Health Check:        http://127.0.0.1:8000/"
Write-Host "    - Phuc che:            http://127.0.0.1:8000/api/restore"
Write-Host "    - To mau:              http://127.0.0.1:8000/api/colorize"
Write-Host "    - Phuc che + To mau:   http://127.0.0.1:8000/api/restore-and-colorize"
Write-Host ""
