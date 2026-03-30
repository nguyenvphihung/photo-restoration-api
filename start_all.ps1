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
$ENV_FILE = Join-Path $PROJECT_ROOT "api\.env"

function Get-EnvConfig {
    param(
        [string]$FilePath
    )

    $config = @{}
    if (-not (Test-Path $FilePath)) {
        return $config
    }

    Get-Content $FilePath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }

        $parts = $line -split "=", 2
        if ($parts.Count -eq 2) {
            $config[$parts[0].Trim()] = $parts[1].Trim()
        }
    }

    return $config
}

function Get-RequiredEnvs {
    $envConfig = Get-EnvConfig -FilePath $ENV_FILE

    return @{
        ZeroScratches = if ($envConfig.ContainsKey("ENV_ZEROSCRATCHES")) { $envConfig["ENV_ZEROSCRATCHES"] } else { "rs-clean" }
        GFPGAN        = if ($envConfig.ContainsKey("ENV_GFPGAN")) { $envConfig["ENV_GFPGAN"] } else { "gfpgan-clean" }
        API           = if ($envConfig.ContainsKey("ENV_API")) { $envConfig["ENV_API"] } else { "api" }
    }
}

function Get-CondaEnvNames {
    $condaJson = conda env list --json | ConvertFrom-Json
    return @($condaJson.envs_details.PSObject.Properties | ForEach-Object { $_.Value.name })
}

function Assert-CondaEnvsExist {
    param(
        [hashtable]$RequiredEnvs
    )

    $existingEnvs = Get-CondaEnvNames
    $missingEnvs = @($RequiredEnvs.Values | Select-Object -Unique | Where-Object { $_ -notin $existingEnvs })

    if ($missingEnvs.Count -eq 0) {
        return
    }

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "   THIEU CONDA ENV - KHONG THE KHOI DONG HE THONG" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Dang thieu cac environment sau:"
    $missingEnvs | ForEach-Object { Write-Host "  - $_" }
    Write-Host ""
    Write-Host "Conda env hien co tren may:"
    $existingEnvs | ForEach-Object { Write-Host "  - $_" }
    Write-Host ""
    Write-Host "Lenh tao goi y:"
    if ($missingEnvs -contains $RequiredEnvs.ZeroScratches) {
        Write-Host "  conda env create -f environments\rs-clean.yml"
    }
    if ($missingEnvs -contains $RequiredEnvs.GFPGAN) {
        Write-Host "  conda env create -f environments\gfpgan-clean.yml"
    }
    if ($missingEnvs -contains $RequiredEnvs.API) {
        Write-Host "  conda env create -f environments\api.yml"
    }
    Write-Host ""
    throw "Missing required conda environments."
}

# Load environment names
$RequiredEnvs = Get-RequiredEnvs
$ENV_ZEROSCRATCHES = $RequiredEnvs.ZeroScratches
$ENV_GFPGAN = $RequiredEnvs.GFPGAN
$ENV_API = $RequiredEnvs.API

# Fail fast if environments are missing
Assert-CondaEnvsExist -RequiredEnvs $RequiredEnvs

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
        [string]$Command,
        [string]$WorkDir
    )

    $scriptFile = Join-Path $TEMP_SCRIPTS "$Name.ps1"

    @"
`$Host.UI.RawUI.WindowTitle = '$Name'
Set-Location '$WorkDir'
Write-Host '========================================'
Write-Host '  $Name'
Write-Host '  Conda Env: $CondaEnv'
Write-Host '========================================'
Write-Host ''
conda run --no-capture-output -n $CondaEnv powershell -NoProfile -Command "$Command"
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
Write-Host "  ZeroScratches Env: $ENV_ZEROSCRATCHES"
Write-Host "  GFPGAN Env:        $ENV_GFPGAN"
Write-Host "  API Env:           $ENV_API"
Write-Host ""

# --- Worker 1: ZeroScratches (port 8001) ---
Write-Host "[1/5] Khoi dong ZeroScratches Worker (port 8001)..."
Start-Worker `
    -Name "Worker1_ZeroScratches_8001" `
    -CondaEnv $ENV_ZEROSCRATCHES `
    -Command "python api/workers/zeroscratches_worker.py" `
    -WorkDir $PROJECT_ROOT
Write-Host "  [OK] ZeroScratches"

Start-Sleep -Seconds 2

# --- Worker 2: Colorization (port 8002) ---
Write-Host "[2/5] Khoi dong Colorization Worker (port 8002)..."
Start-Worker `
    -Name "Worker2_Colorization_8002" `
    -CondaEnv $ENV_GFPGAN `
    -Command "python api/workers/colorization_worker.py" `
    -WorkDir $PROJECT_ROOT
Write-Host "  [OK] Colorization"

Start-Sleep -Seconds 2

# --- Worker 3: GFPGAN (port 8003) ---
Write-Host "[3/5] Khoi dong GFPGAN Worker (port 8003)..."
Start-Worker `
    -Name "Worker3_GFPGAN_8003" `
    -CondaEnv $ENV_GFPGAN `
    -Command "python api/workers/gfpgan_worker.py" `
    -WorkDir $PROJECT_ROOT
Write-Host "  [OK] GFPGAN"

Start-Sleep -Seconds 3

# --- Worker 4: CodeFormer (port 8004) ---
Write-Host "[4/5] Khoi dong CodeFormer Worker (port 8004)..."
Start-Worker `
    -Name "Worker4_CodeFormer_8004" `
    -CondaEnv $ENV_GFPGAN `
    -Command "python api/workers/codeformer_worker.py" `
    -WorkDir $PROJECT_ROOT
Write-Host "  [OK] CodeFormer"

Start-Sleep -Seconds 3

# --- API Gateway (port 8000) ---
Write-Host "[5/5] Khoi dong FastAPI Gateway (port 8000)..."
Start-Worker `
    -Name "API_Gateway_8000" `
    -CondaEnv $ENV_API `
    -Command "uvicorn app:app --host 0.0.0.0 --port 8000 --reload" `
    -WorkDir (Join-Path $PROJECT_ROOT "api")
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
Write-Host "    - Swagger:             http://127.0.0.1:8000/docs"
Write-Host "    - Phuc che:            http://127.0.0.1:8000/api/restore"
Write-Host "    - To mau:              http://127.0.0.1:8000/api/colorize"
Write-Host "    - Phuc che + To mau:   http://127.0.0.1:8000/api/restore-and-colorize"
Write-Host ""
