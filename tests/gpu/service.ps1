# tests/gpu/service.ps1
# Usage:
#   # from repo root
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\tests\gpu\service.ps1 up     # start (checks NVIDIA runtime) & wait until healthy
#   .\tests\gpu\service.ps1 down   # stop & remove containers + volumes
#
# Notes:
#   - Requires NVIDIA drivers + nvidia-container-toolkit (Docker GPU runtime).
#   - Compose file: tests/gpu/docker-compose.yml
#   - API: http://localhost:9002

param(
  [ValidateSet('up','down')]
  [string]$Action = 'up'
)

$ErrorActionPreference = "Stop"
$compose = "docker compose -f tests/gpu/docker-compose.yml"
$baseUrl = "http://localhost:9002"   # asr_ocr runs on 9002

if ($Action -eq 'up') {
  # Verify NVIDIA runtime
  $dockerInfo = docker info 2>$null
  if ($LASTEXITCODE -ne 0 -or ($dockerInfo -notmatch "nvidia")) {
    Write-Error "NVIDIA runtime not detected. Install nvidia-container-toolkit and ensure 'docker info' shows 'Runtimes: nvidia'."
    exit 1
  }

  # Start container (detached)
  iex "$compose up -d"

  # Wait until /health returns 'ok' (max ~5 min)
  $deadline = (Get-Date).AddMinutes(5)
  do {
    try {
      $res = Invoke-RestMethod -Uri "$baseUrl/health" -Method GET -TimeoutSec 3
      if ($res -eq 'ok') {
        Write-Host "Service healthy (GPU)."
        break
      }
    } catch { Start-Sleep -Seconds 3 }
  } while ((Get-Date) -lt $deadline)

  Write-Host "Ready. API at $baseUrl"
}
elseif ($Action -eq 'down') {
  iex "$compose down -v"
  Write-Host "Service stopped and volumes removed."
}
