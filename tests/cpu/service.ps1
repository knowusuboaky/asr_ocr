# tests/cpu/service.ps1
# Usage:
#   # from repo root
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\tests\cpu\service.ps1 up     # start & wait until healthy
#   .\tests\cpu\service.ps1 down   # stop & remove containers + volumes

param(
  [ValidateSet('up','down')]
  [string]$Action = 'up'
)

$ErrorActionPreference = "Stop"
$compose = "docker compose -f tests/cpu/docker-compose.yml"
$baseUrl = "http://localhost:9002"   # asr_ocr runs on 9002

if ($Action -eq 'up') {
  # Start container (detached)
  iex "$compose up -d"

  # Wait until /health returns 'ok' (max ~5 min)
  $deadline = (Get-Date).AddMinutes(5)
  do {
    try {
      $res = Invoke-RestMethod -Uri "$baseUrl/health" -Method GET -TimeoutSec 3
      if ($res -eq 'ok') {
        Write-Host "Service healthy."
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
