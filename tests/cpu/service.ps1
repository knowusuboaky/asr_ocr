# tests/cpu/service.ps1
# Usage (from repo root):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\tests\cpu\service.ps1 up
#   .\tests\cpu\service.ps1 down
#   .\tests\cpu\service.ps1 health
#   .\tests\cpu\service.ps1 logs
#   .\tests\cpu\service.ps1 ps
#   .\tests\cpu\service.ps1 restart
param(
  [ValidateSet('up','down','health','logs','ps','restart')]
  [string]$Action = 'up',

  # Compose file path for the CPU stack
  [string]$ComposeFile = 'tests/cpu/docker-compose.yml',

  # Base URL for health probe
  [string]$BaseUrl = 'http://localhost:9002',

  # Total time to wait for health (seconds)
  [int]$TimeoutSec = 300,

  # Name of the primary service (used for tailing logs on failures)
  [string]$ServiceName = 'asr_ocr'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$compose = "docker compose -f `"$ComposeFile`""

function Invoke-Compose([string]$args) {
  Write-Host ">> $compose $args" -ForegroundColor DarkGray
  iex "$compose $args"
}

function Test-Health {
  param(
    [string]$Url,
    [int]$Timeout = 300
  )
  $deadline = (Get-Date).AddSeconds($Timeout)
  $attempt = 0
  $lastErr = $null
  while ((Get-Date) -lt $deadline) {
    $attempt++
    try {
      $res = Invoke-RestMethod -Uri "$Url/health" -Method GET -TimeoutSec 5
      if ($res -eq 'ok') {
        Write-Host "Service healthy (attempt $attempt)." -ForegroundColor Green
        return $true
      }
    } catch {
      $lastErr = $_
    }
    Start-Sleep -Seconds ([Math]::Min(5, 1 + [Math]::Floor($attempt / 2)))
  }
  if ($lastErr) { Write-Warning "Last error during health checks: $($lastErr.Exception.Message)" }
  return $false
}

switch ($Action) {
  'up' {
    # Pull latest image (safe if already present)
    try { Invoke-Compose 'pull' } catch { Write-Warning "Pull failed or skipped: $($_.Exception.Message)" }

    # Start detached
    Invoke-Compose 'up -d'

    # Wait for health
    if (-not (Test-Health -Url $BaseUrl -Timeout $TimeoutSec)) {
      Write-Warning "Service did not become healthy within $TimeoutSec seconds. Showing recent logs:"
      try { Invoke-Compose "logs --no-color --tail=200 $ServiceName" } catch {}
      throw "Service failed health check."
    }
    Write-Host "Ready. API at $BaseUrl" -ForegroundColor Cyan
  }

  'down' {
    Invoke-Compose 'down -v'
    Write-Host "Service stopped and volumes removed."
  }

  'health' {
    if (Test-Health -Url $BaseUrl -Timeout $TimeoutSec) {
      Write-Host "Healthy at $BaseUrl/health" -ForegroundColor Green
    } else {
      Write-Error "Unhealthy or not reachable at $BaseUrl/health"
    }
  }

  'logs' {
    Invoke-Compose "logs -f --tail=200 $ServiceName"
  }

  'ps' {
    Invoke-Compose 'ps'
  }

  'restart' {
    Invoke-Compose 'up -d'
    if (-not (Test-Health -Url $BaseUrl -Timeout $TimeoutSec)) {
      Write-Warning "Service did not become healthy within $TimeoutSec seconds after restart. Showing recent logs:"
      try { Invoke-Compose "logs --no-color --tail=200 $ServiceName" } catch {}
      throw "Service failed health check after restart."
    }
    Write-Host "Restarted and healthy. API at $BaseUrl" -ForegroundColor Cyan
  }
}
