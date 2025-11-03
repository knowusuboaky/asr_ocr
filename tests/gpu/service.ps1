# tests/gpu/service.ps1
# Usage:
#   # from repo root
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\tests\gpu\service.ps1 up        # start (checks NVIDIA runtime) & wait until healthy
#   .\tests\gpu\service.ps1 down      # stop & remove containers + volumes
#   .\tests\gpu\service.ps1 pull      # docker compose pull
#   .\tests\gpu\service.ps1 health    # poll /health until success (or timeout)
#   .\tests\gpu\service.ps1 logs      # tail service logs
#   .\tests\gpu\service.ps1 ps        # show compose ps
#   .\tests\gpu\service.ps1 restart   # up -d and re-check health
#   .\tests\gpu\service.ps1 smoke     # bring up, run basic checks (incl. CUDA hint), bring down
#
# Notes:
#   - Requires NVIDIA drivers + nvidia-container-toolkit (Docker GPU runtime),
#     unless you set $env:SKIP_NVIDIA_CHECK=1 to bypass host checks.
#   - Compose file: tests/gpu/docker-compose.yml
#   - API: http://localhost:9002

param(
  [ValidateSet('up','down','pull','health','logs','ps','restart','smoke')]
  [string]$Action = 'up'
)

$ErrorActionPreference = "Stop"

# ----- Config (overridable via env) -----
$ComposeFile  = $env:COMPOSE_FILE
if (-not $ComposeFile -or -not (Test-Path $ComposeFile)) { $ComposeFile = "tests/gpu/docker-compose.yml" }
$ServiceName  = $env:SERVICE_NAME; if (-not $ServiceName) { $ServiceName = "asr_ocr" }
$BaseUrl      = $env:BASE_URL;     if (-not $BaseUrl)     { $BaseUrl = "http://localhost:9002" }
$TimeoutSec   = [int]($env:TIMEOUT_SEC | ForEach-Object { if ($_ -as [int]) { $_ } else { 300 } })
$LogTail      = [int]($env:LOG_TAIL   | ForEach-Object { if ($_ -as [int]) { $_ } else { 200 } })

# Prefer Docker Compose v2 plugin; fallback to docker-compose if needed
$compose =
  if ((Get-Command docker -ErrorAction SilentlyContinue) -and
      (docker compose version 2>$null)) { "docker compose -f `"$ComposeFile`"" }
  else { "docker-compose -f `"$ComposeFile`"" }

function Invoke-Compose {
  param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
  $cmd = "$compose $($Args -join ' ')"
  Write-Host ">> $cmd"
  iex $cmd
}

function Test-NvidiaRuntime {
  if ($env:SKIP_NVIDIA_CHECK -eq '1') {
    Write-Host "SKIP_NVIDIA_CHECK=1 → skipping NVIDIA runtime check."
    return $true
  }

  $hasNvidiaSMI = (Get-Command nvidia-smi -ErrorAction SilentlyContinue) -ne $null
  $dockerInfo = docker info 2>$null
  $runtimeHints = @(
    ($dockerInfo -match 'Runtimes:\s*.*nvidia'),
    ($dockerInfo -match 'Default Runtime:\s*nvidia'),
    ($dockerInfo -match 'NVIDIA')
  )
  if ($hasNvidiaSMI -or ($runtimeHints -contains $true)) { return $true }

  Write-Error "NVIDIA runtime not detected. Install nvidia-container-toolkit and ensure 'docker info' shows a 'nvidia' runtime (or set SKIP_NVIDIA_CHECK=1 to bypass)."
  return $false
}

function Wait-Healthy {
  Write-Host "Waiting for service health at $BaseUrl/health (timeout ${TimeoutSec}s)..."
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    try {
      $res = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET -TimeoutSec 5
      if ($res -eq 'ok') { Write-Host "Healthy!"; return }
    } catch { Start-Sleep -Seconds 2 }
  }
  Write-Warning "Service did not become healthy within ${TimeoutSec}s. Recent logs:"
  Invoke-Compose logs --no-color --tail $LogTail $ServiceName
  throw "Health check timed out."
}

function Print-RootSummary {
  Write-Host "Root summary (/):"
  try {
    $root = Invoke-RestMethod -Uri "$BaseUrl/" -Method GET -TimeoutSec 5 -ErrorAction Stop
    $root.ToString().Split("`n") | ForEach-Object { "  $_" } | Write-Host
  } catch {
    Write-Warning "Failed to fetch root summary: $($_.Exception.Message)"
  }
}

function Test-TranscribeUrl {
  Write-Host "HTML extract test (/transcribe_url → example.com):"
  try {
    $body = @{ url = "https://example.com" } | ConvertTo-Json -Compress
    $resp = Invoke-RestMethod -Uri "$BaseUrl/transcribe_url" -Method POST -ContentType "application/json" -Body $body -TimeoutSec 30
    $lines = $resp.ToString().Split("`n")
    $lines[0..([Math]::Min($lines.Count-1, 29))] | ForEach-Object { $_ } | Write-Host
  } catch {
    Write-Warning "transcribe_url failed: $($_.Exception.Message)"
  }
}

function Test-ExtractTxt {
  Write-Host "TXT ingest test (/extract_document with inline TXT):"
  try {
    $tmp = New-TemporaryFile
    "Hello from GPU smoke test." | Out-File -FilePath $tmp -Encoding utf8 -NoNewline
    $headers = @{ "X-Filename" = "note.txt" }
    $resp = Invoke-RestMethod -Uri "$BaseUrl/extract_document?ocr=none&vlm=none" -Method POST -InFile $tmp -Headers $headers -TimeoutSec 30
    Remove-Item $tmp -Force
    $lines = $resp.ToString().Split("`n")
    $lines[0..([Math]::Min($lines.Count-1, 24))] | ForEach-Object { $_ } | Write-Host
  } catch {
    Write-Warning "extract_document failed: $($_.Exception.Message)"
  }
}

function New-TinyPngPath {
  $b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAuMBg6i1r6kAAAAASUVORK5CYII=' # 1x1 PNG
  $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("tiny_" + [System.IO.Path]::GetRandomFileName() + ".png")
  [IO.File]::WriteAllBytes($tmp, [Convert]::FromBase64String($b64))
  return $tmp
}

function Test-TranscribeImage {
  Write-Host "Image OCR test (/transcribe_image with 1x1 PNG):"
  try {
    $png = New-TinyPngPath
    $headers = @{ "X-Filename" = "tiny.png" }
    $resp = Invoke-RestMethod -Uri "$BaseUrl/transcribe_image" -Method POST -InFile $png -Headers $headers -TimeoutSec 30
    Remove-Item $png -Force
    $lines = $resp.ToString().Split("`n")
    $lines[0..([Math]::Min($lines.Count-1, 24))] | ForEach-Object { $_ } | Write-Host
  } catch {
    Write-Warning "transcribe_image failed: $($_.Exception.Message)"
  }
}

function Assert-CudaHintInRoot {
  Write-Host "Checking root summary for CUDA hint…"
  try {
    $root = Invoke-RestMethod -Uri "$BaseUrl/" -Method GET -TimeoutSec 5
    $txt = $root.ToString()
    if ($txt -match '(cuda|CUDA)') {
      Write-Host "Detected CUDA hint in service banner."
    } else {
      Write-Warning "No explicit CUDA hint in banner. Service may be running on CPU. (This is informational.)"
    }
  } catch {
    Write-Warning "Could not verify CUDA hint: $($_.Exception.Message)"
  }
}

switch ($Action) {
  'up' {
    if (-not (Test-NvidiaRuntime)) { exit 1 }
    Invoke-Compose up -d
    Wait-Healthy
    Write-Host "Ready. API at $BaseUrl"
  }

  'down' {
    Invoke-Compose down -v
    Write-Host "Service stopped and volumes removed."
  }

  'pull' {
    Invoke-Compose pull
  }

  'health' {
    Wait-Healthy
  }

  'logs' {
    Invoke-Compose logs -f --tail $LogTail $ServiceName
  }

  'ps' {
    Invoke-Compose ps
  }

  'restart' {
    if (-not (Test-NvidiaRuntime)) { exit 1 }
    Invoke-Compose up -d
    Wait-Healthy
    Write-Host "Restarted and healthy. API at $BaseUrl"
  }

  'smoke' {
    # Ensure cleanup even on failure
    $cleanup = {
      Write-Host "Bringing stack down..."
      try { Invoke-Compose down -v } catch { }
    }
    try {
      if (-not (Test-NvidiaRuntime)) { exit 1 }
      Invoke-Compose pull
      Invoke-Compose up -d
      Wait-Healthy

      Print-RootSummary
      Assert-CudaHintInRoot
      Test-TranscribeUrl
      Test-ExtractTxt
      Test-TranscribeImage

      Write-Host "GPU smoke test complete!"
    } finally { & $cleanup }
  }
}
