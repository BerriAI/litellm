$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$scripts = Join-Path $root ".venv\Scripts"
$litellm = Join-Path $scripts "litellm.exe"
$config = Join-Path $root "litellm_config.yaml"
$pgData = "C:\Users\18747\scoop\persist\postgresql\data"
$pgLog = Join-Path $root "postgresql.log"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "$scripts;$env:PATH"

function Test-TcpPort {
  param(
    [string] $HostName,
    [int] $Port
  )

  $client = $null
  try {
    $client = [System.Net.Sockets.TcpClient]::new()
    $asyncResult = $client.BeginConnect($HostName, $Port, $null, $null)
    if (-not $asyncResult.AsyncWaitHandle.WaitOne(1000, $false)) {
      return $false
    }
    $client.EndConnect($asyncResult)
    return $true
  } catch {
    return $false
  } finally {
    if ($client -ne $null) {
      $client.Close()
    }
  }
}

$pgListening = Test-TcpPort -HostName "127.0.0.1" -Port 5432
if (-not $pgListening) {
  pg_ctl status -D $pgData *> $null
}
if (-not $pgListening -and $LASTEXITCODE -ne 0) {
  Write-Host "Starting PostgreSQL..."
  pg_ctl start -D $pgData -l $pgLog
  if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "PostgreSQL failed to start."
    Write-Host "Try running this from a normal PowerShell window, not an Administrator window."
    Write-Host "Log file: $pgLog"
    throw "PostgreSQL failed to start"
  }
}

& $litellm --config $config --port 4000 @args
