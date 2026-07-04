$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$logPath = Join-Path $PSScriptRoot "litellm_proxy_4000.combined.log"

function Write-ProxyLog {
    param([string]$Message)
    "[$(Get-Date -Format o)] $Message" | Out-File -LiteralPath $logPath -Append -Encoding utf8
}

function Load-DotEnv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line.Length -eq 0 -or $line.StartsWith("#")) {
            return
        }

        $idx = $line.IndexOf("=")
        if ($idx -le 0) {
            return
        }

        $name = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

try {
    Write-ProxyLog "bootstrap pid=$PID cwd=$PSScriptRoot"
    Load-DotEnv -Path (Join-Path $PSScriptRoot ".env")
    [Environment]::SetEnvironmentVariable("ANTHROPIC_BASE_URL", "https://api.kimi.com/coding", "Process")
    [Environment]::SetEnvironmentVariable("ANTHROPIC_API_BASE", "", "Process")
    [Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "Process")
    [Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "Process")
    [Environment]::SetEnvironmentVariable("PYTHONLEGACYWINDOWSSTDIO", "0", "Process")

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = Join-Path $PSScriptRoot ".venv\Scripts\litellm.exe"
    $psi.WorkingDirectory = $PSScriptRoot
    $psi.Arguments = "--config litellm_config.yaml --port 4000 --detailed_debug"
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi

    $logHandler = [System.Diagnostics.DataReceivedEventHandler] {
        param($sender, $eventArgs)
        if ($null -ne $eventArgs.Data) {
            $eventArgs.Data | Out-File -LiteralPath $logPath -Append -Encoding utf8
        }
    }
    $proc.add_OutputDataReceived($logHandler)
    $proc.add_ErrorDataReceived($logHandler)

    Write-ProxyLog "starting litellm on port 4000"
    [void]$proc.Start()
    Write-ProxyLog "litellm child pid=$($proc.Id)"
    $proc.BeginOutputReadLine()
    $proc.BeginErrorReadLine()
    $proc.StandardInput.AutoFlush = $true

    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 5
    }

    Write-ProxyLog "litellm exited code=$($proc.ExitCode)"
    exit $proc.ExitCode
}
catch {
    Write-ProxyLog "bootstrap failed: $($_.Exception.Message)"
    Write-ProxyLog $_.ScriptStackTrace
    exit 1
}
