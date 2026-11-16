# Moved to litellm/scripts/quickstart.ps1.
param([switch]$Yes)

function Forward-LiteLLMQuickstart {
  param([switch]$Yes, [object[]]$Rest)
  try {
    $script = Invoke-RestMethod 'https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/scripts/quickstart.ps1' -UseBasicParsing -ErrorAction Stop
    & ([scriptblock]::Create($script)) -Yes:$Yes @Rest
  } catch {
    Write-Error "LiteLLM: could not run the quickstart script. $_"
    $global:LASTEXITCODE = 1
  }
}
Forward-LiteLLMQuickstart -Yes:$Yes -Rest $args
$hostArgs = [Environment]::GetCommandLineArgs()
$oneShot = ($hostArgs -match '^-(c|command)$') -and -not ($hostArgs -match '^-noexit$')
if ($LASTEXITCODE -ne 0 -and ($PSCommandPath -or $oneShot)) { exit $LASTEXITCODE }
