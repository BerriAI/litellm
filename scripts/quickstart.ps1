# Moved to litellm/scripts/quickstart.ps1. Old links run the current script with the same arguments.
& ([scriptblock]::Create((Invoke-RestMethod -UseBasicParsing 'https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/scripts/quickstart.ps1'))) @args
