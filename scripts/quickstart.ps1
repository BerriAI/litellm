# LiteLLM Gateway quickstart for Windows: the gateway, Postgres, and the admin UI in one command.
#   powershell -c "irm https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.ps1 | iex"
# On macOS and Linux, scripts/quickstart.sh does the same.
#
# To read it before running it:
#   irm https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.ps1 -OutFile quickstart.ps1
#   notepad quickstart.ps1
#   powershell -ExecutionPolicy Bypass -File quickstart.ps1
#
# Asks where to keep the files, then offers to copy the admin password and
# open the admin UI; every question has a default you accept by pressing Enter.
# It asks nothing when input is not a console, under CI or Claude Code, or
# with -Yes.
#
#   -Yes, or LITELLM_YES=1   no questions: install to ~\litellm-gateway, don't open a browser
#   LITELLM_DIR              folder to install into (skips the folder question)
#   LITELLM_PORT             port for the gateway (default 4000, or the next free one)
#   NO_COLOR                 plain output, which agents, CI, and log files always get
#
# New installs listen on this machine only (127.0.0.1). To reach the gateway
# from other machines, remove LITELLM_BIND from .env and put it behind TLS.
#
# Keys and the database password are random, written only to .env, which only
# your Windows account can read, and never printed. When you ask, the admin
# password goes straight to your clipboard. Needs Docker Desktop, Podman, or
# Rancher Desktop.
# Works in Windows PowerShell 5.1 and PowerShell 7. Everything runs inside
# Invoke-LiteLLMQuickstart, so a partial download runs nothing.
param([switch]$Yes)

function Invoke-LiteLLMQuickstart {
  param([switch]$Yes)
  $ErrorActionPreference = 'Stop'
  # Windows PowerShell draws a progress bar for every web request, which makes them crawl.
  $ProgressPreference = 'SilentlyContinue'
  Set-StrictMode -Version 2

  $composeUrl = 'https://raw.githubusercontent.com/BerriAI/litellm/main/docker/docker-compose.quickstart.yml'
  if ($env:LITELLM_COMPOSE_URL) { $composeUrl = $env:LITELLM_COMPOSE_URL }
  $onWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or ((Test-Path variable:IsWindows) -and $IsWindows)

  # ------------------------------------------------------------ output

  # A person at a console gets colors, step marks, and a spinner. Agents, CI,
  # log files, and NO_COLOR get the same lines as plain text.
  $style = (-not [Console]::IsOutputRedirected) -and (-not $env:NO_COLOR) -and (-not $env:CI) -and
    (-not $env:CLAUDECODE) -and ($env:TERM -ne 'dumb')
  # Escape sequences give the brand blue and bold; older consoles get the
  # sixteen console colors instead.
  $vt = $style -and $Host.UI.PSObject.Properties['SupportsVirtualTerminal'] -and $Host.UI.SupportsVirtualTerminal
  # Symbols need a console font that has them: Windows Terminal, VS Code, or
  # any macOS or Linux terminal. The classic console gets plain ASCII.
  $unicode = $style -and ((-not $onWindows) -or $env:WT_SESSION -or ($env:TERM_PROGRAM -eq 'vscode'))
  if ($unicode) {
    $sym = @{ Ok = [char]0x2713; Warn = '!'; Err = [char]0x2717; Head = [char]0x25C6; Ask = '?'; Pointer = [char]0x276F
      Frames = @([char]0x280B, [char]0x2819, [char]0x2839, [char]0x2838, [char]0x283C, [char]0x2834, [char]0x2826, [char]0x2827, [char]0x2807, [char]0x280F)
      Hint = "$([char]0x2191)/$([char]0x2193) to move, Enter to choose"; Dot = [char]0x00B7
      TL = [char]0x256D; TR = [char]0x256E; BL = [char]0x2570; BR = [char]0x256F; H = [char]0x2500; V = [char]0x2502 }
  } else {
    $sym = @{ Ok = '+'; Warn = '!'; Err = 'x'; Head = '*'; Ask = '?'; Pointer = '>'; Frames = @('|', '/', '-', '\')
      Hint = 'Up/Down to move, Enter to choose'; Dot = '-'; TL = '+'; TR = '+'; BL = '+'; BR = '+'; H = '-'; V = '|' }
  }
  $e = [char]27
  $ansi = @{ a = "$e[38;2;91;108;255m"; b = "$e[1m"; d = "$e[90m"; g = "$e[32m"; y = "$e[33m"; r = "$e[1;31m"; u = "$e[1;38;2;91;108;255m"; off = "$e[0m" }
  $colors = @{ a = 'Blue'; b = $null; d = 'DarkGray'; g = 'Green'; y = 'Yellow'; r = 'Red'; u = 'Blue' }

  # Say "text with {a:accent} {b:bold} {d:dim} {g:green} {y:yellow} {r:red} {u:link} spans"
  function Say {
    param([string]$Text, [switch]$Err, [switch]$NoNewline)
    $parts = [regex]::Split($Text, '(\{[abdgyru]:[^}]*\})')
    # Errors go out plain when stderr is redirected, so a log file gets no escape codes.
    if (-not $style -or ($Err -and [Console]::IsErrorRedirected)) {
      $plain = ($parts | ForEach-Object { if ($_ -match '^\{[abdgyru]:(.*)\}$') { $Matches[1] } else { $_ } }) -join ''
      $stream = if ($Err) { [Console]::Error } else { [Console]::Out }
      if ($NoNewline) { $stream.Write($plain) } else { $stream.WriteLine($plain) }
      return
    }
    if ($vt) {
      $line = ($parts | ForEach-Object { if ($_ -match '^\{([abdgyru]):(.*)\}$') { $ansi[$Matches[1]] + $Matches[2] + $ansi.off } else { $_ } }) -join ''
      Write-Host $line -NoNewline:$NoNewline
      return
    }
    foreach ($p in $parts) {
      if ($p -match '^\{([abdgyru]):(.*)\}$') {
        $c = $colors[$Matches[1]]
        if ($c) { Write-Host $Matches[2] -NoNewline -ForegroundColor $c } else { Write-Host $Matches[2] -NoNewline }
      } elseif ($p) { Write-Host $p -NoNewline }
    }
    if (-not $NoNewline) { Write-Host '' }
  }
  # Markup characters in a value (a path, a URL) must not open a span.
  function Lit([string]$s) { return $s.Replace('{', '(').Replace('}', ')') }
  function Step([string]$Text) { if ($style) { Say ("{g:$($sym.Ok)} " + $Text) } else { Say $Text } }
  function Warn([string]$Text) { if ($style) { Say ("{y:$($sym.Warn)} " + $Text) } else { Say $Text } }
  function Fail([string]$Head, [string]$Rest = '') {
    $tail = if ($Rest) { " $Rest" } else { '' }
    if ($style) { Say ("{r:$($sym.Err) $Head}" + $tail) -Err } else { Say ($Head + $tail) -Err }
  }
  function Tildify([string]$Path) {
    if ($Path -eq $HOME) { return '~' }
    if ($Path.StartsWith($HOME + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { return '~' + $Path.Substring($HOME.Length) }
    return $Path
  }
  function Elapsed([datetime]$Since) {
    $s = [int]((Get-Date) - $Since).TotalSeconds
    if ($s -lt 60) { return "${s}s" } else { return "$([math]::Floor($s / 60))m $($s % 60)s" }
  }
  function Clear-Line {
    if ($vt) { Write-Host "`r$e[2K" -NoNewline } else { Write-Host ("`r" + (' ' * ([Console]::WindowWidth - 1)) + "`r") -NoNewline }
  }
  function Show-Cursor([bool]$Shown) { try { [Console]::CursorVisible = $Shown } catch { <# a host without a cursor #> } }

  # Run a native program; returns its exit code and stdout lines. stderr is
  # dropped so Windows PowerShell does not turn it into a terminating error.
  function Invoke-Native {
    param([string]$File, [string[]]$Arguments)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
      $out = & $File @Arguments 2>$null
      $code = $LASTEXITCODE
    } catch {
      $out = @(); $code = 1
    } finally {
      $ErrorActionPreference = $prev
    }
    return [pscustomobject]@{ Code = $code; Out = @($out) }
  }

  # Write a text file as UTF-8 without a byte order mark, with LF line endings,
  # which is what Docker Compose and git expect.
  function Write-PlainText([string]$Path, [string]$Text, [switch]$Append) {
    $enc = New-Object System.Text.UTF8Encoding $false
    if ($Append) { [IO.File]::AppendAllText($Path, $Text, $enc) } else { [IO.File]::WriteAllText($Path, $Text, $enc) }
  }

  # The POSIX cksum of a string's UTF-8 bytes: CRC-32 over the bytes and then
  # their length. quickstart.sh names projects with the same number, so both
  # scripts find the same database for a folder. Int64 keeps the arithmetic
  # unsigned; hex literals with the top bit set would be negative here.
  function Get-Cksum([string]$Text) {
    $bytes = New-Object System.Collections.Generic.List[byte]
    $bytes.AddRange([Text.Encoding]::UTF8.GetBytes($Text))
    for ($n = [int64]$bytes.Count; $n -gt 0; $n = $n -shr 8) { $bytes.Add([byte]($n -band 255)) }
    [int64]$crc = 0
    foreach ($b in $bytes) {
      $crc = $crc -bxor ([int64]$b -shl 24)
      for ($i = 0; $i -lt 8; $i++) {
        $crc = if ($crc -band 2147483648) { (($crc -shl 1) -bxor 79764919) -band 4294967295 } else { ($crc -shl 1) -band 4294967295 }
      }
    }
    return (-bnot $crc) -band 4294967295
  }

  # ------------------------------------------------------------ questions

  $interactive = (-not $Yes) -and (-not $env:LITELLM_YES) -and (-not $env:CI) -and (-not $env:CLAUDECODE) -and
    (-not [Console]::IsInputRedirected) -and [Environment]::UserInteractive

  # Menu "Question" Default @(@('label', 'note'), ...) -> the 1-based pick.
  # ReadKey holds on to Ctrl+C, so take it as a key and stop the way PowerShell
  # does; finally blocks still run and put the terminal back.
  function Read-Key {
    $saved = [Console]::TreatControlCAsInput
    try {
      [Console]::TreatControlCAsInput = $true
      $key = [Console]::ReadKey($true)
    } finally {
      [Console]::TreatControlCAsInput = $saved
    }
    if ($key.Key -eq 'C' -and ($key.Modifiers -band [ConsoleModifiers]::Control)) {
      throw [System.Management.Automation.PipelineStoppedException]::new()
    }
    return $key
  }

  function Menu {
    param([string]$Question, [int]$Default, [object[]]$Options)
    if (-not $interactive) { return $Default }
    $count = $Options.Count
    $pad = ($Options | ForEach-Object { $_[0].Length } | Measure-Object -Maximum).Maximum
    Write-Host ''
    if ($style) { Say ("{a:$($sym.Ask)} {b:" + (Lit $Question) + '}') } else { Write-Host $Question }
    $choice = $Default
    try {
      Show-Cursor $false
      $first = $true
      $top = 0
      while ($true) {
        # Escape-code consoles move up relative to the cursor; the classic
        # Windows console jumps back to the row the menu started on.
        if (-not $first) { if ($vt) { Write-Host "$e[$($count + 1)A" -NoNewline } else { [Console]::SetCursorPosition(0, $top) } }
        $first = $false
        for ($i = 1; $i -le $count; $i++) {
          $label = (Lit $Options[$i - 1][0]).PadRight($pad)
          $note = if ($Options[$i - 1].Count -gt 1) { Lit $Options[$i - 1][1] } else { '' }
          Clear-Line
          if ($i -eq $choice) { Say "  {u:$($sym.Pointer) $label}  {d:$note}" } else { Say "    $label  {d:$note}" }
        }
        Clear-Line
        Say "  {d:$($sym.Hint)}"
        if (-not $vt) { $top = [Console]::CursorTop - ($count + 1) }
        $key = Read-Key
        if ($key.Key -eq 'UpArrow' -or $key.KeyChar -eq 'k') { if ($choice -gt 1) { $choice-- } }
        elseif ($key.Key -eq 'DownArrow' -or $key.KeyChar -eq 'j') { if ($choice -lt $count) { $choice++ } }
        elseif ($key.KeyChar -match '^[1-9]$' -and [int]"$($key.KeyChar)" -le $count) { $choice = [int]"$($key.KeyChar)" }
        elseif ($key.Key -eq 'Enter') { break }
      }
      if ($vt) { Write-Host "$e[1A" -NoNewline } else { [Console]::SetCursorPosition(0, $top + $count) }
      Clear-Line
    } catch [System.Management.Automation.PipelineStoppedException] {
      throw
    } catch {
      # No cursor control here: fall back to a numbered list.
      for ($i = 1; $i -le $count; $i++) { Write-Host ("  $i) " + $Options[$i - 1][0]) }
      $answer = Read-Host "Choose [$Default]"
      if ($answer -match '^[1-9]$' -and [int]$answer -le $count) { $choice = [int]$answer }
    } finally {
      Show-Cursor $true
    }
    return $choice
  }

  # ------------------------------------------------------------ steps

  function Test-PortFree([int]$Port) {
    # Binding answers right away and is accurate. Probing with a connect is
    # not: the WSL2 localhost relay (Podman, Rancher Desktop) swallows the
    # refusal on closed ports, so every connect waits out its timeout and
    # closed ports look taken.
    $listener = New-Object System.Net.Sockets.TcpListener([Net.IPAddress]::Loopback, $Port)
    try {
      $listener.Start()
      return $true
    } catch {
      return $false
    } finally {
      $listener.Stop()
    }
  }

  function Test-Ready([int]$Port) {
    try {
      $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "http://127.0.0.1:$Port/health/readiness"
      return $r.StatusCode -eq 200
    } catch { return $false }
  }

  # Run a program with a spinner and the elapsed time, keeping its output to
  # show if it fails. Without styling it runs in the open.
  function Invoke-WithSpinner {
    param([string]$Label, [string]$Detail, [string]$File, [string[]]$Arguments)
    if (-not $style) {
      $prev = $ErrorActionPreference
      $ErrorActionPreference = 'Continue'
      try { & $File @Arguments 2>&1 | ForEach-Object { [Console]::Out.WriteLine("$_") }; $code = $LASTEXITCODE }
      finally { $ErrorActionPreference = $prev }
      return $code
    }
    $out = [IO.Path]::GetTempFileName(); $err = [IO.Path]::GetTempFileName()
    try {
      $p = Start-Process -FilePath $File -ArgumentList $Arguments -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
      $null = $p.Handle  # keeps ExitCode available after exit in Windows PowerShell
      $start = Get-Date
      $n = 0
      Show-Cursor $false
      try {
        while (-not $p.HasExited) {
          $s = [int]((Get-Date) - $start).TotalSeconds
          $clock = '{0}:{1:00}' -f [math]::Floor($s / 60), ($s % 60)
          $extra = if ($Detail) { "$Detail $($sym.Dot) " } else { '' }
          Clear-Line
          Say ("{a:$($sym.Frames[$n % $sym.Frames.Count])} $Label {d:$($sym.Dot) $extra$clock}") -NoNewline
          $n++
          Start-Sleep -Milliseconds 100
        }
        $p.WaitForExit()
        Clear-Line
      } finally { Show-Cursor $true }
      $code = $p.ExitCode
      if ($code -ne 0) {
        Get-Content $out, $err -ErrorAction SilentlyContinue | ForEach-Object { [Console]::Error.WriteLine($_) }
      }
      return $code
    } finally {
      # Also on Ctrl+C, so a cancelled run leaves no temporary files.
      Remove-Item -LiteralPath $out, $err -Force -ErrorAction SilentlyContinue
    }
  }

  function Wait-Ready([int]$Port) {
    # Up to 3 minutes, like the shell version. A spinner when styled.
    $start = Get-Date
    $n = 0
    $nextCheck = Get-Date
    if ($style) { Show-Cursor $false }
    try {
      while (((Get-Date) - $start).TotalSeconds -lt 180) {
        if ((Get-Date) -ge $nextCheck) {
          if (Test-Ready $Port) { if ($style) { Clear-Line }; return $true }
          $nextCheck = (Get-Date).AddSeconds(2)
        }
        if ($style) {
          $s = [int]((Get-Date) - $start).TotalSeconds
          Clear-Line
          Say ("{a:$($sym.Frames[$n % $sym.Frames.Count])} Waiting for the gateway to be ready {d:$($sym.Dot) $('{0}:{1:00}' -f [math]::Floor($s / 60), ($s % 60))}") -NoNewline
          $n++
        }
        Start-Sleep -Milliseconds 100
      }
      if ($style) { Clear-Line }
      return $false
    } finally { if ($style) { Show-Cursor $true } }
  }

  # The closing summary. Plain output, and a console too narrow for the frame,
  # get the same lines without it.
  function Open-Url([string]$Url) {
    try {
      if ($onWindows) { Start-Process $Url } elseif (Get-Command open -ErrorAction SilentlyContinue) { & open $Url } else { & xdg-open $Url }
      Step "Opened {a:$Url} in your browser"
    } catch {
      # No browser to open; print the address instead.
      Say "  {d:Open} {a:$Url} {d:when you are ready.}"
    }
  }

  # The admin password is the master key. It goes straight to the clipboard,
  # never to the screen.
  function Copy-Password([string]$Folder) {
    $pw = Get-Content -LiteralPath (Join-Path $Folder '.env') |
      Where-Object { $_.StartsWith('LITELLM_MASTER_KEY=') } | Select-Object -First 1
    if (-not $pw) { Warn ('No LITELLM_MASTER_KEY in ' + (Lit (Join-Path (Tildify $Folder) '.env')) + '.'); return }
    try {
      Set-Clipboard -Value $pw.Substring('LITELLM_MASTER_KEY='.Length) -ErrorAction Stop
      Step 'Copied the admin password. Paste it into the sign-in page.'
    } catch {
      # Another app can hold the Windows clipboard open.
      Warn ('Could not copy it. The password is LITELLM_MASTER_KEY in ' + (Lit (Join-Path (Tildify $Folder) '.env')) + '.')
    } finally {
      $pw = $null
    }
  }

  function Show-Box([string]$Title, [object[]]$Rows) {
    $width = $Title.Length
    foreach ($r in $Rows) { if (11 + $r[1].Length -gt $width) { $width = 11 + $r[1].Length } }
    $cols = 0
    try { $cols = [Console]::WindowWidth } catch { <# no console: print without the frame #> }
    if (-not $style -or $cols -lt $width + 4) {
      if ($style) { Say "{u:$Title}" } else { Say "$Title." }
      foreach ($r in $Rows) { Say ('  ' + ($r[0] + ':').PadRight(12) + (Lit $r[1])) }
      return
    }
    $rule = "$($sym.H)" * ($width + 2)
    Say "{a:$($sym.TL)$rule$($sym.TR)}"
    Say ("{a:$($sym.V)} {u:" + $Title.PadRight($width) + "} {a:$($sym.V)}")
    foreach ($r in $Rows) {
      $value = (Lit $r[1]).PadRight($width - 11)
      if ($r[0] -eq 'Admin UI') { $value = "{u:$value}" }
      Say ("{a:$($sym.V)} {d:" + $r[0].PadRight(10) + "} $value {a:$($sym.V)}")
    }
    Say "{a:$($sym.BL)$rule$($sym.BR)}"
  }

  # ------------------------------------------------------------ main

  $savedEncoding = $null
  try {
    if ($unicode -and $onWindows) {
      # Windows PowerShell writes the console in the OEM code page, which has no check marks.
      $savedEncoding = [Console]::OutputEncoding
      [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
    }
    if ($PSVersionTable.PSEdition -eq 'Desktop') {
      # Windows PowerShell can default to TLS 1.0, which GitHub refuses.
      [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    }

    if ($style) {
      Write-Host ''
      Say "{u:$($sym.Head) LiteLLM quickstart}"
      Say '{d:  The gateway, Postgres, and the admin UI in one command}'
      Write-Host ''
    } else {
      Say 'LiteLLM quickstart'
    }

    # Podman and Rancher Desktop (nerdctl in containerd mode; its dockerd mode
    # already provides a docker CLI) work the same way through Compose v2.
    $engineCmd = foreach ($cand in 'docker', 'podman', 'nerdctl') {
      $c = Get-Command $cand -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($c) { $c; break }
    }
    if (-not $engineCmd) {
      Fail 'No container engine found (docker, podman, or nerdctl).' 'The LiteLLM Gateway runs in a container alongside a Postgres database.'
      $install = if ($onWindows) { 'https://docs.docker.com/desktop/setup/install/windows-install/' } else { 'https://docs.docker.com/get-docker/' }
      Say '' -Err
      Say "  Install Docker Desktop ($install)," -Err
      Say '  Podman (https://podman.io), or Rancher Desktop (https://rancherdesktop.io), then run this again.' -Err
      Say '  Or deploy in one click (Railway or Render):  https://docs.litellm.ai/docs/proxy/docker_quick_start' -Err
      Say '  Only need to call models from Python?  pip install litellm' -Err
      return 1
    }
    $docker = $engineCmd.Source
    $engine = $engineCmd.Name -replace '\.exe$', ''
    $engineName = switch ($engine) { 'podman' { 'Podman' } 'nerdctl' { 'nerdctl' } default { 'Docker' } }
    $compose = Invoke-Native $docker @('compose', 'version', '--short')
    if ($compose.Code -ne 0) { Fail "Compose v2 ('$engine compose') is required."; return 1 }
    if ((Invoke-Native $docker @('info')).Code -ne 0) {
      $startHint = switch ($engine) {
        'podman' { 'Start it (podman machine start) and run this again.' }
        'nerdctl' { 'Start Rancher Desktop and run this again.' }
        default { 'Start Docker Desktop and run this again.' }
      }
      Fail "$engineName is installed but not running." $startHint
      return 1
    }
    $server = Invoke-Native $docker @('version', '--format', '{{.Server.Version}}')
    $engineVersion = if ($server.Code -eq 0 -and $server.Out.Count) { "$($server.Out[0]) " } else { '' }
    $composeVersion = "$($compose.Out[0])".TrimStart('v')
    Step "$engineName ${engineVersion}with Compose $composeVersion is running"

    $homeDir = Join-Path $HOME 'litellm-gateway'
    $hereDir = Join-Path (Get-Location).ProviderPath 'litellm-gateway'
    $found = $false
    if ($env:LITELLM_DIR) {
      $dir = $env:LITELLM_DIR
    } elseif (Test-Path -LiteralPath (Join-Path $hereDir '.env')) {
      $dir = $hereDir; $found = $true   # installed in this folder before
    } elseif (Test-Path -LiteralPath (Join-Path $homeDir '.env')) {
      $dir = $homeDir; $found = $true   # installed in the home folder before
    } elseif ($hereDir -eq $homeDir) {
      $dir = $homeDir
    } else {
      $pick = Menu 'Where should LiteLLM keep its files (.env with your keys, and the compose file)?' 1 @(
        , @((Tildify $homeDir), 'recommended, reruns always find it')
        , @((Tildify $hereDir), 'this folder'))
      $dir = if ($pick -eq 2) { $hereDir } else { $homeDir }
    }
    $created = -not (Test-Path -LiteralPath $dir)
    $null = New-Item -ItemType Directory -Force -Path $dir
    $dir = (Resolve-Path -LiteralPath $dir).ProviderPath
    Set-Location -LiteralPath $dir
    if ($found) { Step ('Found your install in {b:' + (Lit (Tildify $dir)) + '}') } else { Step ('Files go in {b:' + (Lit (Tildify $dir)) + '}') }

    $git = Get-Command git -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $inRepo = $git -and (Invoke-Native $git.Source @('rev-parse', '--is-inside-work-tree')).Code -eq 0
    if ($inRepo -and -not (Test-Path -LiteralPath (Join-Path $dir '.env')) -and
      (Invoke-Native $git.Source @('ls-files', '--error-unmatch', '.env')).Code -eq 0) {
      # An ignore rule does not cover a tracked file, so new keys written here
      # would show up as a change to commit.
      Fail 'Git tracks a .env file in this folder,' 'so your keys could be committed.'
      Say '' -Err
      Say '  Install into another folder (set LITELLM_DIR), or stop tracking the file' -Err
      Say '  first with:  git rm --cached .env' -Err
      return 1
    }
    if ($created) {
      # A folder this script made holds only its own files, so keep all of it out of git.
      Write-PlainText (Join-Path $dir '.gitignore') "*`n"
    } elseif ($inRepo -and (Invoke-Native $git.Source @('check-ignore', '-q', '.env')).Code -ne 0) {
      # In a folder that already existed, such as a repository root, leave the
      # tracked .gitignore alone and add only .env to this clone's local exclude
      # list, so the generated keys cannot be committed.
      $exclude = "$((Invoke-Native $git.Source @('rev-parse', '--git-path', 'info/exclude')).Out[0])"
      $exclude = [IO.Path]::GetFullPath([IO.Path]::Combine($dir, $exclude))
      $null = New-Item -ItemType Directory -Force -Path (Split-Path $exclude)
      $prefix = "$((Invoke-Native $git.Source @('rev-parse', '--show-prefix')).Out | Select-Object -First 1)"
      Write-PlainText $exclude "/$prefix.env`n" -Append
      Step ("Kept .env out of git {d:(added it to this repository's local exclude list, " + (Lit (Tildify $exclude)) + ')}')
    } elseif (-not $inRepo) {
      # An existing folder outside git: ignore only .env, so it stays out of
      # commits if the folder becomes a repository later.
      $ignore = Join-Path $dir '.gitignore'
      $current = if (Test-Path -LiteralPath $ignore) { [IO.File]::ReadAllText($ignore) } else { '' }
      if (($current -split "`r?`n") -notcontains '.env') {
        # Start on a new line if the file does not end with one.
        $lead = if ($current.Length -gt 0 -and -not $current.EndsWith("`n")) { "`n" } else { '' }
        Write-PlainText $ignore "$lead.env`n" -Append
      }
    }

    # The engine names containers and the database volume after the project, so
    # an install outside the home folder gets its own name and never shares a
    # database with another litellm-gateway folder. Windows paths ignore case,
    # so there the name does too.
    $envFile = Join-Path $dir '.env'
    $project = 'litellm-gateway'
    if ($dir -ne $homeDir) {
      $key = if ($onWindows) { $dir.ToLowerInvariant() } else { $dir }
      $project = "litellm-gateway-$(Get-Cksum $key)"
    }
    if (-not (Test-Path -LiteralPath $envFile)) {
      # Postgres keeps the password it was created with, so a new password over an
      # old database volume would lock the gateway out. Stop and explain instead.
      if ((Invoke-Native $docker @('volume', 'inspect', "${project}_postgres_data")).Code -eq 0) {
        Fail 'Found a database from an earlier install' "($engineName volume ${project}_postgres_data)"
        Say ('but no ' + (Lit (Join-Path (Tildify $dir) '.env')) + ' with its password.') -Err
        Say '' -Err
        Say '  Restore that .env file and run this again to keep your models and keys, or' -Err
        Say '  delete the old database and start fresh (this removes its models and keys):' -Err
        Say "    $engine volume rm ${project}_postgres_data" -Err
        return 1
      }
    }

    Invoke-WebRequest -UseBasicParsing -Uri $composeUrl -OutFile (Join-Path $dir 'docker-compose.quickstart.yml')
    Step 'Downloaded docker-compose.quickstart.yml'

    $saved = ''
    if (Test-Path -LiteralPath $envFile) {
      $m = Get-Content -LiteralPath $envFile | Where-Object { $_ -match '^LITELLM_PORT=(.*)$' } | Select-Object -Last 1
      if ($m) { $saved = ($m -split '=', 2)[1] }
    }
    if ($env:LITELLM_PORT) {
      $port = [int]$env:LITELLM_PORT
    } elseif ($saved) {
      $port = [int]$saved
    } elseif (Test-Path -LiteralPath $envFile) {
      $port = 4000   # an existing install without a saved port runs on the compose default
    } else {
      $port = 4000
      while (-not (Test-PortFree $port)) {
        $port++
        if ($port -gt 4099) {
          Fail 'Ports 4000 to 4099 are all in use.' 'Set LITELLM_PORT to a free port and run this again.'
          return 1
        }
      }
      if ($port -eq 4000) { Step 'Port {b:4000} is free' } else { Warn "Port 4000 is in use, so LiteLLM will use {b:$port}" }
    }

    if (Test-Path -LiteralPath $envFile) {
      Step 'Reusing .env, so existing keys and data keep working'
    } else {
      $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
      $hex = {
        param([int]$n)
        $bytes = New-Object byte[] $n
        $rng.GetBytes($bytes)
        -join ($bytes | ForEach-Object { $_.ToString('x2') })
      }
      # Lock a temporary file down before anything secret goes into it, then
      # rename it, so a failed write never leaves a partial .env that a rerun
      # would mistake for a finished install.
      $tmp = "$envFile.tmp"
      try {
        Write-PlainText $tmp ''
        if ($onWindows) {
          $acl = New-Object System.Security.AccessControl.FileSecurity
          $acl.SetAccessRuleProtection($true, $false)
          $me = [Security.Principal.WindowsIdentity]::GetCurrent().User
          $acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($me, 'FullControl', 'Allow')))
          Set-Acl -LiteralPath $tmp -AclObject $acl
        } else {
          & chmod 600 $tmp
          if ($LASTEXITCODE -ne 0) { throw "Could not restrict the permissions of $tmp." }
        }
        Write-PlainText $tmp (("LITELLM_MASTER_KEY=sk-{0}`nLITELLM_SALT_KEY=sk-{1}`nPOSTGRES_PASSWORD={2}`n" +
          "LITELLM_PORT={3}`nLITELLM_BIND=127.0.0.1:`nCOMPOSE_PROJECT_NAME={4}`n") -f (& $hex 32), (& $hex 32), (& $hex 24), $port, $project)
        [IO.File]::Move($tmp, $envFile)
      } finally {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
      }
      Step 'Generated .env with your master key, salt key, and database password {d:(keep this file; only you can read it)}'
    }

    # Compose prefers values already set in the environment over .env, so drop
    # inherited ones: .env stays the only source for keys and the project name.
    foreach ($name in 'LITELLM_MASTER_KEY', 'LITELLM_SALT_KEY', 'POSTGRES_PASSWORD', 'COMPOSE_PROJECT_NAME') {
      Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
    # The bind address follows .env when .env sets it. For an older .env without
    # it, a value already in the environment is kept.
    if (Get-Content -LiteralPath $envFile | Where-Object { $_ -match '^LITELLM_BIND=' }) { Remove-Item Env:LITELLM_BIND -ErrorAction SilentlyContinue }
    $env:LITELLM_PORT = "$port"

    $started = Get-Date
    $detail = ''
    foreach ($image in (Invoke-Native $docker @('compose', '-f', 'docker-compose.quickstart.yml', 'config', '--images')).Out) {
      if ($image -and (Invoke-Native $docker @('image', 'inspect', "$image")).Code -ne 0) { $detail = 'downloading images, first run only' }
    }
    if (-not $style) { Say 'Starting LiteLLM and Postgres (the first run downloads the images)...' }
    $code = Invoke-WithSpinner 'Starting LiteLLM and Postgres' $detail $docker @('compose', '-f', 'docker-compose.quickstart.yml', 'up', '-d')
    if ($code -ne 0) {
      Fail "$engineName could not start LiteLLM and Postgres." 'Its output is above.'
      return 1
    }
    if (-not (Wait-Ready $port)) {
      Fail 'The gateway did not become ready in 3 minutes.' 'See what it logged:'
      Say ('    cd ' + (Lit (Tildify $dir)) + "; $engine compose -f docker-compose.quickstart.yml logs litellm") -Err
      return 1
    }
    Step ("LiteLLM and Postgres are up {d:$($sym.Dot) $(Elapsed $started)}")

    $url = "http://localhost:$port/ui"
    $where = Tildify $dir
    if ($where -match ' ') { $where = "`"$where`"" }
    Write-Host ''
    # Over SSH the clipboard would be the server's, not yours. On Linux,
    # Set-Clipboard without xclip quietly keeps the text inside PowerShell, so
    # only Windows and macOS, which always have a clipboard, get the option.
    $canCopy = $interactive -and (-not $env:SSH_CONNECTION) -and
      ($onWindows -or ((Test-Path variable:IsMacOS) -and $IsMacOS))
    $password = "the LITELLM_MASTER_KEY value in $(Join-Path (Tildify $dir) '.env')"
    if ($canCopy) { $password = "copy it below, or LITELLM_MASTER_KEY in $(Join-Path (Tildify $dir) '.env')" }
    Show-Box 'LiteLLM is running' @(
      , @('Admin UI', $url)
      , @('Username', 'admin')
      , @('Password', $password)
      , @('Next', 'in the UI, open Models + Endpoints > Add Model and paste a provider API key')
      , @('Stop it', "cd $where; $engine compose -f docker-compose.quickstart.yml down"))

    if ($canCopy) {
      # Come back to the menu after each choice, with the next step as the default.
      $pick = 1
      while ($true) {
        $pick = Menu 'What next?' $pick @(
          , @('Copy the admin password', 'to the clipboard, ready to paste')
          , @('Open the admin UI in your browser')
          , @('Done'))
        if ($pick -eq 1) { Copy-Password $dir; $pick = 2 }
        elseif ($pick -eq 2) { Open-Url $url; $pick = 3 }
        else { break }
      }
    } elseif ($interactive) {
      if ((Menu 'Open the admin UI in your browser?' 1 @(, @('Yes'), @('No'))) -eq 1) { Open-Url $url }
    }
    return 0
  } finally {
    if ($savedEncoding) { [Console]::OutputEncoding = $savedEncoding }
  }
}

$code = Invoke-LiteLLMQuickstart -Yes:$Yes | Select-Object -Last 1
$global:LASTEXITCODE = $code
# Run from a file, or as `powershell -c "irm ... | iex"`, the exit code should
# reach the caller. Pasted into an open window, `exit` would close that window,
# so there it only sets $LASTEXITCODE.
$hostArgs = [Environment]::GetCommandLineArgs()
$oneShot = ($hostArgs -match '^-(c|command)$') -and -not ($hostArgs -match '^-noexit$')
if ($code -ne 0 -and ($PSCommandPath -or $oneShot)) { exit $code }
