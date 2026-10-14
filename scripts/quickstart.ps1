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
# Run it again for a menu to open, stop, start, update, or remove the install.
# It asks nothing when input is not a console, under CI or Claude Code, or
# with -Yes.
#
# It also takes an action, which the litellm-gateway command it installs takes too:
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.ps1))) stop
#   status, start (up), stop (down), restart, logs, open, update, uninstall
#
#   -Yes, or LITELLM_YES=1   no questions: install to ~\litellm-gateway, don't open a browser,
#                            and let uninstall delete the database without asking
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
  param([switch]$Yes, [object[]]$Rest = @())
  $ErrorActionPreference = 'Stop'
  # Windows PowerShell draws a progress bar for every web request, which makes them crawl.
  $ProgressPreference = 'SilentlyContinue'
  Set-StrictMode -Version 2

  $composeUrl = 'https://raw.githubusercontent.com/BerriAI/litellm/main/docker/docker-compose.quickstart.yml'
  if ($env:LITELLM_COMPOSE_URL) { $composeUrl = $env:LITELLM_COMPOSE_URL }
  $scriptUrl = 'https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.ps1'
  if ($env:LITELLM_SCRIPT_URL) { $scriptUrl = $env:LITELLM_SCRIPT_URL }
  $onWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or ((Test-Path variable:IsWindows) -and $IsWindows)

  # A step that cannot go on stops the whole run with exit code 1, from any depth.
  $exitSignal = 'LITELLM_QUICKSTART_EXIT'
  function Stop-Quickstart { throw $exitSignal }

  function Show-Usage([IO.TextWriter]$To) {
    @(
      'Usage: quickstart.ps1 [-Yes] [action]'
      ''
      'With no action it installs the LiteLLM Gateway, or shows a menu to manage the'
      'install it finds. Actions:'
      '  status      whether it is running, and where'
      '  start       start it (also: up)'
      '  stop        stop it; models, keys, and logs are kept (also: down)'
      '  restart     stop it and start it again'
      "  logs        show the gateway's logs"
      '  open        open the admin UI in your browser'
      '  update      download the latest image and restart on it'
      '  uninstall   remove its containers, its database, and its files'
      '  help        show this help (also: -h, --help)'
      ''
      '  -Yes, -y    ask nothing; without a console, uninstall needs it'
      ''
      'From the install command:'
      '  & ([scriptblock]::Create((irm https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.ps1))) status'
    ) | ForEach-Object { $To.WriteLine($_) }
  }

  # Windows PowerShell passes -y and --yes through as plain arguments.
  $noQuestions = [bool]$Yes
  # A port set by the caller is never moved; later steps set LITELLM_PORT too.
  $portPinned = [bool]$env:LITELLM_PORT
  $action = ''
  foreach ($arg in $Rest) {
    $a = "$arg"
    if ($a -in '-y', '--yes', '-yes') {
      $noQuestions = $true
    } elseif ($a -in '-h', '--help', '-help', 'help') {
      Show-Usage ([Console]::Out)
      return 0
    } elseif ($a -in 'status', 'start', 'up', 'stop', 'down', 'restart', 'logs', 'open', 'update', 'uninstall') {
      if ($action) {
        [Console]::Error.WriteLine("Choose one action, not both $action and $a")
        return 1
      }
      $action = $a.ToLowerInvariant()
    } else {
      [Console]::Error.WriteLine("Unknown option: $a")
      Show-Usage ([Console]::Error)
      return 1
    }
  }
  if ($action -eq 'up') { $action = 'start' } elseif ($action -eq 'down') { $action = 'stop' }
  if ($env:LITELLM_YES) { $noQuestions = $true }

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

  # Run a native program with its output on the console, never in this
  # function's results; returns its exit code.
  function Invoke-Shown {
    param([string]$File, [string[]]$Arguments)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $File @Arguments 2>&1 | ForEach-Object { [Console]::Out.WriteLine("$_") }; return $LASTEXITCODE }
    finally { $ErrorActionPreference = $prev }
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

  $interactive = (-not $noQuestions) -and (-not $env:CI) -and (-not $env:CLAUDECODE) -and
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
    # Windows lets a second program bind a port that another one holds on a
    # different address, so ask it what is listening first.
    if ($onWindows -and (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) {
      try {
        if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) { return $false }
      } catch { <# no answer: binding below decides #> }
    }
    # Binding answers right away and is accurate. Probing with a connect is
    # not: the WSL2 localhost relay (Podman, Rancher Desktop) swallows the
    # refusal on closed ports, so every connect waits out its timeout and
    # closed ports look taken. A listener on all addresses only blocks the
    # second bind, so try both.
    foreach ($address in [Net.IPAddress]::Loopback, [Net.IPAddress]::Any) {
      $listener = New-Object System.Net.Sockets.TcpListener($address, $Port)
      try {
        $listener.Start()
      } catch {
        return $false
      } finally {
        $listener.Stop()
      }
    }
    return $true
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
    if (-not $style) { return Invoke-Shown $File $Arguments }
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

  # ------------------------------------------------------------ the install

  # The install this run manages, filled in by Use-Install.
  $inst = @{ Dir = ''; Project = ''; Port = 0; Url = ''; Command = '' }
  $cf = @('compose', '-f', 'docker-compose.quickstart.yml')

  function Read-EnvValue([string]$Name) {
    $p = Join-Path (Get-Location).ProviderPath '.env'
    if (-not (Test-Path -LiteralPath $p)) { return '' }
    $m = Get-Content -LiteralPath $p |
      Where-Object { $_.StartsWith("$Name=") } | Select-Object -Last 1
    if ($m) { return ($m -split '=', 2)[1] }
    return ''
  }

  # An existing install: LITELLM_DIR, this folder, .\litellm-gateway, or
  # ~\litellm-gateway, whichever has a .env first. Empty when there is none.
  function Find-Install {
    $here = (Get-Location).ProviderPath
    if ($env:LITELLM_DIR) {
      if (Test-Path -LiteralPath (Join-Path $env:LITELLM_DIR '.env')) { return $env:LITELLM_DIR }
    } elseif ((Test-Path -LiteralPath (Join-Path $here '.env')) -and (Test-Path -LiteralPath (Join-Path $here 'docker-compose.quickstart.yml'))) {
      return $here
    } elseif (Test-Path -LiteralPath (Join-Path $here 'litellm-gateway/.env')) {
      return (Join-Path $here 'litellm-gateway')
    } elseif (Test-Path -LiteralPath (Join-Path $HOME 'litellm-gateway/.env')) {
      return (Join-Path $HOME 'litellm-gateway')
    }
    return ''
  }

  # What holds a port: Owner describes it, FreeIt is the command that frees it,
  # and Project is set when it is another quickstart install. All stay empty
  # when the holder cannot be named, such as another user's process.
  function Get-PortOwner([int]$Port) {
    $o = @{ Owner = ''; FreeIt = ''; Project = '' }
    $name = ''
    foreach ($line in (Invoke-Native $docker @('ps', '--format', '{{.Names}} {{.Ports}}')).Out) {
      if ("$line" -match "[:]$Port->") { $name = ("$line" -split ' ')[0]; break }
    }
    if ($name) {
      # Labels come as JSON: Windows PowerShell mangles the quotes an index template needs.
      $labels = Invoke-Native $docker @('inspect', '--format', '{{json .Config.Labels}}', $name)
      if ($labels.Code -eq 0) {
        try {
          $json = ($labels.Out -join "`n") | ConvertFrom-Json
          if ($json -and $json.PSObject.Properties['com.docker.compose.project']) { $o.Project = "$($json.'com.docker.compose.project')" }
        } catch { <# unreadable labels: treat it as a plain container #> }
      }
      # Only another quickstart install keeps its project, since that is the one
      # this script offers to stop. Whoever started the container set the label,
      # so it must also be a name Compose would accept.
      if ($o.Project -cmatch '^litellm-gateway[a-z0-9_-]*$') {
        $o.Owner = "another LiteLLM quickstart install ($engineName project $($o.Project))"
        $o.FreeIt = "$engine compose -p $($o.Project) stop"
      } else {
        $o.Project = ''
        $o.Owner = "the $engineName container $name"; $o.FreeIt = "$engine stop $name"
      }
      return $o
    }
    $procId = 0; $name = ''; $args0 = ''
    if ($onWindows) {
      try {
        $conn = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($conn) {
          $procId = [int]$conn.OwningProcess
          $name = (Get-Process -Id $procId -ErrorAction SilentlyContinue).ProcessName
          $args0 = "$((Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue).CommandLine)"
        }
      } catch { <# not allowed to look: leave it unnamed #> }
      if ($procId) { $o.FreeIt = "Stop-Process -Id $procId" }
    } elseif (Get-Command lsof -CommandType Application -ErrorAction SilentlyContinue) {
      $found = (Invoke-Native lsof @('-nP', "-iTCP:$Port", '-sTCP:LISTEN', '-t')).Out | Select-Object -First 1
      if ($found) {
        $procId = [int]"$found"
        $name = "$((Invoke-Native ps @('-o', 'comm=', '-p', "$procId")).Out | Select-Object -First 1)"
        if ($name) { $name = Split-Path -Leaf $name.Trim() }
        $args0 = "$((Invoke-Native ps @('-o', 'args=', '-p', "$procId")).Out | Select-Object -First 1)"
        $o.FreeIt = "kill $procId"
      }
    }
    if (-not $procId) { return $o }
    if ($args0 -match 'litellm') {
      $o.Owner = "a litellm server (pid $procId), maybe one started after pip install litellm"
    } else {
      if (-not $name) { $name = 'a program' }
      $o.Owner = "$name (pid $procId)"
    }
    return $o
  }

  function Select-Port {
    $saved = Read-EnvValue 'LITELLM_PORT'
    if ($env:LITELLM_PORT) {
      $port = [int]$env:LITELLM_PORT
    } elseif ($saved) {
      $port = [int]$saved
    } elseif (Test-Path -LiteralPath '.env') {
      $port = 4000   # an existing install without a saved port runs on the compose default
    } else {
      $port = 4000
      while (-not (Test-PortFree $port)) {
        $port++
        if ($port -gt 4099) {
          Fail 'Ports 4000 to 4099 are all in use.' 'Set LITELLM_PORT to a free port and run this again.'
          Stop-Quickstart
        }
      }
      if ($port -eq 4000) {
        Step 'Port {b:4000} is free'
      } else {
        $o = Get-PortOwner 4000
        $by = if ($o.Owner) { ' by ' + (Lit $o.Owner) } else { '' }
        Warn "Port 4000 is in use$by, so LiteLLM will use {b:$port}"
      }
    }
    $env:LITELLM_PORT = "$port"
    return $port
  }

  # Compose prefers values already set in the environment over .env, so drop
  # inherited ones: .env stays the only source for keys and the project name.
  function Clear-ComposeEnv {
    foreach ($name in 'LITELLM_MASTER_KEY', 'LITELLM_SALT_KEY', 'POSTGRES_PASSWORD', 'COMPOSE_PROJECT_NAME') {
      Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
    # The bind address follows .env when .env sets it. For an older .env without
    # it, a value already in the environment is kept.
    if (Get-Content -LiteralPath '.env' | Where-Object { $_ -match '^LITELLM_BIND=' }) { Remove-Item Env:LITELLM_BIND -ErrorAction SilentlyContinue }
  }

  function Use-Install([string]$Dir) {
    Set-Location -LiteralPath $Dir
    $inst.Dir = (Get-Location).ProviderPath
    $inst.Project = Read-EnvValue 'COMPOSE_PROJECT_NAME'
    # Installs from before COMPOSE_PROJECT_NAME have Compose's default, the folder name.
    if (-not $inst.Project) { $inst.Project = (Split-Path -Leaf $inst.Dir).ToLowerInvariant() -replace '[^a-z0-9_-]', '' }
    $inst.Port = Select-Port
    Clear-ComposeEnv
    $inst.Url = "http://localhost:$($inst.Port)/ui"
  }

  function Save-Compose {
    Invoke-WebRequest -UseBasicParsing -Uri $composeUrl -OutFile (Join-Path (Get-Location).ProviderPath 'docker-compose.quickstart.yml')
  }

  # Where a symbolic link points, or nothing when the path is not one.
  function Get-LinkTarget([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($item -and $item.LinkType -eq 'SymbolicLink') { return "$(@($item.Target)[0])" }
    return ''
  }

  function Test-BinOnPath {
    $bin = Join-Path $HOME '.local/bin'
    return (-not $onWindows) -and (@($env:PATH -split ':') -contains $bin) -and (Test-Path -LiteralPath $bin -PathType Container)
  }

  # The litellm-gateway command: the copy of this script kept beside .env, run
  # against this folder, so up, down, status, and the rest work from anywhere.
  function Install-Command {
    $dir = $inst.Dir
    $tmp = Join-Path $dir ('quickstart.ps1.' + [IO.Path]::GetRandomFileName())
    try {
      Invoke-WebRequest -UseBasicParsing -Uri $scriptUrl -OutFile $tmp
      if ([IO.File]::ReadAllText($tmp) -match '(?m)^\$code = Invoke-LiteLLMQuickstart') {
        Move-Item -LiteralPath $tmp -Destination (Join-Path $dir 'quickstart.ps1') -Force
      }
    } catch {
      # Offline, or a partial download: keep the copy already here, if any.
    } finally {
      Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path -LiteralPath (Join-Path $dir 'quickstart.ps1'))) { return }
    if ($onWindows) {
      # cmd reads a batch file line by line as it runs, so the last line exits
      # before uninstall has deleted the file under it.
      $path = Join-Path $dir 'litellm-gateway.cmd'
      $text = (@(
          '@echo off'
          'setlocal'
          'rem Manages the LiteLLM Gateway in this folder. Run it for a menu, or with one of'
          'rem status, up, down, restart, logs, open, update, uninstall'
          'set "LITELLM_DIR=%~dp0."'
          'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0quickstart.ps1" %* & exit /b'
        ) -join "`r`n") + "`r`n"
    } else {
      $path = Join-Path $dir 'litellm-gateway'
      $shell = if (Get-Command pwsh -CommandType Application -ErrorAction SilentlyContinue) { 'pwsh' } else { (Get-Process -Id $PID).Path }
      $text = (@(
          '#!/bin/sh'
          '# Manages the LiteLLM Gateway in this folder. Run it for a menu, or with one of'
          '# status, up, down, restart, logs, open, update, uninstall'
          ("export LITELLM_DIR='" + $dir.Replace("'", "'\''") + "'")
          ("exec '" + $shell.Replace("'", "'\''") + "' -NoProfile -File `"`$LITELLM_DIR/quickstart.ps1`" `"`$@`"")
        ) -join "`n") + "`n"
    }
    $current = if (Test-Path -LiteralPath $path) { [IO.File]::ReadAllText($path) } else { '' }
    if ($current -ne $text) { Write-PlainText $path $text }
    if ($onWindows) { return }
    $null = Invoke-Native chmod @('755', $path)
    # Linked into ~/.local/bin when that is on PATH. A link another install made
    # stays, unless that install is gone.
    if (-not (Test-BinOnPath)) { return }
    $bin = Join-Path $HOME '.local/bin'
    $link = Join-Path $bin 'litellm-gateway'
    $target = Get-LinkTarget $link
    if ($target) {
      if ($target -eq $path -or (Test-Path -LiteralPath ([IO.Path]::Combine($bin, $target)))) { return }
      Remove-Item -LiteralPath $link -Force
    } elseif (Test-Path -LiteralPath $link) {
      return
    }
    try { $null = New-Item -ItemType SymbolicLink -Path $link -Target $path } catch { <# not allowed: the full path still works #> }
  }

  # How this install's litellm-gateway command runs from anywhere, or nothing.
  function Find-Command {
    $name = if ($onWindows) { 'litellm-gateway.cmd' } else { 'litellm-gateway' }
    $path = Join-Path $inst.Dir $name
    if (-not (Test-Path -LiteralPath $path)) { return '' }
    if ((Test-BinOnPath) -and (Get-LinkTarget (Join-Path $HOME '.local/bin/litellm-gateway')) -eq $path) { return 'litellm-gateway' }
    $command = Tildify $path
    if ($command -match ' ') { $command = "& `"$command`"" }
    return $command
  }

  # The command for an action: the litellm-gateway command, or Compose itself.
  function Hint([string]$What) {
    if ($inst.Command) { return "$($inst.Command) $What" }
    $verb = switch ($What) { 'up' { 'up -d' } 'down' { 'stop' } 'uninstall' { 'down --volumes' } 'logs' { 'logs litellm' } default { $What } }
    $where = Tildify $inst.Dir
    if ($where -match ' ') { $where = "`"$where`"" }
    return "cd $where; $engine compose -f docker-compose.quickstart.yml $verb"
  }

  # The gateway container's status, such as "Up 2 hours", or nothing when there is none.
  function Get-GatewayStatus {
    $r = Invoke-Native $docker @('ps', '-a', '--filter', "label=com.docker.compose.project=$($inst.Project)",
      '--filter', 'label=com.docker.compose.service=litellm', '--format', '{{.Status}}')
    return "$($r.Out | Select-Object -First 1)"
  }

  function Test-Running { return (Get-GatewayStatus) -like 'Up*' }

  function Test-HasContainers([switch]$All) {
    $a = @('ps', '-q')
    if ($All) { $a += '-a' }
    $a += @('--filter', "label=com.docker.compose.project=$($inst.Project)")
    return [bool](@((Invoke-Native $docker $a).Out | Where-Object { $_ }).Count)
  }

  # The running version, from the API description: the first "version" in it is
  # the gateway's, and it needs no key.
  function Get-GatewayVersion {
    try {
      $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 -Uri "http://127.0.0.1:$($inst.Port)/openapi.json"
      if ("$($r.Content)" -match '"version": *"([^"]*)"') { return $Matches[1] }
    } catch { <# not up yet #> }
    return ''
  }

  # How many requests the gateway has handled, failed ones included, counted
  # from its request logs. It needs the admin key. Empty when the gateway cannot say.
  function Get-RequestCount {
    $key = Read-EnvValue 'LITELLM_MASTER_KEY'
    if (-not $key) { return '' }
    $today = [DateTime]::UtcNow.ToString('yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture)
    try {
      $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 -Headers @{ Authorization = "Bearer $key" } `
        -Uri "http://127.0.0.1:$($inst.Port)/global/activity?start_date=2000-01-01&end_date=$today"
      if ("$($r.Content)" -match '"sum_api_requests": *([0-9]+)') { return $Matches[1] }
    } catch { <# an older gateway, or no database yet #> }
    return ''
  }

  function Get-Images {
    return @((Invoke-Native $docker ($cf + @('config', '--images'))).Out | Where-Object { $_ } | ForEach-Object { "$_" })
  }

  # Before starting: if something else holds the port, say what and how to free it.
  # Moves this install to another port and saves it in .env. Writing the file in
  # place keeps its permissions, so only you can still read the keys.
  function Move-Port([int]$Port) {
    $p = Join-Path $inst.Dir '.env'
    $lines = @(Get-Content -LiteralPath $p -ErrorAction Stop | Where-Object { -not $_.StartsWith('LITELLM_PORT=') })
    Write-PlainText $p ((($lines + "LITELLM_PORT=$Port") -join "`n") + "`n")
    $env:LITELLM_PORT = "$Port"
    $inst.Port = $Port
    $inst.Url = "http://localhost:$Port/ui"
  }

  # Before starting: if something else holds the port, stop it when it is another
  # quickstart install, or move to a free port when it is anything else.
  function Confirm-PortFree {
    $port = $inst.Port
    if ((Test-Running) -or (Test-PortFree $port)) { return }
    $o = Get-PortOwner $port
    $who = if ($o.Owner) { $o.Owner } else { 'another program' }
    $next = 0
    if ($o.Project) {
      if ($interactive) {
        $choice = Menu "Port $port is in use by $who. Stop it and start this one?" 1 @(@('Yes, stop it', $o.FreeIt), @('No'))
        if ($choice -eq 1 -and (Invoke-WithSpinner 'Stopping it' '' $docker @('compose', '-p', $o.Project, 'stop')) -eq 0) {
          Step ('Stopped ' + (Lit $o.Owner))
          return
        }
      }
    } elseif (-not $portPinned) {
      foreach ($candidate in ($port + 1)..($port + 100)) {
        if (Test-PortFree $candidate) { $next = $candidate; break }
      }
      if ($next -and $interactive) {
        $where = 'saved in ' + (Join-Path (Tildify $inst.Dir) '.env')
        $choice = Menu "Port $port is in use by $who. Use port $next instead?" 1 @(@("Yes, use $next", $where), @('No'))
        if ($choice -eq 1) {
          Move-Port $next
          Step "LiteLLM moved from port $port to {b:$next}"
          return
        }
      }
    }
    $instead = if ($next) { ", such as $next" } else { '' }
    Fail (Lit "Port $port is in use by $who.") ('Free it, or set LITELLM_PORT in ' + (Lit (Join-Path (Tildify $inst.Dir) '.env')) + " to another port$instead.")
    if ($o.FreeIt) { Say ('  To free it:  ' + (Lit $o.FreeIt)) -Err }
    Stop-Quickstart
  }

  function Start-Gateway {
    Confirm-PortFree
    if (-not (Test-Path -LiteralPath 'docker-compose.quickstart.yml')) { Save-Compose }
    $started = Get-Date
    $detail = ''
    foreach ($image in Get-Images) {
      if ((Invoke-Native $docker @('image', 'inspect', $image)).Code -ne 0) { $detail = 'downloading images, first run only' }
    }
    if (-not $style) { if ($detail) { Say "Starting LiteLLM and Postgres ($detail)..." } else { Say 'Starting LiteLLM and Postgres...' } }
    if ((Invoke-WithSpinner 'Starting LiteLLM and Postgres' $detail $docker ($cf + @('up', '-d'))) -ne 0) {
      Fail "$engineName could not start LiteLLM and Postgres." 'Its output is above.'
      Stop-Quickstart
    }
    if (-not (Wait-Ready $inst.Port)) {
      Fail 'The gateway did not become ready in 3 minutes.' 'See what it logged:'
      Say ('    ' + (Lit (Hint 'logs'))) -Err
      Stop-Quickstart
    }
    Step "LiteLLM is up at {a:$($inst.Url)} {d:$($sym.Dot) $(Elapsed $started)}"
  }

  function Stop-Gateway {
    if (-not (Test-HasContainers)) { Step 'LiteLLM is already stopped'; return }
    if ((Invoke-WithSpinner 'Stopping LiteLLM and Postgres' '' $docker ($cf + @('stop'))) -ne 0) {
      Fail "$engineName could not stop LiteLLM and Postgres." 'Its output is above.'
      Stop-Quickstart
    }
    Step ("Stopped LiteLLM. Your models, keys, and logs are kept {d:$($sym.Dot) start it again: " + (Lit (Hint 'up')) + '}')
  }

  function Restart-Gateway {
    if ((Test-HasContainers) -and (Invoke-WithSpinner 'Stopping LiteLLM and Postgres' '' $docker ($cf + @('stop'))) -ne 0) {
      Fail "$engineName could not stop LiteLLM and Postgres." 'Its output is above.'
      Stop-Quickstart
    }
    Start-Gateway
  }

  function Update-Gateway {
    $before = ''
    if (Test-Running) { $before = Get-GatewayVersion }
    Save-Compose
    Install-Command
    $inst.Command = Find-Command
    $images = Get-Images
    $old = @(foreach ($image in $images) {
        $r = Invoke-Native $docker @('image', 'inspect', '--format', '{{.Id}}', $image)
        if ($r.Code -eq 0) { "$($r.Out | Select-Object -First 1)" }
      })
    if (-not $style) { Say 'Downloading the latest LiteLLM and Postgres images...' }
    if ((Invoke-WithSpinner 'Downloading the latest images' '' $docker ($cf + @('pull'))) -ne 0) {
      Fail "$engineName could not download the images." 'Its output is above.'
      Stop-Quickstart
    }
    Start-Gateway
    # Delete the images this replaced. Removal fails, and they stay, while any
    # container still uses them or another tag points at them.
    $now = @(foreach ($image in $images) { "$((Invoke-Native $docker @('image', 'inspect', '--format', '{{.Id}}', $image)).Out | Select-Object -First 1)" })
    foreach ($id in $old) {
      if ($id -and $now -notcontains $id) { $null = Invoke-Native $docker @('image', 'rm', $id) }
    }
    $after = Get-GatewayVersion
    if ($before -and $before -eq $after) {
      Step "LiteLLM $after is already the latest"
    } elseif ($before -and $after) {
      Step "Updated LiteLLM from $before to $after"
    } elseif ($after) {
      Step "LiteLLM $after is on the latest image"
    } else {
      Step 'LiteLLM is on the latest image'
    }
  }

  function Show-Logs([switch]$Follow) {
    if (-not (Test-HasContainers -All)) { Step 'No logs yet: start LiteLLM first'; return }
    if ($Follow) {
      Say '{d:The gateway logs. Press Ctrl+C to stop following them.}'
      $null = Invoke-Shown $docker ($cf + @('logs', '-f', '--tail', '100', 'litellm'))
      return
    }
    $null = Invoke-Shown $docker ($cf + @('logs', '--tail', '100', 'litellm'))
  }

  function Get-DesktopNote {
    if ($engine -eq 'docker' -and ((Invoke-Native $docker @('info', '--format', '{{.OperatingSystem}}')).Out -match 'Docker Desktop')) {
      return "listed as $($inst.Project) in Docker Desktop"
    }
    return ''
  }

  function Show-Status {
    $status = Get-GatewayStatus
    $image = "$(Get-Images | Where-Object { $_ -match 'litellm' } | Select-Object -First 1)"
    $desktop = Get-DesktopNote
    $rows = New-Object System.Collections.ArrayList
    if ($status -like 'Up*') {
      $uptime = ($status -replace '^Up ', '') -replace ' \(.*$', ''
      $version = Get-GatewayVersion
      if (-not $version) { $version = 'starting' }
      $requests = Get-RequestCount
      $null = $rows.Add(@('Admin UI', $inst.Url))
      $null = $rows.Add(@('Version', $version))
      $null = $rows.Add(@('Uptime', $uptime))
      if ($requests -eq '0') { $requests = 'none yet' } elseif ($requests) { $requests = "$requests handled" }
      if ($requests) { $null = $rows.Add(@('Requests', $requests)) }
      $null = $rows.Add(@('Folder', (Tildify $inst.Dir)))
      $null = $rows.Add(@('Image', $image))
      if ($desktop) { $null = $rows.Add(@('Desktop', $desktop)) }
      $null = $rows.Add(@('Stop it', (Hint 'down')))
      Show-Box 'LiteLLM is running' $rows.ToArray()
    } else {
      $null = $rows.Add(@('Folder', (Tildify $inst.Dir)))
      $null = $rows.Add(@('Image', $image))
      if ($desktop) { $null = $rows.Add(@('Desktop', $desktop)) }
      $null = $rows.Add(@('Start it', (Hint 'up')))
      $null = $rows.Add(@('Remove it', (Hint 'uninstall')))
      Show-Box 'LiteLLM is stopped' $rows.ToArray()
    }
  }

  function Open-UI {
    if (Test-Running) { Open-Url $inst.Url } else { Warn ('LiteLLM is stopped. Start it with: ' + (Lit (Hint 'up'))) }
  }

  # Returns $false when the person keeps the install, so the menu can carry on.
  function Remove-Gateway {
    $dir = $inst.Dir
    $volume = "$($inst.Project)_postgres_data"
    if (-not (Test-Path -LiteralPath 'docker-compose.quickstart.yml')) { Save-Compose }
    $images = Get-Images
    $link = ''
    if (-not $onWindows) {
      $bin = Join-Path $HOME '.local/bin'
      if ((Get-LinkTarget (Join-Path $bin 'litellm-gateway')) -eq (Join-Path $dir 'litellm-gateway')) { $link = Join-Path $bin 'litellm-gateway' }
    }
    if ($interactive) {
      Write-Host ''
      Say ('{b:This removes LiteLLM from ' + (Lit (Tildify $dir)) + ':}')
      Say "  - its containers ($engineName project $($inst.Project))"
      Say "  - its database, the $engineName volume $volume, with your models, keys, and spend logs"
      Say ('  - its files in ' + (Lit (Tildify $dir)) + ', including .env with the admin password')
      if ($link) { Say ('  - the litellm-gateway command in ' + (Lit (Tildify $bin))) }
      if ((Menu 'Remove LiteLLM and delete its database?' 2 @(@('Yes, remove it'), @('No, keep it'))) -ne 1) {
        Step 'Kept LiteLLM and its data'
        return $false
      }
    } elseif (-not $noQuestions) {
      Fail "Removing LiteLLM deletes its database ($engineName volume $volume)." 'To confirm, run it again with -Yes.'
      Stop-Quickstart
    }

    if (-not $style) { Say 'Removing the LiteLLM and Postgres containers and the database...' }
    if ((Invoke-WithSpinner 'Removing the containers and the database' '' $docker ($cf + @('down', '--volumes', '--remove-orphans'))) -ne 0) {
      Fail "$engineName could not remove them." 'Its output is above.'
      Stop-Quickstart
    }
    Step "Removed the containers and the database {d:($volume)}"

    $kept = New-Object System.Collections.ArrayList
    foreach ($image in $images) {
      if ((Invoke-Native $docker @('image', 'inspect', $image)).Code -eq 0) { $null = $kept.Add($image) }
    }
    if ($kept.Count -and $interactive) {
      # The size on disk, as the images list shows it (1.68GB, or 1.23 GB on Podman).
      $bytes = 0.0
      foreach ($image in $kept) {
        $size = "$((Invoke-Native $docker @('images', '--format', '{{.Size}}', $image)).Out | Select-Object -First 1)"
        if ($size -match '^([\d.]+)\s*([kKMGT]?)') {
          $unit = switch -CaseSensitive ($Matches[2]) { 'k' { 1e3 } 'K' { 1e3 } 'M' { 1e6 } 'G' { 1e9 } 'T' { 1e12 } default { 1 } }
          $bytes += [double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture) * $unit
        }
      }
      $total = if ($bytes -gt 0) { ' (' + ($bytes / 1e9).ToString('0.0', [Globalization.CultureInfo]::InvariantCulture) + ' GB)' } else { '' }
      $choice = Menu "Also delete the LiteLLM and Postgres images$($total)?" 2 @(@('Yes, delete them'), @('No, keep them', 'a reinstall starts faster'))
      if ($choice -eq 1) {
        foreach ($image in @($kept.ToArray())) {
          if ((Invoke-Native $docker @('image', 'rm', $image)).Code -eq 0) {
            Step "Deleted the image $image"
            $kept.Remove($image)
          } else {
            Warn "Kept the image $image, which other containers use"
          }
        }
      }
    }

    foreach ($name in '.env', 'docker-compose.quickstart.yml', 'quickstart.ps1', 'quickstart.sh', 'litellm-gateway', 'litellm-gateway.cmd') {
      Remove-Item -LiteralPath (Join-Path $dir $name) -Force -ErrorAction SilentlyContinue
    }
    $ignore = Join-Path $dir '.gitignore'
    if ((Test-Path -LiteralPath $ignore) -and
      -not ([IO.File]::ReadAllText($ignore) -split "`r?`n" | Where-Object { $_ -and $_ -ne '*' -and $_ -ne '.env' })) {
      Remove-Item -LiteralPath $ignore -Force
    }
    if ($link) { Remove-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue }
    # Windows will not delete the folder a program is working in, this one included.
    Set-Location -LiteralPath $HOME
    [Environment]::CurrentDirectory = $HOME
    try {
      [IO.Directory]::Delete($dir, $false)
      Step ('Deleted ' + (Lit (Tildify $dir)))
    } catch {
      $why = if (@(Get-ChildItem -LiteralPath $dir -Force -ErrorAction SilentlyContinue).Count) { 'it has other files' } else { 'a program is using it' }
      Step ('Deleted the LiteLLM files in ' + (Lit (Tildify $dir)) + " {d:(the folder stays: $why)}")
    }

    $left = New-Object System.Collections.ArrayList
    if (Test-HasContainers -All) { $null = $left.Add('containers') }
    if ((Invoke-Native $docker @('volume', 'inspect', $volume)).Code -eq 0) { $null = $left.Add("volume $volume") }
    if (Test-Path -LiteralPath (Join-Path $dir '.env')) { $null = $left.Add((Join-Path (Tildify $dir) '.env')) }
    if ($left.Count) {
      Warn ('Still there: ' + (Lit ($left.ToArray() -join ', ')))
      return $true
    }
    Step 'LiteLLM is removed. Nothing is left running, and its data is deleted'
    if ($kept.Count) {
      Say ("  {d:The images stay, so a reinstall starts faster. To delete them:} $engine image rm " + ($kept.ToArray() -join ' '))
    }
    return $true
  }

  # Over SSH the clipboard would be the server's, not yours. On Linux,
  # Set-Clipboard without xclip quietly keeps the text inside PowerShell, so
  # only Windows and macOS, which always have a clipboard, get the option.
  function Test-CanCopy {
    return $interactive -and (-not $env:SSH_CONNECTION) -and (-not $env:SSH_TTY) -and
      ($onWindows -or ((Test-Path variable:IsMacOS) -and $IsMacOS))
  }

  # The menu a rerun shows for an existing install. Its options follow whether
  # LiteLLM is running, and come back after each choice until Done.
  function Show-ManageMenu {
    Show-Status
    $canCopy = Test-CanCopy
    $pick = 1
    while ($true) {
      $options = New-Object System.Collections.ArrayList
      if (Test-Running) {
        $question = "LiteLLM is running at $($inst.Url). What next?"
        $null = $options.Add(@('Open the admin UI in your browser'))
        if ($canCopy) { $null = $options.Add(@('Copy the admin password', 'to the clipboard, ready to paste')) }
        $null = $options.Add(@('Stop it', 'keeps your models, keys, and logs'))
        $null = $options.Add(@('Restart it'))
        $null = $options.Add(@('Show the logs', 'the last 100 lines'))
        $null = $options.Add(@('Update it', 'download the latest image and restart'))
        $null = $options.Add(@('Remove it', 'delete its containers, database, and files'))
        $null = $options.Add(@('Done'))
      } else {
        $question = 'LiteLLM is stopped. What next?'
        $null = $options.Add(@('Start it', "at $($inst.Url)"))
        $null = $options.Add(@('Update it', 'download the latest image and start'))
        $null = $options.Add(@('Show the logs', 'the last 100 lines'))
        $null = $options.Add(@('Remove it', 'delete its containers, database, and files'))
        $null = $options.Add(@('Done'))
      }
      if ($pick -gt $options.Count) { $pick = 1 }
      $choice = Menu $question $pick $options.ToArray()
      $picked = $options[$choice - 1][0]
      $pick = 1
      switch -Wildcard ($picked) {
        'Open*' { Open-Url $inst.Url; $pick = $options.Count }
        'Copy*' { Copy-Password $inst.Dir }
        'Start*' { Start-Gateway }
        'Stop*' { Stop-Gateway }
        'Restart*' { Restart-Gateway }
        'Show*' { Show-Logs }
        'Update*' { Update-Gateway }
        'Remove*' { if (Remove-Gateway) { return } }
        default { return }
      }
    }
  }

  # ------------------------------------------------------------ main

  $savedEncoding = $null
  $savedLocation = (Get-Location).ProviderPath
  $savedPort = $env:LITELLM_PORT
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

    if ($action) {
      # An action prints only its own result.
    } elseif ($style) {
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

    if ($action) {
      $existing = Find-Install
      if (-not $existing) {
        $s = [IO.Path]::DirectorySeparatorChar
        $places = if ($env:LITELLM_DIR) { $env:LITELLM_DIR } else { "this folder, .${s}litellm-gateway, or ~${s}litellm-gateway" }
        Fail 'No LiteLLM install found' ('in ' + (Lit $places) + '.')
        $runner = if ($onWindows) { 'powershell' } else { 'pwsh' }
        Say "  Install it with:  $runner -c `"irm $scriptUrl | iex`"" -Err
        return 1
      }
      Use-Install $existing
      $inst.Command = Find-Command
      switch ($action) {
        'status' { Show-Status }
        'start' { if (Test-Running) { Step "LiteLLM is already running at {a:$($inst.Url)}" } else { Start-Gateway } }
        'stop' { Stop-Gateway }
        'restart' { Restart-Gateway }
        'logs' { if ($interactive) { Show-Logs -Follow } else { Show-Logs } }
        'open' { Open-UI }
        'update' { Update-Gateway }
        'uninstall' { $null = Remove-Gateway }
      }
      return 0
    }

    $server = Invoke-Native $docker @('version', '--format', '{{.Server.Version}}')
    $engineVersion = if ($server.Code -eq 0 -and $server.Out.Count) { "$($server.Out[0]) " } else { '' }
    $composeVersion = "$($compose.Out[0])".TrimStart('v')
    Step "$engineName ${engineVersion}with Compose $composeVersion is running"

    $homeDir = Join-Path $HOME 'litellm-gateway'
    $hereDir = Join-Path (Get-Location).ProviderPath 'litellm-gateway'
    $dir = Find-Install
    $found = [bool]$dir
    if ($found) {
      # installed before
    } elseif ($env:LITELLM_DIR) {
      $dir = $env:LITELLM_DIR
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
      if (($current -split "`r?`n") -notcontains '.env' -and ($current -split "`r?`n") -notcontains '*') {
        # Start on a new line if the file does not end with one.
        $lead = if ($current.Length -gt 0 -and -not $current.EndsWith("`n")) { "`n" } else { '' }
        Write-PlainText $ignore "$lead.env`n" -Append
      }
    }

    if ($found -and $interactive) {
      Use-Install $dir
      if (-not (Test-Path -LiteralPath 'quickstart.ps1') -or -not (Find-Command)) { Install-Command }
      $inst.Command = Find-Command
      Show-ManageMenu
      return 0
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

    Save-Compose
    Step 'Downloaded docker-compose.quickstart.yml'

    if (Test-Path -LiteralPath $envFile) {
      Step 'Reusing .env, so existing keys and data keep working'
    } else {
      $port = Select-Port
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

    Use-Install $dir
    Install-Command
    $inst.Command = Find-Command
    Start-Gateway

    $url = $inst.Url
    Write-Host ''
    $canCopy = Test-CanCopy
    $password = "the LITELLM_MASTER_KEY value in $(Join-Path (Tildify $dir) '.env')"
    if ($canCopy) { $password = "copy it below, or LITELLM_MASTER_KEY in $(Join-Path (Tildify $dir) '.env')" }
    # A rerun finds ~\litellm-gateway on its own; any other folder needs naming.
    $rerun = 'rerun the install command'
    if ($dir -ne $homeDir) { $rerun += " with LITELLM_DIR=$(Tildify $dir)" }
    $manage = if ($inst.Command) { "run $($inst.Command), or $rerun" } else { $rerun }
    $rows = New-Object System.Collections.ArrayList
    $null = $rows.Add(@('Admin UI', $url))
    $null = $rows.Add(@('Username', 'admin'))
    $null = $rows.Add(@('Password', $password))
    $null = $rows.Add(@('Next', 'in the UI, open Models + Endpoints > Add Model and paste a provider API key'))
    $null = $rows.Add(@('Stop it', (Hint 'down')))
    $null = $rows.Add(@('Start it', (Hint 'up')))
    $null = $rows.Add(@('Remove it', (Hint 'uninstall')))
    $desktop = Get-DesktopNote
    if ($desktop) { $null = $rows.Add(@('Desktop', $desktop)) }
    $null = $rows.Add(@('Manage', $manage))
    Show-Box 'LiteLLM is running' $rows.ToArray()

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
      if ((Menu 'Open the admin UI in your browser?' 1 @(@('Yes'), @('No'))) -eq 1) { Open-Url $url }
    }
    return 0
  } catch {
    if ($_.Exception.Message -eq $exitSignal) { return 1 }
    throw
  } finally {
    if ($savedEncoding) { [Console]::OutputEncoding = $savedEncoding }
    if ($savedPort) { $env:LITELLM_PORT = $savedPort } else { Remove-Item Env:LITELLM_PORT -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $savedLocation) { Set-Location -LiteralPath $savedLocation }
  }
}

$code = Invoke-LiteLLMQuickstart -Yes:$Yes -Rest $args | Select-Object -Last 1
$global:LASTEXITCODE = $code
# Run from a file, or as `powershell -c "irm ... | iex"`, the exit code should
# reach the caller. Pasted into an open window, `exit` would close that window,
# so there it only sets $LASTEXITCODE.
$hostArgs = [Environment]::GetCommandLineArgs()
$oneShot = ($hostArgs -match '^-(c|command)$') -and -not ($hostArgs -match '^-noexit$')
if ($code -ne 0 -and ($PSCommandPath -or $oneShot)) { exit $code }
