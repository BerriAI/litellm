#!/bin/sh
# LiteLLM Gateway quickstart: the gateway, Postgres, and the admin UI in one command.
#   curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.sh | sh
# On Windows, scripts/quickstart.ps1 does the same in PowerShell.
#
# To read it before running it:
#   curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.sh -o quickstart.sh
#   less quickstart.sh
#   sh quickstart.sh
#
# Asks where to keep the files, then offers to copy the admin password and
# open the admin UI; every question has a default you accept by pressing Enter.
# It asks nothing when there is no terminal, under CI or Claude Code, or when
# run with --yes.
#
#   --yes, -y        no questions: install to ~/litellm-gateway, don't open a browser
#   LITELLM_DIR      folder to install into (skips the folder question)
#   LITELLM_PORT     port for the gateway (default 4000, or the next free one)
#   NO_COLOR         plain output, which agents, CI, and log files always get
#
# New installs listen on this machine only (127.0.0.1). To reach the gateway
# from other machines, remove LITELLM_BIND from .env and put it behind TLS.
#
# Keys and the database password are random (openssl rand), written only to
# .env with permissions 600, and never printed. When you ask, the admin
# password goes straight to your clipboard. Needs Docker, Podman, or Rancher
# Desktop, each with Compose v2.
# Everything runs inside main(), so a partial download runs nothing.
set -eu

COMPOSE_URL="${LITELLM_COMPOSE_URL:-https://raw.githubusercontent.com/BerriAI/litellm/main/docker/docker-compose.quickstart.yml}"

# ---------------------------------------------------------------- output

# A person watching a terminal gets colors, step marks, and a spinner. Agents,
# CI, log files, and NO_COLOR get the same lines as plain text.
STYLE=0
ERR_STYLE=0 # stderr is styled only when it is a terminal too
C_ACC='' C_OK='' C_WARN='' C_ERR='' C_DIM='' C_BOLD='' C_OFF=''
POINTER='>' S_OK='+' S_WARN='!' S_ERR='x' S_HEAD='*' S_ASK='?'
# shellcheck disable=SC1003 # the last frame is a backslash
SPIN_FRAMES='| / - \'
HINT='Up/Down to move, Enter to choose'
BOX_TL='+' BOX_TR='+' BOX_BL='+' BOX_BR='+' BOX_H='-' BOX_V='|'
SPIN_PID='' SPIN_LOG='' ENV_TMP=''

setup_output() {
  case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
    *UTF-8* | *utf-8* | *UTF8* | *utf8*)
      POINTER='❯' S_OK='✓' S_WARN='!' S_ERR='✗' S_HEAD='◆' S_ASK='?'
      SPIN_FRAMES='⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏'
      HINT='↑/↓ to move, Enter to choose'
      BOX_TL='╭' BOX_TR='╮' BOX_BL='╰' BOX_BR='╯' BOX_H='─' BOX_V='│'
      ;;
  esac
  if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ] &&
    [ -z "${CI:-}" ] && [ -z "${CLAUDECODE:-}" ]; then
    STYLE=1
    e="$(printf '\033')"
    # The accent is the LiteLLM blue, lightened so it reads on dark and light themes.
    case "${COLORTERM:-}" in
      truecolor | 24bit) C_ACC="${e}[38;2;91;108;255m" ;;
      *) C_ACC="${e}[94m" ;;
    esac
    C_OK="${e}[32m" C_WARN="${e}[33m" C_ERR="${e}[31m" C_DIM="${e}[90m" C_BOLD="${e}[1m" C_OFF="${e}[0m"
    if [ -t 2 ]; then ERR_STYLE=1; fi
  fi
}

step() {
  if [ "$STYLE" = 1 ]; then
    printf '%s%s%s %s%s\n' "${C_OK}" "$S_OK" "${C_OFF}" "$1" "${C_OFF}"
  else
    printf '%s\n' "$1"
  fi
}

warn() {
  if [ "$STYLE" = 1 ]; then
    printf '%s%s%s %s%s\n' "${C_WARN}" "$S_WARN" "${C_OFF}" "$1" "${C_OFF}"
  else
    printf '%s\n' "$1"
  fi
}

fail() {
  if [ "$ERR_STYLE" = 1 ]; then
    printf '%s%s %s%s%s\n' "${C_ERR}${C_BOLD}" "$S_ERR" "$1" "${C_OFF}" "${2:+ $2}" >&2
  else
    printf '%s%s\n' "$1" "${2:+ $2}" >&2
  fi
}

tildify() {
  case "$1" in
    "$HOME") printf '~' ;;
    "$HOME"/*) printf '~%s' "${1#"$HOME"}" ;;
    *) printf '%s' "$1" ;;
  esac
}

elapsed_since() {
  s=$(($(date +%s) - $1))
  if [ "$s" -lt 60 ]; then printf '%ss' "$s"; else printf '%sm %ss' "$((s / 60))" "$((s % 60))"; fi
}

# spin "label" "detail" command...: run the command behind a spinner with the
# elapsed time, keeping its output to show if it fails. Without styling the
# command runs in the open, as before.
spin() {
  label="$1" detail="$2"
  shift 2
  if [ "$STYLE" != 1 ]; then
    "$@"
    return
  fi
  SPIN_LOG="$(mktemp)"
  "$@" >"$SPIN_LOG" 2>&1 &
  SPIN_PID=$!
  start="$(date +%s)" s=0 n=0
  frames="$SPIN_FRAMES "
  printf '\033[?25l'
  while kill -0 "$SPIN_PID" 2>/dev/null; do
    # Rotate the frames in the shell and read the clock once a second, so a
    # redraw starts no process besides sleep.
    frame="${frames%% *}"
    frames="${frames#* }$frame "
    if [ $((n % 10)) = 0 ]; then s=$(($(date +%s) - start)); fi
    printf '\r\033[2K%s%s%s %s %s· %s%d:%02d%s' "${C_ACC}" "$frame" "${C_OFF}" "$label" "${C_DIM}" \
      "${detail:+$detail · }" "$((s / 60))" "$((s % 60))" "${C_OFF}"
    n=$((n + 1))
    sleep 0.1
  done
  rc=0
  wait "$SPIN_PID" || rc=$?
  SPIN_PID=''
  printf '\r\033[2K\033[?25h'
  if [ "$rc" != 0 ]; then cat "$SPIN_LOG" >&2; fi
  rm -f "$SPIN_LOG"
  SPIN_LOG=''
  return "$rc"
}

# box "Title" "Label|value"...: the closing summary. Plain output, and a
# terminal too narrow for the frame, get the same lines without it.
box() {
  title="$1"
  shift
  cols=''
  if [ "$STYLE" = 1 ]; then cols="$( (stty size </dev/tty) 2>/dev/null | cut -d ' ' -f 2)" || cols=''; fi
  width=${#title}
  for row in "$@"; do
    value="${row#*|}"
    if [ $((11 + ${#value})) -gt "$width" ]; then width=$((11 + ${#value})); fi
  done
  if [ "$STYLE" != 1 ] || [ -z "$cols" ] || [ $((width + 4)) -gt "$cols" ]; then
    if [ "$STYLE" = 1 ]; then printf '%s%s%s\n' "${C_ACC}${C_BOLD}" "$title" "${C_OFF}"; else printf '%s.\n' "$title"; fi
    for row in "$@"; do
      printf '  %-12s%s\n' "${row%%|*}:" "${row#*|}"
    done
    return 0
  fi
  rule="$(printf "%$((width + 2))s" '' | sed "s/ /$BOX_H/g")"
  printf '%s%s%s%s%s\n' "${C_ACC}" "$BOX_TL" "$rule" "$BOX_TR" "${C_OFF}"
  printf '%s%s%s %s%-*s%s %s%s%s\n' "${C_ACC}" "$BOX_V" "${C_OFF}" "${C_ACC}${C_BOLD}" "$width" "$title" "${C_OFF}" \
    "${C_ACC}" "$BOX_V" "${C_OFF}"
  for row in "$@"; do
    label="${row%%|*}" value="${row#*|}" shade=''
    if [ "$label" = "Admin UI" ]; then shade="${C_ACC}${C_BOLD}"; fi
    printf '%s%s%s %s%-10s%s %s%-*s%s %s%s%s\n' "${C_ACC}" "$BOX_V" "${C_OFF}" "${C_DIM}" "$label" "${C_OFF}" \
      "$shade" "$((width - 11))" "$value" "${C_OFF}" "${C_ACC}" "$BOX_V" "${C_OFF}"
  done
  printf '%s%s%s%s%s\n' "${C_ACC}" "$BOX_BL" "$rule" "$BOX_BR" "${C_OFF}"
}

# ---------------------------------------------------------------- terminal

INTERACTIVE=0   # a person is at a terminal we can ask
ARROWS=0        # that terminal supports the arrow-key menu
STTY_SAVED=""

detect_terminal() {
  # Piped from curl, stdin is the script itself, so questions go to /dev/tty.
  if (exec </dev/tty) 2>/dev/null && [ "${TERM:-dumb}" != "dumb" ]; then
    INTERACTIVE=1
    if STTY_SAVED="$(stty -g </dev/tty 2>/dev/null)" && [ -n "$STTY_SAVED" ]; then
      ARROWS=1
    fi
  fi
}

restore_terminal() {
  if [ -n "$STTY_SAVED" ]; then
    stty "$STTY_SAVED" </dev/tty 2>/dev/null || true
  fi
  if [ "$STYLE" = 1 ] || [ -n "$STTY_SAVED" ]; then
    printf '\033[?25h' >/dev/tty 2>/dev/null || true
  fi
}

on_interrupt() {
  if [ -n "$SPIN_PID" ]; then kill "$SPIN_PID" 2>/dev/null || true; fi
  if [ -n "$SPIN_LOG" ]; then rm -f "$SPIN_LOG"; fi
  if [ -n "$ENV_TMP" ]; then rm -f "$ENV_TMP"; fi
  if [ "$STYLE" = 1 ]; then printf '\r\033[2K'; fi
  restore_terminal
  printf '\nCancelled.\n' >&2
  exit 130
}

read_key() {
  # One keypress in raw mode. Enter comes back empty (command substitution
  # drops the newline); arrows come back as "up" or "down".
  k="$(dd bs=1 count=1 2>/dev/null </dev/tty)"
  if [ "$k" = "$(printf '\033')" ]; then
    rest="$(dd bs=1 count=2 2>/dev/null </dev/tty)"
    case "$rest" in
      '[A' | 'OA') k=up ;;
      '[B' | 'OB') k=down ;;
      *) k=other ;;
    esac
  fi
  printf '%s' "$k"
}

# menu "Question" DEFAULT OPTION... -> sets CHOICE to the 1-based pick.
# An option may carry a note after a tab, shown dimmed and lined up.
menu() {
  question="$1"
  CHOICE="$2"
  shift 2
  count=$#
  if [ "$INTERACTIVE" != 1 ]; then return 0; fi

  tab="$(printf '\t')"
  pad=0
  for opt in "$@"; do
    label="${opt%%"$tab"*}"
    if [ ${#label} -gt "$pad" ]; then pad=${#label}; fi
  done

  if [ "$STYLE" = 1 ]; then
    printf '\n%s%s%s %s%s%s\n' "${C_ACC}" "$S_ASK" "${C_OFF}" "${C_BOLD}" "$question" "${C_OFF}" >/dev/tty
  else
    printf '\n%s\n' "$question" >/dev/tty
  fi
  if [ "$ARROWS" = 1 ]; then
    stty -icanon -echo min 1 time 0 </dev/tty
    printf '\033[?25l' >/dev/tty
    first=1
    while :; do
      [ "$first" = 1 ] || printf '\033[%sA' "$((count + 1))" >/dev/tty
      first=0
      i=1
      for opt in "$@"; do
        label="${opt%%"$tab"*}" note=''
        [ "$label" = "$opt" ] || note="${opt#*"$tab"}"
        if [ "$i" = "$CHOICE" ]; then
          printf '\033[2K  %s%s %-*s%s  %s%s%s\n' "${C_ACC}${C_BOLD}" "$POINTER" "$pad" "$label" "${C_OFF}" \
            "${C_DIM}" "$note" "${C_OFF}" >/dev/tty
        else
          printf '\033[2K    %-*s  %s%s%s\n' "$pad" "$label" "${C_DIM}" "$note" "${C_OFF}" >/dev/tty
        fi
        i=$((i + 1))
      done
      printf '\033[2K  %s%s%s\n' "${C_DIM}" "$HINT" "${C_OFF}" >/dev/tty
      key="$(read_key)"
      case "$key" in
        up | k) [ "$CHOICE" -gt 1 ] && CHOICE=$((CHOICE - 1)) ;;
        down | j) [ "$CHOICE" -lt "$count" ] && CHOICE=$((CHOICE + 1)) ;;
        [1-9]) [ "$key" -le "$count" ] && CHOICE="$key" ;;
        '' | "$(printf '\r')") break ;;
      esac
    done
    printf '\033[1A\033[2K' >/dev/tty
    stty "$STTY_SAVED" </dev/tty 2>/dev/null || true
    printf '\033[?25h' >/dev/tty
  else
    i=1
    for opt in "$@"; do
      label="${opt%%"$tab"*}" note=''
      [ "$label" = "$opt" ] || note="   ${opt#*"$tab"}"
      printf '  %s) %s%s\n' "$i" "$label" "$note" >/dev/tty
      i=$((i + 1))
    done
    printf 'Choose [%s]: ' "$CHOICE" >/dev/tty
    answer=""
    read -r answer </dev/tty || answer=""
    case "$answer" in
      '' | *[!0-9]*) ;;
      *) [ "$answer" -ge 1 ] && [ "$answer" -le "$count" ] && CHOICE="$answer" ;;
    esac
  fi
  return 0
}

# ---------------------------------------------------------------- steps

port_free() {
  # curl exits 7 when nothing accepts the connection.
  rc=0
  curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$1/" || rc=$?
  [ "$rc" = 7 ]
}

pick_folder() {
  home_dir="$HOME/litellm-gateway"
  here_dir="$(pwd)/litellm-gateway"
  found=0
  if [ -n "${LITELLM_DIR:-}" ]; then
    DIR="$LITELLM_DIR"
  elif [ -f "$here_dir/.env" ]; then
    DIR="$here_dir" found=1   # installed in this folder before
  elif [ -f "$home_dir/.env" ]; then
    DIR="$home_dir" found=1   # installed in the home folder before
  elif [ "$here_dir" = "$home_dir" ]; then
    DIR="$home_dir"
  else
    tab="$(printf '\t')"
    menu "Where should LiteLLM keep its files (.env with your keys, and the compose file)?" 1 \
      "$(tildify "$home_dir")${tab}recommended, reruns always find it" \
      "$(tildify "$here_dir")${tab}this folder"
    if [ "$CHOICE" = 2 ]; then DIR="$here_dir"; else DIR="$home_dir"; fi
  fi
  created=0
  [ -d "$DIR" ] || created=1
  mkdir -p "$DIR"
  cd "$DIR"
  DIR="$(pwd)"
  if [ "$found" = 1 ]; then
    step "Found your install in ${C_BOLD}$(tildify "$DIR")${C_OFF}"
  else
    step "Files go in ${C_BOLD}$(tildify "$DIR")${C_OFF}"
  fi
  if [ ! -f .env ] && command -v git >/dev/null 2>&1 &&
    git ls-files --error-unmatch .env >/dev/null 2>&1; then
    # An ignore rule does not cover a tracked file, so new keys written here
    # would show up as a change to commit.
    fail "Git tracks a .env file in this folder," "so your keys could be committed."
    cat >&2 <<'EOF'

  Install into another folder (set LITELLM_DIR), or stop tracking the file
  first with:  git rm --cached .env
EOF
    exit 1
  fi
  if [ "$created" = 1 ]; then
    # A folder this script made holds only its own files, so keep all of it out of git.
    printf '*\n' >.gitignore
  elif command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1 &&
    ! git check-ignore -q .env 2>/dev/null; then
    # In a folder that already existed, such as a repository root, leave the
    # tracked .gitignore alone and add only .env to this clone's local exclude
    # list, so the generated keys cannot be committed.
    exclude="$(git rev-parse --git-path info/exclude)"
    mkdir -p "$(dirname "$exclude")"
    exclude="$(cd "$(dirname "$exclude")" && pwd)/exclude"
    printf '/%s.env\n' "$(git rev-parse --show-prefix)" >>"$exclude"
    step "Kept .env out of git ${C_DIM}(added it to this repository's local exclude list, $(tildify "$exclude"))"
  elif ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    # An existing folder outside git: ignore only .env, so it stays out of
    # commits if the folder becomes a repository later.
    if ! grep -qxF '.env' .gitignore 2>/dev/null; then
      # Start on a new line if the file does not end with one.
      if [ -s .gitignore ] && [ -n "$(tail -c 1 .gitignore)" ]; then printf '\n' >>.gitignore; fi
      printf '.env\n' >>.gitignore
    fi
  fi
}

pick_port() {
  saved=""
  [ -f .env ] && saved="$(sed -n 's/^LITELLM_PORT=//p' .env | tail -n 1)"
  if [ -n "${LITELLM_PORT:-}" ]; then
    PORT="$LITELLM_PORT"
  elif [ -n "$saved" ]; then
    PORT="$saved"
  elif [ -f .env ]; then
    # An existing install without a saved port runs on the compose default.
    PORT=4000
  else
    PORT=4000
    while ! port_free "$PORT"; do
      PORT=$((PORT + 1))
      if [ "$PORT" -gt 4099 ]; then
        fail "Ports 4000 to 4099 are all in use." "Set LITELLM_PORT to a free port and run this again."
        exit 1
      fi
    done
    if [ "$PORT" = 4000 ]; then
      step "Port ${C_BOLD}4000${C_OFF} is free"
    else
      warn "Port 4000 is in use, so LiteLLM will use ${C_BOLD}$PORT${C_OFF}"
    fi
  fi
  export LITELLM_PORT="$PORT"
}

# Docker names containers and the database volume after the project, so an
# install outside the home folder gets its own name and never shares a
# database with another litellm-gateway folder.
check_new_install() {
  project=litellm-gateway
  [ "$DIR" = "$HOME/litellm-gateway" ] || project="litellm-gateway-$(printf '%s' "$DIR" | cksum | cut -d ' ' -f 1)"
  # Postgres keeps the password it was created with, so a new password over an
  # old database volume would lock the gateway out. Stop and explain instead.
  if "$ENGINE" volume inspect "${project}_postgres_data" >/dev/null 2>&1; then
    fail "Found a database from an earlier install" "($ENGINE_NAME volume ${project}_postgres_data)"
    cat >&2 <<EOF
but no $(tildify "$DIR")/.env with its password.

  Restore that .env file and run this again to keep your models and keys, or
  delete the old database and start fresh (this removes its models and keys):
    $ENGINE volume rm ${project}_postgres_data
EOF
    exit 1
  fi
}

start_stack() {
  "$ENGINE" compose -f docker-compose.quickstart.yml up -d
}

wait_ready() {
  i=0
  until curl -fsS "http://127.0.0.1:$PORT/health/readiness" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 90 ]; then return 1; fi
    sleep 2
  done
}

open_url() {
  for opener in open xdg-open; do
    if command -v "$opener" >/dev/null 2>&1 && "$opener" "$1" >/dev/null 2>&1; then
      step "Opened ${C_ACC}$1${C_OFF} in your browser"
      return 0
    fi
  done
  printf '  %sOpen%s %s%s%s %swhen you are ready.%s\n' "${C_DIM}" "${C_OFF}" "${C_ACC}" "$1" "${C_OFF}" "${C_DIM}" "${C_OFF}"
}

# A clipboard to copy the admin password to, or nothing. Over SSH the
# clipboard would be the server's, not yours.
find_clipboard() {
  [ -z "${SSH_CONNECTION:-}${SSH_TTY:-}" ] || return 0
  if command -v pbcopy >/dev/null 2>&1; then CLIP=pbcopy
  elif [ -n "${WAYLAND_DISPLAY:-}" ] && command -v wl-copy >/dev/null 2>&1; then CLIP=wl-copy
  elif [ -n "${DISPLAY:-}" ] && command -v xclip >/dev/null 2>&1; then CLIP="xclip -selection clipboard"
  elif [ -n "${DISPLAY:-}" ] && command -v xsel >/dev/null 2>&1; then CLIP="xsel --clipboard --input"
  elif command -v clip.exe >/dev/null 2>&1; then CLIP=clip.exe
  fi
}

# The admin password is the master key. It goes through a pipe, so it is
# never on screen or in the process list.
copy_password() {
  pw="$(sed -n 's/^LITELLM_MASTER_KEY=//p' .env | head -n 1)"
  if [ -z "$pw" ]; then
    warn "No LITELLM_MASTER_KEY in $(tildify "$DIR")/.env."
    return 0
  fi
  # shellcheck disable=SC2086 # CLIP is a command and its flags
  if printf '%s' "$pw" | $CLIP >/dev/null 2>&1; then
    step "Copied the admin password. Paste it into the sign-in page."
  else
    warn "Could not copy it. The password is LITELLM_MASTER_KEY in $(tildify "$DIR")/.env."
  fi
  pw=''
}

# After the summary: copy the password, open the admin UI, or finish. The
# menu comes back after each choice with the next step as its default.
next_steps() {
  url="$1"
  [ "$INTERACTIVE" = 1 ] || return 0
  if [ -z "$CLIP" ]; then
    menu "Open the admin UI in your browser?" 1 "Yes" "No"
    if [ "$CHOICE" = 1 ]; then open_url "$url"; fi
    return 0
  fi
  tab="$(printf '\t')"
  pick=1
  while :; do
    menu "What next?" "$pick" \
      "Copy the admin password${tab}to the clipboard, ready to paste" \
      "Open the admin UI in your browser" \
      "Done"
    case "$CHOICE" in
      1) copy_password; pick=2 ;;
      2) open_url "$url"; pick=3 ;;
      *) return 0 ;;
    esac
  done
}

main() {
  NO_QUESTIONS=0
  for arg in "$@"; do
    case "$arg" in
      -y | --yes) NO_QUESTIONS=1 ;;
      *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
  done

  setup_output
  detect_terminal
  # Agents and CI get the defaults even inside a terminal, so nothing waits on a keypress.
  if [ "$NO_QUESTIONS" = 1 ] || [ -n "${CI:-}" ] || [ -n "${CLAUDECODE:-}" ]; then INTERACTIVE=0; fi
  trap restore_terminal EXIT
  trap on_interrupt INT TERM

  if [ "$STYLE" = 1 ]; then
    printf '\n%s%s LiteLLM quickstart%s\n' "${C_ACC}${C_BOLD}" "$S_HEAD" "${C_OFF}"
    printf '%s  The gateway, Postgres, and the admin UI in one command%s\n\n' "${C_DIM}" "${C_OFF}"
  else
    echo "LiteLLM quickstart"
  fi

  # Podman and Rancher Desktop (nerdctl in containerd mode; its dockerd mode
  # already provides a docker CLI) work the same way through Compose v2.
  ENGINE=''
  for cand in docker podman nerdctl; do
    if command -v "$cand" >/dev/null 2>&1; then
      ENGINE="$cand"
      break
    fi
  done
  if [ -z "$ENGINE" ]; then
    fail "No container engine found (docker, podman, or nerdctl)." "The LiteLLM Gateway runs in a container alongside a Postgres database."
    cat >&2 <<'EOF'

  Install Docker (https://docs.docker.com/get-docker/), Podman (https://podman.io),
  or Rancher Desktop (https://rancherdesktop.io), then run this again.
  Or deploy in one click (Railway or Render):  https://docs.litellm.ai/docs/proxy/docker_quick_start
  Only need to call models from Python?  pip install litellm
EOF
    exit 1
  fi
  case "$ENGINE" in
    podman) ENGINE_NAME=Podman ;;
    nerdctl) ENGINE_NAME=nerdctl ;;
    *) ENGINE_NAME=Docker ;;
  esac
  compose_version="$("$ENGINE" compose version --short 2>/dev/null)" ||
    { fail "Compose v2 ('$ENGINE compose') is required."; exit 1; }
  if ! "$ENGINE" info >/dev/null 2>&1; then
    if [ "$ENGINE" = podman ]; then
      fail "Podman is installed but not running." "Start it (podman machine start) and run this again."
    else
      fail "$ENGINE_NAME is installed but not running." "Start it and run this again."
    fi
    exit 1
  fi
  command -v openssl >/dev/null 2>&1 || { fail "openssl is required to generate keys."; exit 1; }
  engine_version="$("$ENGINE" version --format '{{.Server.Version}}' 2>/dev/null)" || engine_version=''
  step "$ENGINE_NAME ${engine_version:+$engine_version }with Compose ${compose_version#v} is running"

  pick_folder
  [ -f .env ] || check_new_install
  curl -fsSL -o docker-compose.quickstart.yml "$COMPOSE_URL"
  step "Downloaded docker-compose.quickstart.yml"
  pick_port

  if [ -f .env ]; then
    step "Reusing .env, so existing keys and data keep working"
  else
    master="$(openssl rand -hex 32)"
    salt="$(openssl rand -hex 32)"
    db_password="$(openssl rand -hex 24)"
    # Write a fresh mktemp file (mode 600, never a reused one) and rename it, so a
    # failed write never leaves a partial .env that a rerun would mistake for a
    # finished install.
    if ! ENV_TMP="$(mktemp .env.XXXXXX)" ||
      ! printf 'LITELLM_MASTER_KEY=sk-%s\nLITELLM_SALT_KEY=sk-%s\nPOSTGRES_PASSWORD=%s\nLITELLM_PORT=%s\nLITELLM_BIND=127.0.0.1:\nCOMPOSE_PROJECT_NAME=%s\n' \
        "$master" "$salt" "$db_password" "$PORT" "$project" >"$ENV_TMP" || ! mv -f "$ENV_TMP" .env; then
      if [ -n "$ENV_TMP" ]; then rm -f "$ENV_TMP"; fi
      fail "Could not write $(tildify "$DIR")/.env."
      exit 1
    fi
    ENV_TMP=''
    unset master salt db_password
    step "Generated .env with your master key, salt key, and database password ${C_DIM}(keep this file; only you can read it)"
  fi

  # Compose prefers values already set in the shell over .env, so drop any
  # inherited ones: .env stays the only source for keys and the project name.
  unset LITELLM_MASTER_KEY LITELLM_SALT_KEY POSTGRES_PASSWORD COMPOSE_PROJECT_NAME
  # The bind address follows .env when .env sets it (every install this script
  # creates does). For an older .env without it, a value exported in the shell
  # is kept, so an intentional LITELLM_BIND=127.0.0.1: is not dropped.
  if grep -q '^LITELLM_BIND=' .env; then unset LITELLM_BIND; fi

  started="$(date +%s)"
  detail=''
  for image in $("$ENGINE" compose -f docker-compose.quickstart.yml config --images 2>/dev/null); do
    "$ENGINE" image inspect "$image" >/dev/null 2>&1 || detail="downloading images, first run only"
  done
  [ "$STYLE" = 1 ] || echo "Starting LiteLLM and Postgres (the first run downloads the images)..."
  if ! spin "Starting LiteLLM and Postgres" "$detail" start_stack; then
    fail "$ENGINE_NAME could not start LiteLLM and Postgres." "Its output is above."
    exit 1
  fi
  if ! spin "Waiting for the gateway to be ready" "" wait_ready; then
    fail "The gateway did not become ready in 3 minutes." "See what it logged:"
    echo "    cd $(tildify "$DIR") && $ENGINE compose -f docker-compose.quickstart.yml logs litellm" >&2
    exit 1
  fi
  step "LiteLLM and Postgres are up ${C_DIM}· $(elapsed_since "$started")"

  url="http://localhost:$PORT/ui"
  password="the LITELLM_MASTER_KEY value in $(tildify "$DIR")/.env"
  CLIP=''
  if [ "$INTERACTIVE" = 1 ]; then find_clipboard; fi
  if [ -n "$CLIP" ]; then
    password="copy it below, or LITELLM_MASTER_KEY in $(tildify "$DIR")/.env"
  fi
  echo
  box "LiteLLM is running" \
    "Admin UI|$url" \
    "Username|admin" \
    "Password|$password" \
    "Next|in the UI, open Models + Endpoints > Add Model and paste a provider API key" \
    "Stop it|cd $(tildify "$DIR") && $ENGINE compose -f docker-compose.quickstart.yml down"

  next_steps "$url"
}

main "$@"
