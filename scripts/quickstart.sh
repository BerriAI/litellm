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
# Run it again for a menu to open, stop, start, update, or remove the install.
# It asks nothing when there is no terminal, under CI or Claude Code, or when
# run with --yes.
#
# It also takes an action, which the litellm-gateway command it installs takes too:
#   curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.sh | sh -s -- stop
#   status, start (up), stop (down), restart, logs, open, update, uninstall
#
#   --yes, -y        no questions: install to ~/litellm-gateway, don't open a browser,
#                    and let uninstall delete the database without asking
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
SCRIPT_URL="${LITELLM_SCRIPT_URL:-https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.sh}"
# A port set by the caller is never moved; later steps export LITELLM_PORT too.
PORT_PINNED="${LITELLM_PORT:+1}"

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

usage() {
  cat <<'USAGE'
Usage: quickstart.sh [--yes] [action]

With no action it installs the LiteLLM Gateway, or shows a menu to manage the
install it finds. Actions:
  status      whether it is running, and where
  start       start it (also: up)
  stop        stop it; models, keys, and logs are kept (also: down)
  restart     stop it and start it again
  logs        show the gateway's logs
  open        open the admin UI in your browser
  update      download the latest image and restart on it
  uninstall   remove its containers, its database, and its files
  help        show this help (also: -h, --help)

  --yes, -y   ask nothing; without a terminal, uninstall needs it

From the install command:
  curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.sh | sh -s -- status
USAGE
}

compose() {
  "$ENGINE" compose -f docker-compose.quickstart.yml "$@"
}

port_free() {
  # curl exits 7 when nothing accepts the connection.
  rc=0
  curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$1/" || rc=$?
  [ "$rc" = 7 ]
}

# What holds a port: sets OWNER to a description, FREE_IT to the command that
# frees it, and OWNER_PROJECT when it is another quickstart install. All stay
# empty when the holder cannot be named, such as another user's process.
port_owner() {
  OWNER='' FREE_IT='' OWNER_PROJECT=''
  name="$("$ENGINE" ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | grep -E "[:]$1->" | head -n 1 | cut -d ' ' -f 1)"
  if [ -n "$name" ]; then
    OWNER_PROJECT="$("$ENGINE" inspect --format '{{index .Config.Labels "com.docker.compose.project"}}' "$name" 2>/dev/null)" ||
      OWNER_PROJECT=''
    # Only another quickstart install keeps its project, since that is the one
    # this script offers to stop. Whoever started the container set the label,
    # so it must also be a name Compose would accept.
    case "$OWNER_PROJECT" in
      *[!a-z0-9_-]*) OWNER_PROJECT='' ;;
    esac
    case "$OWNER_PROJECT" in
      litellm-gateway*)
        OWNER="another LiteLLM quickstart install ($ENGINE_NAME project $OWNER_PROJECT)"
        FREE_IT="$ENGINE compose -p $OWNER_PROJECT stop"
        ;;
      *) OWNER_PROJECT='' OWNER="the $ENGINE_NAME container $name" FREE_IT="$ENGINE stop $name" ;;
    esac
    return 0
  fi
  command -v lsof >/dev/null 2>&1 || return 0
  pid="$(lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -n 1)"
  [ -n "$pid" ] || return 0
  FREE_IT="kill $pid"
  case "$(ps -o args= -p "$pid" 2>/dev/null)" in
    *litellm*) OWNER="a litellm server (pid $pid), maybe one started after pip install litellm" ;;
    *) OWNER="$(basename "$(ps -o comm= -p "$pid" 2>/dev/null || echo a program)") (pid $pid)" ;;
  esac
}

# Sets DIR to an existing install: LITELLM_DIR, this folder, ./litellm-gateway,
# or ~/litellm-gateway, whichever has a .env first.
locate_install() {
  DIR=''
  if [ -n "${LITELLM_DIR:-}" ]; then
    if [ -f "$LITELLM_DIR/.env" ]; then DIR="$LITELLM_DIR"; fi
  elif [ -f .env ] && [ -f docker-compose.quickstart.yml ]; then
    DIR="$(pwd)"
  elif [ -f litellm-gateway/.env ]; then
    DIR="$(pwd)/litellm-gateway"
  elif [ -f "$HOME/litellm-gateway/.env" ]; then
    DIR="$HOME/litellm-gateway"
  fi
  [ -n "$DIR" ]
}

pick_folder() {
  home_dir="$HOME/litellm-gateway"
  here_dir="$(pwd)/litellm-gateway"
  found=0
  if locate_install; then
    found=1
  elif [ -n "${LITELLM_DIR:-}" ]; then
    DIR="$LITELLM_DIR"
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
    cat >&2 <<'MSG'

  Install into another folder (set LITELLM_DIR), or stop tracking the file
  first with:  git rm --cached .env
MSG
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
    if ! grep -qxE '\*|\.env' .gitignore 2>/dev/null; then
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
      port_owner 4000
      warn "Port 4000 is in use${OWNER:+ by $OWNER}, so LiteLLM will use ${C_BOLD}$PORT${C_OFF}"
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
    cat >&2 <<MSG
but no $(tildify "$DIR")/.env with its password.

  Restore that .env file and run this again to keep your models and keys, or
  delete the old database and start fresh (this removes its models and keys):
    $ENGINE volume rm ${project}_postgres_data
MSG
    exit 1
  fi
}

# Compose prefers values already set in the shell over .env, so drop any
# inherited ones: .env stays the only source for keys and the project name.
clean_env() {
  unset LITELLM_MASTER_KEY LITELLM_SALT_KEY POSTGRES_PASSWORD COMPOSE_PROJECT_NAME
  # The bind address follows .env when .env sets it (every install this script
  # creates does). For an older .env without it, a value exported in the shell
  # is kept, so an intentional LITELLM_BIND=127.0.0.1: is not dropped.
  if grep -q '^LITELLM_BIND=' .env; then unset LITELLM_BIND; fi
}

use_install() {
  cd "$DIR"
  DIR="$(pwd)"
  project="$(sed -n 's/^COMPOSE_PROJECT_NAME=//p' .env | tail -n 1)"
  # Installs from before COMPOSE_PROJECT_NAME have Compose's default, the folder name.
  [ -n "$project" ] || project="$(basename "$DIR" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')"
  pick_port
  clean_env
  url="http://localhost:$PORT/ui"
}

fetch_compose() {
  curl -fsSL -o docker-compose.quickstart.yml "$COMPOSE_URL"
}

# The litellm-gateway command: the copy of this script kept beside .env, run
# against this folder, so up, down, status, and the rest work from anywhere.
install_command() {
  if tmp="$(mktemp quickstart.sh.XXXXXX)"; then
    if curl -fsSL -o "$tmp" "$SCRIPT_URL" 2>/dev/null && grep -q '^main "\$@"$' "$tmp"; then
      mv -f "$tmp" quickstart.sh
    else
      rm -f "$tmp"
    fi
  fi
  [ -f quickstart.sh ] || return 0
  quoted="$(printf '%s' "$DIR" | sed "s/'/'\\\\''/g")"
  {
    printf '#!/bin/sh\n'
    printf '# Manages the LiteLLM Gateway in this folder. Run it for a menu, or with one of\n'
    printf '# status, up, down, restart, logs, open, update, uninstall\n'
    printf "export LITELLM_DIR='%s'\n" "$quoted"
    # shellcheck disable=SC2016 # expanded when the command runs
    printf 'exec sh "$LITELLM_DIR/quickstart.sh" "$@"\n'
  } >litellm-gateway
  chmod 755 litellm-gateway
  # Linked into ~/.local/bin when that is on PATH. A link another install made
  # stays, unless that install is gone.
  link="$HOME/.local/bin/litellm-gateway"
  case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) return 0 ;; esac
  [ -d "$HOME/.local/bin" ] || return 0
  if [ -L "$link" ]; then
    if [ "$(readlink "$link")" = "$DIR/litellm-gateway" ] || [ -e "$link" ]; then return 0; fi
    rm -f "$link"
  elif [ -e "$link" ]; then
    return 0
  fi
  ln -s "$DIR/litellm-gateway" "$link"
}

# Sets COMMAND to how this install's litellm-gateway command runs from anywhere.
find_command() {
  COMMAND=''
  [ -x "$DIR/litellm-gateway" ] || return 0
  COMMAND="$(tildify "$DIR")/litellm-gateway"
  case ":$PATH:" in
    *":$HOME/.local/bin:"*)
      if [ "$(readlink "$HOME/.local/bin/litellm-gateway" 2>/dev/null)" = "$DIR/litellm-gateway" ]; then
        COMMAND=litellm-gateway
      fi
      ;;
  esac
}

hint() {
  if [ -n "$COMMAND" ]; then
    printf '%s %s' "$COMMAND" "$1"
    return 0
  fi
  case "$1" in
    up) set -- up -d ;;
    down) set -- stop ;;
    uninstall) set -- down --volumes ;;
    logs) set -- logs litellm ;;
  esac
  printf 'cd %s && %s compose -f docker-compose.quickstart.yml %s' "$(tildify "$DIR")" "$ENGINE" "$*"
}

# The gateway container's status, such as "Up 2 hours", or nothing when there is none.
gateway_status() {
  "$ENGINE" ps -a --filter "label=com.docker.compose.project=$project" \
    --filter label=com.docker.compose.service=litellm --format '{{.Status}}' 2>/dev/null | head -n 1
}

is_running() {
  case "$(gateway_status)" in
    Up*) return 0 ;;
    *) return 1 ;;
  esac
}

project_containers() {
  "$ENGINE" ps -q "$@" --filter "label=com.docker.compose.project=$project" 2>/dev/null
}

# The running version, from the API description: the first "version" in it is
# the gateway's, and it needs no key.
gateway_version() {
  curl -fsS --max-time 5 "http://127.0.0.1:$PORT/openapi.json" 2>/dev/null |
    grep -o '"version": *"[^"]*"' | head -n 1 | sed 's/.*"\([^"]*\)"$/\1/'
}

# How many requests the gateway has handled, failed ones included, counted from
# its request logs. It needs the admin key, which curl reads from stdin so it
# stays off the command line. Empty when the gateway cannot say.
gateway_requests() {
  key="$(sed -n 's/^LITELLM_MASTER_KEY=//p' .env | tail -n 1)"
  [ -n "$key" ] || return 0
  printf 'header = "Authorization: Bearer %s"\n' "$key" |
    curl -fsS --max-time 5 -K - \
      "http://127.0.0.1:$PORT/global/activity?start_date=2000-01-01&end_date=$(date -u +%Y-%m-%d)" 2>/dev/null |
    grep -o '"sum_api_requests": *[0-9]*' | grep -o '[0-9]*$'
}

# Stops the other quickstart install port_owner found.
stop_owner() {
  "$ENGINE" compose -p "$OWNER_PROJECT" stop
}

# Sets NEXT_PORT to the first free port after $1, or empty when the next 100
# are all taken.
next_free_port() {
  NEXT_PORT=$(($1 + 1))
  while ! port_free "$NEXT_PORT"; do
    NEXT_PORT=$((NEXT_PORT + 1))
    if [ "$NEXT_PORT" -gt $(($1 + 100)) ]; then
      NEXT_PORT=''
      return 0
    fi
  done
}

# Moves this install to NEXT_PORT and saves it in .env, through a fresh mktemp
# file (mode 600) and a rename so the keys are never half written.
move_port() {
  ENV_TMP="$(mktemp .env.XXXXXX)" || return 1
  if ! awk '!/^LITELLM_PORT=/' .env >"$ENV_TMP" ||
    ! printf 'LITELLM_PORT=%s\n' "$NEXT_PORT" >>"$ENV_TMP" || ! mv -f "$ENV_TMP" .env; then
    rm -f "$ENV_TMP"
    ENV_TMP=''
    return 1
  fi
  ENV_TMP=''
  PORT="$NEXT_PORT"
  export LITELLM_PORT="$PORT"
  url="http://localhost:$PORT/ui"
}

# Before starting: if something else holds the port, stop it when it is another
# quickstart install, or move to a free port when it is anything else.
check_port() {
  if is_running || port_free "$PORT"; then return 0; fi
  port_owner "$PORT"
  tab="$(printf '\t')" NEXT_PORT=''
  if [ -n "$OWNER_PROJECT" ]; then
    if [ "$INTERACTIVE" = 1 ]; then
      menu "Port $PORT is in use by $OWNER. Stop it and start this one?" 1 \
        "Yes, stop it${tab}$FREE_IT" "No"
      if [ "$CHOICE" = 1 ] && spin "Stopping it" "" stop_owner; then
        step "Stopped $OWNER"
        return 0
      fi
    fi
  elif [ -z "$PORT_PINNED" ]; then
    next_free_port "$PORT"
    if [ -n "$NEXT_PORT" ] && [ "$INTERACTIVE" = 1 ]; then
      menu "Port $PORT is in use by ${OWNER:-another program}. Use port $NEXT_PORT instead?" 1 \
        "Yes, use $NEXT_PORT${tab}saved in $(tildify "$DIR")/.env" "No"
      if [ "$CHOICE" = 1 ]; then
        old="$PORT"
        if move_port; then
          step "LiteLLM moved from port $old to ${C_BOLD}$PORT${C_OFF}"
          return 0
        fi
        fail "Could not write $(tildify "$DIR")/.env."
        exit 1
      fi
    fi
  fi
  fail "Port $PORT is in use by ${OWNER:-another program}." \
    "Free it, or set LITELLM_PORT in $(tildify "$DIR")/.env to another port${NEXT_PORT:+, such as $NEXT_PORT}."
  if [ -n "$FREE_IT" ]; then printf '  To free it:  %s\n' "$FREE_IT" >&2; fi
  exit 1
}

wait_ready() {
  i=0
  until curl -fsS "http://127.0.0.1:$PORT/health/readiness" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 90 ]; then return 1; fi
    sleep 2
  done
}

launch() {
  check_port
  [ -f docker-compose.quickstart.yml ] || fetch_compose
  started="$(date +%s)"
  detail=''
  for image in $(compose config --images 2>/dev/null); do
    "$ENGINE" image inspect "$image" >/dev/null 2>&1 || detail="downloading images, first run only"
  done
  [ "$STYLE" = 1 ] || echo "Starting LiteLLM and Postgres${detail:+ ($detail)}..."
  if ! spin "Starting LiteLLM and Postgres" "$detail" compose up -d; then
    fail "$ENGINE_NAME could not start LiteLLM and Postgres." "Its output is above."
    exit 1
  fi
  if ! spin "Waiting for the gateway to be ready" "" wait_ready; then
    fail "The gateway did not become ready in 3 minutes." "See what it logged:"
    echo "    $(hint logs)" >&2
    exit 1
  fi
  step "LiteLLM is up at ${C_ACC}$url${C_OFF} ${C_DIM}· $(elapsed_since "$started")"
}

stop_stack() {
  if [ -z "$(project_containers)" ]; then
    step "LiteLLM is already stopped"
    return 0
  fi
  if ! spin "Stopping LiteLLM and Postgres" "" compose stop; then
    fail "$ENGINE_NAME could not stop LiteLLM and Postgres." "Its output is above."
    exit 1
  fi
  step "Stopped LiteLLM. Your models, keys, and logs are kept ${C_DIM}· start it again: $(hint up)"
}

restart_stack() {
  if [ -n "$(project_containers)" ] && ! spin "Stopping LiteLLM and Postgres" "" compose stop; then
    fail "$ENGINE_NAME could not stop LiteLLM and Postgres." "Its output is above."
    exit 1
  fi
  launch
}

update_stack() {
  before=''
  if is_running; then before="$(gateway_version)"; fi
  fetch_compose
  install_command
  find_command
  images="$(compose config --images 2>/dev/null)" || images=''
  old=''
  for image in $images; do
    old="$old $("$ENGINE" image inspect --format '{{.Id}}' "$image" 2>/dev/null)"
  done
  [ "$STYLE" = 1 ] || echo "Downloading the latest LiteLLM and Postgres images..."
  if ! spin "Downloading the latest images" "" compose pull; then
    fail "$ENGINE_NAME could not download the images." "Its output is above."
    exit 1
  fi
  launch
  # Delete the images this replaced. Removal fails, and they stay, while any
  # container still uses them or another tag points at them.
  for id in $old; do
    for image in $images; do
      [ "$id" != "$("$ENGINE" image inspect --format '{{.Id}}' "$image" 2>/dev/null)" ] || continue 2
    done
    "$ENGINE" image rm "$id" >/dev/null 2>&1 || true
  done
  after="$(gateway_version)"
  if [ -n "$before" ] && [ "$before" = "$after" ]; then
    step "LiteLLM $after is already the latest"
  elif [ -n "$before" ] && [ -n "$after" ]; then
    step "Updated LiteLLM from $before to $after"
  else
    step "LiteLLM ${after:+$after }is on the latest image"
  fi
}

show_logs() {
  if [ -z "$(project_containers -a)" ]; then
    step "No logs yet: start LiteLLM first"
    return 0
  fi
  if [ "${1:-}" = follow ]; then
    printf '%sThe gateway logs. Press Ctrl+C to stop following them.%s\n' "${C_DIM}" "${C_OFF}"
    exec "$ENGINE" compose -f docker-compose.quickstart.yml logs -f --tail 100 litellm
  fi
  compose logs --tail 100 litellm
}

desktop_row() {
  if [ "$ENGINE" = docker ] && docker info --format '{{.OperatingSystem}}' 2>/dev/null | grep -q 'Docker Desktop'; then
    printf 'Desktop|listed as %s in Docker Desktop' "$project"
  fi
}

show_status() {
  status="$(gateway_status)"
  image="$(compose config --images 2>/dev/null | grep -m 1 litellm)" || image=''
  desktop="$(desktop_row)"
  case "$status" in
    Up*)
      uptime="${status#Up }"
      version="$(gateway_version)"
      requests="$(gateway_requests)" || requests=''
      case "$requests" in
        '') ;;
        0) requests='Requests|none yet' ;;
        *) requests="Requests|$requests handled" ;;
      esac
      box "LiteLLM is running" "Admin UI|$url" "Version|${version:-starting}" "Uptime|${uptime%% (*}" \
        ${requests:+"$requests"} "Folder|$(tildify "$DIR")" "Image|$image" ${desktop:+"$desktop"} "Stop it|$(hint down)"
      ;;
    *)
      box "LiteLLM is stopped" "Folder|$(tildify "$DIR")" "Image|$image" ${desktop:+"$desktop"} \
        "Start it|$(hint up)" "Remove it|$(hint uninstall)"
      ;;
  esac
}

open_ui() {
  if is_running; then
    open_url "$url"
  else
    warn "LiteLLM is stopped. Start it with: $(hint up)"
  fi
}

# Returns 1 when the person keeps the install, so the menu can carry on.
uninstall() {
  volume="${project}_postgres_data"
  [ -f docker-compose.quickstart.yml ] || fetch_compose
  images="$(compose config --images 2>/dev/null)" || images=''
  link=''
  if [ "$(readlink "$HOME/.local/bin/litellm-gateway" 2>/dev/null)" = "$DIR/litellm-gateway" ]; then
    link="$HOME/.local/bin/litellm-gateway"
  fi
  if [ "$INTERACTIVE" = 1 ]; then
    printf '\n%sThis removes LiteLLM from %s:%s\n' "${C_BOLD}" "$(tildify "$DIR")" "${C_OFF}"
    printf '  - its containers (%s project %s)\n' "$ENGINE_NAME" "$project"
    printf '  - its database, the %s volume %s, with your models, keys, and spend logs\n' "$ENGINE_NAME" "$volume"
    printf '  - its files in %s, including .env with the admin password\n' "$(tildify "$DIR")"
    if [ -n "$link" ]; then printf '  - the litellm-gateway command in %s\n' "$(tildify "$HOME/.local/bin")"; fi
    menu "Remove LiteLLM and delete its database?" 2 "Yes, remove it" "No, keep it"
    if [ "$CHOICE" != 1 ]; then
      step "Kept LiteLLM and its data"
      return 1
    fi
  elif [ "$NO_QUESTIONS" != 1 ]; then
    fail "Removing LiteLLM deletes its database ($ENGINE_NAME volume $volume)." "To confirm, run it again with --yes."
    exit 1
  fi

  [ "$STYLE" = 1 ] || echo "Removing the LiteLLM and Postgres containers and the database..."
  if ! spin "Removing the containers and the database" "" compose down --volumes --remove-orphans; then
    fail "$ENGINE_NAME could not remove them." "Its output is above."
    exit 1
  fi
  step "Removed the containers and the database ${C_DIM}($volume)"

  kept=''
  for image in $images; do
    "$ENGINE" image inspect "$image" >/dev/null 2>&1 && kept="$kept $image"
  done
  if [ -n "$kept" ] && [ "$INTERACTIVE" = 1 ]; then
    # The size on disk, as the images list shows it (1.68GB, or 1.23 GB on Podman).
    size="$(for image in $kept; do "$ENGINE" images --format '{{.Size}}' "$image" 2>/dev/null | head -n 1; done |
      awk '{ n = $1 + 0; u = $0; gsub(/[0-9. ]/, "", u); m = 1
        if (u ~ /^[kK]/) m = 1e3; else if (u ~ /^M/) m = 1e6; else if (u ~ /^G/) m = 1e9; else if (u ~ /^T/) m = 1e12
        s += n * m } END { if (s > 0) printf " (%.1f GB)", s / 1e9 }')"
    tab="$(printf '\t')"
    menu "Also delete the LiteLLM and Postgres images${size}?" 2 \
      "Yes, delete them" "No, keep them${tab}a reinstall starts faster"
    if [ "$CHOICE" = 1 ]; then
      for image in $kept; do
        if "$ENGINE" image rm "$image" >/dev/null 2>&1; then
          step "Deleted the image $image"
          kept="$(printf '%s' "$kept" | sed "s| $image||")"
        else
          warn "Kept the image $image, which other containers use"
        fi
      done
    fi
  fi

  rm -f .env docker-compose.quickstart.yml quickstart.sh quickstart.ps1 litellm-gateway litellm-gateway.cmd
  if [ -f .gitignore ] && ! grep -qvxE '\*|\.env' .gitignore; then rm -f .gitignore; fi
  if [ -n "$link" ]; then rm -f "$link"; fi
  cd "$HOME"
  if rmdir "$DIR" 2>/dev/null; then
    step "Deleted $(tildify "$DIR")"
  else
    step "Deleted the LiteLLM files in $(tildify "$DIR") ${C_DIM}(the folder stays: it has other files)"
  fi

  left=''
  if [ -n "$(project_containers -a)" ]; then left=" containers,"; fi
  if "$ENGINE" volume inspect "$volume" >/dev/null 2>&1; then left="$left volume $volume,"; fi
  if [ -f "$DIR/.env" ]; then left="$left $(tildify "$DIR")/.env,"; fi
  if [ -n "$left" ]; then
    warn "Still there:${left%,}"
    return 0
  fi
  step "LiteLLM is removed. Nothing is left running, and its data is deleted"
  if [ -n "$kept" ]; then
    printf '  %sThe images stay, so a reinstall starts faster. To delete them:%s %s image rm%s\n' \
      "${C_DIM}" "${C_OFF}" "$ENGINE" "$kept"
  fi
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
  CLIP=''
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

# The menu a rerun shows for an existing install. Its options follow whether
# LiteLLM is running, and come back after each choice until Done.
manage() {
  show_status
  find_clipboard
  tab="$(printf '\t')"
  pick=1
  while :; do
    if is_running; then
      question="LiteLLM is running at $url. What next?"
      set -- "Open the admin UI in your browser"
      [ -z "$CLIP" ] || set -- "$@" "Copy the admin password${tab}to the clipboard, ready to paste"
      set -- "$@" "Stop it${tab}keeps your models, keys, and logs" "Restart it" \
        "Show the logs${tab}the last 100 lines" "Update it${tab}download the latest image and restart" \
        "Remove it${tab}delete its containers, database, and files" "Done"
    else
      question="LiteLLM is stopped. What next?"
      set -- "Start it${tab}at $url" "Update it${tab}download the latest image and start" \
        "Show the logs${tab}the last 100 lines" "Remove it${tab}delete its containers, database, and files" "Done"
    fi
    [ "$pick" -le $# ] || pick=1
    menu "$question" "$pick" "$@"
    picked=''
    eval "picked=\${$CHOICE}"
    pick=1
    case "$picked" in
      Open*) open_url "$url"; pick=$# ;;
      Copy*) copy_password ;;
      Start*) launch ;;
      Stop*) stop_stack ;;
      Restart*) restart_stack ;;
      Show*) show_logs ;;
      Update*) update_stack ;;
      Remove*) if uninstall; then return 0; fi ;;
      *) return 0 ;;
    esac
  done
}

main() {
  NO_QUESTIONS=0 ACTION=''
  for arg in "$@"; do
    case "$arg" in
      -y | --yes) NO_QUESTIONS=1 ;;
      -h | --help | help) usage; exit 0 ;;
      status | start | up | stop | down | restart | logs | open | update | uninstall)
        if [ -n "$ACTION" ]; then
          echo "Choose one action, not both $ACTION and $arg" >&2
          exit 1
        fi
        ACTION="$arg"
        ;;
      *) echo "Unknown option: $arg" >&2; usage >&2; exit 1 ;;
    esac
  done
  case "$ACTION" in
    up) ACTION=start ;;
    down) ACTION=stop ;;
  esac

  setup_output
  detect_terminal
  # Agents and CI get the defaults even inside a terminal, so nothing waits on a keypress.
  if [ "$NO_QUESTIONS" = 1 ] || [ -n "${CI:-}" ] || [ -n "${CLAUDECODE:-}" ]; then INTERACTIVE=0; fi
  trap restore_terminal EXIT
  trap on_interrupt INT TERM

  if [ -n "$ACTION" ]; then
    :
  elif [ "$STYLE" = 1 ]; then
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
    cat >&2 <<'MSG'

  Install Docker (https://docs.docker.com/get-docker/), Podman (https://podman.io),
  or Rancher Desktop (https://rancherdesktop.io), then run this again.
  Or deploy in one click (Railway or Render):  https://docs.litellm.ai/docs/proxy/docker_quick_start
  Only need to call models from Python?  pip install litellm
MSG
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

  if [ -n "$ACTION" ]; then
    if ! locate_install; then
      fail "No LiteLLM install found" "in ${LITELLM_DIR:-this folder, ./litellm-gateway, or ~/litellm-gateway}."
      printf '  Install it with:  curl -fsSL %s | sh\n' "$SCRIPT_URL" >&2
      exit 1
    fi
    use_install
    find_command
    case "$ACTION" in
      status) show_status ;;
      start) if is_running; then step "LiteLLM is already running at ${C_ACC}$url${C_OFF}"; else launch; fi ;;
      stop) stop_stack ;;
      restart) restart_stack ;;
      logs) if [ "$INTERACTIVE" = 1 ]; then show_logs follow; else show_logs; fi ;;
      open) open_ui ;;
      update) update_stack ;;
      uninstall) uninstall || true ;;
    esac
    return 0
  fi

  engine_version="$("$ENGINE" version --format '{{.Server.Version}}' 2>/dev/null)" || engine_version=''
  step "$ENGINE_NAME ${engine_version:+$engine_version }with Compose ${compose_version#v} is running"

  pick_folder
  if [ "$found" = 1 ] && [ "$INTERACTIVE" = 1 ]; then
    use_install
    if [ ! -x litellm-gateway ] || [ ! -f quickstart.sh ]; then install_command; fi
    find_command
    manage
    return 0
  fi
  if [ "$found" = 0 ]; then
    command -v openssl >/dev/null 2>&1 || { fail "openssl is required to generate keys."; exit 1; }
    check_new_install
  fi
  fetch_compose
  step "Downloaded docker-compose.quickstart.yml"

  if [ "$found" = 1 ]; then
    step "Reusing .env, so existing keys and data keep working"
  else
    pick_port
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

  use_install
  install_command
  find_command
  launch

  password="the LITELLM_MASTER_KEY value in $(tildify "$DIR")/.env"
  CLIP=''
  if [ "$INTERACTIVE" = 1 ]; then find_clipboard; fi
  if [ -n "$CLIP" ]; then
    password="copy it below, or LITELLM_MASTER_KEY in $(tildify "$DIR")/.env"
  fi
  desktop="$(desktop_row)"
  # A rerun finds ~/litellm-gateway on its own; any other folder needs naming.
  rerun="rerun the install command"
  [ "$DIR" = "$HOME/litellm-gateway" ] || rerun="$rerun with LITELLM_DIR=$(tildify "$DIR")"
  echo
  box "LiteLLM is running" \
    "Admin UI|$url" \
    "Username|admin" \
    "Password|$password" \
    "Next|in the UI, open Models + Endpoints > Add Model and paste a provider API key" \
    "Stop it|$(hint down)" \
    "Start it|$(hint up)" \
    "Remove it|$(hint uninstall)" \
    ${desktop:+"$desktop"} \
    "Manage|${COMMAND:+run $COMMAND, or }$rerun"

  next_steps
}

main "$@"
