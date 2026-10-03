#!/bin/sh
# LiteLLM Gateway quickstart: the gateway, Postgres, and the admin UI in one command.
#   curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.sh | sh
#
# To read it before running it:
#   curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.sh -o quickstart.sh
#   less quickstart.sh
#   sh quickstart.sh
#
# Asks at most two questions (where to keep the files, and whether to open the
# admin UI), each with a default you accept by pressing Enter. It asks nothing
# when there is no terminal, under CI or Claude Code, or when run with --yes.
#
#   --yes, -y        no questions: install to ~/litellm-gateway, don't open a browser
#   LITELLM_DIR      folder to install into (skips the folder question)
#   LITELLM_PORT     port for the gateway (default 4000, or the next free one)
#
# New installs listen on this machine only (127.0.0.1). To reach the gateway
# from other machines, remove LITELLM_BIND from .env and put it behind TLS.
#
# Keys and the database password are random (openssl rand), written only to
# .env with permissions 600, and never printed. Needs Docker with Compose v2.
# Everything runs inside main(), so a partial download runs nothing.
set -eu

COMPOSE_URL="${LITELLM_COMPOSE_URL:-https://raw.githubusercontent.com/BerriAI/litellm/main/docker/docker-compose.quickstart.yml}"

# ---------------------------------------------------------------- terminal

INTERACTIVE=0   # a person is at a terminal we can ask
ARROWS=0        # that terminal supports the arrow-key menu
STTY_SAVED=""
POINTER='>'

detect_terminal() {
  # Piped from curl, stdin is the script itself, so questions go to /dev/tty.
  if (exec </dev/tty) 2>/dev/null && [ "${TERM:-dumb}" != "dumb" ]; then
    INTERACTIVE=1
    if STTY_SAVED="$(stty -g </dev/tty 2>/dev/null)" && [ -n "$STTY_SAVED" ]; then
      ARROWS=1
    fi
  fi
  case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
    *UTF-8* | *utf-8* | *UTF8* | *utf8*) POINTER='❯' ;;
  esac
}

restore_terminal() {
  if [ -n "$STTY_SAVED" ]; then
    stty "$STTY_SAVED" </dev/tty 2>/dev/null || true
    printf '\033[?25h' >/dev/tty 2>/dev/null || true
  fi
}

on_interrupt() {
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
menu() {
  question="$1"
  CHOICE="$2"
  shift 2
  count=$#
  if [ "$INTERACTIVE" != 1 ]; then return 0; fi

  printf '\n%s\n' "$question" >/dev/tty
  if [ "$ARROWS" = 1 ]; then
    trap on_interrupt INT TERM
    stty -icanon -echo min 1 time 0 </dev/tty
    printf '\033[?25l' >/dev/tty
    first=1
    while :; do
      [ "$first" = 1 ] || printf '\033[%sA' "$count" >/dev/tty
      first=0
      i=1
      for opt in "$@"; do
        if [ "$i" = "$CHOICE" ]; then
          printf '\033[2K  \033[1;36m%s %s\033[0m\n' "$POINTER" "$opt" >/dev/tty
        else
          printf '\033[2K    %s\n' "$opt" >/dev/tty
        fi
        i=$((i + 1))
      done
      key="$(read_key)"
      case "$key" in
        up | k) [ "$CHOICE" -gt 1 ] && CHOICE=$((CHOICE - 1)) ;;
        down | j) [ "$CHOICE" -lt "$count" ] && CHOICE=$((CHOICE + 1)) ;;
        [1-9]) [ "$key" -le "$count" ] && CHOICE="$key" ;;
        '' | "$(printf '\r')") break ;;
      esac
    done
    restore_terminal
    trap - INT TERM
  else
    i=1
    for opt in "$@"; do
      printf '  %s) %s\n' "$i" "$opt" >/dev/tty
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
  if [ -n "${LITELLM_DIR:-}" ]; then
    DIR="$LITELLM_DIR"
  elif [ -f "$here_dir/.env" ]; then
    DIR="$here_dir"   # installed in this folder before
  elif [ -f "$home_dir/.env" ]; then
    DIR="$home_dir"   # installed in the home folder before
  elif [ "$here_dir" = "$home_dir" ]; then
    DIR="$home_dir"
  else
    menu "Where should LiteLLM keep its files (.env with your keys, and the compose file)?" 1 \
      "$home_dir   recommended, reruns always find it" \
      "$here_dir   this folder"
    if [ "$CHOICE" = 2 ]; then DIR="$here_dir"; else DIR="$home_dir"; fi
  fi
  created=0
  [ -d "$DIR" ] || created=1
  mkdir -p "$DIR"
  cd "$DIR"
  DIR="$(pwd)"
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
    echo "Added .env to this repository's local git exclude list ($exclude), so your keys stay out of commits."
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
        echo "Ports 4000 to 4099 are all in use. Set LITELLM_PORT to a free port and run this again." >&2
        exit 1
      fi
    done
    [ "$PORT" = 4000 ] || echo "Port 4000 is in use, so LiteLLM will use $PORT."
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
  if docker volume inspect "${project}_postgres_data" >/dev/null 2>&1; then
    cat >&2 <<EOF
Found a database from an earlier install (Docker volume ${project}_postgres_data)
but no $DIR/.env with its password.

  Restore that .env file and run this again to keep your models and keys, or
  delete the old database and start fresh (this removes its models and keys):
    docker volume rm ${project}_postgres_data
EOF
    exit 1
  fi
}

open_browser() {
  url="$1"
  menu "Open the admin UI in your browser?" 1 "Yes" "No"
  [ "$INTERACTIVE" = 1 ] && [ "$CHOICE" = 1 ] || return 0
  if command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  fi
}

main() {
  NO_QUESTIONS=0
  for arg in "$@"; do
    case "$arg" in
      -y | --yes) NO_QUESTIONS=1 ;;
      *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
  done

  detect_terminal
  # Agents and CI get the defaults even inside a terminal, so nothing waits on a keypress.
  if [ "$NO_QUESTIONS" = 1 ] || [ -n "${CI:-}" ] || [ -n "${CLAUDECODE:-}" ]; then INTERACTIVE=0; fi
  trap restore_terminal EXIT

  if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Docker is not installed. The LiteLLM Gateway runs in Docker alongside a Postgres database.

  Install Docker, then run this again:  https://docs.docker.com/get-docker/
  Or deploy in one click (Railway or Render):  https://docs.litellm.ai/docs/proxy/docker_quick_start
  Only need to call models from Python?  pip install litellm
EOF
    exit 1
  fi
  docker compose version >/dev/null 2>&1 || { echo "Docker Compose v2 ('docker compose') is required." >&2; exit 1; }
  docker info >/dev/null 2>&1 || { echo "Docker is installed but not running. Start it and run this again." >&2; exit 1; }
  command -v openssl >/dev/null 2>&1 || { echo "openssl is required to generate keys." >&2; exit 1; }

  echo "LiteLLM quickstart"
  pick_folder
  [ -f .env ] || check_new_install
  curl -fsSL -o docker-compose.quickstart.yml "$COMPOSE_URL"
  pick_port

  if [ -f .env ]; then
    echo "Reusing $DIR/.env, so existing keys and data keep working."
  else
    (umask 077 && printf 'LITELLM_MASTER_KEY=sk-%s\nLITELLM_SALT_KEY=sk-%s\nPOSTGRES_PASSWORD=%s\nLITELLM_PORT=%s\nLITELLM_BIND=127.0.0.1:\nCOMPOSE_PROJECT_NAME=%s\n' \
      "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" "$(openssl rand -hex 24)" "$PORT" "$project" >.env)
    echo "Generated $DIR/.env with your master key, salt key, and database password. Keep this file."
  fi

  # Compose prefers values already set in the shell over .env, so drop any
  # inherited ones: .env stays the only source for keys and the project name.
  unset LITELLM_MASTER_KEY LITELLM_SALT_KEY POSTGRES_PASSWORD COMPOSE_PROJECT_NAME
  # The bind address follows .env when .env sets it (every install this script
  # creates does). For an older .env without it, a value exported in the shell
  # is kept, so an intentional LITELLM_BIND=127.0.0.1: is not dropped.
  if grep -q '^LITELLM_BIND=' .env; then unset LITELLM_BIND; fi

  echo "Starting LiteLLM and Postgres (the first run downloads the images)..."
  docker compose -f docker-compose.quickstart.yml up -d

  i=0
  until curl -fsS "http://127.0.0.1:$PORT/health/readiness" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 90 ]; then
      echo "The gateway did not become ready in 3 minutes. Check: cd $DIR && docker compose -f docker-compose.quickstart.yml logs litellm" >&2
      exit 1
    fi
    sleep 2
  done

  echo
  echo "LiteLLM is running."
  echo "  Admin UI:   http://localhost:$PORT/ui"
  echo "  Username:   admin"
  echo "  Password:   the LITELLM_MASTER_KEY value in $DIR/.env"
  echo "  Next:       in the UI, open Models + Endpoints > Add Model and paste a provider API key"
  echo "  Stop it:    cd $DIR && docker compose -f docker-compose.quickstart.yml down"

  open_browser "http://localhost:$PORT/ui"
}

main "$@"
