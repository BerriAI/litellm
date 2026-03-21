#!/usr/bin/env bash
# LiteLLM Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/install.sh | sh
#
# NOTE: set -e without pipefail for POSIX sh compatibility (dash on Ubuntu/Debian
# ignores the shebang when invoked as `sh` and does not support `pipefail`).
set -eu

MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=9

# NOTE: before merging, this must stay as "litellm[proxy]" to install from PyPI.
LITELLM_PACKAGE="litellm[proxy]"

# ── colours ────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
  BOLD='\033[1m'
  GREEN='\033[38;2;78;186;101m'
  GREY='\033[38;2;153;153;153m'
  RESET='\033[0m'
else
  BOLD='' GREEN='' GREY='' RESET=''
fi

info()    { printf "${GREY}  %s${RESET}\n" "$*"; }
success() { printf "${GREEN}  ✔ %s${RESET}\n" "$*"; }
header()  { printf "${BOLD}  %s${RESET}\n" "$*"; }
die()     { printf "\n  Error: %s\n\n" "$*" >&2; exit 1; }

# ── banner ─────────────────────────────────────────────────────────────────
echo ""
cat << 'EOF'
  ██╗     ██╗████████╗███████╗██╗     ██╗     ███╗   ███╗
  ██║     ██║╚══██╔══╝██╔════╝██║     ██║     ████╗ ████║
  ██║     ██║   ██║   █████╗  ██║     ██║     ██╔████╔██║
  ██║     ██║   ██║   ██╔══╝  ██║     ██║     ██║╚██╔╝██║
  ███████╗██║   ██║   ███████╗███████╗███████╗██║ ╚═╝ ██║
  ╚══════╝╚═╝   ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝     ╚═╝
EOF
printf "  ${BOLD}LiteLLM Installer${RESET}  ${GREY}— unified gateway for 100+ LLM providers${RESET}\n\n"

# ── OS detection ───────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin)  PLATFORM="macOS ($ARCH)" ;;
  Linux)   PLATFORM="Linux ($ARCH)" ;;
  *)       die "Unsupported OS: $OS. LiteLLM supports macOS and Linux." ;;
esac

info "Platform: $PLATFORM"

# ── Python detection ───────────────────────────────────────────────────────
PYTHON_BIN=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    major="$("$candidate" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || true)"
    minor="$("$candidate" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || true)"
    if [ "${major:-0}" -ge "$MIN_PYTHON_MAJOR" ] && [ "${minor:-0}" -ge "$MIN_PYTHON_MINOR" ]; then
      PYTHON_BIN="$(command -v "$candidate")"
      info "Python: $("$candidate" --version 2>&1)"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  die "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ is required but not found.
  Install it from https://python.org/downloads or via your package manager:
    macOS:  brew install python@3
    Ubuntu: sudo apt install python3 python3-pip"
fi

# ── pip detection ──────────────────────────────────────────────────────────
if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  die "pip is not available. Install it with:
    $PYTHON_BIN -m ensurepip --upgrade"
fi

# ── install ────────────────────────────────────────────────────────────────
echo ""
header "Installing litellm[proxy]…"
echo ""

LITELLM_BIN=""
SCRIPTS_DIR=""
PIP_ERR="/tmp/litellm_pip_err.$$"
trap 'rm -f "$PIP_ERR"' EXIT

if "$PYTHON_BIN" -m pip install --upgrade "${LITELLM_PACKAGE}" 2>"$PIP_ERR"; then
  # pip succeeded — find litellm in Python's script dir
  SCRIPTS_DIR="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
  LITELLM_BIN="${SCRIPTS_DIR}/litellm"
  if [ ! -x "$LITELLM_BIN" ]; then
    USER_BIN="$("$PYTHON_BIN" -c 'import site; print(site.getuserbase())')/bin"
    LITELLM_BIN="${USER_BIN}/litellm"
    SCRIPTS_DIR="${USER_BIN}"
  fi
else
  # pip failed — check for PEP 668 externally-managed-environment
  if grep -qi "externally-managed\|EXTERNALLY-MANAGED" "$PIP_ERR" 2>/dev/null; then
    info "Detected externally-managed Python (PEP 668). Using isolated install…"
    if command -v pipx >/dev/null 2>&1; then
      info "Installing via pipx…"
      if pipx install "${LITELLM_PACKAGE}"; then
        PIPX_BIN="${PIPX_BIN_DIR:-${HOME}/.local/bin}"
        LITELLM_BIN="${PIPX_BIN}/litellm"
        SCRIPTS_DIR="${PIPX_BIN}"
      fi
    fi
    if [ -z "$LITELLM_BIN" ] || [ ! -x "$LITELLM_BIN" ]; then
      # pipx not available or failed — create venv in ~/.litellm/venv
      LITELLM_VENV="${LITELLM_VENV:-${HOME}/.litellm/venv}"
      info "Creating virtual environment at $LITELLM_VENV…"
      mkdir -p "$(dirname "$LITELLM_VENV")"
      "$PYTHON_BIN" -m venv "$LITELLM_VENV"
      "$LITELLM_VENV/bin/pip" install --upgrade --quiet "${LITELLM_PACKAGE}"
      LITELLM_BIN="${LITELLM_VENV}/bin/litellm"
      SCRIPTS_DIR="${LITELLM_VENV}/bin"
    fi
  fi
  if [ -z "$LITELLM_BIN" ] || [ ! -x "$LITELLM_BIN" ]; then
    cat "$PIP_ERR" >&2
    die "pip install failed. On macOS with Homebrew Python, try: brew install pipx && pipx install '${LITELLM_PACKAGE}'"
  fi
fi

if [ ! -x "$LITELLM_BIN" ]; then
  die "litellm binary not found after install. Try: $PYTHON_BIN -m pip install --user '${LITELLM_PACKAGE}'"
fi

# ── success banner ─────────────────────────────────────────────────────────
echo ""
success "LiteLLM installed"

installed_ver="$("$LITELLM_BIN" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
[ -n "$installed_ver" ] && info "Version: $installed_ver"

# ── PATH hint ──────────────────────────────────────────────────────────────
if ! command -v litellm >/dev/null 2>&1; then
  info "Note: add litellm to your PATH:  export PATH=\"\$PATH:${SCRIPTS_DIR}\""
fi

# ── launch setup wizard ────────────────────────────────────────────────────
echo ""
printf "  ${BOLD}Run the interactive setup wizard?${RESET} ${GREY}(Y/n)${RESET}: "
# /dev/tty may be unavailable in Docker/CI — default to yes if it can't be read
answer=""
if [ -r /dev/tty ]; then
  read -r answer </dev/tty || answer=""
fi

if [ -z "$answer" ] || [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
  echo ""
  # Use /dev/tty for interactive input when available (stdin is a pipe from curl)
  if [ -r /dev/tty ]; then
    exec "$LITELLM_BIN" --setup </dev/tty
  else
    exec "$LITELLM_BIN" --setup
  fi
else
  echo ""
  header "Quick start:"
  echo ""
  info "  litellm --setup          # interactive wizard"
  info "  litellm --model gpt-4o   # single-model quickstart"
  echo ""
  info "Docs: https://docs.litellm.ai"
  echo ""
fi
