#!/usr/bin/env bash
# LiteLLM Installer
# Usage: curl -fsSL https://litellm.ai/install.sh | sh
set -euo pipefail

LITELLM_PACKAGE="litellm[proxy]"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=9

# ── colours ────────────────────────────────────────────────────────────────
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
  ORANGE='\033[38;2;215;119;87m'
  BOLD='\033[1m'
  GREEN='\033[38;2;78;186;101m'
  GREY='\033[38;2;153;153;153m'
  RESET='\033[0m'
else
  ORANGE='' BOLD='' GREEN='' GREY='' RESET=''
fi

info()    { printf "${GREY}  %s${RESET}\n" "$*"; }
success() { printf "${GREEN}  ✔ %s${RESET}\n" "$*"; }
header()  { printf "${ORANGE}  %s${RESET}\n" "$*"; }
die()     { printf "\n  Error: %s\n\n" "$*" >&2; exit 1; }

# ── banner ─────────────────────────────────────────────────────────────────
echo ""
printf "${ORANGE}"
cat << 'EOF'
  ██╗     ██╗████████╗███████╗██╗     ██╗     ███╗   ███╗
  ██║     ██║╚══██╔══╝██╔════╝██║     ██║     ████╗ ████║
  ██║     ██║   ██║   █████╗  ██║     ██║     ██╔████╔██║
  ██║     ██║   ██║   ██╔══╝  ██║     ██║     ██║╚██╔╝██║
  ███████╗██║   ██║   ███████╗███████╗███████╗██║ ╚═╝ ██║
  ╚══════╝╚═╝   ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝     ╚═╝
EOF
printf "${RESET}"
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
    py_ver="$("$candidate" -c 'import sys; print(sys.version_info[:2])'  2>/dev/null || true)"
    major="$("$candidate" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || true)"
    minor="$("$candidate" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || true)"
    if [ "${major:-0}" -ge "$MIN_PYTHON_MAJOR" ] && [ "${minor:-0}" -ge "$MIN_PYTHON_MINOR" ]; then
      PYTHON_BIN="$candidate"
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
    $PYTHON_BIN -m ensurepip --upgrade
  or:
    curl https://bootstrap.pypa.io/get-pip.py | $PYTHON_BIN"
fi

# ── install ────────────────────────────────────────────────────────────────
echo ""
header "Installing ${LITELLM_PACKAGE}…"
echo ""

# Use --quiet to avoid wall of pip output; keep --progress-bar off for cleaner CI
"$PYTHON_BIN" -m pip install --quiet --progress-bar off "${LITELLM_PACKAGE}" \
  || die "pip install failed. Try manually: $PYTHON_BIN -m pip install '${LITELLM_PACKAGE}'"

# Verify litellm is on PATH (or accessible as python -m litellm)
LITELLM_BIN="$(command -v litellm 2>/dev/null || true)"
if [ -z "$LITELLM_BIN" ]; then
  # Might be installed in a user PATH that isn't active yet
  USER_BIN="$("$PYTHON_BIN" -c 'import site,os; print(site.getuserbase())')/bin"
  if [ -x "$USER_BIN/litellm" ]; then
    LITELLM_BIN="$USER_BIN/litellm"
    info "Note: $LITELLM_BIN is not in your PATH yet."
    info "Add this to your shell profile:"
    info "  export PATH=\"\$PATH:$USER_BIN\""
  fi
fi

echo ""
success "LiteLLM installed"

# ── version check ──────────────────────────────────────────────────────────
if [ -n "$LITELLM_BIN" ]; then
  installed_ver="$("$LITELLM_BIN" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
  [ -n "$installed_ver" ] && info "Version: $installed_ver"
fi

# ── launch setup wizard ────────────────────────────────────────────────────
echo ""
printf "  ${BOLD}Run the interactive setup wizard?${RESET} ${GREY}(Y/n)${RESET}: "
read -r answer </dev/tty

if [ -z "$answer" ] || [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
  echo ""
  if [ -n "$LITELLM_BIN" ]; then
    exec "$LITELLM_BIN" --setup
  else
    exec "$PYTHON_BIN" -m litellm --setup
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
