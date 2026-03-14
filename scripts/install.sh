#!/usr/bin/env bash
# LiteLLM Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/install.sh | sh
set -euo pipefail

MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=9

# NOTE: set to "litellm[proxy]" before merging to main (once --setup ships in a PyPI release).
# On this branch we install from git so the installer gets the --setup flag.
LITELLM_PACKAGE="litellm[proxy] @ git+https://github.com/BerriAI/litellm.git@worktree-dynamic-tickling-journal"

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
    $PYTHON_BIN -m ensurepip --upgrade"
fi

# ── install ────────────────────────────────────────────────────────────────
echo ""
if [ -n "$LITELLM_BRANCH" ]; then
  header "Installing litellm from branch '${LITELLM_BRANCH}'…"
else
  header "Installing litellm[proxy]…"
fi
echo ""

"$PYTHON_BIN" -m pip install --quiet --progress-bar off "${LITELLM_PACKAGE}" \
  || die "pip install failed. Try manually: $PYTHON_BIN -m pip install '${LITELLM_PACKAGE}'"

# ── find litellm binary ────────────────────────────────────────────────────
LITELLM_BIN="$(command -v litellm 2>/dev/null || true)"
if [ -z "$LITELLM_BIN" ]; then
  USER_BIN="$("$PYTHON_BIN" -c 'import site; print(site.getuserbase())')/bin"
  if [ -x "$USER_BIN/litellm" ]; then
    LITELLM_BIN="$USER_BIN/litellm"
    info "Note: $LITELLM_BIN is not in your PATH yet."
    info "Add this to your shell profile:  export PATH=\"\$PATH:$USER_BIN\""
  fi
fi

echo ""
success "LiteLLM installed"

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
