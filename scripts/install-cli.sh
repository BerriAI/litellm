#!/usr/bin/env bash
# LiteLLM CLI Installer (the thin `lite` client)
# Usage: curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/install-cli.sh | sh
#
# Installs only litellm[cli]: the `lite` command for authenticating to a LiteLLM
# proxy and running coding agents (lite claude / codex / opencode) through it.
# None of the proxy server runtime is pulled in. To run a proxy server instead,
# use scripts/install.sh, which installs litellm[proxy].
#
# Needs only curl: uv is bootstrapped if missing, and uv provisions a compatible
# Python itself (honouring litellm's requires-python), downloading a managed one
# when the host has no suitable interpreter.
#
# NOTE: set -e without pipefail for POSIX sh compatibility (dash on Ubuntu/Debian
# ignores the shebang when invoked as `sh` and does not support `pipefail`).
set -eu

# NOTE: before merging, this must stay as "litellm[cli]" to install from PyPI.
LITELLM_PACKAGE="litellm[cli]"
UV_VERSION="0.10.9"

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
  ██╗     ██╗████████╗███████╗
  ██║     ██║╚══██╔══╝██╔════╝
  ██║     ██║   ██║   █████╗
  ██║     ██║   ██║   ██╔══╝
  ███████╗██║   ██║   ███████╗
  ╚══════╝╚═╝   ╚═╝   ╚══════╝
EOF
printf "  ${BOLD}LiteLLM CLI Installer${RESET}  ${GREY}the thin 'lite' client for your proxy${RESET}\n\n"

# ── OS detection ───────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin)  PLATFORM="macOS ($ARCH)" ;;
  Linux)   PLATFORM="Linux ($ARCH)" ;;
  *)       die "Unsupported OS: $OS. LiteLLM supports macOS and Linux." ;;
esac

info "Platform: $PLATFORM"

# ── uv detection / install ────────────────────────────────────────────────
UV_BIN=""
CURRENT_UV_VERSION=""
for candidate in uv "$HOME/.local/bin/uv"; do
  if command -v "$candidate" >/dev/null 2>&1; then
    UV_BIN="$(command -v "$candidate")"
    break
  elif [ -x "$candidate" ]; then
    UV_BIN="$candidate"
    break
  fi
done

if [ -n "$UV_BIN" ]; then
  CURRENT_UV_VERSION="$("$UV_BIN" --version 2>/dev/null | awk '{print $2}' | head -1 || true)"
fi

if [ -z "$UV_BIN" ] || [ "${CURRENT_UV_VERSION:-}" != "$UV_VERSION" ]; then
  header "Installing uv…"
  if [ -n "${CURRENT_UV_VERSION:-}" ]; then
    info "Upgrading uv from ${CURRENT_UV_VERSION} to ${UV_VERSION}"
  fi
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_NO_MODIFY_PATH=1 sh \
    || die "uv installation failed. Try manually: curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh"
  UV_BIN="$HOME/.local/bin/uv"
fi

# ── install ────────────────────────────────────────────────────────────────
# --python-preference system: reuse a compatible system Python when present,
# otherwise download a managed one. Either way uv honours litellm's requires-python,
# so an unsupported system Python is skipped, not forced.
echo ""
header "Installing litellm[cli]…"
echo ""

"$UV_BIN" tool install --python-preference system --force "${LITELLM_PACKAGE}" \
  || die "uv tool install failed. Try manually: $UV_BIN tool install '${LITELLM_PACKAGE}'"

# ── find the lite binary installed by uv tool ──────────────────────────────
SCRIPTS_DIR="$("$UV_BIN" tool dir --bin)"
LITE_BIN="${SCRIPTS_DIR}/lite"

if [ ! -x "$LITE_BIN" ]; then
  die "lite binary not found after install. Try: $UV_BIN tool install '${LITELLM_PACKAGE}'"
fi

# ── success banner ─────────────────────────────────────────────────────────
echo ""
success "LiteLLM CLI installed"

installed_ver="$("$LITE_BIN" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
[ -n "$installed_ver" ] && info "Version: $installed_ver"

# ── PATH hint ──────────────────────────────────────────────────────────────
if ! command -v lite >/dev/null 2>&1; then
  info "Note: add lite to your PATH:  export PATH=\"\$PATH:${SCRIPTS_DIR}\""
fi

# ── next steps ─────────────────────────────────────────────────────────────
echo ""
header "Next steps:"
echo ""
info "  export LITELLM_PROXY_URL=https://your-proxy   # point at your gateway"
info "  lite login                                    # authenticate via SSO"
info "  lite claude                                   # run Claude Code through the proxy"
echo ""
info "Docs: https://docs.litellm.ai/docs/proxy/management_cli"
echo ""
