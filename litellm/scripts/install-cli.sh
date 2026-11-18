#!/usr/bin/env bash
# LiteLLM CLI Installer (the thin `lite` client)
# Usage: curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/scripts/install-cli.sh | sh
#
# Installs only litellm[cli]: the `lite` command for authenticating to a LiteLLM
# proxy and running coding agents (lite claude / codex / opencode) through it.
# None of the proxy server runtime is pulled in. To run a proxy server instead,
# use litellm/scripts/install.sh, which installs litellm[proxy].
#
# Needs only curl: uv is bootstrapped if missing, and uv provisions a compatible
# Python itself (honouring litellm's requires-python), downloading a managed one
# when the host has no suitable interpreter.
#
# To try an unreleased branch instead of the latest PyPI release (for example, to
# QA a CLI feature before it ships), set LITELLM_CLI_REF to a branch, tag, or commit:
#   curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/<branch>/litellm/scripts/install-cli.sh | \
#     LITELLM_CLI_REF=<branch> sh
#
# NOTE: set -e without pipefail for POSIX sh compatibility (dash on Ubuntu/Debian
# ignores the shebang when invoked as `sh` and does not support `pipefail`).
set -eu

# Defaults to the PyPI release; LITELLM_CLI_REF opts into installing from source instead.
if [ -n "${LITELLM_CLI_REF:-}" ]; then
  LITELLM_PACKAGE="litellm[cli] @ git+https://github.com/BerriAI/litellm.git@${LITELLM_CLI_REF}"
else
  LITELLM_PACKAGE="litellm[cli]"
fi
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
  case "$OS-$ARCH" in
    Darwin-arm64)              UV_TARGET=aarch64-apple-darwin           UV_SHA256=a92f61e9ac9b0f29668c15f56152e4a60143fca148ff5bfadb86718472c3f376 ;;
    Darwin-x86_64)             UV_TARGET=x86_64-apple-darwin            UV_SHA256=9cc2de7d195fa157f98b306a8a1cb151ded93f488939b93363cebc8b9d598c28 ;;
    Linux-x86_64)              UV_TARGET=x86_64-unknown-linux-musl      UV_SHA256=433e56874739e92c7cfd661ba9e5f287b376ca612c08c8194a41a98a13158aea ;;
    Linux-aarch64|Linux-arm64) UV_TARGET=aarch64-unknown-linux-musl     UV_SHA256=05b0d3087e913ebe11756365a90dd47c05d6728752fdbe129ad4c3ccd769826d ;;
    Linux-i686|Linux-i386)     UV_TARGET=i686-unknown-linux-musl        UV_SHA256=1d42b0d0a037b3d658b11ec889154686db3ab269ba2b789bdbc45d36e3549f34 ;;
    Linux-armv7l)              UV_TARGET=armv7-unknown-linux-musleabihf UV_SHA256=f702e821b80e371e14987a886d58ee103c5948b7b096fa49a552624c24d7e073 ;;
    Linux-armv6l)              UV_TARGET=arm-unknown-linux-musleabihf   UV_SHA256=6220fa3eb5f8212cae4ec3a5053060914aaa829549cf706dde9f9cc344f75f61 ;;
    Linux-ppc64le)             UV_TARGET=powerpc64le-unknown-linux-gnu  UV_SHA256=e804f4a7d0659e09ef806365f04bdd33c940603fab903e925402748d05dd109a ;;
    Linux-s390x)               UV_TARGET=s390x-unknown-linux-gnu        UV_SHA256=a589d4a8930c82fa7225daec19c632651b3c84f50f770efe758056b387e5f0dd ;;
    Linux-riscv64)             UV_TARGET=riscv64gc-unknown-linux-gnu    UV_SHA256=1541596da45855e34202130027a613a2ace7d441e04d747cb4dd9f2590461c9a ;;
    *) die "No pinned uv ${UV_VERSION} download for ${PLATFORM}. Install uv ${UV_VERSION} (https://docs.astral.sh/uv/getting-started/installation/) and rerun." ;;
  esac
  UV_TMP="$(mktemp -d)"
  trap 'rm -rf "$UV_TMP"' EXIT
  curl -fsSL -o "$UV_TMP/uv.tar.gz" "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${UV_TARGET}.tar.gz" \
    || die "Could not download uv ${UV_VERSION} for ${UV_TARGET}."
  if command -v sha256sum >/dev/null 2>&1; then
    UV_DOWNLOAD_SHA256="$(sha256sum "$UV_TMP/uv.tar.gz" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    UV_DOWNLOAD_SHA256="$(shasum -a 256 "$UV_TMP/uv.tar.gz" | awk '{print $1}')"
  else
    die "Need sha256sum or shasum to verify the uv download."
  fi
  [ "$UV_DOWNLOAD_SHA256" = "$UV_SHA256" ] \
    || die "uv ${UV_VERSION} download failed its SHA-256 check (expected ${UV_SHA256}, got ${UV_DOWNLOAD_SHA256}). Nothing was installed."
  tar -xzf "$UV_TMP/uv.tar.gz" -C "$UV_TMP" || die "Could not unpack the uv ${UV_VERSION} download."
  mkdir -p "$HOME/.local/bin"
  cp "$UV_TMP/uv-${UV_TARGET}/uv" "$UV_TMP/uv-${UV_TARGET}/uvx" "$HOME/.local/bin/" \
    || die "Could not install uv into $HOME/.local/bin."
  UV_BIN="$HOME/.local/bin/uv"
fi

# ── install ────────────────────────────────────────────────────────────────
# --python mirrors requires-python in pyproject.toml (keep in sync): uv selects the
# interpreter before resolving, so an unconstrained request accepts a too-old system
# Python (stock macOS ships 3.9) and fails resolution instead of downloading a
# managed one. --python-preference system still reuses a compatible system Python.
echo ""
if [ -n "${LITELLM_CLI_REF:-}" ]; then
  header "Installing litellm[cli] from ${LITELLM_CLI_REF}…"
else
  header "Installing litellm[cli]…"
fi
echo ""

"$UV_BIN" tool install --python '>=3.10,<3.15' --python-preference system --force "${LITELLM_PACKAGE}" \
  || die "uv tool install failed. Try manually: $UV_BIN tool install --python '>=3.10,<3.15' '${LITELLM_PACKAGE}'"

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
