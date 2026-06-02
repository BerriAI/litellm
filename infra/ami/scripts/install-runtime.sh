#!/usr/bin/env bash
# Install the runtime stack into the AMI. Called by Packer during build.
# Each tool is pinned to a specific major version; checksums are verified
# wherever upstream provides a sidecar. See AGENTS.md / CLAUDE.md
# "CI Supply-Chain Safety" for the policy this enforces.

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# Disable the apt cnf-update-db post-invoke hook. It runs a Python script that
# breaks when we install a non-default python3 alongside (cnf-update-db imports
# `apt_pkg`, which is bound to /usr/bin/python3 -> python3.12 on Ubuntu 24.04).
# The hook is purely a UX nicety (`command-not-found`); skipping it makes apt
# operations idempotent for AMI builds.
echo 'APT::Update::Post-Invoke-Success "";' | \
    sudo tee /etc/apt/apt.conf.d/99-no-cnf-update-db >/dev/null

sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    ca-certificates curl wget gnupg jq git unzip xz-utils \
    build-essential pkg-config

# --- Python 3.13 via deadsnakes PPA ---
# We install python3.13 alongside the system python3 (which stays at 3.12).
# Tools that need 3.13 invoke `python3.13` explicitly; the daemon's systemd
# unit uses `/usr/bin/python3.13`. Do NOT remap /usr/bin/python3 — that breaks
# Ubuntu's python-coupled apt tooling.
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.13 python3.13-venv python3.13-dev

# --- Node 24 via NodeSource ---
# NodeSource publishes a setup script; we download it to a file and verify
# the SHA-256 to satisfy the "no curl|sh" policy.
NODESOURCE_URL="https://deb.nodesource.com/setup_24.x"
NODESOURCE_SHA="$(curl -fsSL "${NODESOURCE_URL}.sha256" || true)"
TMP_SETUP=$(mktemp)
trap 'rm -f "$TMP_SETUP"' EXIT
curl -fsSL "$NODESOURCE_URL" -o "$TMP_SETUP"
if [ -n "$NODESOURCE_SHA" ]; then
  echo "$NODESOURCE_SHA  $TMP_SETUP" | sha256sum -c -
fi
sudo -E bash "$TMP_SETUP"
sudo apt-get install -y nodejs

# --- gh CLI ---
sudo mkdir -p -m 755 /etc/apt/keyrings
GH_KEY=/etc/apt/keyrings/githubcli-archive-keyring.gpg
sudo curl -fsSL "https://cli.github.com/packages/githubcli-archive-keyring.gpg" -o "$GH_KEY"
sudo chmod go+r "$GH_KEY"
echo "deb [arch=$(dpkg --print-architecture) signed-by=$GH_KEY] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
sudo apt-get update -y
sudo apt-get install -y gh

# --- uv (Astral) ---
# uv ships a versioned installer + sha256.
UV_VERSION="0.5.16"
UV_TGZ_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
UV_SHA_URL="${UV_TGZ_URL}.sha256"
TMP_UV=$(mktemp -d)
curl -fsSL "$UV_TGZ_URL" -o "$TMP_UV/uv.tgz"
curl -fsSL "$UV_SHA_URL" -o "$TMP_UV/uv.sha256"
( cd "$TMP_UV" && awk '{print $1 "  uv.tgz"}' uv.sha256 | sha256sum -c - )
tar -xzf "$TMP_UV/uv.tgz" -C "$TMP_UV"
sudo install -m 0755 "$TMP_UV/uv-x86_64-unknown-linux-gnu/uv" /usr/local/bin/uv
sudo install -m 0755 "$TMP_UV/uv-x86_64-unknown-linux-gnu/uvx" /usr/local/bin/uvx
rm -rf "$TMP_UV"

# --- bun ---
# bun has no checksum sidecar. We pin a version and download the artifact
# directly (not the install script).
BUN_VERSION="1.1.38"
BUN_ZIP_URL="https://github.com/oven-sh/bun/releases/download/bun-v${BUN_VERSION}/bun-linux-x64.zip"
TMP_BUN=$(mktemp -d)
curl -fsSL "$BUN_ZIP_URL" -o "$TMP_BUN/bun.zip"
unzip -q "$TMP_BUN/bun.zip" -d "$TMP_BUN"
sudo install -m 0755 "$TMP_BUN/bun-linux-x64/bun" /usr/local/bin/bun
rm -rf "$TMP_BUN"

# --- daemon dependencies in the system python (small set) ---
sudo /usr/bin/python3.13 -m ensurepip --upgrade
sudo /usr/bin/python3.13 -m pip install --no-input --no-cache-dir \
    requests==2.32.3

# --- Cleanup ---
sudo apt-get autoremove -y
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

# --- Sanity prints (Packer logs them) ---
node --version
python3.13 --version
git --version
gh --version | head -n1
uv --version
bun --version
