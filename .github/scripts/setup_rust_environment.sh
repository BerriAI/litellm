#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
toolchain=$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$root/litellm/rust/rust-toolchain.toml")
test -n "$toolchain"
rustup toolchain install "$toolchain" --profile minimal --component rustfmt --component clippy --no-self-update
{
  echo "RUSTUP_TOOLCHAIN=$toolchain"
  echo "CARGO_NET_RETRY=5"
  echo "CARGO_HTTP_MULTIPLEXING=false"
  echo "SQLX_OFFLINE=${SQLX_OFFLINE:-true}"
} >> "${GITHUB_ENV:?GITHUB_ENV is required}"
