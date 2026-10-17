#!/usr/bin/env bash
# Sync the environment without building litellm, then install the editable Rust
# bridge that .github/actions/rust-bridge put in rust-bridge-dist.
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <uv sync args...>" >&2
  exit 2
fi

uv sync "$@" --no-install-project
venv_python=.venv/bin/python
[[ -x "${venv_python}" ]] || venv_python=.venv/Scripts/python.exe
uv pip install --no-deps --python "${venv_python}" rust-bridge-dist/*.whl
cp rust-bridge-dist/litellm/rust_bridge/_native.* litellm/rust_bridge/
uv run --no-sync python -c "import importlib.metadata, litellm.rust_bridge._native; print(importlib.metadata.version('litellm'))"
