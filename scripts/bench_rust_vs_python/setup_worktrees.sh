#!/usr/bin/env bash
# Usage: setup_worktrees.sh <worktree root> <name>=<sha> [<name>=<sha> ...]
set -euo pipefail

root="$1"
shift
repo="$(git rev-parse --show-toplevel)"
python_version="${BENCH_PYTHON:-3.12}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/target}"
export RUSTFLAGS="-C force-frame-pointers=yes"
export CARGO_PROFILE_RELEASE_DEBUG=line-tables-only
export CARGO_PROFILE_RELEASE_STRIP=none

mkdir -p "$root"
for spec in "$@"; do
  name="${spec%%=*}"
  sha="${spec#*=}"
  tree="$root/$name"
  if [ ! -d "$tree" ]; then
    git -C "$repo" worktree add --detach "$tree" "$sha"
  fi
  [ "$(git -C "$tree" rev-parse HEAD)" = "$(git -C "$repo" rev-parse "$sha^{commit}")" ] || {
    echo "$tree is not at $sha" >&2
    exit 1
  }
  (
    cd "$tree"
    UV_PROJECT_ENVIRONMENT="$tree/.venv" uv sync --frozen --no-install-project --python "$python_version" \
      --extra proxy
    locked() { grep -A1 "^name = \"$1\"$" uv.lock | sed -n 's/^version = "\(.*\)"$/\1/p'; }
    uv pip install --python "$tree/.venv/bin/python" "maturin==1.15.0" \
      "prometheus-client==$(locked prometheus-client)" "boto3==$(locked boto3)"
    VIRTUAL_ENV="$tree/.venv" "$tree/.venv/bin/maturin" develop --release --uv
    if [ -d "$tree/litellm/cli" ] && [ -f "$tree/.github/packaging/cli_package.py" ]; then
      site="$("$tree/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
      "$tree/.venv/bin/python" -c '
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/.github/packaging")
from cli_package import editable_support
for name, data in editable_support(Path(sys.argv[1]) / "litellm" / "cli"):
    (Path(sys.argv[2]) / name).write_bytes(data)
' "$tree" "$site"
      "$tree/.venv/bin/python" -c "import litellm_cli.main"
    fi
    "$tree/.venv/bin/python" -c "import litellm.rust_bridge._native"
  )
  echo "ready: $name $sha"
done
