#!/usr/bin/env bash
set -euo pipefail

root="${MCP_CONFORMANCE_ROOT:?Set MCP_CONFORMANCE_ROOT to a fresh directory}"
test ! -e "$root"
archive="$(mktemp)"
trap 'rm -f "$archive"' EXIT
curl --fail --location --silent --show-error \
  'https://codeload.github.com/modelcontextprotocol/conformance/tar.gz/7169291ec0b68eb370fddcd9947313ab0d5e4156' \
  --output "$archive"
PYTHONPATH=tests .venv/bin/python - "$archive" <<'PY'
import sys
from pathlib import Path
from integration._support.conformance import verify_archive

verify_archive(Path(sys.argv[1]), "51c1e27027f36be5b5f067746eb3a0f240fc3fbb7adcbd64bc4a79d86db65cd8")
PY
mkdir -p "$root"
tar -xzf "$archive" --strip-components=1 -C "$root"
npm ci --ignore-scripts --prefix "$root"
# The current reference misclassifies legacy initialize _meta as stateless traffic.
# Use the last official legacy reference; leave its source and lockfile untouched.
curl --fail --location --silent --show-error \
  'https://codeload.github.com/modelcontextprotocol/conformance/tar.gz/8f3994c75ff1aed1e39f91cff9358e2bc2c81dcd' \
  --output "$archive"
PYTHONPATH=tests .venv/bin/python - "$archive" <<'PYREF'
import sys
from pathlib import Path
from integration._support.conformance import verify_archive

verify_archive(Path(sys.argv[1]), "181119bd222f29db208b2952537430fcc865f91b64f2ac3d7525a67f490ebaa1")
PYREF
mkdir "$root/legacy-reference"
tar -xzf "$archive" --strip-components=1 -C "$root/legacy-reference"
npm ci --ignore-scripts --prefix "$root/legacy-reference/examples/servers/typescript"
