#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <bundle-dir> <out-tarball>" >&2
  exit 2
fi

BUNDLE_DIR="$1"
OUT_TARBALL="$2"

MANIFEST="${BUNDLE_DIR}/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "no ${MANIFEST}: refusing to publish a bundle with no manifest (record produced nothing)" >&2
  exit 1
fi

echo "packing fixture bundle from ${BUNDLE_DIR}"
python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print('  format_version', m['format_version'], 'recorded_at', m['recorded_at'], 'harness', m['harness_version'])" "${MANIFEST}"

TEST_DIRS=$(find "${BUNDLE_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
if [[ "${TEST_DIRS}" -eq 0 ]]; then
  echo "bundle at ${BUNDLE_DIR} has a manifest but no recorded interactions; refusing to publish an empty bundle" >&2
  exit 1
fi
echo "  ${TEST_DIRS} recorded test director(ies)"

mkdir -p "$(dirname "${OUT_TARBALL}")"
tar czf "${OUT_TARBALL}" -C "${BUNDLE_DIR}" .

OUT_DIR="$(cd "$(dirname "${OUT_TARBALL}")" && pwd)"
OUT_BASE="$(basename "${OUT_TARBALL}")"
( cd "${OUT_DIR}" && sha256sum "${OUT_BASE}" > "${OUT_BASE}.sha256" )

echo "wrote ${OUT_TARBALL} ($(du -h "${OUT_TARBALL}" | cut -f1)) and ${OUT_BASE}.sha256"
cat "${OUT_DIR}/${OUT_BASE}.sha256"
