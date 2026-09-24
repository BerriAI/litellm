#!/usr/bin/env bash
# Downloads one Claude Code CLI release into a directory and verifies it.
#
#   install_claude_code.sh <version> <dest-dir>
#
# The binary comes from the native release channel the official
# installer (https://claude.ai/install.sh) reads, and its sha256 is
# checked against the `linux-x64` entry of that release's manifest.json
# before anything is executed (AGENTS.md, "CI supply-chain safety").
# The freshly downloaded binary then runs `--version` once, under a
# scrubbed environment so its first execution sees none of the run's
# provider keys, and the install fails unless it reports the requested
# version.

set -Eeuo pipefail

RELEASES_URL="https://downloads.claude.ai/claude-code-releases"

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# -eq 2 ]] || die "usage: $(basename "$0") <version> <dest-dir>"
VERSION="$1"
DEST_DIR="$2"
[[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "not a Claude Code release version: '${VERSION}'"

mkdir -p "${DEST_DIR}"
MANIFEST="${DEST_DIR}/manifest.json"
curl -fsSL --retry 3 --retry-all-errors --output "${MANIFEST}" "${RELEASES_URL}/${VERSION}/manifest.json" \
  || die "no release manifest for claude code ${VERSION} at ${RELEASES_URL}"
CHECKSUM="$(jq -r '.platforms["linux-x64"].checksum // empty' "${MANIFEST}")"
[[ "${CHECKSUM}" =~ ^[0-9a-f]{64}$ ]] || die "manifest for claude code ${VERSION} carries no linux-x64 sha256"

log "downloading claude code ${VERSION} (linux-x64)"
DOWNLOAD="${DEST_DIR}/claude.download"
curl -fsSL --retry 3 --retry-all-errors --output "${DOWNLOAD}" "${RELEASES_URL}/${VERSION}/linux-x64/claude"
echo "${CHECKSUM}  ${DOWNLOAD}" | sha256sum -c - >/dev/null \
  || die "claude code ${VERSION} sha256 mismatch; refusing to install"
chmod 0755 "${DOWNLOAD}"
mv "${DOWNLOAD}" "${DEST_DIR}/claude"

PROBE_HOME="$(mktemp -d -t claude-probe-home.XXXXXX)"
trap 'rm -rf "${PROBE_HOME}"' EXIT
REPORTED="$(
  env -i HOME="${PROBE_HOME}" PATH="${PATH}" DISABLE_AUTOUPDATER=1 \
    "${DEST_DIR}/claude" --version | awk '{print $1}'
)" || die "claude code ${VERSION} could not run --version"
[[ "${REPORTED}" == "${VERSION}" ]] \
  || die "installed claude code reports '${REPORTED}', expected ${VERSION}"
log "installed claude code ${VERSION} at ${DEST_DIR}/claude"
