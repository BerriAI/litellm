#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-${GITHUB_REPOSITORY:?REPO required}}"
ARTIFACT_NAME="${2:-e2e-fixtures-bundle}"
BASE_BRANCH="${3:?base branch required}"
DEST_DIR="${4:?destination bundle dir required}"

: "${GH_TOKEN:?GH_TOKEN required to query and download artifacts}"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

echo "resolving newest non-expired '${ARTIFACT_NAME}' artifact on ${REPO}@${BASE_BRANCH}"

SELECTED="$(
  gh api "repos/${REPO}/actions/artifacts" -X GET -f per_page=100 --paginate \
    --jq ".artifacts[] | select(.name == \"${ARTIFACT_NAME}\" and .expired == false and .workflow_run.head_branch == \"${BASE_BRANCH}\") | {id, digest, created_at, run_id: .workflow_run.id, run_number: .workflow_run.run_number}" \
    | jq -s 'sort_by(.created_at) | reverse | .[0] // empty'
)"

if [[ -z "${SELECTED}" ]]; then
  echo "no usable '${ARTIFACT_NAME}' artifact on ${BASE_BRANCH}: the last record run produced none (a red Saturday), so there is nothing fresh to replay; failing loudly instead of replaying a stale bundle" >&2
  exit 1
fi

RUN_ID="$(echo "${SELECTED}" | jq -r '.run_id')"
RUN_NUMBER="$(echo "${SELECTED}" | jq -r '.run_number')"
ARTIFACT_ID="$(echo "${SELECTED}" | jq -r '.id')"
GH_DIGEST="$(echo "${SELECTED}" | jq -r '.digest // "unknown"')"
CREATED_AT="$(echo "${SELECTED}" | jq -r '.created_at')"

echo "pinned bundle: run #${RUN_NUMBER} (run_id=${RUN_ID}, artifact_id=${ARTIFACT_ID}), recorded ${CREATED_AT}, github digest ${GH_DIGEST}"

gh run download "${RUN_ID}" --repo "${REPO}" -n "${ARTIFACT_NAME}" -D "${WORKDIR}"

TARBALL="$(find "${WORKDIR}" -name '*.tar.gz' -type f | head -n 1)"
if [[ -z "${TARBALL}" ]]; then
  echo "downloaded artifact contained no tarball" >&2
  exit 1
fi
SIDECAR="${TARBALL}.sha256"
if [[ ! -f "${SIDECAR}" ]]; then
  echo "downloaded artifact has no ${SIDECAR}: cannot verify the bundle digest" >&2
  exit 1
fi

echo "verifying bundle against its recorded sha256 digest"
( cd "$(dirname "${TARBALL}")" && sha256sum -c "$(basename "${SIDECAR}")" )

mkdir -p "${DEST_DIR}"
tar xzf "${TARBALL}" -C "${DEST_DIR}"

echo "extracted bundle into ${DEST_DIR}"
python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print('  recorded_at', m['recorded_at'], 'harness', m['harness_version'], 'format_version', m['format_version'])" "${DEST_DIR}/manifest.json"
