#!/bin/sh
set -eu

CHARTSNAP_VERSION="v0.6.0"
HELM_IMAGE="ghcr.io/appuio/helm-v4:4.1.0"
CHART_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${CHARTSNAP_WORK_DIR:-$PWD}"

if ! command -v docker >/dev/null 2>&1; then
  echo "chartsnap runs inside ${HELM_IMAGE} (pinned helm + locale) so snapshots are" >&2
  echo "identical across hosts, and docker is required. Install docker, then re-run." >&2
  exit 1
fi

exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HELM_DATA_HOME=/tmp/helm \
  -e HOME=/tmp \
  -e LC_ALL=C.UTF-8 \
  -e LANG=C.UTF-8 \
  -v "$WORK_DIR":"$WORK_DIR" \
  -v "$CHART_DIR":"$CHART_DIR" \
  -w "$WORK_DIR" \
  --entrypoint sh \
  "$HELM_IMAGE" \
  -c "helm plugin install https://github.com/jlandowner/helm-chartsnap --version ${CHARTSNAP_VERSION} --verify=false >/dev/null 2>&1 || true; exec helm chartsnap \"\$@\"" -- "$@"