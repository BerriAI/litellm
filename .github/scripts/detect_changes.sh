#!/usr/bin/env bash
set -uo pipefail

readonly API_FILE_CEILING=3000
readonly CATEGORY="${CATEGORY:-backend}"

decide() {
  echo "detect-changes[${CATEGORY}]: decision=$1"
  [ -z "${GITHUB_OUTPUT:-}" ] || echo "decision=$1" >>"${GITHUB_OUTPUT}"
  exit 0
}

run_full() {
  echo "detect-changes[${CATEGORY}]: $1; running job"
  decide run
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
classify="${here}/../../.circleci/scripts/classify_changes.sh"

[ -n "${PR_NUMBER:-}" ] || run_full "not a pull_request event"
[ -n "${REPO:-}" ] || run_full "no repository in the environment"

case "${CHANGED_FILE_COUNT:-}" in
'' | *[!0-9]*) run_full "the event payload carries no changed_files count" ;;
esac
[ "${CHANGED_FILE_COUNT}" -le "${API_FILE_CEILING}" ] ||
  run_full "PR #${PR_NUMBER} changes ${CHANGED_FILE_COUNT} files, past the ${API_FILE_CEILING}-file listing ceiling"

changed="$(gh api "repos/${REPO}/pulls/${PR_NUMBER}/files" --paginate --jq '.[].filename')" ||
  run_full "could not list the files on PR #${PR_NUMBER}"
[ -n "${changed}" ] || run_full "the API listed no files on PR #${PR_NUMBER}"

echo "detect-changes[${CATEGORY}]: files changed by PR #${PR_NUMBER}:"
printf '%s\n' "${changed}" | sed 's/^/  /'

decision="$(printf '%s\n' "${changed}" | bash "${classify}" "${CATEGORY}")" ||
  run_full "classify_changes.sh failed"
case "${decision}" in
run | skip) decide "${decision}" ;;
*) run_full "classify_changes.sh printed an unexpected decision: ${decision}" ;;
esac
