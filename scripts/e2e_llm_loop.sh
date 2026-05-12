#!/usr/bin/env bash
# e2e_llm_loop.sh — run e2e LLM-translation tests with coverage (and
# optionally mutation) scoped to ONE provider's source dir.
#
# Drives the loop described in tests/llm_translation/E2E_AGENT.md.
#
# Usage:
#   scripts/e2e_llm_loop.sh coverage <provider> [--fail-under N] [pytest-args...]
#   scripts/e2e_llm_loop.sh mutation <provider> [mutmut-args...]
#
# Example:
#   scripts/e2e_llm_loop.sh coverage anthropic --fail-under 90
#   scripts/e2e_llm_loop.sh mutation anthropic
#
# Provider name = subdirectory of litellm/llms/ (e.g. anthropic, openai,
# bedrock). Test files are auto-discovered from
# tests/llm_translation/test_<provider>*.py.
#
# Coverage scope is litellm/llms/<provider>/ only — everything else is
# excluded from the report so the threshold reflects the provider's
# translation code, not the whole repo.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

mode="${1:-}"
provider="${2:-}"

if [[ -z "${mode}" || -z "${provider}" ]]; then
  echo "Usage: $0 {coverage|mutation} <provider> [extra args...]" >&2
  exit 2
fi
shift 2

PROVIDER_SRC="litellm/llms/${provider}"
if [[ ! -d "${PROVIDER_SRC}" ]]; then
  echo "No source dir: ${PROVIDER_SRC}" >&2
  exit 2
fi

# Test files: test_<provider>*.py under tests/llm_translation/. Glob may
# expand to multiple files; that's fine for pytest.
shopt -s nullglob
test_files=( tests/llm_translation/test_${provider}*.py )
shopt -u nullglob
if [[ ${#test_files[@]} -eq 0 ]]; then
  echo "No test files matching tests/llm_translation/test_${provider}*.py" >&2
  exit 2
fi

case "${mode}" in
  coverage)
    fail_under=90
    pytest_extra=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --fail-under)
          fail_under="$2"
          shift 2
          ;;
        --fail-under=*)
          fail_under="${1#*=}"
          shift
          ;;
        *)
          pytest_extra+=( "$1" )
          shift
          ;;
      esac
    done

    out_dir="e2e_llm_coverage/${provider}"
    mkdir -p "${out_dir}"

    echo "==> Coverage run: provider=${provider} fail_under=${fail_under}"
    echo "    Tests:        ${test_files[*]}"
    echo "    Source scope: ${PROVIDER_SRC}"

    uv run --no-sync pytest \
      "${test_files[@]}" \
      --cov="${PROVIDER_SRC}" \
      --cov-branch \
      --cov-report="term-missing:skip-covered" \
      --cov-report="html:${out_dir}/html" \
      --cov-report="xml:${out_dir}/coverage.xml" \
      --cov-fail-under="${fail_under}" \
      "${pytest_extra[@]}"
    ;;

  mutation)
    out_dir="e2e_llm_mutation/${provider}"
    mkdir -p "${out_dir}"

    echo "==> Mutation run: provider=${provider}"
    echo "    Tests:        ${test_files[*]}"
    echo "    Mutate scope: ${PROVIDER_SRC}"
    echo "    Output:       ${out_dir}"
    echo ""
    echo "NOTE: mutmut config in pyproject.toml [tool.mutmut] is the"
    echo "      default scope (proxy/management_endpoints). This script"
    echo "      overrides it via env vars MUTMUT_PATHS / MUTMUT_TESTS"
    echo "      consumed by the wrapper below."

    # mutmut reads paths_to_mutate / tests_dir from pyproject.toml. We
    # override at invocation by exporting and using --paths-to-mutate.
    uv run --no-sync --with mutmut==3.3.1 mutmut run \
      --paths-to-mutate="${PROVIDER_SRC}" \
      --tests-dir="tests/llm_translation" \
      "$@"

    # Persist a JSON report alongside coverage output.
    uv run --no-sync --with mutmut==3.3.1 mutmut results > "${out_dir}/results.txt" || true
    echo "Wrote ${out_dir}/results.txt"
    ;;

  *)
    echo "Unknown mode: ${mode}" >&2
    echo "Usage: $0 {coverage|mutation} <provider> [extra args...]" >&2
    exit 2
    ;;
esac
