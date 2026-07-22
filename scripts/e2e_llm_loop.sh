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

# Test files: test_<provider>.py and test_<provider>_*.py under
# tests/llm_translation/. The two patterns avoid cross-matching unrelated
# providers that share a name prefix (e.g. azure vs azure_ai, bedrock vs
# bedrock_mantle, openai vs openai_like).
shopt -s nullglob
test_files=(
  tests/llm_translation/test_${provider}.py
  tests/llm_translation/test_${provider}_*.py
)
shopt -u nullglob
if [[ ${#test_files[@]} -eq 0 ]]; then
  echo "No test files matching tests/llm_translation/test_${provider}.py or test_${provider}_*.py" >&2
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
      ${pytest_extra[@]+"${pytest_extra[@]}"}
    ;;

  mutation)
    out_dir="e2e_llm_mutation/${provider}"
    mkdir -p "${out_dir}"

    echo "==> Mutation run: provider=${provider}"
    echo "    Tests:        ${test_files[*]}"
    echo "    Mutate scope: ${PROVIDER_SRC}"
    echo "    Output:       ${out_dir}"
    echo ""
    echo "NOTE: mutmut 3.x reads paths_to_mutate / tests_dir from"
    echo "      pyproject.toml [tool.mutmut] and no longer accepts those"
    echo "      as CLI flags. This script temporarily rewrites the"
    echo "      [tool.mutmut] section for the duration of the run and"
    echo "      restores the original file on exit."

    # mutmut 3.x removed --paths-to-mutate / --tests-dir; all scope config
    # must come from pyproject.toml. Back up the file, rewrite the
    # [tool.mutmut] section in-place, and restore it (even on error) so
    # the default proxy/management_endpoints scope is preserved for CI.
    pyproject_backup="$(mktemp)"
    cp pyproject.toml "${pyproject_backup}"
    trap 'mv "${pyproject_backup}" pyproject.toml' EXIT

    # Newline-delimited list of provider-scoped test files, consumed by the
    # python heredoc below. Using an env var avoids quoting headaches when
    # paths are interpolated into a TOML array.
    TEST_FILES="$(printf '%s\n' "${test_files[@]}")" \
    PROVIDER_SRC="${PROVIDER_SRC}" uv run --no-sync python - <<'PYEOF'
import json
import os
import re

provider_src = os.environ["PROVIDER_SRC"]
test_files = [p for p in os.environ["TEST_FILES"].splitlines() if p]
with open("pyproject.toml", "r") as f:
    content = f.read()

test_selection = ", ".join(json.dumps(p) for p in test_files)
new_section = (
    "[tool.mutmut]\n"
    f'paths_to_mutate = ["{provider_src}/"]\n'
    f'pytest_add_cli_args_test_selection = [{test_selection}]\n'
    'also_copy = ["litellm/"]\n'
    'pytest_add_cli_args = ["-p", "no:retry", "-p", "no:rerunfailures", "-p", "no:xdist"]\n'
)

# Replace the existing [tool.mutmut] section (up to but not including the
# next top-level [section] header or end of file).
pattern = re.compile(r"\[tool\.mutmut\].*?(?=\n\[[^\]]+\]|\Z)", re.DOTALL)
if not pattern.search(content):
    raise SystemExit("Could not find [tool.mutmut] section in pyproject.toml")
content = pattern.sub(new_section.rstrip("\n"), content, count=1)

with open("pyproject.toml", "w") as f:
    f.write(content)
PYEOF

    uv run --no-sync --with mutmut==3.3.1 mutmut run "$@"

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
