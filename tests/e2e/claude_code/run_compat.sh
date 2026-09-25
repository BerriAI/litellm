#!/usr/bin/env bash
# Run the Claude Code compat matrix end-to-end against a live LiteLLM
# proxy, with per-provider rate limits applied via the cross-process
# token bucket in `tests/e2e/claude_code/rate_limiter.py`.
#
# Designed for binary-searching the ideal X / Y / Z req/s per provider:
#   1. Pick an initial rate (e.g. 5/s for everyone).
#   2. Run this script.
#   3. Read `compat-rate-limit-summary.json` to see whether any provider
#      hit a 429-shaped error during the run.
#   4. If a provider has `rate_limited > 0`, halve its rate; else, double it.
#   5. Repeat until the highest no-429 rate is found.
#
# Required env (proxy connection), same names as the rest of tests/e2e:
#   LITELLM_PROXY_URL        e.g. http://localhost:4000
#   LITELLM_MASTER_KEY       e.g. sk-1234
#
# Optional env (rate limits, all default to 5 req/s; 0 disables a column):
#   LITELLM_COMPAT_RATE_ANTHROPIC
#   LITELLM_COMPAT_RATE_AZURE
#   LITELLM_COMPAT_RATE_VERTEX_AI
#   LITELLM_COMPAT_RATE_BEDROCK_CONVERSE
#   LITELLM_COMPAT_RATE_BEDROCK_INVOKE
#   LITELLM_COMPAT_RATE_OPENAI
#   LITELLM_COMPAT_RATE_AZURE_OPENAI
#   LITELLM_COMPAT_RATE_BEDROCK_MANTLE
#   LITELLM_COMPAT_RATE_BURST            override per-bucket burst
#
# Optional env (GPT-5.6 columns):
#   COMPAT_MANTLE_CELLS=1                 opt the Bedrock Mantle GPT-5.6
#                                         cells in; without it they skip
#                                         and publish as not_tested
#
# Optional env (parallelism):
#   COMPAT_XDIST_WORKERS                  passed to `pytest -n` (default: auto)
#
# Optional env (artifacts):
#   COMPAT_RESULTS_PATH                   default: compat-results.json
#   COMPAT_RATE_LIMIT_SUMMARY_PATH        default: compat-rate-limit-summary.json

set -euo pipefail

if [[ -z "${LITELLM_PROXY_URL:-}" || -z "${LITELLM_MASTER_KEY:-}" ]]; then
    echo "error: LITELLM_PROXY_URL and LITELLM_MASTER_KEY must be set" >&2
    exit 64
fi

# Reset the cross-process rate-limiter state from any prior run. Stale
# token-bucket files would let a previous run's accumulated budget bleed
# into the new one, which subtly biases the binary search.
state_dir="${LITELLM_COMPAT_RATE_STATE_DIR:-${TMPDIR:-/tmp}/litellm-claude-compat-ratelimit}"
if [[ -d "$state_dir" ]]; then
    rm -rf "$state_dir"
fi

# Worker count. `auto` picks one worker per CPU; the rate limiter
# enforces aggregate provider rates regardless of worker count, so
# this is a "go as fast as the limiter allows" knob, not a tuning knob.
workers="${COMPAT_XDIST_WORKERS:-auto}"

# Where the artifacts land. We resolve them now so the summary file is
# always at a known path the caller can grep, even if they didn't set
# the env explicitly.
results_path="${COMPAT_RESULTS_PATH:-compat-results.json}"
summary_path="${COMPAT_RATE_LIMIT_SUMMARY_PATH:-compat-rate-limit-summary.json}"

echo "[run_compat] rates:"
for provider in ANTHROPIC AZURE VERTEX_AI BEDROCK_CONVERSE BEDROCK_INVOKE OPENAI AZURE_OPENAI BEDROCK_MANTLE; do
    var="LITELLM_COMPAT_RATE_${provider}"
    echo "  ${provider}=${!var:-default(5/s)}"
done
echo "  BURST=${LITELLM_COMPAT_RATE_BURST:-default(=rate)}"
echo "[run_compat] xdist workers: ${workers}"
echo "[run_compat] results:  ${results_path}"
echo "[run_compat] summary:  ${summary_path}"

# Run only the per-feature live tests; skip the unit-test directories
# (they're under directories starting with `_`). The dist=loadfile
# scheduler keeps each test file pinned to a single worker, which is
# what we want — every test in a file shares a single ThreadPoolExecutor
# fanout, and we don't gain anything by splitting it across workers.
start=$(date +%s)
set +e
COMPAT_RESULTS_PATH="${results_path}" \
COMPAT_RATE_LIMIT_SUMMARY_PATH="${summary_path}" \
PATH="$HOME/.local/bin:$PATH" \
uv run pytest \
    tests/e2e/claude_code/basic_messaging_non_streaming \
    tests/e2e/claude_code/basic_messaging_streaming \
    tests/e2e/claude_code/thinking \
    tests/e2e/claude_code/tool_use \
    tests/e2e/claude_code/vision \
    tests/e2e/claude_code/prompt_caching_5m \
    -n "${workers}" \
    --dist=loadfile \
    -q \
    "$@"

exit_code=$?
set -e
end=$(date +%s)
echo "[run_compat] wall time: $((end - start))s"

# Surface the rate-limit summary inline so a human reader doesn't have
# to `cat` the JSON file. The full file is still on disk for the binary
# search loop.
if [[ -f "${summary_path}" ]]; then
    echo "[run_compat] summary: ${summary_path}"
    if command -v jq >/dev/null 2>&1; then
        jq '.totals, .per_provider' "${summary_path}"
    else
        cat "${summary_path}"
    fi
fi

exit "${exit_code}"
