#!/usr/bin/env bash
set -euo pipefail

: "${ANTHROPIC_BASE_URL:?ANTHROPIC_BASE_URL is required}"
: "${ANTHROPIC_AUTH_TOKEN:?ANTHROPIC_AUTH_TOKEN is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"

OUTPUT_FILE="${CLAUDE_CODE_OUTPUT_FILE:-/tmp/claude-code-output.txt}"
PROMPT="${CLAUDE_CODE_PROMPT:-Respond with exactly this text and nothing else: Hello from LiteLLM Claude Code.}"

echo "Running Claude Code against ${ANTHROPIC_BASE_URL} with model ${MODEL_NAME}"

claude -p "${PROMPT}" --model "${MODEL_NAME}" >"${OUTPUT_FILE}"

if [[ ! -s "${OUTPUT_FILE}" ]]; then
  echo "Claude Code produced no output."
  exit 1
fi

echo "Claude Code output:"
sed -n '1,20p' "${OUTPUT_FILE}"
