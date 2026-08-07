#!/usr/bin/env bash
set -euo pipefail

: "${ANTHROPIC_BASE_URL:?ANTHROPIC_BASE_URL is required}"
: "${ANTHROPIC_AUTH_TOKEN:?ANTHROPIC_AUTH_TOKEN is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"

OUTPUT_FILE="${CLAUDE_CODE_OUTPUT_FILE:-/tmp/claude-code-output.txt}"

echo "--- Test 1: Basic Request (Back-to-back 1) ---"
echo "Running Claude Code against ${ANTHROPIC_BASE_URL} with model ${MODEL_NAME}"
claude -p "Respond with exactly this text and nothing else: Hello from LiteLLM Claude Code Request 1." --model "${MODEL_NAME}" >"${OUTPUT_FILE}"

if [[ ! -s "${OUTPUT_FILE}" ]]; then
  echo "Claude Code produced no output for request 1."
  exit 1
fi
echo "Claude Code output 1:"
cat "${OUTPUT_FILE}"

echo "--- Test 2: Basic Request (Back-to-back 2) ---"
claude -p "Respond with exactly this text and nothing else: Hello from LiteLLM Claude Code Request 2." --model "${MODEL_NAME}" >"${OUTPUT_FILE}"

if [[ ! -s "${OUTPUT_FILE}" ]]; then
  echo "Claude Code produced no output for request 2."
  exit 1
fi
echo "Claude Code output 2:"
cat "${OUTPUT_FILE}"

echo "All back-to-back Claude Code integration tests passed."
