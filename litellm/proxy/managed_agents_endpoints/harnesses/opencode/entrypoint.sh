#!/usr/bin/env bash
set -euo pipefail

: "${REPO_URL:?REPO_URL required}"
: "${LITELLM_API_KEY:?LITELLM_API_KEY required}"
: "${LITELLM_API_BASE:?LITELLM_API_BASE required}"
: "${LITELLM_DEFAULT_MODEL:?LITELLM_DEFAULT_MODEL required}"

: "${BRANCH:=main}"
: "${PORT:=4096}"
: "${REPO_DIR:=/work/repo}"

# Normalize base URL: strip trailing slash, ensure /v1 suffix
BASE="${LITELLM_API_BASE%/}"
case "$BASE" in
  */v1) ;;
  *) BASE="${BASE}/v1" ;;
esac

# Clone. Token (if present) is fed via stdin to credential helper so it never
# lands in argv, env of child processes, .git/config, or shell history.
if [ ! -d "$REPO_DIR/.git" ]; then
  if [ -n "${GIT_TOKEN:-}" ]; then
    git -c credential.helper= \
        -c "credential.helper=!f() { echo username=x-access-token; echo password=$GIT_TOKEN; }; f" \
        clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
  else
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
  fi
fi

# Wipe token from env so opencode shell tool can't `printenv GIT_TOKEN`.
unset GIT_TOKEN

cd "$REPO_DIR"

# Belt-and-suspenders: ensure .git/config has clean remote (no embedded creds).
git remote set-url origin "$REPO_URL" 2>/dev/null || true

# Wire LiteLLM as OpenAI-compatible provider
cat > opencode.json <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "${BASE}",
        "apiKey": "${LITELLM_API_KEY}"
      },
      "models": {
        "${LITELLM_DEFAULT_MODEL}": {}
      }
    }
  },
  "model": "litellm/${LITELLM_DEFAULT_MODEL}"
}
EOF

if [ -n "${AGENT_PROMPT:-}" ]; then
  mkdir -p .opencode/agent
  cat > .opencode/agent/default.md <<EOF
---
description: sandbox agent
---
${AGENT_PROMPT}
EOF
fi

echo "[entrypoint] booting opencode serve on 0.0.0.0:${PORT}"
echo "[entrypoint] base=${BASE} model=${LITELLM_DEFAULT_MODEL} repo=${REPO_DIR}"

exec opencode serve --hostname 0.0.0.0 --port "$PORT"
