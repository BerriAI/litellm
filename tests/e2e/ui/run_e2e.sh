#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# UI E2E Test Runner
#
# Two modes:
#   1. LITELLM_PROXY_URL set: run Playwright against that live gateway
#      (the same one the tests/e2e python suites target). globalSetup
#      seeds all e2e-* users/teams/keys/models through the management
#      API with LITELLM_MASTER_KEY, so the gateway only needs a master
#      key and a database (store_model_in_db: true).
#   2. LITELLM_PROXY_URL unset: provision a disposable gateway first
#      (dockerized postgres, dashboard built from source, proxy on
#      port 4000), then run the suite against it.
#
# Usage:
#   ./run_e2e.sh                                   # provision + run
#   LITELLM_PROXY_URL=http://localhost:4000 ./run_e2e.sh   # reuse a gateway
#   ./run_e2e.sh --repeat-each=5                   # extra args go to playwright
#
# In CI (CI=true), expects:
#   - PostgreSQL already running on 127.0.0.1:5432
#   - DATABASE_URL already set
#   - Python/uv already installed
#   - Node.js/npx already available
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DASHBOARD_DIR="$REPO_ROOT/ui/litellm-dashboard"
IS_CI="${CI:-false}"
CONTAINER_NAME="litellm-e2e-postgres-$$"
PROXY_PID=""
PROXY_LOG=""

# --- Ensure common tool paths are available (local dev only) ---
if [ "$IS_CI" = "false" ]; then
  for p in /usr/local/bin /opt/homebrew/bin "$HOME/.local/bin" /opt/homebrew/opt/postgresql@14/bin /opt/homebrew/opt/libpq/bin; do
    [ -d "$p" ] && export PATH="$p:$PATH"
  done
  [ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"
fi

# Provider keys for the real-model deployments globalSetup registers via
# /model/new; the shared harness .env is the canonical place they live.
if [ -f "$REPO_ROOT/tests/e2e/.env" ]; then
  set -a
  source "$REPO_ROOT/tests/e2e/.env"
  set +a
fi

run_playwright() {
  echo "=== Installing Playwright dependencies ==="
  cd "$SCRIPT_DIR"
  npm install --silent 2>/dev/null || true
  npx playwright install chromium --with-deps 2>/dev/null || npx playwright install chromium

  echo "=== Running Playwright tests ==="
  npx playwright test --config playwright.config.ts "$@"
}

# --- Mode 1: reuse an already-running gateway ---
if [ -n "${LITELLM_PROXY_URL:-}" ]; then
  echo "=== Using existing gateway at $LITELLM_PROXY_URL ==="
  export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-1234}"
  curl -fs "$LITELLM_PROXY_URL/health/liveliness" >/dev/null || {
    echo "Error: no live proxy at $LITELLM_PROXY_URL"
    exit 1
  }
  run_playwright "$@"
  exit $?
fi

# --- Mode 2: provision a disposable gateway ---
cleanup() {
  echo "Cleaning up..."
  [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true
  [ -n "$PROXY_LOG" ] && rm -f "$PROXY_LOG" || true
  if [ "$IS_CI" = "false" ]; then
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
  fi
  echo "Done."
}
trap cleanup EXIT INT TERM

# --- Pre-flight checks ---
for cmd in python3 npx uv; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found."; exit 1; }
done

# --- Database setup ---
if [ "$IS_CI" = "false" ]; then
  for cmd in docker; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found."; exit 1; }
  done
  for port in 4000 5432; do
    if lsof -ti ":$port" >/dev/null 2>&1; then
      echo "Error: port $port is in use"
      exit 1
    fi
  done

  export POSTGRES_USER="e2euser"
  export POSTGRES_PASSWORD="$(openssl rand -hex 32)"
  export POSTGRES_DB="litellm_e2e"
  export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:5432/${POSTGRES_DB}"

  echo "=== Starting PostgreSQL ==="
  docker run -d --rm --name "$CONTAINER_NAME" \
    -e POSTGRES_USER -e POSTGRES_PASSWORD -e POSTGRES_DB \
    -p 127.0.0.1:5432:5432 \
    postgres:16

  echo "Waiting for PostgreSQL..."
  for i in $(seq 1 30); do
    if docker exec "$CONTAINER_NAME" pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
else
  echo "=== Using CI PostgreSQL service ==="
  : "${DATABASE_URL:?DATABASE_URL must be set in CI}"
fi

# --- Credentials ---
export LITELLM_MASTER_KEY="sk-1234"
export DISABLE_SCHEMA_UPDATE="true"
# Ensure the proxy serves UI at /ui (not behind a subpath)
export SERVER_ROOT_PATH=""
# Boot with an external logout URL so proxyLogoutUrl.spec.ts can assert the
# redirect. This same value is exported to the Playwright process below (the
# spec's skip guard reads it). Safe for the rest of the suite — nothing else
# performs a logout.
export PROXY_LOGOUT_URL="https://www.example.com"
# Forward LITELLM_LICENSE if set in the outer env so premium-gated UI flows
# (e.g. Team-BYOK Model switch) can be exercised. Tests that depend on a
# premium proxy gate themselves on process.env.LITELLM_LICENSE.
export LITELLM_LICENSE="${LITELLM_LICENSE:-}"

# --- Rebuild UI from source ---
echo "=== Building UI from source ==="
cd "$DASHBOARD_DIR"
npm install --silent 2>/dev/null || true
npm run build
# Copy the fresh build to the proxy's static UI directory
cp -r "$DASHBOARD_DIR/out/" "$REPO_ROOT/litellm/proxy/_experimental/out/"

# Restructure HTML files so extensionless routes work (e.g. /ui/login)
# Next.js export produces login.html; the proxy expects login/index.html
find "$REPO_ROOT/litellm/proxy/_experimental/out" -name '*.html' ! -name 'index.html' | while read -r htmlfile; do
  target_dir="${htmlfile%.html}"
  target_path="$target_dir/index.html"
  mkdir -p "$target_dir"
  mv "$htmlfile" "$target_path"
done
echo "UI build copied and restructured"

# --- Python environment ---
echo "=== Setting up Python environment ==="
cd "$REPO_ROOT"
export UV_PYTHON="${UV_PYTHON:-3.13}"
uv sync --group dev --group proxy-dev --extra proxy --frozen --quiet
uv run --no-sync python -m prisma generate --schema litellm/proxy/schema.prisma

echo "=== Pushing Prisma schema to database ==="
uv run --no-sync python -m prisma db push --schema litellm/proxy/schema.prisma --accept-data-loss

# --- LiteLLM proxy ---
echo "=== Starting LiteLLM proxy ==="
cd "$REPO_ROOT"
PROXY_LOG="${TMPDIR:-/tmp}/litellm-e2e-proxy-$$.log"
uv run --no-sync python -m litellm.proxy.proxy_cli \
  --config "$SCRIPT_DIR/fixtures/config.yml" \
  --port 4000 >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!

echo "Waiting for proxy (logs: $PROXY_LOG)..."
PROXY_READY=0
for i in $(seq 1 180); do
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "Error: proxy process exited unexpectedly. Proxy output:"
    tail -n 100 "$PROXY_LOG"
    exit 1
  fi
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:4000/health -H "Authorization: Bearer $LITELLM_MASTER_KEY" 2>/dev/null || true)
  if [ "$HTTP_CODE" = "200" ]; then
    PROXY_READY=1
    break
  fi
  sleep 1
done
if [ "$PROXY_READY" -ne 1 ]; then
  echo "Error: proxy did not become healthy within 180 seconds. Proxy output:"
  tail -n 100 "$PROXY_LOG"
  exit 1
fi
echo "Proxy is ready."

run_playwright "$@"
exit $?
