#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# UI E2E Test Runner (Consolidated)
# Starts postgres, seeds DB, starts mock + proxy, runs Playwright.
# All tests target the proxy on port 4000 (which serves both API
# and UI from the built Next.js static export).
#
# Usage:
#   ./run_e2e.sh                    # Run once
#   ./run_e2e.sh --repeat-each=5    # Run each test 5 times
#   ./run_e2e.sh --headed           # Run with browser visible
#
# Ports default to 4000 / 5432 / 8090 and can be moved when another checkout
# already holds them:
#   PROXY_PORT=4100 POSTGRES_PORT=5532 MOCK_LLM_PORT=8190 ./run_e2e.sh
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
MOCK_PID=""
PROXY_PID=""
PROXY_LOG=""

# Ports, overridable so two checkouts can run this harness at the same time --
# otherwise a second run aborts on "port 4000 is in use" and the only way out is
# to stop someone else's stack. Defaults are the historical values, so an unset
# environment behaves exactly as before (CI, the CircleCI job and the docs all
# assume 4000/5432/8090).
PROXY_PORT="${PROXY_PORT:-4000}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
MOCK_LLM_PORT="${MOCK_LLM_PORT:-8090}"
export MOCK_LLM_PORT

# --- Ensure common tool paths are available (local dev only) ---
if [ "$IS_CI" = "false" ]; then
  for p in /usr/local/bin /opt/homebrew/bin "$HOME/.local/bin" /opt/homebrew/opt/postgresql@14/bin /opt/homebrew/opt/libpq/bin; do
    [ -d "$p" ] && export PATH="$p:$PATH"
  done
  # Sourcing nvm only makes `nvm` available -- it leaves you on whatever the
  # default alias points at, which is frequently an older Node than the
  # dashboard's engines allow. `npm install` then fails EBADENGINE, npm exits
  # non-zero, and because the install below is `--silent ... || true` the error
  # is swallowed and the run dies later with the far less obvious
  # "sh: next: command not found".
  #
  # So select a Node that satisfies ui/litellm-dashboard's engines.node, and if
  # none is available say so here rather than 200 lines downstream.
  if [ -s "$HOME/.nvm/nvm.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.nvm/nvm.sh"
    required_major="$(sed -nE 's/.*"node"[[:space:]]*:[[:space:]]*">=?([0-9]+).*/\1/p' \
      "$DASHBOARD_DIR/package.json" 2>/dev/null | head -1)"
    if [ -n "$required_major" ]; then
      current_major="$(node --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/')"
      if [ -z "$current_major" ] || [ "$current_major" -lt "$required_major" ]; then
        echo "Node $(node --version 2>/dev/null || echo 'not found') is below the dashboard's required v${required_major}; selecting a newer one via nvm"
        nvm use "$required_major" >/dev/null 2>&1 || nvm use --lts >/dev/null 2>&1 || true
        current_major="$(node --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/')"
        if [ -z "$current_major" ] || [ "$current_major" -lt "$required_major" ]; then
          echo "Error: ui/litellm-dashboard requires Node >= v${required_major}, and no such version is installed."
          echo "       Install one with:  nvm install ${required_major}"
          exit 1
        fi
      fi
      echo "Using Node $(node --version) / npm $(npm --version)"
    fi
  fi
fi

# --- Cleanup on exit ---
cleanup() {
  echo "Cleaning up..."
  [ -n "$MOCK_PID" ] && kill "$MOCK_PID" 2>/dev/null || true
  [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true
  [ -n "$PROXY_LOG" ] && rm -f "$PROXY_LOG" || true
  if [ "$IS_CI" = "false" ]; then
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
  fi
  echo "Done."
}
on_signal() {
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

# --- Pre-flight checks ---
for cmd in python3 npx uv; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found."; exit 1; }
done

# --- Database setup ---
if [ "$IS_CI" = "false" ]; then
  for cmd in docker psql; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found."; exit 1; }
  done
  # Only a LISTENER conflicts with us. Without -sTCP:LISTEN this also matches
  # ESTABLISHED sockets, so an unrelated *outbound* connection from this machine
  # to someone else's :5432 (a psql session, a running app, a Prisma engine
  # talking to a remote database) aborts the run with "port 5432 is in use"
  # while nothing is actually bound locally.
  for port in "$PROXY_PORT" "$POSTGRES_PORT" "$MOCK_LLM_PORT"; do
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "Error: port $port is in use (override with PROXY_PORT / POSTGRES_PORT / MOCK_LLM_PORT)"
      exit 1
    fi
  done

  export POSTGRES_USER="e2euser"
  export POSTGRES_PASSWORD="$(openssl rand -hex 32)"
  export POSTGRES_DB="litellm_e2e"
  export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:${POSTGRES_PORT}/${POSTGRES_DB}"

  echo "=== Starting PostgreSQL ==="
  docker run -d --rm --name "$CONTAINER_NAME" \
    -e POSTGRES_USER -e POSTGRES_PASSWORD -e POSTGRES_DB \
    -p "127.0.0.1:${POSTGRES_PORT}:5432" \
    postgres:16

  echo "Waiting for PostgreSQL..."
  for i in $(seq 1 30); do
    if PGPASSWORD="$POSTGRES_PASSWORD" pg_isready -h 127.0.0.1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
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
export MOCK_LLM_URL="http://127.0.0.1:${MOCK_LLM_PORT}/v1"
export DISABLE_SCHEMA_UPDATE="true"
# The suite resolves its target from E2E_UI_BASE_URL (constants.ts), which
# otherwise defaults to :4000 -- so without this a relocated stack would be
# built and booted correctly and then tested against whatever happens to be
# listening on the default port.
export E2E_UI_BASE_URL="${E2E_UI_BASE_URL:-http://127.0.0.1:${PROXY_PORT}}"
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
# NOT silenced, and NOT `|| true`. Swallowing this is what turns a one-line
# EBADENGINE ("dashboard requires node >=24, you have v20") into the
# considerably less helpful "sh: next: command not found" from the build below,
# because the deps that provide `next` were never installed.
npm install
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

# --- Mock LLM server ---
echo "=== Starting mock LLM server ==="
uv run --no-sync python "$SCRIPT_DIR/fixtures/mock_llm_server/server.py" &
MOCK_PID=$!

for i in $(seq 1 15); do
  if curl -sf http://127.0.0.1:${MOCK_LLM_PORT}/health >/dev/null 2>&1; then break; fi
  sleep 1
done

# --- LiteLLM proxy ---
echo "=== Starting LiteLLM proxy ==="
cd "$REPO_ROOT"
PROXY_LOG="${TMPDIR:-/tmp}/litellm-e2e-proxy-$$.log"
uv run --no-sync python -m litellm.proxy.proxy_cli \
  --config "$SCRIPT_DIR/fixtures/config.yml" \
  --port "$PROXY_PORT" >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!

echo "Waiting for proxy (logs: $PROXY_LOG)..."
PROXY_READY=0
for i in $(seq 1 180); do
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "Error: proxy process exited unexpectedly. Proxy output:"
    tail -n 100 "$PROXY_LOG"
    exit 1
  fi
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:${PROXY_PORT}/health -H "Authorization: Bearer $LITELLM_MASTER_KEY" 2>/dev/null || true)
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

# --- Seed database ---
echo "=== Seeding database ==="
DB_USER=$(echo "$DATABASE_URL" | sed -n 's|.*://\([^:]*\):.*|\1|p')
DB_PASS=$(echo "$DATABASE_URL" | sed -n 's|.*://[^:]*:\([^@]*\)@.*|\1|p')
DB_HOST=$(echo "$DATABASE_URL" | sed -n 's|.*@\([^:]*\):.*|\1|p')
DB_PORT=$(echo "$DATABASE_URL" | sed -n 's|.*:\([0-9]*\)/.*|\1|p')
DB_NAME=$(echo "$DATABASE_URL" | sed -n 's|.*/\([^?]*\).*|\1|p')

PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  -f "$SCRIPT_DIR/fixtures/seed.sql"

# --- Playwright ---
echo "=== Installing Playwright dependencies ==="
cd "$SCRIPT_DIR"
# Same reasoning as the dashboard install above: a failure here means the suite
# has no @playwright/test, and the run should say that rather than fail later.
npm install
npx playwright install chromium --with-deps 2>/dev/null || npx playwright install chromium

# Authoring a new spec means running it over and over against a stack that is
# already up -- rebuilding the UI and re-seeding for every iteration costs
# minutes each time. E2E_KEEP_ALIVE brings the stack up, then blocks, so you can
# run `npx playwright test <spec>` yourself from another shell against it.
# Ctrl-C here tears everything down through the usual trap.
if [ "${E2E_KEEP_ALIVE:-0}" = "1" ]; then
  cat <<EOF

=== Stack is up (E2E_KEEP_ALIVE=1); not running tests ===
  UI / API : http://127.0.0.1:${PROXY_PORT}
  Mock LLM : http://127.0.0.1:${MOCK_LLM_PORT}/v1
  Database : $DATABASE_URL
  Proxy log: $PROXY_LOG

Run specs against it from $SCRIPT_DIR:
  npx playwright test --config playwright.config.ts <spec>

Press Ctrl-C to tear the stack down.
EOF
  while kill -0 "$PROXY_PID" 2>/dev/null; do
    sleep 5
  done
  echo "Error: proxy process exited unexpectedly. Proxy output:"
  tail -n 100 "$PROXY_LOG"
  exit 1
fi

echo "=== Running Playwright tests ==="
npx playwright test --config playwright.config.ts "$@"
EXIT_CODE=$?

exit $EXIT_CODE
