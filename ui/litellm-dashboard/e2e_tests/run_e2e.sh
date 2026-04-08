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
# In CI (CI=true), expects:
#   - PostgreSQL already running on 127.0.0.1:5432
#   - DATABASE_URL already set
#   - Python/Poetry already installed
#   - Node.js/npx already available
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
IS_CI="${CI:-false}"
CONTAINER_NAME="litellm-e2e-postgres-$$"
MOCK_PID=""
PROXY_PID=""

# --- Ensure common tool paths are available (local dev only) ---
if [ "$IS_CI" = "false" ]; then
  for p in /usr/local/bin /opt/homebrew/bin "$HOME/.local/bin" /opt/homebrew/opt/postgresql@14/bin /opt/homebrew/opt/libpq/bin; do
    [ -d "$p" ] && export PATH="$p:$PATH"
  done
  [ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"
fi

# --- Cleanup on exit ---
cleanup() {
  echo "Cleaning up..."
  [ -n "$MOCK_PID" ] && kill "$MOCK_PID" 2>/dev/null || true
  [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true
  if [ "$IS_CI" = "false" ]; then
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
  fi
  echo "Done."
}
trap cleanup EXIT INT TERM

# --- Pre-flight checks ---
for cmd in python3 npx poetry; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found."; exit 1; }
done

# --- Database setup ---
if [ "$IS_CI" = "false" ]; then
  for cmd in docker psql; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found."; exit 1; }
  done
  for port in 4000 5432 8090; do
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
export MOCK_LLM_URL="http://127.0.0.1:8090/v1"
export DISABLE_SCHEMA_UPDATE="true"
# Ensure the proxy serves UI at /ui (not behind a subpath)
export SERVER_ROOT_PATH=""
# Prevent logout from redirecting to an external URL
export PROXY_LOGOUT_URL=""

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
if ! poetry run python3 -c "import prisma" 2>/dev/null; then
  echo "Installing Python dependencies (first run)..."
  poetry install --with dev,proxy-dev --extras "proxy" --quiet
  poetry run pip install nodejs-wheel-binaries 2>/dev/null || true
  poetry run prisma generate --schema litellm/proxy/schema.prisma
fi

echo "=== Pushing Prisma schema to database ==="
poetry run prisma db push --schema litellm/proxy/schema.prisma --accept-data-loss

# --- Mock LLM server ---
echo "=== Starting mock LLM server ==="
poetry run python3 "$SCRIPT_DIR/fixtures/mock_llm_server/server.py" &
MOCK_PID=$!

for i in $(seq 1 15); do
  if curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1; then break; fi
  sleep 1
done

# --- LiteLLM proxy ---
echo "=== Starting LiteLLM proxy ==="
cd "$REPO_ROOT"
poetry run python3 -m litellm.proxy.proxy_cli \
  --config "$SCRIPT_DIR/fixtures/config.yml" \
  --port 4000 &
PROXY_PID=$!

echo "Waiting for proxy..."
PROXY_READY=0
for i in $(seq 1 180); do
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "Error: proxy process exited unexpectedly"
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
  echo "Error: proxy did not become healthy within 180 seconds"
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
cd "$DASHBOARD_DIR"
npm install --silent 2>/dev/null || true
npx playwright install chromium --with-deps 2>/dev/null || npx playwright install chromium

echo "=== Running Playwright tests ==="
npx playwright test --config e2e_tests/playwright.config.ts "$@"
EXIT_CODE=$?

exit $EXIT_CODE
