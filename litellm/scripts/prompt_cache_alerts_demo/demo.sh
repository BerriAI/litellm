#!/usr/bin/env bash
# Seeded prompt cache alerts demo. Usage: demo.sh up | run | reset | logs | down
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
checkout="$(git -C "$here" rev-parse --show-toplevel)"
DEMO_COMMIT="$(git -C "$checkout" rev-parse HEAD)"
state_dir="${DEMO_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/litellm-prompt-cache-demo}"
DEMO_ENV_FILE="$state_dir/demo.env"
export DEMO_COMMIT DEMO_ENV_FILE

compose() {
  docker compose --project-name litellm-prompt-cache-demo --file "$here/compose.yaml" "$@"
}

random_hex() {
  od -An -tx1 -N32 /dev/urandom | tr -d ' \n'
}

ensure_credentials() {
  if [ -f "$DEMO_ENV_FILE" ]; then
    return
  fi
  (
    umask 077
    mkdir -p "$state_dir"
    master_key="sk-$(random_hex)"
    {
      printf 'LITELLM_MASTER_KEY=%s\n' "$master_key"
      printf 'UI_PASSWORD=%s\n' "$master_key"
      printf 'LITELLM_SALT_KEY=%s\n' "$(random_hex)"
    } > "$DEMO_ENV_FILE"
  )
}

forget_credentials() {
  rm -f "$DEMO_ENV_FILE"
}

require_clean_checkout() {
  if [ -n "$(git -C "$checkout" status --porcelain)" ]; then
    echo "The checkout has uncommitted changes, so the image would not match $DEMO_COMMIT. Commit or stash them first." >&2
    exit 1
  fi
}

case "${1:-}" in
  up)
    require_clean_checkout
    ensure_credentials
    compose up --build --detach --wait --wait-timeout 900 proxy
    echo "Proxy built from $DEMO_COMMIT is ready at http://localhost:${DEMO_PROXY_PORT:-4100}/ui"
    echo "Sign in as admin with the UI_PASSWORD in $DEMO_ENV_FILE"
    ;;
  run)
    compose --profile driver run --rm driver
    ;;
  reset)
    ensure_credentials
    compose --profile driver down --volumes --remove-orphans
    forget_credentials
    "$0" up
    ;;
  logs)
    ensure_credentials
    compose logs --no-color --tail 200 proxy upstream notifier
    ;;
  down)
    ensure_credentials
    compose --profile driver down --volumes --remove-orphans
    forget_credentials
    ;;
  *)
    echo "usage: $0 up | run | reset | logs | down" >&2
    exit 2
    ;;
esac
