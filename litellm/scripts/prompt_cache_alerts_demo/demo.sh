#!/usr/bin/env bash
# Seeded prompt cache alerts demo. Usage: demo.sh up | run | reset | logs | down
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
checkout="$(git -C "$here" rev-parse --show-toplevel)"
DEMO_COMMIT="$(git -C "$checkout" rev-parse HEAD)"
export DEMO_COMMIT

compose() {
  docker compose --project-name litellm-prompt-cache-demo --file "$here/compose.yaml" "$@"
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
    compose up --build --detach --wait --wait-timeout 900 proxy
    echo "Proxy built from $DEMO_COMMIT is ready at http://localhost:${DEMO_PROXY_PORT:-4100}/ui (admin / sk-prompt-cache-demo)"
    ;;
  run)
    compose --profile driver run --rm driver
    ;;
  reset)
    compose --profile driver down --volumes --remove-orphans
    "$0" up
    ;;
  logs)
    compose logs --no-color --tail 200 proxy upstream notifier
    ;;
  down)
    compose --profile driver down --volumes --remove-orphans
    ;;
  *)
    echo "usage: $0 up | run | reset | logs | down" >&2
    exit 2
    ;;
esac
