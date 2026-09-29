#!/usr/bin/env bash
# Wrapper script for running the Google Drive MCP server in Docker.
# Reuses an existing container instead of creating a new one each time.
#
# Usage:
#   ./scripts/docker-mcp.sh [container-name]
#
# The container name defaults to "google-drive-mcp".

set -euo pipefail

CONTAINER_NAME="${1:-google-drive-mcp}"
IMAGE_NAME="google-drive-mcp"

# Verify the image exists before attempting to create a container
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Image '$IMAGE_NAME' not found. Build it first: docker build -t $IMAGE_NAME ." >&2
  exit 1
fi

STATE="$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)" || STATE="missing"

# If a container exists, check whether its image is still current.
# When the user rebuilds the image (e.g. to test a feature), the old
# container would keep running stale code.  Detect the mismatch and
# replace the container so the new image is actually used.
if [ "$STATE" != "missing" ]; then
  CONTAINER_IMAGE="$(docker inspect --format '{{.Image}}' "$CONTAINER_NAME" 2>/dev/null)" || CONTAINER_IMAGE=""
  CURRENT_IMAGE="$(docker image inspect --format '{{.Id}}' "$IMAGE_NAME" 2>/dev/null)" || CURRENT_IMAGE=""
  if [ -n "$CURRENT_IMAGE" ] && [ "$CONTAINER_IMAGE" != "$CURRENT_IMAGE" ]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
    STATE="missing"
  fi
fi

case "$STATE" in
  running)
    # Container already running — fall through to the readiness check below
    ;;
  exited|created)
    # Container exists but stopped — start it
    docker start "$CONTAINER_NAME" >/dev/null 2>&1
    ;;
  missing)
    # No container yet — create one in the background with sleep so it stays alive
    docker run -d \
      --name "$CONTAINER_NAME" \
      --entrypoint sleep \
      -v "${GOOGLE_DRIVE_OAUTH_CREDENTIALS:-$HOME/.config/google-drive-mcp/gcp-oauth.keys.json}:/config/gcp-oauth.keys.json:ro" \
      -v "${GOOGLE_DRIVE_MCP_TOKEN_PATH:-$HOME/.config/google-drive-mcp/tokens.json}:/config/tokens.json" \
      "$IMAGE_NAME" \
      infinity >/dev/null 2>&1
    ;;
  *)
    echo "Unexpected container state: $STATE" >&2
    exit 1
    ;;
esac

# Wait for the container to be running (covers start and run -d cases)
for _i in 1 2 3 4 5 6 7 8 9 10; do
  S="$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)" || S=""
  [ "$S" = "running" ] && break
  sleep 0.5
done
if [ "$S" != "running" ]; then
  echo "Container $CONTAINER_NAME failed to reach running state (current: $S)" >&2
  exit 1
fi

# Kill any already-running MCP server inside the container
docker exec "$CONTAINER_NAME" pkill -f 'node dist/index.js' >/dev/null 2>&1 || true

# Run the MCP server via exec — stdin/stdout are connected to the client
exec docker exec -i "$CONTAINER_NAME" node dist/index.js
