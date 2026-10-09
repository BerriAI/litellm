#!/usr/bin/env bash
set -euo pipefail

image="${1:?pass the built image reference}"
release="${2:?pass the expected release tag}"
version="$(docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges "$image" --version)"
test "$version" = "litellm-lens $release protocol=7"
container="$(docker run -d --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 64 --memory 2g --cpus 2 \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  -e LITELLM_URL=http://127.0.0.1:1 \
  -e CLICKHOUSE_URL=http://127.0.0.1:1 \
  -e LITELLM_LENS_SERVICE_TOKEN=isolated-runtime-smoke-secret-32-characters \
  "$image")"
trap 'docker rm -f "$container" >/dev/null' EXIT
test "$(docker exec "$container" id -u)" = 65532
docker exec -i "$container" python3.13 -I -S - <<'PY'
import time
import urllib.error
import urllib.request

for attempt in range(50):
    try:
        with urllib.request.urlopen("http://127.0.0.1:4318/health/live", timeout=1) as response:
            assert response.status == 200
        break
    except urllib.error.URLError:
        if attempt == 49:
            raise
        time.sleep(0.1)

for path, expected in (("health/ready", 503), ("internal/status", 401)):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:4318/{path}", timeout=1)
    except urllib.error.HTTPError as error:
        assert error.code == expected, (path, error.code)
    else:
        raise AssertionError(f"{path} should return {expected}")
print("Unprivileged Lens service remains live with unavailable dependencies")
PY
