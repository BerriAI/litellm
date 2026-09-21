import os
import time
from typing import Final

import httpx
from redis import Redis


def main() -> None:
    primary: Final = os.environ["INTEGRATION_PROXY_URL"]
    peer: Final = os.environ.get("INTEGRATION_PEER_URL")
    proxies: Final = (primary, peer) if peer else (primary,)
    deadline: Final = time.monotonic() + 90
    headers: Final = {"Authorization": f"Bearer {os.environ['INTEGRATION_MASTER_KEY']}"}
    with httpx.Client(trust_env=False, timeout=2) as client, Redis(
        host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), socket_timeout=2
    ) as cache:
        while True:
            try:
                ready: Final = (
                    client.get(f"{os.environ['INTEGRATION_UPSTREAM_URL']}/health").status_code == 200
                    and all(client.get(f"{url}/health/readiness").status_code == 200 for url in proxies)
                )
                if ready:
                    for url in proxies:
                        response: Final = client.get(f"{url}/cache/ping", headers=headers)
                        response.raise_for_status()
                        result: Final = response.json()
                        assert result["status"] == "healthy", result
                        assert result["cache_type"] == "redis", result
                        assert result["ping_response"] is True, result
                        assert result["set_cache_response"] == "success", result
                    if cache.pubsub_numsub("litellm_proxy.auth_cache_invalidation")[0][1] >= len(proxies):
                        return
            except httpx.TransportError:
                pass
            if time.monotonic() >= deadline:
                raise SystemExit("Integration services or auth-cache subscribers did not become ready")
            time.sleep(0.2)


if __name__ == "__main__":
    main()
