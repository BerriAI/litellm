import os
from contextlib import ExitStack
from typing import Final

import httpx
from integration._support.client import Gateway
from integration._support.workers import wait_for_services
from redis import Redis


def main() -> None:
    primary: Final = os.environ["INTEGRATION_PROXY_URL"]
    peer: Final = os.environ.get("INTEGRATION_PEER_URL")
    urls: Final = (primary, peer) if peer else (primary,)
    upstream: Final = os.environ["INTEGRATION_UPSTREAM_URL"]
    with ExitStack() as stack:
        proxies: Final = tuple(
            Gateway(
                stack.enter_context(httpx.Client(base_url=url, trust_env=False, timeout=2)),
                os.environ["INTEGRATION_MASTER_KEY"],
                upstream,
            )
            for url in urls
        )
        cache: Final = stack.enter_context(
            Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), socket_timeout=2)
        )
        wait_for_services(upstream, proxies, cache)


if __name__ == "__main__":
    main()
