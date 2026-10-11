from dataclasses import dataclass
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.workers import readiness_failure

HEALTHY: Final = {
    "status": "healthy",
    "cache_type": "redis",
    "ping_response": True,
    "set_cache_response": "success",
}


@dataclass(frozen=True, slots=True)
class Subscribers:
    count: int

    def pubsub_numsub(self, *channels: str) -> tuple[tuple[str, int], ...]:
        return tuple((channel, self.count) for channel in channels)


@pytest.mark.parametrize("subscribers", (0, 1, 2))
def test_readiness_requires_a_subscriber_for_each_proxy(subscribers: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/cache/ping":
            assert request.headers["authorization"] == "Bearer owned-key"
            return httpx.Response(200, json=HEALTHY)
        return httpx.Response(200)

    with httpx.Client(base_url="http://127.0.0.1:4000", transport=httpx.MockTransport(respond)) as client:
        primary: Final = Gateway(client, "owned-key", "http://127.0.0.1:8190")
        failure: Final = readiness_failure(client, primary.upstream_url, (primary, primary), Subscribers(subscribers))
    assert failure == (None if subscribers == 2 else f"auth-cache subscribers {subscribers}/2")


@pytest.mark.parametrize(
    ("field", "value"),
    (("status", "unhealthy"), ("cache_type", "local"), ("ping_response", False), ("set_cache_response", "failure")),
)
def test_readiness_rejects_a_proxy_without_working_shared_redis(field: str, value: str | bool) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**HEALTHY, field: value})

    with httpx.Client(base_url="http://127.0.0.1:4000", transport=httpx.MockTransport(respond)) as client:
        primary: Final = Gateway(client, "owned-key", "http://127.0.0.1:8190")
        failure: Final = readiness_failure(client, primary.upstream_url, (primary,), Subscribers(1))
    assert failure == "proxy 0 Redis read/write health"
