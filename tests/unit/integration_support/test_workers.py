from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.workers import GatewayProfile, gateway_profile, readiness_failure
from tests.integration.run import worker_settings

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


@pytest.mark.parametrize(
    ("group", "peer", "cost_map"),
    (
        ("management", True, False),
        ("mcp", True, False),
        ("cost", False, True),
        ("security", False, False),
        ("extensions", False, False),
    ),
)
def test_each_group_gets_the_worker_gateway_its_shared_proxy_used_to_provide(
    group: str, peer: bool, cost_map: bool
) -> None:
    assert gateway_profile(worker_settings(group, 4)) == GatewayProfile(
        peer=peer, replica=False, cost_map=cost_map, coverage_file=None
    )


def test_replica_runs_and_coverage_runs_reach_every_worker_gateway() -> None:
    profile: Final = gateway_profile(
        {
            **worker_settings("management", 2),
            "INTEGRATION_PROXY_READ_REPLICA_URL": "postgresql://reader@127.0.0.1/circle_test",
            "INTEGRATION_COVERAGE_FILE": "/results/coverage/data",
        }
    )
    assert profile == GatewayProfile(
        peer=True, replica=True, cost_map=False, coverage_file=Path("/results/coverage/data")
    )
