"""Live e2e: the proxy keeps answering while every Redis command times out.

Runs only against a proxy booted from tests/e2e/gateway/redis_timeout_ci_config.yml, which
points cache_params at a real Redis with socket_timeout 0.001 so commands time out and the
circuit breaker opens. Each request fails its primary deployment, whose api_base is a closed
port, retries, falls back to the backup and succeeds, so it carries retry breadcrumbs; its cost
tracking then fails on the spend counter increment and stringifies the request metadata into a
failed-tracking alert. On v1.100.0 that string doubled per request until the worker hung
(LIT-6780). Deselected unless E2E_REDIS_TIMEOUT is set, since it needs that dedicated proxy.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import pytest
from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import NoBody, Result, Success
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, KeyGenerateBody
from proxy_client import ProxyClient
from pydantic import BaseModel

pytestmark = [pytest.mark.e2e, pytest.mark.redis_timeout]

PRIMARY_MODEL: Final = "redis-timeout-primary"
BACKUP_MODEL: Final = "redis-timeout-backup"
REQUESTS: Final = 20
MAX_SECONDS_PER_REQUEST: Final = 10.0
MAX_LATENCY_GROWTH_RATIO: Final = 3.0
MAX_LIVELINESS_SECONDS: Final = 2.0


class ResponsesBody(BaseModel):
    model: str
    input: str
    max_output_tokens: int = 5


class ResponsesObject(BaseModel):
    id: str | None = None
    status: str | None = None
    output: list[object] = []


@dataclass(frozen=True, slots=True)
class Endpoint:
    name: str
    send: Callable[[ProxyClient, str, str], Result[BaseModel]]
    served: Callable[[BaseModel], bool]


def _send_chat(proxy: ProxyClient, key: str, marker: str) -> Result[BaseModel]:
    return proxy.transport.post(
        "/chat/completions",
        headers=proxy.transport.bearer(key),
        json=ChatBody(model=PRIMARY_MODEL, messages=[ChatMessage(role="user", content=marker)], max_tokens=5),
        response_type=ChatResponse,
        timeout=MAX_SECONDS_PER_REQUEST,
    )


def _send_responses(proxy: ProxyClient, key: str, marker: str) -> Result[BaseModel]:
    return proxy.transport.post(
        "/v1/responses",
        headers=proxy.transport.bearer(key),
        json=ResponsesBody(model=PRIMARY_MODEL, input=marker),
        response_type=ResponsesObject,
        timeout=MAX_SECONDS_PER_REQUEST,
    )


ENDPOINTS: Final = (
    Endpoint(
        name="chat_completions",
        send=_send_chat,
        served=lambda data: isinstance(data, ChatResponse) and bool(data.choices),
    ),
    Endpoint(
        name="responses",
        send=_send_responses,
        served=lambda data: isinstance(data, ResponsesObject) and bool(data.output),
    ),
)


class TestRedisTimeout:
    @pytest.mark.parametrize("endpoint", ENDPOINTS, ids=[endpoint.name for endpoint in ENDPOINTS])
    @pytest.mark.covers(
        "reliability.circuit_breaker.redis_timeout.stays_responsive",
        exercised_on=["chat_completions", "responses"],
    )
    def test_retries_under_redis_timeouts_keep_answering(
        self, client: ComplexityRouterClient, resources: ResourceManager, endpoint: Endpoint
    ) -> None:
        proxy = client.proxy
        key = proxy.generate_key(
            KeyGenerateBody(
                models=[PRIMARY_MODEL, BACKUP_MODEL], key_alias=f"e2e-redis-timeout-{endpoint.name}-{unique_marker()}"
            )
        )
        resources.defer(lambda: proxy.delete_key(key))

        latencies: list[float] = []
        for request_number in range(1, REQUESTS + 1):
            started = time.monotonic()
            result = endpoint.send(proxy, key, f"redis timeout {unique_marker()} {request_number}")
            elapsed = time.monotonic() - started
            assert isinstance(result, Success), (
                f"{endpoint.name} request {request_number} failed after {elapsed:.1f}s with Redis timing out: {result}; "
                f"earlier requests took {[round(seconds, 2) for seconds in latencies]}"
            )
            assert endpoint.served(result.data), (
                f"{endpoint.name} request {request_number}: fallback to {BACKUP_MODEL} returned no output"
            )
            assert elapsed < MAX_SECONDS_PER_REQUEST, (
                f"{endpoint.name} request {request_number} took {elapsed:.1f}s with Redis timing out; "
                f"earlier requests took {[round(seconds, 2) for seconds in latencies]}"
            )
            latencies.append(elapsed)

        third = REQUESTS // 3
        early = sum(latencies[:third]) / third
        late = sum(latencies[-third:]) / third
        assert late <= max(early * MAX_LATENCY_GROWTH_RATIO, 0.5), (
            f"{endpoint.name} per-request latency grew from {early:.2f}s to {late:.2f}s across {REQUESTS} requests "
            "while Redis timed out; the proxy is paying more for each failed request"
        )

        started = time.monotonic()
        probe = proxy.transport.probe("/health/liveliness", params=NoBody())
        liveliness_seconds = time.monotonic() - started
        assert probe.healthy, f"/health/liveliness returned {probe.status_code} after the Redis timeout loop"
        assert liveliness_seconds < MAX_LIVELINESS_SECONDS, (
            f"/health/liveliness took {liveliness_seconds:.1f}s after the loop; the worker is stalled"
        )

        rows = proxy.poll_logs_for_key(key, min_rows=REQUESTS)
        assert len(rows) >= REQUESTS, (
            f"only {len(rows)} of {REQUESTS} {endpoint.name} requests reached the spend log; "
            "a Redis outage must not lose spend rows"
        )
