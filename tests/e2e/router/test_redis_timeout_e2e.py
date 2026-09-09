"""Live e2e: the proxy keeps answering while every Redis command times out.

Runs only against a proxy booted from tests/e2e/gateway/redis_timeout_ci_config.yml, which
points cache_params at a real Redis with socket_timeout 0.001 so every command times out and
the circuit breaker opens. Each request fails its primary deployment, retries, falls back to the
backup and succeeds, so it carries retry breadcrumbs; its cost tracking then fails on the spend
counter increment and stringifies the request metadata into a failed-tracking alert. On v1.100.0
that string doubled per request until the worker hung (LIT-6780). Deselected unless
E2E_REDIS_TIMEOUT is set, since it needs that dedicated proxy.
"""

from __future__ import annotations

import time
from typing import Final

import pytest
from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import NoBody, Success
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, KeyGenerateBody

pytestmark = [pytest.mark.e2e, pytest.mark.redis_timeout]

PRIMARY_MODEL: Final = "redis-timeout-primary"
BACKUP_MODEL: Final = "redis-timeout-backup"
REQUESTS: Final = 20
MAX_SECONDS_PER_REQUEST: Final = 10.0
MAX_LATENCY_GROWTH_RATIO: Final = 3.0
MAX_LIVELINESS_SECONDS: Final = 2.0


class TestRedisTimeout:
    @pytest.mark.covers(
        "reliability.circuit_breaker.redis_timeout.stays_responsive",
        exercised_on=["chat_completions"],
    )
    def test_retries_under_redis_timeouts_keep_answering(
        self, client: ComplexityRouterClient, resources: ResourceManager
    ) -> None:
        proxy = client.proxy
        key = proxy.generate_key(
            KeyGenerateBody(models=[PRIMARY_MODEL, BACKUP_MODEL], key_alias=f"e2e-redis-timeout-{unique_marker()}")
        )
        resources.defer(lambda: proxy.delete_key(key))

        latencies: list[float] = []
        for request_number in range(1, REQUESTS + 1):
            started = time.monotonic()
            result = proxy.transport.post(
                "/chat/completions",
                headers=proxy.transport.bearer(key),
                json=ChatBody(
                    model=PRIMARY_MODEL,
                    messages=[ChatMessage(role="user", content=f"redis timeout {unique_marker()} {request_number}")],
                    max_tokens=5,
                ),
                response_type=ChatResponse,
                timeout=MAX_SECONDS_PER_REQUEST,
            )
            elapsed = time.monotonic() - started
            assert isinstance(result, Success), (
                f"request {request_number} failed after {elapsed:.1f}s with Redis timing out: {result}; "
                f"earlier requests took {[round(seconds, 2) for seconds in latencies]}"
            )
            assert result.data.choices, f"request {request_number}: fallback to {BACKUP_MODEL} returned no choices"
            assert elapsed < MAX_SECONDS_PER_REQUEST, (
                f"request {request_number} took {elapsed:.1f}s with Redis timing out; "
                f"earlier requests took {[round(seconds, 2) for seconds in latencies]}"
            )
            latencies.append(elapsed)

        third = REQUESTS // 3
        early = sum(latencies[:third]) / third
        late = sum(latencies[-third:]) / third
        assert late <= max(early * MAX_LATENCY_GROWTH_RATIO, 0.5), (
            f"per-request latency grew from {early:.2f}s to {late:.2f}s across {REQUESTS} requests "
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
            f"only {len(rows)} of {REQUESTS} requests reached the spend log; a Redis outage must not lose spend rows"
        )
