"""Live e2e: the proxy keeps answering while every Redis command times out.

Runs only against a proxy booted from tests/e2e/gateway/redis_timeout_ci_config.yml, which
points cache_params at a real Redis with socket_timeout 0.001. The test holds that Redis in
CLIENT PAUSE WRITE for its duration, so every write the proxy sends, the spend counter increment
included, hangs past the timeout, and it proves the degradation was real from the breaker metrics on /metrics: fresh timeouts, a breaker transition,
or an already-open breaker rejecting every call, which is the state a customer's worker sits in. The test registers two deployments through /model/new: a primary whose api_base is a closed port
and a backup that answers with a mock. Each request fails the primary, retries, falls back and succeeds, so it carries retry breadcrumbs; its cost
tracking then fails on the spend counter increment and stringifies the request metadata into a
failed-tracking alert. On v1.100.0 that string doubled per request until the worker hung
(LIT-6780). Deselected unless E2E_REDIS_TIMEOUT is set, since it needs that dedicated proxy.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Final

import pytest
import redis
from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import NoBody, Result, Success
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, KeyGenerateBody, LiteLLMParamsBody
from proxy_client import ProxyClient
from pydantic import BaseModel

pytestmark = [pytest.mark.e2e, pytest.mark.redis_timeout]

PRIMARY_MODEL: Final = "redis-timeout-primary"
BACKUP_MODEL: Final = "redis-timeout-backup"
BACKING_MODEL: Final = "openai/gpt-5-mini"
CLOSED_PORT_API_BASE: Final = "http://127.0.0.1:1"
REQUESTS: Final = 20
MAX_SECONDS_PER_REQUEST: Final = 10.0
MAX_LATENCY_GROWTH_RATIO: Final = 3.0
MAX_LIVELINESS_SECONDS: Final = 2.0
REDIS_PAUSE_MS: Final = 600_000
BREAKER_FAILURE_THRESHOLD: Final = 5
TIMEOUT_FAILURES_RE: Final = re.compile(
    r'^litellm_redis_circuit_breaker_failures_total\{failure_class="timeout"\} ([0-9.e+]+)$', re.M
)


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
BREAKER_OPEN_RE: Final = re.compile(r'^litellm_redis_circuit_breaker_state\{state="open"\} ([0-9.e+]+)$', re.M)
BREAKER_TRANSITIONS_RE: Final = re.compile(
    r'^litellm_redis_circuit_breaker_transitions_total\{state="[a-z_]+"\} ([0-9.e+]+)$', re.M
)


@pytest.fixture
def paused_redis() -> Iterator[None]:
    """Hold the proxy's Redis in CLIENT PAUSE WRITE so every write it sends outlives the 1 ms socket
    timeout. A loopback Redis otherwise answers many commands inside that budget. Reads stay live so
    this control connection can lift the pause in teardown."""
    host = os.environ.get("REDIS_HOST")
    port = os.environ.get("REDIS_PORT")
    assert host and port, "REDIS_HOST and REDIS_PORT must name the Redis the proxy under test uses"
    control = redis.Redis(host=host, port=int(port), socket_timeout=5)
    control.client_pause(REDIS_PAUSE_MS, all=False)  # pyright: ignore[reportUnknownMemberType]  # redis-py stubs return Any
    try:
        yield
    finally:
        control.client_unpause()  # pyright: ignore[reportUnknownMemberType]  # redis-py stubs return Any
        control.close()


def _metric(proxy: ProxyClient, pattern: re.Pattern[str]) -> float:
    body = proxy.probe("/metrics", params=NoBody()).body
    return sum(float(match.group(1)) for match in pattern.finditer(body))


class TestRedisTimeout:
    @pytest.mark.parametrize("endpoint", ENDPOINTS, ids=[endpoint.name for endpoint in ENDPOINTS])
    @pytest.mark.covers(
        "reliability.circuit_breaker.redis_timeout.stays_responsive",
        exercised_on=["chat_completions", "responses"],
    )
    def test_retries_under_redis_timeouts_keep_answering(
        self, client: ComplexityRouterClient, resources: ResourceManager, endpoint: Endpoint, paused_redis: None
    ) -> None:
        proxy = client.proxy
        primary_id = proxy.create_model(
            PRIMARY_MODEL,
            LiteLLMParamsBody(
                model=BACKING_MODEL, api_key="sk-redis-timeout-primary-not-used", api_base=CLOSED_PORT_API_BASE
            ),
        )
        resources.defer(lambda: proxy.delete_model(primary_id))
        backup_id = proxy.create_model(
            BACKUP_MODEL,
            LiteLLMParamsBody(model=BACKING_MODEL, api_key="sk-redis-timeout-backup-not-used", mock_response="ok"),
        )
        resources.defer(lambda: proxy.delete_model(backup_id))
        timeouts_before = _metric(proxy, TIMEOUT_FAILURES_RE)
        transitions_before = _metric(proxy, BREAKER_TRANSITIONS_RE)
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

        timeouts_total = _metric(proxy, TIMEOUT_FAILURES_RE)
        timeouts = timeouts_total - timeouts_before
        transitions = _metric(proxy, BREAKER_TRANSITIONS_RE) - transitions_before
        breaker_open = _metric(proxy, BREAKER_OPEN_RE) >= 1
        assert timeouts_total >= BREAKER_FAILURE_THRESHOLD, (
            f"the proxy counted only {timeouts_total:.0f} Redis timeouts in its lifetime; the write-paused Redis "
            "never made its spend counter writes time out, so this run proved nothing"
        )
        assert timeouts >= REQUESTS or transitions >= 1 or breaker_open, (
            f"during {REQUESTS} {endpoint.name} requests the breaker counted {timeouts:.0f} new timeouts, "
            f"{transitions:.0f} state transitions, and ended {'open' if breaker_open else 'closed'}; "
            "Redis was healthy for this case, so it proved nothing"
        )

        rows = proxy.poll_logs_for_key(key, min_rows=REQUESTS)
        assert len(rows) >= REQUESTS, (
            f"only {len(rows)} of {REQUESTS} {endpoint.name} requests reached the spend log; "
            "a Redis outage must not lose spend rows"
        )
