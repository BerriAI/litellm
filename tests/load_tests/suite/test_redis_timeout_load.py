"""Load test: retries against a failing upstream while every Redis command times out.

Reproduces the v1.100.0 OOM (LIT-6780). Every request fails its primary deployment, retries,
falls back to a healthy deployment and succeeds, so each one leaves retry breadcrumbs in its
metadata. Its cost tracking then increments spend counters in a Redis where every command
times out, the circuit breaker opens, the increment raises, and the cost callback's
except block turns the whole request metadata into the body of a failed-tracking alert. On
v1.100.0 that body roughly doubled with every request in the process. This test asserts it
stays flat.

Part of the load test suite (see conftest.py in this directory). Per-request numbers are logged at INFO:

    make test-load
"""

import asyncio
import gc
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Final

import psutil
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

import litellm
from litellm import Router
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from tests._fault_injecting_redis import FaultInjectingRedis, FaultInjectingRedisCache, always

REQUESTS: Final = 12
MAX_SECONDS_PER_REQUEST: Final = 8.0
MAX_RSS_GROWTH_BYTES: Final = 256 * 1024 * 1024
MAX_ALERT_BODY_BYTES: Final = 200 * 1024
MAX_ALERT_GROWTH_RATIO: Final = 2.0
MAX_LATENCY_GROWTH_RATIO: Final = 3.0
ALERT_POLL_SECONDS: Final = 0.02
MODEL: Final = "openai/gpt-5-mini"
NOISY_LOGGERS: Final = ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router")
_log: Final = logging.getLogger(__name__)


@dataclass(slots=True)
class AlertRecorder:
    """Stands in for the Slack alerter so the failed-tracking alert body is observable."""

    body_sizes: list[int] = field(default_factory=list)

    async def failed_tracking_alert(self, error_message: str, failing_model: str) -> None:
        self.body_sizes.append(len(error_message))


@dataclass(frozen=True, slots=True)
class RequestSample:
    number: int
    seconds: float
    rss_growth_bytes: int
    breadcrumbs: int
    alert_body_bytes: int


@pytest.fixture
def failing_redis_cache() -> FaultInjectingRedisCache:
    return FaultInjectingRedisCache(
        FaultInjectingRedis({}, default=always(RedisTimeoutError("injected: Redis stalled")))
    )


@pytest.fixture
def quiet_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in NOISY_LOGGERS:
        monkeypatch.setattr(logging.getLogger(name), "level", logging.CRITICAL)


@pytest.fixture
def alert_recorder(
    monkeypatch: pytest.MonkeyPatch, failing_redis_cache: FaultInjectingRedisCache, quiet_loggers: None
) -> AlertRecorder:
    from litellm.proxy import proxy_server

    recorder: Final = AlertRecorder()
    monkeypatch.setattr(proxy_server.spend_counter_cache, "redis_cache", failing_redis_cache)
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "alerting", ["slack"])
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "slack_alerting_instance", recorder)
    monkeypatch.setattr(litellm, "callbacks", [_ProxyDBLogger()])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    return recorder


def _retrying_router_with_fallback() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": MODEL,
                    "api_key": "sk-primary-not-real",
                    "mock_response": "litellm.InternalServerError",
                },
            },
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": MODEL,
                    "api_key": "sk-backup-not-real",
                    "mock_response": "ok",
                },
            },
        ],
        fallbacks=[{"primary": ["backup"]}],
        num_retries=1,
        set_verbose=False,
    )


async def _wait_for_alert(recorder: AlertRecorder, expected_count: int, deadline: float) -> None:
    """Cost tracking runs as a background task after the response returns, so the alert it
    emits is the signal that the except block has finished stringifying the metadata."""
    while len(recorder.body_sizes) < expected_count:
        if time.perf_counter() > deadline:
            return
        await asyncio.sleep(ALERT_POLL_SECONDS)


async def _one_proxy_shaped_request(
    router: Router, request_number: int, recorder: AlertRecorder, process: psutil.Process, baseline_rss: int
) -> RequestSample:
    """The proxy hands the router a metadata dict and a request snapshot whose body is a shallow
    copy of the request, so ``body["metadata"]`` is the same dict the router stamps breadcrumbs
    onto. That alias is what made every breadcrumb point back at the breadcrumb list."""
    metadata: Final[dict[str, object]] = {
        "user_api_key": f"hashed-key-{request_number % 3}",
        "user_api_key_user_id": "load-user",
        "user_api_key_team_id": "load-team",
        "request_marker": request_number,
    }
    messages: Final = [{"role": "user", "content": f"request {request_number}"}]
    started: Final = time.perf_counter()
    await router.acompletion(
        model="primary",
        messages=messages,
        metadata=metadata,
        proxy_server_request={
            "url": "http://localhost:4000/v1/chat/completions",
            "method": "POST",
            "headers": {},
            "body": {"model": "primary", "messages": messages, "metadata": metadata},
        },
    )
    await _wait_for_alert(recorder, expected_count=request_number, deadline=started + MAX_SECONDS_PER_REQUEST)
    breadcrumbs: Final = metadata.get("previous_models")
    return RequestSample(
        number=request_number,
        seconds=time.perf_counter() - started,
        rss_growth_bytes=process.memory_info().rss - baseline_rss,
        breadcrumbs=len(breadcrumbs) if isinstance(breadcrumbs, (list, tuple)) else 0,
        alert_body_bytes=recorder.body_sizes[-1] if len(recorder.body_sizes) >= request_number else 0,
    )


async def _drive(router: Router, recorder: AlertRecorder) -> AsyncIterator[RequestSample]:
    process: Final = psutil.Process(os.getpid())
    gc.collect()
    baseline_rss: Final = process.memory_info().rss
    for request_number in range(1, REQUESTS + 1):
        sample: Final = await _one_proxy_shaped_request(router, request_number, recorder, process, baseline_rss)
        _log.info(
            "request %d: %.2fs, rss %+.1f MB, breadcrumbs=%d, alert body=%d bytes",
            sample.number,
            sample.seconds,
            sample.rss_growth_bytes / 2**20,
            sample.breadcrumbs,
            sample.alert_body_bytes,
        )
        if sample.alert_body_bytes == 0:
            pytest.fail(
                f"request {sample.number}: cost tracking did not fail within {MAX_SECONDS_PER_REQUEST:.0f}s, "
                "so either the failing Redis never reached the cost callback or stringifying the metadata "
                "took longer than the budget"
            )
        if sample.rss_growth_bytes > MAX_RSS_GROWTH_BYTES:
            pytest.fail(f"request {sample.number}: RSS grew {sample.rss_growth_bytes / 2**20:.0f} MB over baseline")
        yield sample


@pytest.mark.asyncio
@pytest.mark.no_parallel
async def test_retries_under_redis_timeouts_stay_flat(
    alert_recorder: AlertRecorder, failing_redis_cache: FaultInjectingRedisCache
) -> None:
    router: Final = _retrying_router_with_fallback()

    samples: Final = [sample async for sample in _drive(router, alert_recorder)]

    assert failing_redis_cache._circuit_breaker.is_open(), "the injected timeouts never tripped the circuit breaker"
    breadcrumb_counts: Final = frozenset(sample.breadcrumbs for sample in samples)
    assert len(breadcrumb_counts) == 1 and min(breadcrumb_counts) >= 2, (
        f"every request should carry the same handful of its own failed attempts, got {sorted(breadcrumb_counts)}"
    )

    third: Final = REQUESTS // 3
    early_latency: Final = sum(sample.seconds for sample in samples[:third]) / third
    late_latency: Final = sum(sample.seconds for sample in samples[-third:]) / third
    assert late_latency <= max(early_latency * MAX_LATENCY_GROWTH_RATIO, 0.5), (
        f"per-request latency grew from {early_latency:.2f}s to {late_latency:.2f}s across {REQUESTS} failing requests"
    )
    alert_sizes: Final = tuple(sample.alert_body_bytes for sample in samples)
    assert max(alert_sizes) <= MAX_ALERT_BODY_BYTES, f"failed-tracking alert body reached {max(alert_sizes)} bytes"
    assert max(alert_sizes) <= min(alert_sizes) * MAX_ALERT_GROWTH_RATIO, (
        f"failed-tracking alert body grew from {min(alert_sizes)} to {max(alert_sizes)} bytes across requests"
    )
