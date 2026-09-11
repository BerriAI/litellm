"""Live e2e: the proxy under load keeps serving every request while Redis writes time out.

Runs against a proxy booted from tests/e2e/gateway/redis_chaos_ci_config.yml, which points
cache_params at a real Redis with litellm's default socket_timeout. That one client backs all
three Redis touchpoints on the request path: the virtual-key auth cache, the response cache,
and the cross-pod spend counter the cost-tracking callback awaits.

The load runs in two phases against one model group of three mock deployments. The two at
order 1 raise InternalServerError and the one at order 2 serves, so every request burns its
retries on the failing pair (a 500 is retryable, so retries keep re-picking inside the lowest
order) and the router's order-based fallback then re-targets order 2. Every request is expected
to succeed, and each one carries retry breadcrumbs into cost tracking.

Phase A is a baseline with Redis healthy; phase B holds Redis in CLIENT PAUSE WRITE, so the
spend counter increment times out and the callback stringifies the request metadata,
breadcrumbs included, into a failed-tracking alert. On v1.100.0 that string doubled per request
until the worker hung (LIT-6780), which is what the per-phase RSS and CPU percentiles are here
to catch.

Needs the proxy on the same host, since RSS and CPU come from psutil on its process tree:
a multi-worker proxy serves /metrics from the prometheus multiprocess collector, which drops
the process collector's memory and CPU series. Deselected unless E2E_REDIS_CHAOS is set.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
import redis
from e2e_config import PROXY_BASE_URL, unique_marker
from e2e_http import NoBody
from lifecycle import ResourceManager
from load_client import LoadClient
from locust_load import LoadResult, run_chat_load
from models import KeyGenerateBody, LiteLLMParamsBody
from proxy_client import ProxyClient
from proxy_usage import ProxyUsageSampler, UsageWindow

pytestmark = [pytest.mark.e2e, pytest.mark.redis_chaos]

MODEL_GROUP: Final = "redis-chaos-fable"
MOCK_MODEL: Final = "anthropic/claude-fable-5-1"
FAILING_DEPLOYMENTS: Final = 2
SERVING_DEPLOYMENTS: Final = 1
FAILING_ORDER: Final = 1
SERVING_ORDER: Final = 2
KEY_POOL_SIZE: Final = 8
LOCUST_USERS: Final = 50
LOCUST_SPAWN_RATE: Final = 50.0
BASELINE_SECONDS: Final = 60.0
CHAOS_SECONDS: Final = 90.0
REDIS_PAUSE_MS: Final = 600_000
BASELINE_TIMEOUT_RATE_CEILING: Final = 0.05
CHAOS_TIMEOUT_RATE_FLOOR: Final = 0.20

TIMEOUT_FAILURES_RE: Final = re.compile(
    r'^litellm_redis_circuit_breaker_failures_total\{failure_class="timeout"\} ([0-9.e+]+)$', re.M
)
BREAKER_OPEN_RE: Final = re.compile(r'^litellm_redis_circuit_breaker_state\{state="open"\} ([0-9.e+]+)$', re.M)
BREAKER_TRANSITIONS_RE: Final = re.compile(
    r'^litellm_redis_circuit_breaker_transitions_total\{state="[a-z_]+"\} ([0-9.e+]+)$', re.M
)
RETRIES_RE: Final = re.compile(r"^litellm_deployment_failure_responses_total\{[^}]*\} ([0-9.e+]+)$", re.M)
COOLDOWN_RE: Final = re.compile(r"^litellm_deployment_cooled_down_total\{[^}]*\} ([0-9.e+]+)$", re.M)


@dataclass(frozen=True, slots=True)
class Phase:
    """One load phase's traffic and what the proxy's process tree did during it."""

    name: str
    load: LoadResult
    usage: UsageWindow

    def report(self) -> str:
        return (
            f"{self.name}: {self.load.requests} requests, {self.load.failures} failures, "
            f"{self.load.requests_per_second:.0f} rps, {self.load.latency_summary()}; {self.usage.summary()}"
        )


def _failing_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=MOCK_MODEL,
        api_key="sk-redis-chaos-not-used",
        mock_response="litellm.InternalServerError",
        order=FAILING_ORDER,
    )


def _serving_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=MOCK_MODEL,
        api_key="sk-redis-chaos-not-used",
        mock_response="redis chaos ok",
        order=SERVING_ORDER,
    )


@pytest.fixture
def proxy_pid() -> int:
    """The proxy's PID, which the workflow exports after starting it.

    Required rather than discovered: picking a process out of the table by name would be
    ambiguous on a developer machine running more than one proxy.
    """
    pid: Final = os.environ.get("E2E_PROXY_PID")
    assert pid and pid.isdigit(), (
        "E2E_PROXY_PID must hold the PID of the proxy under test; RSS and CPU are read from "
        "its process tree because a multi-worker proxy does not report them on /metrics"
    )
    return int(pid)


@pytest.fixture
def redis_control() -> Iterator[redis.Redis[bytes]]:
    """A control connection to the proxy's Redis, which unpauses writes in teardown.

    Only writes are paused: CLIENT PAUSE ALL would freeze this connection too, leaving
    nothing able to lift the pause.
    """
    host: Final = os.environ.get("REDIS_HOST")
    port: Final = os.environ.get("REDIS_PORT")
    assert host and port, "REDIS_HOST and REDIS_PORT must name the Redis the proxy under test uses"
    control: Final = redis.Redis(host=host, port=int(port), socket_timeout=5)
    try:
        yield control
    finally:
        control.client_unpause()  # pyright: ignore[reportUnknownMemberType]  # redis-py stubs return Any
        control.close()


def _metric(proxy: ProxyClient, pattern: re.Pattern[str]) -> float:
    body: Final = proxy.probe("/metrics", params=NoBody()).body
    return sum(float(match.group(1)) for match in pattern.finditer(body))


def _register_deployments(proxy: ProxyClient, resources: ResourceManager) -> None:
    for _ in range(FAILING_DEPLOYMENTS):
        failing_id = proxy.create_model(MODEL_GROUP, _failing_params())
        resources.defer(lambda model_id=failing_id: proxy.delete_model(model_id))
    for _ in range(SERVING_DEPLOYMENTS):
        serving_id = proxy.create_model(MODEL_GROUP, _serving_params())
        resources.defer(lambda model_id=serving_id: proxy.delete_model(model_id))


def _generate_key_pool(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, ...]:
    """A pool of virtual keys so auth and budget lookups are not one permanently warm
    cache entry; each locust user picks one, so Redis auth reads actually happen."""
    keys: Final = tuple(
        proxy.generate_key(
            KeyGenerateBody(models=[MODEL_GROUP], key_alias=f"e2e-redis-chaos-{unique_marker()}-{index}")
        )
        for index in range(KEY_POOL_SIZE)
    )
    for key in keys:
        resources.defer(lambda doomed=key: proxy.delete_key(doomed))
    return keys


def _drive(keys: tuple[str, ...], seconds: float) -> LoadResult:
    return run_chat_load(
        base_url=PROXY_BASE_URL,
        api_keys=keys,
        model=MODEL_GROUP,
        users=LOCUST_USERS,
        spawn_rate=LOCUST_SPAWN_RATE,
        duration_seconds=seconds,
    )


class TestRedisChaos:
    @pytest.mark.covers(
        "reliability.circuit_breaker.redis_timeout.stays_responsive",
        exercised_on=["chat_completions"],
    )
    def test_load_survives_redis_write_timeouts(
        self,
        client: LoadClient,
        resources: ResourceManager,
        proxy_pid: int,
        redis_control: redis.Redis[bytes],
    ) -> None:
        proxy: Final = client.proxy
        _register_deployments(proxy, resources)
        keys: Final = _generate_key_pool(proxy, resources)

        timeouts_at_start: Final = _metric(proxy, TIMEOUT_FAILURES_RE)
        retries_before: Final = _metric(proxy, RETRIES_RE)
        cooldowns_before: Final = _metric(proxy, COOLDOWN_RE)

        with ProxyUsageSampler(proxy_pid) as sampler:
            baseline: Final = Phase(name="baseline", load=_drive(keys, BASELINE_SECONDS), usage=sampler.split())
            timeouts_after_baseline: Final = _metric(proxy, TIMEOUT_FAILURES_RE)

            redis_control.client_pause(REDIS_PAUSE_MS, all=False)  # pyright: ignore[reportUnknownMemberType]  # redis-py stubs return Any
            chaos: Final = Phase(name="chaos", load=_drive(keys, CHAOS_SECONDS), usage=sampler.split())

        report: Final = f"{baseline.report()} | {chaos.report()}"

        for phase in (baseline, chaos):
            assert phase.load.requests > 0, (
                f"{phase.name} drove no traffic at all, so it proved nothing: {phase.load.diagnosis()}. {report}"
            )
            assert phase.load.failures == 0, (
                f"{phase.name} had {phase.load.failures} of {phase.load.requests} requests fail. Every request "
                f"must succeed: the failing deployments sit at order {FAILING_ORDER} and the serving one at order "
                f"{SERVING_ORDER}, so once the retries on order {FAILING_ORDER} are spent the order-based fallback "
                f"lands on the serving deployment. Failures mean it was cooled down, the fallback did not run, or "
                f"a Redis failure reached the response path. {phase.load.diagnosis()}. {report}"
            )

        cooldowns: Final = _metric(proxy, COOLDOWN_RE) - cooldowns_before
        assert cooldowns == 0, (
            f"{cooldowns:.0f} deployments were cooled down during the run; the failing deployments are supposed "
            f"to stay in rotation so every request keeps exercising the retry path. {report}"
        )

        retries: Final = _metric(proxy, RETRIES_RE) - retries_before
        assert retries >= baseline.load.requests + chaos.load.requests, (
            f"only {retries:.0f} deployment failures were counted across "
            f"{baseline.load.requests + chaos.load.requests} requests; the mock deployments did not fail, so no "
            f"request carried retry breadcrumbs into cost tracking and the regression path was never entered. "
            f"{report}"
        )

        baseline_timeout_rate: Final = (timeouts_after_baseline - timeouts_at_start) / baseline.load.requests
        assert baseline_timeout_rate <= BASELINE_TIMEOUT_RATE_CEILING, (
            f"a healthy Redis timed out on {baseline_timeout_rate:.1%} of baseline requests, over the "
            f"{BASELINE_TIMEOUT_RATE_CEILING:.0%} this test tolerates; at litellm's default socket_timeout a "
            f"loaded Redis does time out occasionally, but this much means the baseline is already degraded and "
            f"the two phases are not comparable. {report}"
        )

        chaos_timeouts: Final = _metric(proxy, TIMEOUT_FAILURES_RE) - timeouts_after_baseline
        chaos_timeout_rate: Final = chaos_timeouts / chaos.load.requests
        transitions: Final = _metric(proxy, BREAKER_TRANSITIONS_RE)
        breaker_open: Final = _metric(proxy, BREAKER_OPEN_RE) >= 1
        assert chaos_timeout_rate >= CHAOS_TIMEOUT_RATE_FLOOR or transitions >= 1 or breaker_open, (
            f"with writes paused the breaker saw Redis time out on only {chaos_timeout_rate:.1%} of requests "
            f"against {baseline_timeout_rate:.1%} at baseline, under the {CHAOS_TIMEOUT_RATE_FLOOR:.0%} a real "
            f"outage produces, and it counted {transitions:.0f} state transitions and ended "
            f"{'open' if breaker_open else 'closed'}. The spend counter increment never failed, so this run "
            f"proved nothing. {report}"
        )

        rows: Final = proxy.poll_logs_for_key(keys[0], min_rows=1)
        assert rows, (
            f"no spend rows landed for the first key in the pool; a Redis outage must not cost the proxy its "
            f"spend logs, which are written to Postgres through a queue rather than through Redis. {report}"
        )

        print(f"\nredis chaos load: {report}")  # noqa: T201  # the numbers this test exists to report, read off the CI log
