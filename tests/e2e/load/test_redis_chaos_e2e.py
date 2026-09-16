"""Live e2e: the proxy under load keeps serving every request while Redis is down entirely.

Runs against a proxy booted from tests/e2e/gateway/redis_chaos_ci_config.yml, which points
cache_params at a real Redis with litellm's default socket_timeout. That one client backs all
three Redis touchpoints on the request path: the virtual-key auth cache, the response cache,
and the cross-pod spend counter the cost-tracking callback awaits.

The load runs in two phases against one model group of three mock deployments. The two at
order 1 raise InternalServerError and the one at order 2 serves, so every request burns its
retries on the failing pair (a 500 is retryable, so retries keep re-picking inside the lowest
order) and the router's order-based fallback then re-targets order 2. Every request is expected
to succeed, and each one carries retry breadcrumbs into cost tracking.

Traffic is split round robin between /chat/completions and /v1/messages, one endpoint per
simulated user: the Redis touchpoints and the cost-tracking callback are shared by both, but
the Anthropic Messages route reaches them through its own request path, so a regression that
only shows up there would not surface from chat completions alone.

Phase A is a baseline with Redis healthy; phase B holds Redis in CLIENT PAUSE ALL for the
length of the phase, simulating Redis being down outright rather than merely slow to write.
Every touchpoint times out: the auth cache read falls back to Postgres, the response cache
read and write both fail, and the spend counter increment times out and the callback
stringifies the request metadata, breadcrumbs included, into a failed-tracking alert. On
v1.100.0 that string doubled per request until the worker hung (LIT-6780), which is what the
per-phase RSS, CPU, and log-bytes budgets are here to catch.

Needs the proxy on the same host, since RSS and CPU come from psutil on its process tree:
a multi-worker proxy serves /metrics from the prometheus multiprocess collector, which drops
the process collector's memory and CPU series. Log bytes are read from the file the proxy's
stdout/stderr was redirected to, so the same host requirement covers that too. Deselected
unless E2E_REDIS_CHAOS is set.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final

import pytest
import redis
from e2e_config import PROXY_BASE_URL, unique_marker
from e2e_http import NoBody
from lifecycle import ResourceManager
from load_client import LoadClient
from locust_load import LoadResult, run_gateway_load
from models import KeyGenerateBody, LiteLLMParamsBody
from phase_budget import AbsoluteBudget, Budget, RatioBudget, violations
from proxy_client import ProxyClient
from proxy_usage import ProxyUsageSampler, UsageWindow

pytestmark: Final = pytest.mark.e2e

MODEL_GROUP: Final = f"redis-chaos-fable-{unique_marker()}"
MOCK_MODEL: Final = "anthropic/claude-fable-5-1"
FAILING_DEPLOYMENTS: Final = 2
SERVING_DEPLOYMENTS: Final = 1
FAILING_ORDER: Final = 1
SERVING_ORDER: Final = 2
KEY_POOL_SIZE: Final = 8
LOAD_ENDPOINTS: Final = ("/chat/completions", "/v1/messages")
LOCUST_USERS: Final = 50
LOCUST_SPAWN_RATE: Final = 50.0
BASELINE_SECONDS: Final = 60.0
CHAOS_SECONDS: Final = 90.0
REDIS_PAUSE_MS: Final = int(CHAOS_SECONDS * 1000)

# RSS and CPU are budgeted as a multiple of the same metric in the baseline phase, because both
# are machine-shaped: RSS scales with worker count and CPU with core count, so a number
# calibrated on one runner means nothing on another. RSS moved 0.91x-1.40x across three otherwise
# identical local runs, so it stays loose; CPU per request held steady at 1.33x-1.36x across the
# same runs, so it sits close to what is actually measured. That makes CPU the likeliest of these
# to flake first on a runner whose core count shifts how much of baseline CPU is fixed per-request
# work: loosen it rather than widening the others if a CI run trips it without a real cause.
CHAOS_RSS_RATIO_CEILING: Final = 2.0
CHAOS_CPU_PER_REQUEST_RATIO_CEILING: Final = 2.0

# Latency and log volume get flat ceilings instead, because a ratio cannot bound either one. Once
# the breaker opens, a request skips Redis rather than waiting on its socket timeout, so the chaos
# phase can come in faster than baseline (local runs measured p90 at 0.61x) and a ratio passes on a
# phase that was never slow. What a user actually cares about is the wall-clock number, which these
# hold directly. Calibrated from local runs whose worst chaos phase was p50 0.19s, p90 0.23s, p99
# 0.69s and 3.5 KB of log per request, with several times that left as slack for a shared CI runner.
CHAOS_P50_LATENCY_CEILING_SECONDS: Final = 1.0
CHAOS_P90_LATENCY_CEILING_SECONDS: Final = 2.0
CHAOS_P99_LATENCY_CEILING_SECONDS: Final = 3.0
CHAOS_LOG_BYTES_PER_REQUEST_CEILING: Final = 10_000.0

DRAIN_TIMEOUT_SECONDS: Final = 30.0
DRAIN_POLL_SECONDS: Final = 1.0

TIMEOUT_FAILURES_RE: Final = re.compile(
    r'^litellm_redis_circuit_breaker_failures_total\{failure_class="timeout"\} ([0-9.e+]+)$', re.M
)
# The state gauge carries a pid label under the multiprocess collector, one series per worker,
# so this matches any label order rather than a bare {state="open"} that never appears.
BREAKER_OPEN_RE: Final = re.compile(
    r'^litellm_redis_circuit_breaker_state\{[^}]*state="open"[^}]*\} ([0-9.e+]+)$', re.M
)
BREAKER_TRANSITIONS_RE: Final = re.compile(
    r'^litellm_redis_circuit_breaker_transitions_total\{state="[a-z_]+"\} ([0-9.e+]+)$', re.M
)


def _deployment_metric_re(name: str, model_ids: tuple[str, ...]) -> re.Pattern[str]:
    """A per-deployment counter, narrowed to the deployments one run registered, so traffic
    anything else sends the same proxy during the run cannot pad the retry count."""
    ids: Final = "|".join(re.escape(model_id) for model_id in model_ids)
    return re.compile(rf'^litellm_{name}\{{[^}}]*model_id="(?:{ids})"[^}}]*\}} ([0-9.e+]+)$', re.M)


@dataclass(frozen=True, slots=True)
class Phase:
    """One load phase's traffic and what the proxy's process tree did during it."""

    name: str
    load: LoadResult
    usage: UsageWindow
    redis_timeouts: float
    log_bytes: int

    @property
    def timeouts_per_request(self) -> float:
        return self.redis_timeouts / self.load.requests if self.load.requests else 0.0

    @property
    def cpu_seconds_per_request(self) -> float:
        return self.usage.cpu_seconds_per_request(self.load.requests)

    @property
    def log_bytes_per_request(self) -> float:
        return self.log_bytes / self.load.requests if self.load.requests else 0.0

    def report(self) -> str:
        return (
            f"{self.name}: {self.load.requests} requests, {self.load.failures} failures, "
            f"{self.load.requests_per_second:.0f} rps, {self.load.latency_summary()}; {self.usage.summary()}; "
            f"{self.cpu_seconds_per_request * 1000:.1f} ms CPU per request; "
            f"{self.log_bytes_per_request:.0f} log bytes per request; "
            f"{self.timeouts_per_request:.2f} Redis timeouts per request; "
            f"by endpoint: {self.load.endpoint_summary()}"
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
def proxy_log() -> Path:
    """Path to the proxy's stdout/stderr log, which the workflow captures to a file.

    Required rather than discovered for the same reason as proxy_pid: a developer machine may
    have more than one proxy log around.
    """
    path: Final = os.environ.get("E2E_PROXY_LOG")
    assert path, "E2E_PROXY_LOG must hold the path the proxy's stdout/stderr was redirected to"
    return Path(path)


def _log_bytes(path: Path) -> int:
    return path.stat().st_size


@pytest.fixture
def redis_control() -> Iterator[redis.Redis[bytes]]:
    """A control connection to the proxy's Redis, which unpauses it in teardown as a safety net.

    CLIENT PAUSE ALL freezes every connection including this one, so REDIS_PAUSE_MS is sized
    to the chaos phase: by the time teardown runs, the pause has
    already lapsed on its own and CLIENT UNPAUSE here returns immediately. It only actually
    waits out a lapsed pause if the chaos phase itself overran that duration.
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


def _scrape(proxy: ProxyClient) -> str:
    """One /metrics body, read once per checkpoint so every counter comes from the same instant."""
    scrape: Final = proxy.probe("/metrics", params=NoBody())
    assert scrape.status_code == 200, (
        f"/metrics did not answer ({scrape.status_code}: {scrape.body[:200]}), so no counter can be read; "
        f"a silent 0 here would turn every before-and-after difference negative"
    )
    return scrape.body


def _metric(scrape: str, pattern: re.Pattern[str]) -> float:
    return sum(float(match.group(1)) for match in pattern.finditer(scrape))


def _scrape_after_drain(proxy: ProxyClient, pattern: re.Pattern[str]) -> str:
    """A /metrics body taken once `pattern`'s count has stopped moving.

    `set_llm_deployment_failure_metrics` runs from the async logging callback queue, so a load
    generator that just stopped sending traffic can still have thousands of failure increments
    in flight, and a scrape taken the instant load stops undercounts them. Settling on the
    counter rather than sleeping a fixed duration keeps the wait proportional to how backed up
    the queue actually is.
    """
    deadline: Final = time.monotonic() + DRAIN_TIMEOUT_SECONDS

    def scrapes() -> Iterator[str]:
        yield _scrape(proxy)
        while time.monotonic() < deadline:
            time.sleep(DRAIN_POLL_SECONDS)
            yield _scrape(proxy)

    settled: Final = next(
        (later for earlier, later in pairwise(scrapes()) if _metric(earlier, pattern) == _metric(later, pattern)),
        None,
    )
    return settled if settled is not None else _scrape(proxy)


def _register_deployments(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, ...]:
    """The model ids this run registered, which scope its per-deployment metric reads."""
    params: Final = (
        *(_failing_params() for _ in range(FAILING_DEPLOYMENTS)),
        *(_serving_params() for _ in range(SERVING_DEPLOYMENTS)),
    )
    model_ids: Final = tuple(proxy.create_model(MODEL_GROUP, one) for one in params)
    for model_id in model_ids:
        resources.defer(lambda doomed=model_id: proxy.delete_model(doomed))
    return model_ids


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
    return run_gateway_load(
        base_url=PROXY_BASE_URL,
        api_keys=keys,
        model=MODEL_GROUP,
        endpoints=LOAD_ENDPOINTS,
        users=LOCUST_USERS,
        spawn_rate=LOCUST_SPAWN_RATE,
        duration_seconds=seconds,
    )


def _latency_budget(percentile: str, measured: float, ceiling: float) -> Budget:
    return AbsoluteBudget(name=f"{percentile} latency", measured=measured, ceiling=ceiling, unit="s", decimals=3)


def _rss_budget(percentile: str, baseline: UsageWindow, degraded: UsageWindow, fraction: float) -> Budget:
    return RatioBudget(
        name=f"{percentile} RSS",
        baseline=baseline.rss_percentile(fraction) / 2**20,
        degraded=degraded.rss_percentile(fraction) / 2**20,
        ratio_ceiling=CHAOS_RSS_RATIO_CEILING,
        unit=" MB",
        decimals=0,
    )


def _chaos_budgets(baseline: Phase, chaos: Phase) -> tuple[Budget, ...]:
    """What a Redis outage is allowed to cost.

    Every request still succeeding is the headline assertion, but a proxy can answer every
    request while leaking: the v1.100.0 regression (LIT-6780) served traffic the whole way up
    to a 61 GB worker. These bound the cost of serving it. RSS and CPU are bounded against the
    same run's healthy phase, latency and log bytes against a flat ceiling; see phase_budget
    for why the two kinds of metric cannot share one shape.

    Latency and RSS are budgeted at p50, p90 and p99 so a regression that only shows up in the
    tail (or only in the median) cannot hide behind the other. RSS gets the tightest bound: the
    failure path has no business allocating more per request. CPU and log bytes are each budgeted
    once, as an amount per request rather than per percentile: cores-busy saturates at the worker
    count under load, so its percentiles read the same whether a request costs 10 ms of CPU or
    40, and cannot budget anything; per-request is the figure that actually moves. Log bytes
    isolates the cost of the failed-tracking alert's own noisy error handling from the CPU it
    burns doing useful retry work, since the two would otherwise be indistinguishable in one
    CPU number.
    """
    return (
        _latency_budget("p50", chaos.load.p50_seconds, CHAOS_P50_LATENCY_CEILING_SECONDS),
        _latency_budget("p90", chaos.load.p90_seconds, CHAOS_P90_LATENCY_CEILING_SECONDS),
        _latency_budget("p99", chaos.load.p99_seconds, CHAOS_P99_LATENCY_CEILING_SECONDS),
        _rss_budget("p50", baseline.usage, chaos.usage, 0.5),
        _rss_budget("p90", baseline.usage, chaos.usage, 0.9),
        _rss_budget("p99", baseline.usage, chaos.usage, 0.99),
        RatioBudget(
            name="CPU per request",
            baseline=baseline.cpu_seconds_per_request * 1000,
            degraded=chaos.cpu_seconds_per_request * 1000,
            ratio_ceiling=CHAOS_CPU_PER_REQUEST_RATIO_CEILING,
            unit=" ms",
        ),
        AbsoluteBudget(
            name="log bytes per request",
            measured=chaos.log_bytes_per_request,
            ceiling=CHAOS_LOG_BYTES_PER_REQUEST_CEILING,
            unit=" B",
            decimals=0,
        ),
    )


@pytest.mark.redis_chaos
class TestRedisChaos:
    @pytest.mark.covers(
        "reliability.circuit_breaker.redis_timeout.stays_responsive",
        exercised_on=("chat_completions", "messages"),
    )
    def test_load_survives_redis_being_down(
        self,
        client: LoadClient,
        resources: ResourceManager,
        proxy_pid: int,
        proxy_log: Path,
        redis_control: redis.Redis[bytes],
    ) -> None:
        proxy: Final = client.proxy
        model_ids: Final = _register_deployments(proxy, resources)
        keys: Final = _generate_key_pool(proxy, resources)

        retries_re: Final = _deployment_metric_re("deployment_failure_responses_total", model_ids)
        cooldown_re: Final = _deployment_metric_re("deployment_cooled_down_total", model_ids)

        at_start: Final = _scrape(proxy)
        log_at_start: Final = _log_bytes(proxy_log)

        with ProxyUsageSampler(proxy_pid) as sampler:
            baseline_load: Final = _drive(keys, BASELINE_SECONDS)
            baseline_usage: Final = sampler.split()
            after_baseline: Final = _scrape(proxy)
            log_after_baseline: Final = _log_bytes(proxy_log)

            redis_control.client_pause(REDIS_PAUSE_MS, all=True)  # pyright: ignore[reportUnknownMemberType]  # redis-py stubs return Any
            chaos_load: Final = _drive(keys, CHAOS_SECONDS)
            chaos_usage: Final = sampler.split()
        at_end: Final = _scrape_after_drain(proxy, retries_re)
        log_at_end: Final = _log_bytes(proxy_log)

        baseline: Final = Phase(
            name="baseline",
            load=baseline_load,
            usage=baseline_usage,
            redis_timeouts=_metric(after_baseline, TIMEOUT_FAILURES_RE) - _metric(at_start, TIMEOUT_FAILURES_RE),
            log_bytes=log_after_baseline - log_at_start,
        )
        chaos: Final = Phase(
            name="chaos",
            load=chaos_load,
            usage=chaos_usage,
            redis_timeouts=_metric(at_end, TIMEOUT_FAILURES_RE) - _metric(after_baseline, TIMEOUT_FAILURES_RE),
            log_bytes=log_at_end - log_after_baseline,
        )
        report: Final = f"{baseline.report()} | {chaos.report()}"

        for phase in (baseline, chaos):
            assert phase.load.requests > 0, (
                f"{phase.name} drove no traffic at all, so it proved nothing: {phase.load.diagnosis()}. {report}"
            )
            assert frozenset(endpoint.name for endpoint in phase.load.endpoints) == frozenset(LOAD_ENDPOINTS), (
                f"{phase.name} drove {tuple(endpoint.name for endpoint in phase.load.endpoints)} rather than every "
                f"endpoint in {LOAD_ENDPOINTS}; the round robin hands one endpoint to each simulated user, so a "
                f"missing one means a route never ran and its request path was never exercised. {report}"
            )
            assert phase.load.failures == 0, (
                f"{phase.name} had {phase.load.failures} of {phase.load.requests} requests fail. Every request "
                f"must succeed: the failing deployments sit at order {FAILING_ORDER} and the serving one at order "
                f"{SERVING_ORDER}, so once the retries on order {FAILING_ORDER} are spent the order-based fallback "
                f"lands on the serving deployment. Failures mean it was cooled down, the fallback did not run, or "
                f"a Redis failure reached the response path. {phase.load.diagnosis()}. {report}"
            )

        cooldowns: Final = _metric(at_end, cooldown_re) - _metric(at_start, cooldown_re)
        assert cooldowns == 0, (
            f"{cooldowns:.0f} deployments were cooled down during the run; the failing deployments are supposed "
            f"to stay in rotation so every request keeps exercising the retry path. {report}"
        )

        retries: Final = _metric(at_end, retries_re) - _metric(at_start, retries_re)
        assert retries >= baseline.load.requests + chaos.load.requests, (
            f"only {retries:.0f} deployment failures were counted across "
            f"{baseline.load.requests + chaos.load.requests} requests; the mock deployments did not fail, so no "
            f"request carried retry breadcrumbs into cost tracking and the regression path was never entered. "
            f"{report}"
        )

        transitions: Final = _metric(at_end, BREAKER_TRANSITIONS_RE) - _metric(after_baseline, BREAKER_TRANSITIONS_RE)
        breaker_open: Final = _metric(at_end, BREAKER_OPEN_RE) >= 1
        assert transitions >= 1 or breaker_open, (
            f"pausing Redis produced no circuit breaker state transitions and it ended closed; nothing on the "
            f"request path ever saw Redis fail, so this run proved nothing. {report}"
        )

        blown: Final = violations(_chaos_budgets(baseline, chaos))
        assert not blown, (
            f"pausing Redis cost the proxy more than a Redis outage is allowed to: {'; '.join(blown)}. {report}"
        )

        rows: Final = proxy.poll_logs_for_key(keys[0], min_rows=1)
        assert rows, (
            f"no spend rows landed for the first key in the pool; a Redis outage must not cost the proxy its "
            f"spend logs, which are written to Postgres through a queue rather than through Redis. {report}"
        )

        print(f"\nredis chaos load: {report}")  # noqa: T201  # the numbers this test exists to report, read off the CI log
