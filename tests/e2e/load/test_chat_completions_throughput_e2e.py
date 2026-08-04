import pytest

from e2e_config import (
    LOAD_BASELINE_SECONDS,
    LOAD_DURATION_SECONDS,
    LOAD_MAX_FAILURE_RATIO,
    LOAD_MAX_SERIAL_LATENCY_SECONDS,
    LOAD_MIN_CONCURRENCY_EFFICIENCY,
    LOAD_SPAWN_RATE,
    LOAD_USERS,
    PROXY_BASE_URL,
)
from load_client import LoadClient
from load_constants import LOAD_MODEL
from locust_load import run_chat_load

pytestmark = [pytest.mark.e2e, pytest.mark.load]


class TestChatCompletionsThroughput:
    @pytest.mark.skip(
        reason=(
            "LIT-5119: stage refuses most of the closed-loop load at the ELB (65.9% 502/503 on the "
            "2026-08-02 run) because it idles at ~1 warm gateway replica; the per-replica SLO cannot "
            "get a clean read until the fleet is pre-scaled for the load phase"
        )
    )
    @pytest.mark.covers("reliability.perf.throughput.under_slo")
    def test_sustains_throughput_slo_under_load(self, client: LoadClient, load_key: str) -> None:
        baseline = run_chat_load(
            base_url=PROXY_BASE_URL,
            api_key=load_key,
            model=LOAD_MODEL,
            users=1,
            spawn_rate=1,
            duration_seconds=LOAD_BASELINE_SECONDS,
        )

        assert baseline.requests > 0 and baseline.requests_per_second > 0, (
            f"no requests completed against {PROXY_BASE_URL} in the {LOAD_BASELINE_SECONDS}s serial "
            f"baseline; the load generator never drove traffic (proxy unreachable or model unservable). "
            f"{baseline.diagnosis()}"
        )
        assert baseline.failure_ratio <= LOAD_MAX_FAILURE_RATIO, (
            f"{baseline.failures}/{baseline.requests} requests failed in the serial baseline "
            f"({baseline.failure_ratio:.1%} > {LOAD_MAX_FAILURE_RATIO:.1%} allowed), so it calibrates "
            f"nothing: a request that fails in milliseconds reads as a fast one, passing the latency "
            f"budget and inflating the floor it derives. {baseline.diagnosis()}"
        )
        assert baseline.median_response_seconds <= LOAD_MAX_SERIAL_LATENCY_SECONDS, (
            f"one user with no competing load saw a median of {baseline.median_response_seconds * 1000:.0f}ms "
            f"per request, over the {LOAD_MAX_SERIAL_LATENCY_SECONDS * 1000:.0f}ms budget; the request path "
            f"itself is slow, before any concurrency. {baseline.diagnosis()}"
        )

        replica_rps = baseline.requests_per_second
        min_rps = replica_rps * LOAD_MIN_CONCURRENCY_EFFICIENCY

        result = run_chat_load(
            base_url=PROXY_BASE_URL,
            api_key=load_key,
            model=LOAD_MODEL,
            users=LOAD_USERS,
            spawn_rate=LOAD_SPAWN_RATE,
            duration_seconds=LOAD_DURATION_SECONDS,
        )

        assert result.requests > 0, (
            f"no requests completed against {PROXY_BASE_URL} in {LOAD_DURATION_SECONDS}s; "
            f"the load generator never drove traffic (proxy unreachable or model unservable). "
            f"{result.diagnosis()}"
        )
        assert result.failure_ratio <= LOAD_MAX_FAILURE_RATIO, (
            f"{result.failures}/{result.requests} requests failed "
            f"({result.failure_ratio:.1%} > {LOAD_MAX_FAILURE_RATIO:.1%} allowed); "
            f"throughput of {result.requests_per_second:.1f} RPS is not a clean read under this error rate, "
            f"since closed-loop throughput rises when requests fail fast. {result.diagnosis()}"
        )
        assert result.requests_per_second >= min_rps, (
            f"sustained {result.requests_per_second:.1f} RPS over {LOAD_DURATION_SECONDS}s with "
            f"{LOAD_USERS} users, below the {min_rps:.1f} RPS floor that one replica set on its own "
            f"({replica_rps:.1f} RPS serial at a {baseline.median_response_seconds * 1000:.0f}ms median, "
            f"x {LOAD_MIN_CONCURRENCY_EFFICIENCY:.0%}); concurrency made the request path slower per "
            f"request than a single user did, so the fleet is serialising rather than merely saturated. "
            f"{result.diagnosis()}"
        )
