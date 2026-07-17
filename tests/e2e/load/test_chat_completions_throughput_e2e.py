import pytest

from e2e_config import (
    LOAD_DURATION_SECONDS,
    LOAD_MAX_FAILURE_RATIO,
    LOAD_MIN_RPS,
    LOAD_SPAWN_RATE,
    LOAD_USERS,
    PROXY_BASE_URL,
)
from load_client import LoadClient
from locust_load import run_chat_load

pytestmark = [pytest.mark.e2e, pytest.mark.load]

LOAD_MODEL = "load-mock"


class TestChatCompletionsThroughput:
    @pytest.mark.covers("reliability.perf.throughput.under_slo")
    def test_sustains_throughput_slo_under_load(self, client: LoadClient, load_key: str) -> None:
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
            f"the load generator never drove traffic (proxy unreachable or model unservable)"
        )
        assert result.failure_ratio <= LOAD_MAX_FAILURE_RATIO, (
            f"{result.failures}/{result.requests} requests failed "
            f"({result.failure_ratio:.1%} > {LOAD_MAX_FAILURE_RATIO:.1%} allowed); "
            f"throughput of {result.requests_per_second:.1f} RPS is not a clean read under this error rate"
        )
        assert result.requests_per_second >= LOAD_MIN_RPS, (
            f"sustained {result.requests_per_second:.1f} RPS over {LOAD_DURATION_SECONDS}s with "
            f"{LOAD_USERS} users, below the {LOAD_MIN_RPS} RPS SLO; the proxy request path regressed under load"
        )
