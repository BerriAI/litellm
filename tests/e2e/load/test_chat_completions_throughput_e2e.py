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


class TestChatCompletionsThroughput:
    @pytest.mark.covers("reliability.perf.throughput.under_slo")
    def test_sustains_throughput_slo_under_load(
        self, client: LoadClient, load_key: str, load_model: str
    ) -> None:
        client.preflight_mock_chat(key=load_key, model=load_model)

        result = run_chat_load(
            base_url=PROXY_BASE_URL,
            api_key=load_key,
            model=load_model,
            users=LOAD_USERS,
            spawn_rate=LOAD_SPAWN_RATE,
            duration_seconds=LOAD_DURATION_SECONDS,
        )

        assert result.requests > 0, (
            f"no requests completed against {PROXY_BASE_URL} in {LOAD_DURATION_SECONDS}s; "
            f"the load generator never drove traffic (proxy unreachable or model unservable)"
        )
        assert result.failure_ratio <= LOAD_MAX_FAILURE_RATIO, (
            f"load failure ratio {result.failure_ratio:.1%} exceeds "
            f"{LOAD_MAX_FAILURE_RATIO:.1%} allowed "
            f"({result.failures}/{result.requests} failed, "
            f"{result.requests_per_second:.1f} RPS observed; not a clean SLO read).\n"
            f"failure breakdown:\n{result.format_failure_reasons()}\n"
            f"users={LOAD_USERS} spawn_rate={LOAD_SPAWN_RATE} "
            f"duration={LOAD_DURATION_SECONDS}s model={load_model!r}"
        )
        assert result.requests_per_second >= LOAD_MIN_RPS, (
            f"sustained {result.requests_per_second:.1f} RPS over {LOAD_DURATION_SECONDS}s with "
            f"{LOAD_USERS} users, below the {LOAD_MIN_RPS} RPS SLO; the proxy request path "
            f"regressed under load (failure_ratio={result.failure_ratio:.1%})"
        )
