"""Contract: the committed twin counters and the Prometheus remaining budget gauges that read them (#49199).

The Redis spend counter `spend:key:<hash>` holds charged spend plus every in-flight hold, which is what admission
needs. A dashboard must not show a hold as spent, so while a `PrometheusLogger` is registered the proxy keeps a
charges-only twin for each budgeted key, team, user, organization and end user lifetime counter
(`litellm/proxy/spend_tracking/committed_spend.py`):

    spend:committed:key:<hash>   spend:committed:team:<team_id>   spend:committed:user:<user_id>   ...

Holds never touch a twin, and window counters (`:window:`) have none (`has_committed_twin`). After each served call
the remaining budget gauges (`litellm_remaining_api_key_budget_metric{hashed_api_key=...}`,
`litellm_remaining_team_budget_metric{team=...}`) are set to `max_budget - max(stored spend, committed spend)`
(`prometheus.py::_increment_remaining_budget_metrics`), so a concurrent hold never lowers them.

Price: input free, output 0.004 a token, one served call 0.02 (`hold_model`).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import Gateway, gateway_from_environment, held
from tests.integration.spend.dec1_budgets2_support import committed_key_counter, gauge_reads, owned, team_key
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    background,
    call,
    counter,
    counter_settles,
    held_provider,
    hold_model,
    key_counter,
    redis_client,
)
from tests.integration.spend.dec1_spend_support import hashed, scripted_upstream, served

pytestmark: Final = pytest.mark.shares_worker_state

KEY_REMAINING: Final = "litellm_remaining_api_key_budget_metric"
TEAM_REMAINING: Final = "litellm_remaining_team_budget_metric"


@pytest.fixture(scope="module")
def prometheus_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """An owned proxy on the shared database and Redis with `litellm_settings.callbacks: [prometheus]`."""
    with (
        gateway_from_environment() as shared,
        owned(
            shared,
            tmp_path_factory.mktemp("dec1-budgets2-prometheus"),
            litellm_settings={
                "callbacks": ["prometheus"],
                "cache": True,
                "cache_params": {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
            },
        ) as proxy,
    ):
        yield proxy


@pytest.mark.timeout(180)
def test_a_served_call_adds_its_cost_to_the_committed_key_twin(prometheus_proxy: Gateway) -> None:
    """A key with `max_budget` 1.0 makes one served call. `spend:committed:key:<hash>` reads exactly 0.02, the same
    as `spend:key:<hash>`, once the call has settled."""
    proxy: Final = prometheus_proxy
    with scripted_upstream() as wire, proxy.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        served(call(proxy, model, key, max_tokens=5))
        assert counter_settles(cache, committed_key_counter(key), CHAT_COST) == pytest.approx(CHAT_COST)
        assert counter(cache, key_counter(key)) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(180)
def test_an_in_flight_hold_raises_the_spend_counter_but_not_its_committed_twin(prometheus_proxy: Gateway) -> None:
    """After one served call (0.02), a call with `max_tokens: 10` is held at the provider. While it is in flight
    `spend:key:<hash>` reads 0.06 (0.02 charged plus the 0.04 hold) and `spend:committed:key:<hash>` still reads 0.02.
    A port that writes holds into the twin would make the gauges show unfinished requests as spent."""
    proxy: Final = prometheus_proxy
    with held_provider() as provider, proxy.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        served(call(proxy, model, key, max_tokens=5))
        counter_settles(cache, committed_key_counter(key), CHAT_COST)
        in_flight: Final = run(lambda: call(proxy, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(0.06)
        assert counter(cache, committed_key_counter(key)) == pytest.approx(CHAT_COST)
        provider.release()
        served(in_flight.result(timeout=60))
        assert counter_settles(cache, committed_key_counter(key), 2 * CHAT_COST) == pytest.approx(2 * CHAT_COST)


@pytest.mark.timeout(180)
def test_the_key_remaining_gauge_ignores_a_concurrent_hold(prometheus_proxy: Gateway) -> None:
    """A key with `max_budget` 1.0 has a call with `max_tokens: 10` held in flight (0.04 hold). A second call with
    `max_tokens: 5` is served meanwhile. `litellm_remaining_api_key_budget_metric{hashed_api_key=<hash>}` then reads
    0.98 (1.0 minus the one charged 0.02), not 0.94, because the gauge reads the committed twin."""
    proxy: Final = prometheus_proxy
    with held_provider() as provider, proxy.scenario() as scenario, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        in_flight: Final = run(lambda: call(proxy, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        try:
            served(call(proxy, model, key, max_tokens=5))
            reading: Final = gauge_reads(proxy, KEY_REMAINING, 1.0 - CHAT_COST, hashed_api_key=hashed(key))
        finally:
            provider.release()
        assert reading == pytest.approx(1.0 - CHAT_COST)
        served(in_flight.result(timeout=60))


@pytest.mark.timeout(180)
def test_the_team_twin_and_gauge_track_charged_team_spend(prometheus_proxy: Gateway) -> None:
    """A team with `max_budget` 1.0 and a key of that team make one served call. `spend:committed:team:<team_id>`
    reads 0.02 and `litellm_remaining_team_budget_metric{team=<team_id>}` reads 0.98."""
    proxy: Final = prometheus_proxy
    with scripted_upstream() as wire, proxy.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        team_id, key = team_key(scenario, model, max_budget=1.0)
        served(call(proxy, model, key, max_tokens=5))
        assert counter_settles(cache, f"spend:committed:team:{team_id}", CHAT_COST) == pytest.approx(CHAT_COST)
        assert gauge_reads(proxy, TEAM_REMAINING, 1.0 - CHAT_COST, team=team_id) == pytest.approx(1.0 - CHAT_COST)


@pytest.mark.timeout(180)
def test_a_failed_call_leaves_the_committed_twin_unchanged(prometheus_proxy: Gateway) -> None:
    """After one served call (twin 0.02) the provider fails the next call with 500. The twin stays at exactly 0.02
    for the next three seconds: a released hold is never written to it, in either direction."""
    proxy: Final = prometheus_proxy
    with held_provider() as provider, proxy.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        served(call(proxy, model, key, max_tokens=5))
        counter_settles(cache, committed_key_counter(key), CHAT_COST)
        provider.failing_status = 500
        assert call(proxy, model, key, max_tokens=10).status_code >= 500
        stable: Final = held(
            lambda: counter(cache, committed_key_counter(key)),
            lambda value: value is not None and abs(value - CHAT_COST) < 1e-9,
            holding=3,
        )
        assert stable == pytest.approx(CHAT_COST)


@pytest.mark.timeout(180)
def test_a_key_window_counter_has_no_committed_twin(prometheus_proxy: Gateway) -> None:
    """A key with `max_budget` 1.0 and `budget_limits: [{budget_duration: 1d, max_budget: 1.0}]` makes one call.
    `spend:key:<hash>:window:1d` reads 0.02, `spend:committed:key:<hash>` reads 0.02, and no
    `spend:committed:key:<hash>:window:1d` exists: windows are never twinned (`has_committed_twin`)."""
    proxy: Final = prometheus_proxy
    with scripted_upstream() as wire, proxy.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(
            models=[model], max_budget=1.0, budget_limits=[{"budget_duration": "1d", "max_budget": 1.0}]
        )
        served(call(proxy, model, key, max_tokens=5))
        counter_settles(cache, f"spend:key:{hashed(key)}:window:1d", CHAT_COST)
        counter_settles(cache, committed_key_counter(key), CHAT_COST)
        assert counter(cache, f"spend:committed:key:{hashed(key)}:window:1d") is None


@pytest.mark.timeout(180)
def test_a_key_without_a_budget_gets_no_committed_twin(prometheus_proxy: Gateway) -> None:
    """A key with no `max_budget` makes one served call on the Prometheus proxy. No `spend:committed:key:<hash>`
    appears in the following three seconds: twins are kept only for entities with a budget."""
    proxy: Final = prometheus_proxy
    with scripted_upstream() as wire, proxy.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(call(proxy, model, key, max_tokens=5))
        assert held(lambda: counter(cache, committed_key_counter(key)), lambda value: value is None, holding=3) is None


@pytest.mark.timeout(180)
def test_without_a_prometheus_callback_no_committed_twin_is_written(gateway: Gateway) -> None:
    """On the suite proxy, which registers no Prometheus callback, a key with `max_budget` 1.0 makes one served
    call. `spend:key:<hash>` settles at 0.02 and `spend:committed:key:<hash>` never appears
    (`committed_spend_tracked`). UNVERIFIED: that the suite proxy config has no Prometheus callback in every CI group."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        served(call(gateway, model, key, max_tokens=5))
        counter_settles(cache, key_counter(key), CHAT_COST)
        assert held(lambda: counter(cache, committed_key_counter(key)), lambda value: value is None, holding=3) is None
