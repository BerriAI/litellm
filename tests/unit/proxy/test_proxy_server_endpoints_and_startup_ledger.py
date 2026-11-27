from typing import Final

import pytest

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

from tests.unit.proxy import test_proxy_server_endpoints_and_startup as scenarios
from tests.unit.proxy.spend_counters_ledger_twin import force_ledger_windows, ledger_scenarios, rust_memory_store
from tests.unit.proxy.spend_tracking.native_budgets_replica import native_reservation

_WINDOW_ROW_QUEUE_SCENARIOS: Final = frozenset({"test_a_slow_window_row_read_does_not_hold_back_the_key_spend_counter"})


def _counter_scenario(name: str) -> bool:
    return ("spend_counter" in name or "increment_spend" in name) and name not in _WINDOW_ROW_QUEUE_SCENARIOS


globals().update(ledger_scenarios(scenarios, _counter_scenario))


@pytest.fixture(autouse=True)
def ledger_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    force_ledger_windows(monkeypatch)


@pytest.mark.asyncio
async def test_increment_spend_counters_finalizes_after_unreserved_increments(
    monkeypatch: pytest.MonkeyPatch, native_budgets_on: None
) -> None:
    """Redefined on the Rust step, which never calls the Python prepare helper the original patched to watch the
    reservation while each unreserved counter was prepared. The unreserved team counter takes the charge, the reserved
    key counter is reconciled to it, and the reservation is finalized."""
    counter_cache: Final = rust_memory_store(monkeypatch, {"spend:key:key-finalize-after-increments": 0.0})
    budget_reservation: Final = await native_reservation(
        counter_cache,
        UserAPIKeyAuth(token="key-finalize-after-increments"),
        [
            {
                "counter_key": "spend:key:key-finalize-after-increments",
                "entity_type": "Key",
                "entity_id": "key-finalize-after-increments",
            }
        ],
        0.5,
    )
    assert counter_cache.get_local("spend:key:key-finalize-after-increments") == 0.5
    monkeypatch.setattr(ps, "user_api_key_cache", UserApiKeyCache())

    await ps.increment_spend_counters(
        token="key-finalize-after-increments",
        team_id="team-finalize-after-increments",
        user_id=None,
        response_cost=0.25,
        budget_reservation=budget_reservation,
    )

    assert counter_cache.snapshot() == {
        "spend:key:key-finalize-after-increments": 0.25,
        "spend:team:team-finalize-after-increments": 0.25,
    }
    assert budget_reservation["finalized"] is True
