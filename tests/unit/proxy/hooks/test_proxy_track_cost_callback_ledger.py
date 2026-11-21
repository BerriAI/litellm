from typing import Final

import pytest

from tests.unit.proxy.hooks import test_proxy_track_cost_callback as scenarios
from tests.unit.proxy.spend_counters_ledger_twin import force_ledger_windows, ledger_scenarios

_WINDOW_ROW_QUEUE_SCENARIOS: Final = frozenset(
    {"test_a_failure_spend_row_lands_in_the_key_budget_window_as_of_the_request_start"}
)

globals().update(ledger_scenarios(scenarios, lambda name: name not in _WINDOW_ROW_QUEUE_SCENARIOS))


@pytest.fixture(autouse=True)
def ledger_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    force_ledger_windows(monkeypatch)
