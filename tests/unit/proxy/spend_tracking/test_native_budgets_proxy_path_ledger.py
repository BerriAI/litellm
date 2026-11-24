from typing import Final

import pytest

from tests.unit.proxy.spend_counters_ledger_twin import force_ledger_windows, ledger_scenarios
from tests.unit.proxy.spend_tracking import test_native_budgets_proxy_path as scenarios

# Admit-only scenarios that never reach the post-call fan-out and, on main too, give a different
# race result when they run a second time in one process
_ADMIT_ONLY_SCENARIOS: Final = frozenset({"test_racing_requests_for_the_last_of_a_budget_admit_like_python"})

globals().update(ledger_scenarios(scenarios, lambda name: name not in _ADMIT_ONLY_SCENARIOS))


@pytest.fixture(autouse=True)
def ledger_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    force_ledger_windows(monkeypatch)
