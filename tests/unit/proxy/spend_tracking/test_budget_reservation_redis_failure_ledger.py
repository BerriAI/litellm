import pytest

from tests.unit.proxy.spend_counters_ledger_twin import force_ledger_windows, ledger_scenarios
from tests.unit.proxy.spend_tracking import test_budget_reservation_redis_failure as scenarios

globals().update(ledger_scenarios(scenarios))


@pytest.fixture(autouse=True)
def ledger_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    force_ledger_windows(monkeypatch)
