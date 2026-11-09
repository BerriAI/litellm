from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm._auth import common_budgets
from litellm.proxy import proxy_server
from litellm.proxy.auth.common_budget_lookups import _read_access_group_rows
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limit", "reads", "denied"), ((None, 0, False), (0, 0, False), (-1, 0, False), (1, 1, True), (2, 2, False))
)
async def test_raw_access_group_snapshot_stops_reads_at_native_denial(
    monkeypatch: pytest.MonkeyPatch,
    limit: float | None,
    reads: int,
    denied: bool,
) -> None:
    counter: Final = AsyncMock(return_value=1.0)
    monkeypatch.setattr(proxy_server, "get_current_spend", counter)
    rows: Final = {
        "first": ModelAccessGroupBudget(access_group_name="first", max_budget=limit),
        "second": ModelAccessGroupBudget(access_group_name="second", max_budget=limit),
    }
    snapshots: Final = await _read_access_group_rows(("first", "second"), rows)
    assert common_budgets.check(snapshots[0].facts).denied is denied
    assert len(snapshots) == (1 if denied else 2)
    assert snapshots[0].name == "first"
    assert counter.await_count == reads


@pytest.mark.asyncio
async def test_access_group_counter_failure_is_terminal_before_next_group(monkeypatch: pytest.MonkeyPatch) -> None:
    failure: Final = RuntimeError("counter unavailable")
    counter: Final = AsyncMock(side_effect=failure)
    monkeypatch.setattr(proxy_server, "get_current_spend", counter)
    row: Final = ModelAccessGroupBudget(access_group_name="first", max_budget=1)
    with pytest.raises(RuntimeError, match="counter unavailable") as raised:
        await _read_access_group_rows(("first", "second"), {"first": row, "second": row})
    assert raised.value is failure
    assert counter.await_count == 1
