import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from math import inf
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm._auth.common_budgets import BudgetFacts
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_EndUserTable, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth import auth_checks as reference
from litellm.proxy.auth import common_budget_adapter as native
from litellm.proxy.utils import ProxyLogging


@dataclass(frozen=True, slots=True)
class Observed:
    error: tuple[object, ...] | None
    alerts: tuple[object, ...]
    throttle: float | None


async def observe(run: Callable[[], Awaitable[None]], logging: Mock, key: UserAPIKeyAuth) -> Observed:
    try:
        await run()
    except litellm.BudgetExceededError as error:
        failure: Final = (error.message, error.current_cost, error.max_budget, error.entity_type, error.entity_id)
        await asyncio.sleep(0)
        return Observed(failure, tuple(logging.budget_alerts.call_args_list), key.budget_throttle_pct)
    await asyncio.sleep(0)
    return Observed(None, tuple(logging.budget_alerts.call_args_list), key.budget_throttle_pct)


@pytest.mark.asyncio
@pytest.mark.parametrize(("limit", "spend"), ((None, 2), (1, 0), (1, 1), (1, 2), (0, 0), (-1, 0), (inf, 2)))
async def test_team_error_envelope_and_alert_projection_match_reference(
    monkeypatch: pytest.MonkeyPatch, limit: float | None, spend: float
) -> None:
    reader: Final = AsyncMock(return_value=spend)
    monkeypatch.setattr(proxy_server, "get_current_spend", reader)
    team: Final = LiteLLM_TeamTable(team_id="team", max_budget=limit, spend=0)
    key: Final = UserAPIKeyAuth(token="hash", team_id="team")
    old_log: Final = Mock(spec=ProxyLogging, budget_alerts=AsyncMock())
    new_log: Final = Mock(spec=ProxyLogging, budget_alerts=AsyncMock())
    before: Final = await observe(lambda: reference.team_max_budget_check(team, key, old_log), old_log, key)
    after: Final = await observe(lambda: native.team_max_budget_check(team, key, new_log), new_log, key)
    assert after == before
    assert reader.await_count == (0 if limit is None else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(("limit", "spend"), ((None, 2), (1, 0), (1, 1), (1, 2), (0, 0), (-1, 0), (inf, 2)))
@pytest.mark.parametrize("throttle", (None, 0.5, 2))
async def test_key_error_envelope_alert_and_throttle_mutation_match_reference(
    monkeypatch: pytest.MonkeyPatch, limit: float | None, spend: float, throttle: float | None
) -> None:
    monkeypatch.setattr(proxy_server, "get_current_spend", AsyncMock(return_value=spend))
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", throttle)
    old: Final = UserAPIKeyAuth(
        token="hash", key_alias="alias", max_budget=limit, tpm_limit=10, metadata={"throttle_on_budget_exceeded": True}
    )
    new: Final = old.model_copy(deep=True)
    old_log: Final = Mock(spec=ProxyLogging, budget_alerts=AsyncMock())
    new_log: Final = Mock(spec=ProxyLogging, budget_alerts=AsyncMock())
    before: Final = await observe(lambda: reference.virtual_key_max_budget_check(old, old_log), old_log, old)
    after: Final = await observe(lambda: native.virtual_key_max_budget_check(new, new_log), new_log, new)
    assert after == before


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("/chat/completions", "/key/info"))
@pytest.mark.parametrize(("limit", "spend"), ((None, 2), (1, 1), (1, 2), (0, 0), (-1, 0)))
async def test_end_user_info_skip_and_strict_ceiling_match_reference(
    monkeypatch: pytest.MonkeyPatch, route: str, limit: float | None, spend: float
) -> None:
    reader: Final = AsyncMock(return_value=spend)
    monkeypatch.setattr(proxy_server, "get_current_spend", reader)
    user: Final = LiteLLM_EndUserTable(
        user_id="end", blocked=False, litellm_budget_table=LiteLLM_BudgetTable(max_budget=limit)
    )
    key: Final = UserAPIKeyAuth()
    old_log: Final = Mock(spec=ProxyLogging, budget_alerts=AsyncMock())
    new_log: Final = Mock(spec=ProxyLogging, budget_alerts=AsyncMock())
    before: Final = await observe(lambda: reference.check_end_user_budget(user, route), old_log, key)
    after: Final = await observe(lambda: native.check_end_user_budget(user, route), new_log, key)
    assert after == before


@pytest.mark.asyncio
async def test_spend_io_failure_is_terminal_and_does_not_replay() -> None:
    failure: Final = RuntimeError("counter unavailable")
    reader: Final = AsyncMock(side_effect=failure)
    with pytest.raises(RuntimeError, match="counter unavailable") as raised:
        await native.read_budget(BudgetFacts("Key", 1), native.CounterRead("spend:key:hash", 0, 1), reader)
    assert raised.value is failure
    assert reader.await_count == 1


@pytest.mark.parametrize(
    ("member", "default", "increase", "expiry"),
    (
        (1, 5, 2, datetime(2025, 1, 2)),
        (None, 5, 2, datetime(2025, 1, 2)),
        (1, 5, 2, datetime(2025, 1, 1, tzinfo=timezone.utc)),
        (None, 0, 2, datetime(2025, 1, 2)),
        (None, -1, 2, None),
        (0, 5, None, None),
    ),
)
def test_member_budget_selection_and_temporary_expiry_match_reference(
    member: float | None, default: float | None, increase: float | None, expiry: datetime | None
) -> None:
    now: Final = datetime(2025, 1, 1, tzinfo=timezone.utc)
    row: Final = LiteLLM_BudgetTable(max_budget=member, temp_budget_increase=increase, temp_budget_expiry=expiry)
    team: Final = LiteLLM_BudgetTable(max_budget=default)
    assert native.resolve_effective_team_member_budget(
        row, team, now
    ) == reference.resolve_effective_team_member_budget(row, team, now)


@pytest.mark.parametrize(
    "policy", (reference.resolve_effective_team_member_budget, native.resolve_effective_team_member_budget)
)
def test_naive_now_preserves_datetime_error(policy: Callable[..., float | None]) -> None:
    row: Final = LiteLLM_BudgetTable(max_budget=1, temp_budget_increase=2, temp_budget_expiry=datetime(2025, 1, 2))
    with pytest.raises(TypeError, match="offset-naive"):
        policy(row, None, datetime(2025, 1, 1))
