from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from typing import Final
from unittest.mock import AsyncMock

import pytest

import litellm
from litellm._auth import model_budget_data
from litellm._auth.model_budget_data import ModelBudgetEntity
from litellm.caching.caching import DualCache
from litellm.proxy._types import Litellm_EntityType
from litellm.proxy.auth.model_budget_adapter import (
    ModelBudgetIO,
    enforce_model_budget,
    read_model_budget,
    resolve_model_budget,
)
from litellm.proxy.hooks import model_max_budget_limiter as reference
from litellm.proxy.hooks.model_budget_shared_windows import SharedWindow

from .budget_reference import model_allowed


@dataclass(frozen=True, slots=True)
class Windows:
    value: SharedWindow | None = None
    read: AsyncMock = field(default_factory=AsyncMock)

    async def open(self, spend_key: str, budget_duration: str, now: float) -> float | None:
        raise AssertionError("request auth must not open a window")

    async def add(self, spend_key: str, budget_duration: str, started_at: float, spend: float) -> None:
        raise AssertionError("request auth must not write a counter")


def groups(model: str, team_id: str | None) -> frozenset[str]:
    return frozenset(("group",))


@pytest.mark.parametrize(
    ("model", "budgets"),
    (
        (
            "provider/model",
            {"model": {"max_budget": 1, "budget_duration": "1h"}, "*": {"max_budget": 2, "budget_duration": "1h"}},
        ),
        ("model", {"model": {"budget_duration": "bad"}, "*": {"max_budget": 2, "budget_duration": "1h"}}),
        ("model", {"model": {}, "group": {"max_budget": 2, "budget_duration": "1h"}}),
        (
            "model",
            {"m*": {"max_budget": 2, "budget_duration": "1h"}, "group": {"max_budget": 1, "budget_duration": "1h"}},
        ),
        (
            "mødél",
            {"m*é*": {"max_budget": 2, "budget_duration": "1h"}, "*": {"max_budget": 1, "budget_duration": "1h"}},
        ),
        (
            "model",
            {"model": {"max_budget": None, "budget_duration": "1h"}, "*": {"max_budget": 1, "budget_duration": "1h"}},
        ),
    ),
)
def test_resolution_matches_production_reference(model: str, budgets: Mapping[str, object]) -> None:
    io: Final = ModelBudgetIO(DualCache(), groups, Windows(), 0)
    old: Final = reference.resolve_model_budget(model, budgets, groups)
    result: Final = resolve_model_budget(model, budgets, io, None)
    assert result.read_groups is False
    assert (result.entry.name if result.entry else None) == (old.budget_model if old else None)
    assert (result.entry.max_budget if result.entry else None) == (old.budget_config.max_budget if old else None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entity", "scope"),
    (
        ("Key", Litellm_EntityType.KEY),
        ("User", Litellm_EntityType.USER),
        ("Team", Litellm_EntityType.TEAM),
        ("TeamMember", Litellm_EntityType.TEAM_MEMBER),
        ("EndUser", Litellm_EntityType.END_USER),
    ),
)
@pytest.mark.parametrize(
    ("local", "shared", "now", "limit"),
    ((0, 2, 100, 1), (1, 0, 100, 1), (0, 2, 3700, 1), (0, 0, 100, 0), (0, 2, 100, -1), (0, 2, 100, None)),
)
async def test_counter_and_shared_window_match_production_reference(
    entity: ModelBudgetEntity, scope: Litellm_EntityType, local: float, shared: float, now: float, limit: float | None
) -> None:
    cache: Final = DualCache()
    key: Final = reference.model_budget_spend_cache_key(scope, "id", "model", "1h")
    cache.set_cache(key, local)
    windows: Final = Windows(read=AsyncMock(return_value=SharedWindow(100, shared)))
    budgets: Final = {"model": {"max_budget": limit, "budget_duration": "1h"}}
    old: Final = reference._PROXY_VirtualKeyModelMaxBudgetLimiter(
        cache, clock=lambda: now, model_access_groups=groups, shared_windows=windows
    )
    io: Final = ModelBudgetIO(cache, groups, windows, 0, clock=lambda: now, monotonic=lambda: 0)
    before: Final = await observed_model_result(model_allowed(old, scope, "id", None, budgets, "model", "exceeded"))
    after: Final = await observed_model_result(
        enforce_model_budget(io, entity, "id", None, "model", budgets, "exceeded")
    )
    assert after == before


async def observed_model_result(call: Awaitable[bool]) -> tuple[object, ...]:
    try:
        return (await call,)
    except litellm.BudgetExceededError as error:
        return (error.message, error.current_cost, error.max_budget, error.entity_type, error.entity_id)


@pytest.mark.asyncio
async def test_legacy_counter_is_added_to_the_current_counter_before_shared_read() -> None:
    cache: Final = DualCache()
    cache.set_cache(reference.model_budget_spend_cache_key(Litellm_EntityType.KEY, "id", "model", "1h"), 0.4)
    cache.set_cache(reference.model_budget_spend_cache_key(Litellm_EntityType.KEY, "id", "provider/model", "1h"), 0.6)
    windows: Final = Windows(read=AsyncMock(side_effect=AssertionError("already over local limit")))
    io: Final = ModelBudgetIO(cache, groups, windows, 0, monotonic=lambda: 1)
    reading: Final = await read_model_budget(
        io, "Key", "id", None, "provider/model", {"model": {"max_budget": 1, "budget_duration": "1h"}}
    )
    assert reading.snapshot.local_spend == 0.4
    assert reading.snapshot.legacy_spend == 0.6
    assert model_budget_data.check(reading.snapshot).denied is True
    windows.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_read_failure_is_terminal_and_not_replayed() -> None:
    failure: Final = RuntimeError("database unavailable")
    windows: Final = Windows(read=AsyncMock(side_effect=failure))
    io: Final = ModelBudgetIO(DualCache(), groups, windows, 0, monotonic=lambda: 0)
    with pytest.raises(RuntimeError, match="database unavailable") as raised:
        await read_model_budget(io, "User", "id", None, "model", {"model": {"max_budget": 1, "budget_duration": "1h"}})
    assert raised.value is failure
    assert windows.read.await_count == 1
