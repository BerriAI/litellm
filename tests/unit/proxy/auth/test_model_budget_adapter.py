from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from typing import Final
from unittest.mock import AsyncMock, call

import pytest

import litellm
from litellm._auth import model_budget_data
from litellm._auth.budget_pricing import ModelPriceFacts
from litellm._auth.fallback_candidate import (
    CandidateBudgetRead,
    CandidateBudgetScope,
    FallbackCandidateFacts,
    fallback_candidate,
)
from litellm._auth.model_budget_data import ModelBudgetEntity
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry
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


@pytest.fixture
def alias_catalog(monkeypatch: pytest.MonkeyPatch) -> Mapping[str, object]:
    catalog: Final = {"budget-latest": AliasedCostEntry({}, ("budget-target", "budget-latest"))}
    monkeypatch.setattr(litellm, "model_cost", catalog)
    return catalog


@pytest.mark.asyncio
@pytest.mark.parametrize("entity", ("Key", "User", "Team", "TeamMember", "EndUser"))
@pytest.mark.parametrize(
    ("alias_spend", "target_spend", "expected"),
    ((1.0, 2.0, "budget-latest"), (0.0, 2.0, "budget-target"), (0.0, 0.0, None)),
)
async def test_raw_alias_scope_preserves_first_denial_and_counter_order(
    alias_catalog: Mapping[str, object],
    entity: ModelBudgetEntity,
    alias_spend: float,
    target_spend: float,
    expected: str | None,
) -> None:
    cache: Final = DualCache()
    scope: Final = {
        "Key": Litellm_EntityType.KEY,
        "User": Litellm_EntityType.USER,
        "Team": Litellm_EntityType.TEAM,
        "TeamMember": Litellm_EntityType.TEAM_MEMBER,
        "EndUser": Litellm_EntityType.END_USER,
    }[entity]
    alias_key: Final = reference.model_budget_spend_cache_key(scope, "id", "budget-latest", "1h")
    target_key: Final = reference.model_budget_spend_cache_key(scope, "id", "budget-target", "1h")
    cache.in_memory_cache.set_cache(alias_key, alias_spend)  # pyright: ignore[reportUnknownMemberType]  # legacy cache setter has no annotations
    cache.in_memory_cache.set_cache(target_key, target_spend)  # pyright: ignore[reportUnknownMemberType]  # legacy cache setter has no annotations
    shared_read: Final = AsyncMock(return_value=None)
    windows: Final = Windows(read=shared_read)
    io: Final = ModelBudgetIO(
        cache, groups, windows, 0, clock=lambda: 100, monotonic=lambda: 0, price_entry=alias_catalog.get
    )
    reading: Final = await read_model_budget(
        io,
        entity,
        "id",
        None,
        "budget-latest",
        {
            "budget-latest": {"max_budget": 1, "budget_duration": "1h"},
            "budget-target": {"max_budget": 2, "budget_duration": "1h"},
        },
    )
    decision: Final = model_budget_data.check(reading.snapshot)
    assert decision.denied is (expected is not None)
    assert reading.budget_model == (expected or "budget-target")
    assert decision.current_spend == (alias_spend if expected == "budget-latest" else target_spend)
    candidate: Final = fallback_candidate(
        FallbackCandidateFacts(
            True,
            True,
            True,
            membership="Loaded",
            price=ModelPriceFacts(cached=False),
            budget_models=("budget-latest",),
            budget_scopes=(CandidateBudgetScope(entity, True, True),),
            budgets=(CandidateBudgetRead(0, entity, reading.snapshot),),
        )
    )
    assert candidate.kind == ("Complete" if expected is not None else "ReadAuthorization")
    assert candidate.usable is False
    expected_shared: Final = (
        [] if entity == "EndUser" or alias_spend == 1 else [alias_key] if target_spend == 2 else [alias_key, target_key]
    )
    assert shared_read.call_args_list == [call(spend_key=key, budget_duration="1h") for key in expected_shared]


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", (("budget-latest",), ("budget-target",), ("budget-latest", "budget-target")))
async def test_team_overrides_apply_across_alias_and_target(
    alias_catalog: Mapping[str, object], overrides: tuple[str, ...]
) -> None:
    io: Final = ModelBudgetIO(DualCache(), groups, Windows(), 0, price_entry=alias_catalog.get)
    budgets: Final = {name: {"max_budget": 0, "budget_duration": "1h"} for name in ("budget-latest", "budget-target")}
    key: Final = {name: {"max_budget": 1, "budget_duration": "1h"} for name in overrides}
    reading: Final = await read_model_budget(io, "Team", "id", "id", "budget-latest", budgets, key)
    expected: Final = next((name for name in budgets if name not in overrides), None)
    assert model_budget_data.check(reading.snapshot).denied is (expected is not None)
    assert reading.budget_model == expected


@pytest.mark.asyncio
async def test_shared_native_model_flow_enforces_target_before_user_scope(alias_catalog: Mapping[str, object]) -> None:
    from starlette.requests import Request

    from litellm._auth.decision import Deny, ModelBudgetExceeded
    from litellm._auth.driver import drive_model_budgets
    from litellm._auth.rust_core import RustCore
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import (
        _ModelBudgetEffects,  # pyright: ignore[reportPrivateUsage]  # exercise the production raw effect boundary
    )

    token: Final = UserAPIKeyAuth(
        token="id",
        user_id="owner",
        key_alias="alias-key",
        model_max_budget={"budget-target": {"max_budget": 0, "budget_duration": "1h"}},
        user_model_max_budget={"budget-latest": {"max_budget": 0, "budget_duration": "1h"}},
    )
    effects: Final = _ModelBudgetEffects(
        request=Request({"type": "http", "headers": [], "path": "/chat/completions"}),
        request_data={"model": "budget-latest"},
        valid_token=token,
        route="/chat/completions",
        limiter=reference._PROXY_VirtualKeyModelMaxBudgetLimiter(DualCache(), clock=lambda: 100),  # pyright: ignore[reportPrivateUsage]  # concrete limiter integration
        llm_router=None,
        llm_model_list=None,
    )
    outcome: Final = await drive_model_budgets(RustCore().check_model_budgets(), effects)
    assert isinstance(outcome, Deny) and isinstance(outcome.denial, ModelBudgetExceeded)
    assert (outcome.denial.entity_type, outcome.denial.entity_id, outcome.denial.max_budget) == ("key", "id", 0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization_denied", (False, True))
async def test_fallback_probe_retains_original_denial_without_authorization_retries(
    alias_catalog: Mapping[str, object], authorization_denied: bool
) -> None:
    from starlette.requests import Request

    from litellm._auth.decision import Deny, ModelBudgetExceeded
    from litellm._auth.driver import drive_model_budgets
    from litellm._auth.rust_core import RustCore
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import (
        _ModelBudgetEffects,  # pyright: ignore[reportPrivateUsage]  # exercise the production raw effect boundary
    )

    data: Final[dict[str, object]] = {"model": "initial"}
    token: Final = UserAPIKeyAuth(
        token="id",
        models=["initial", "later"],
        model_max_budget={
            "initial": {"max_budget": 0, "budget_duration": "1h"},
            "budget-target": {"max_budget": 1 if authorization_denied else 0, "budget_duration": "1h"},
        },
        budget_fallbacks={"initial": ["budget-latest", "later"] if authorization_denied else ["budget-latest"]},
    )
    effects: Final = _ModelBudgetEffects(
        request=Request({"type": "http", "headers": [], "path": "/chat/completions"}),
        request_data=data,
        valid_token=token,
        route="/chat/completions",
        limiter=reference._PROXY_VirtualKeyModelMaxBudgetLimiter(  # pyright: ignore[reportPrivateUsage]  # concrete limiter integration
            DualCache(), clock=lambda: 100, shared_windows=Windows(read=AsyncMock(return_value=None))
        ),
        llm_router=None,
        llm_model_list=None,
    )
    outcome: Final = await drive_model_budgets(RustCore().check_model_budgets(), effects)
    assert isinstance(outcome, Deny) and isinstance(outcome.denial, ModelBudgetExceeded)
    assert "model=initial" in outcome.denial.message
    assert data == {"model": "initial"}
