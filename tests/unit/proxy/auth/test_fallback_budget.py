from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
import time_machine

from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import Litellm_EntityType, UserAPIKeyAuth
from litellm.proxy.auth.fallback_budget import (
    RouterFallbackBudgetCheck,
    is_token_within_budget_for_model,
    router_fallback_budget_check,
)
from litellm.proxy.hooks.model_max_budget_limiter import (
    PROXY_VirtualKeyModelMaxBudgetLimiter,
    model_budget_spend_cache_key,
    team_member_budget_entity_id,
)
from tests.unit.test_router.core.scripted_host import ScriptedHost, rate_limit, rust_core

FREE_MODEL = {
    "model_name": "free-model",
    "litellm_params": {
        "model": "ollama/llama2",
        "api_base": "http://localhost:11434",
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
    "model_info": {
        "id": "free-model-id",
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
}

PAID_MODEL = {
    "model_name": "paid-model",
    "litellm_params": {"model": "openai/gpt-4o", "api_key": "k"},
    "model_info": {"id": "paid-model-id"},
}


def _router() -> Router:
    return Router(model_list=[FREE_MODEL, PAID_MODEL], fallbacks=[{"free-model": ["paid-model"]}])


def _token(**overrides) -> UserAPIKeyAuth:
    fields = {
        "api_key": "hashed",
        "token": "hashed",
        "spend": 0.0,
        "max_budget": None,
        "user_id": "u1",
        "user_spend": 0.0,
        "user_max_budget": None,
    }
    fields.update(overrides)
    return UserAPIKeyAuth(**fields)


async def _team_member_limiter(
    spend_by_model: Mapping[str, float] | None = None,
) -> tuple[PROXY_VirtualKeyModelMaxBudgetLimiter, MagicMock]:
    entity_id: Final = team_member_budget_entity_id(user_id="u1", team_id="t1")
    spend_by_key: Final = {
        model_budget_spend_cache_key(
            entity_type=Litellm_EntityType.TEAM_MEMBER,
            entity_id=entity_id,
            budget_model=model,
            budget_duration="1d",
        ): spend
        for model, spend in (spend_by_model or {}).items()
    }
    redis_cache: Final = MagicMock()

    async def get_spend(*, key: str) -> object | None:
        return spend_by_key.get(key)

    redis_cache.async_get_cache = AsyncMock(side_effect=get_spend)
    return PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache(redis_cache=redis_cache)), redis_cache


def _member_budget_spend_keys(models: Sequence[str]) -> tuple[str, ...]:
    entity_id: Final = team_member_budget_entity_id(user_id="u1", team_id="t1")
    return tuple(
        model_budget_spend_cache_key(
            entity_type=Litellm_EntityType.TEAM_MEMBER,
            entity_id=entity_id,
            budget_model=model,
            budget_duration="1d",
        )
        for model in models
    )


ENFORCED = RouterFallbackBudgetCheck(is_enforced=lambda: True)
NOT_ENFORCED = RouterFallbackBudgetCheck(is_enforced=lambda: False)


@pytest.mark.asyncio
async def test_paid_target_allowed_when_under_budget():
    token = _token(spend=1.0, max_budget=50.0, user_spend=1.0, user_max_budget=50.0)
    assert await is_token_within_budget_for_model(model="paid-model", valid_token=token, llm_router=_router()) is True


@pytest.mark.asyncio
async def test_paid_target_refused_when_over_key_budget():
    token = _token(spend=100.0, max_budget=50.0)
    assert await is_token_within_budget_for_model(model="paid-model", valid_token=token, llm_router=_router()) is False


@pytest.mark.asyncio
async def test_paid_target_refused_when_over_user_budget():
    token = _token(user_spend=1900.0, user_max_budget=50.0)
    assert await is_token_within_budget_for_model(model="paid-model", valid_token=token, llm_router=_router()) is False


@pytest.mark.asyncio
async def test_paid_target_loses_temporary_user_budget_at_expiry() -> None:
    expiry: Final = datetime(2030, 1, 1, tzinfo=timezone.utc)
    user: Final = LiteLLM_UserTable(user_id="u1", max_budget=10, temp_budget_increase=20, temp_budget_expiry=expiry)
    token: Final = UserAPIKeyAuth(user_id="u1", user_spend=15, user_max_budget=30)
    token.carry_temporary_user_budget(user)
    router: Final = _router()
    with time_machine.travel(expiry - timedelta(seconds=1), tick=False):
        assert await is_token_within_budget_for_model(model="paid-model", valid_token=token, llm_router=router)
    with time_machine.travel(expiry, tick=False):
        assert not await is_token_within_budget_for_model(model="paid-model", valid_token=token, llm_router=router)


@pytest.mark.asyncio
async def test_team_member_budget_refuses_an_exhausted_paid_target() -> None:
    member_caps: Final = {"paid-model": {"max_budget": 5.0, "budget_duration": "1d"}}
    token: Final = _token(
        user_id="u1",
        team_id="t1",
        team_member_model_max_budget=member_caps,
    )
    limiter, redis_cache = await _team_member_limiter(spend_by_model={"paid-model": 10.0})

    within_budget: Final = await is_token_within_budget_for_model(
        model="paid-model",
        valid_token=token,
        llm_router=_router(),
        team_member_model_budget_limiter=limiter,
    )

    assert within_budget is False
    assert tuple(call.kwargs["key"] for call in redis_cache.async_get_cache.call_args_list) == (
        _member_budget_spend_keys(("paid-model",))
    )


@pytest.mark.asyncio
async def test_router_fallback_budget_check_resolves_model_group_alias_before_member_cap_check() -> None:
    expensive_model: Final = {
        **PAID_MODEL,
        "model_name": "expensive",
        "model_info": {"id": "expensive-model-id"},
    }
    router: Final = Router(
        model_list=[FREE_MODEL, expensive_model],
        model_group_alias={"backup": "expensive"},
    )
    member_caps: Final = {
        "backup": {"max_budget": 100.0, "budget_duration": "1d"},
        "expensive": {"max_budget": 5.0, "budget_duration": "1d"},
    }
    token: Final = _token(user_id="u1", team_id="t1", team_member_model_max_budget=member_caps)
    limiter, redis_cache = await _team_member_limiter(spend_by_model={"backup": 1.0, "expensive": 10.0})
    budget_check: Final = RouterFallbackBudgetCheck(
        is_enforced=lambda: True,
        team_member_model_budget_limiter=limiter,
    )

    within_budget: Final = await budget_check(
        model="backup",
        request_kwargs={"metadata": {"user_api_key_auth": token}},
        ctx=router._core_ctx,
    )

    assert within_budget is False
    assert tuple(call.kwargs["key"] for call in redis_cache.async_get_cache.call_args_list) == (
        _member_budget_spend_keys(("backup", "expensive"))
    )


@pytest.mark.asyncio
async def test_router_fallback_budget_check_refuses_an_exhausted_alias_cap() -> None:
    expensive_model: Final = {
        **PAID_MODEL,
        "model_name": "expensive",
        "model_info": {"id": "expensive-model-id"},
    }
    router: Final = Router(
        model_list=[FREE_MODEL, expensive_model],
        model_group_alias={"backup": "expensive"},
    )
    member_caps: Final = {"backup": {"max_budget": 5.0, "budget_duration": "1d"}}
    token: Final = _token(user_id="u1", team_id="t1", team_member_model_max_budget=member_caps)
    limiter, redis_cache = await _team_member_limiter(spend_by_model={"backup": 10.0})
    budget_check: Final = RouterFallbackBudgetCheck(
        is_enforced=lambda: True,
        team_member_model_budget_limiter=limiter,
    )

    within_budget: Final = await budget_check(
        model="backup",
        request_kwargs={"metadata": {"user_api_key_auth": token}},
        ctx=router._core_ctx,
    )

    assert within_budget is False
    assert tuple(call.kwargs["key"] for call in redis_cache.async_get_cache.call_args_list) == (
        _member_budget_spend_keys(("backup",))
    )


@pytest.mark.asyncio
async def test_router_fallback_budget_check_allows_alias_when_both_names_are_within_cap() -> None:
    expensive_model: Final = {
        **PAID_MODEL,
        "model_name": "expensive",
        "model_info": {"id": "expensive-model-id"},
    }
    router: Final = Router(
        model_list=[FREE_MODEL, expensive_model],
        model_group_alias={"backup": "expensive"},
    )
    member_caps: Final = {
        "backup": {"max_budget": 5.0, "budget_duration": "1d"},
        "expensive": {"max_budget": 5.0, "budget_duration": "1d"},
    }
    token: Final = _token(user_id="u1", team_id="t1", team_member_model_max_budget=member_caps)
    limiter, redis_cache = await _team_member_limiter(spend_by_model={"backup": 1.0, "expensive": 1.0})
    budget_check: Final = RouterFallbackBudgetCheck(
        is_enforced=lambda: True,
        team_member_model_budget_limiter=limiter,
    )

    within_budget: Final = await budget_check(
        model="backup",
        request_kwargs={"metadata": {"user_api_key_auth": token}},
        ctx=router._core_ctx,
    )

    assert within_budget is True
    assert tuple(call.kwargs["key"] for call in redis_cache.async_get_cache.call_args_list) == (
        _member_budget_spend_keys(("backup", "expensive"))
    )


@pytest.mark.asyncio
async def test_router_fallback_budget_check_deduplicates_non_alias_target() -> None:
    router: Final = _router()
    member_caps: Final = {"paid-model": {"max_budget": 5.0, "budget_duration": "1d"}}
    token: Final = _token(user_id="u1", team_id="t1", team_member_model_max_budget=member_caps)
    limiter, redis_cache = await _team_member_limiter(spend_by_model={"paid-model": 1.0})
    budget_check: Final = RouterFallbackBudgetCheck(
        is_enforced=lambda: True,
        team_member_model_budget_limiter=limiter,
    )

    within_budget: Final = await budget_check(
        model="paid-model",
        request_kwargs={"metadata": {"user_api_key_auth": token}},
        ctx=router._core_ctx,
    )

    assert within_budget is True
    assert tuple(call.kwargs["key"] for call in redis_cache.async_get_cache.call_args_list) == (
        _member_budget_spend_keys(("paid-model",))
    )


@pytest.mark.asyncio
async def test_team_member_budget_allows_a_paid_target_when_under_cap() -> None:
    member_caps: Final = {"paid-model": {"max_budget": 5.0, "budget_duration": "1d"}}
    token: Final = _token(
        user_id="u1",
        team_id="t1",
        team_member_model_max_budget=member_caps,
    )
    limiter, redis_cache = await _team_member_limiter(spend_by_model={"paid-model": 1.0})

    within_budget: Final = await is_token_within_budget_for_model(
        model="paid-model",
        valid_token=token,
        llm_router=_router(),
        team_member_model_budget_limiter=limiter,
    )

    assert within_budget is True
    assert tuple(call.kwargs["key"] for call in redis_cache.async_get_cache.call_args_list) == (
        _member_budget_spend_keys(("paid-model",))
    )


@pytest.mark.asyncio
async def test_team_member_caps_are_not_evaluated_without_a_limiter() -> None:
    token: Final = _token(
        user_id="u1",
        team_id="t1",
        team_member_model_max_budget={"paid-model": {"max_budget": 5.0, "budget_duration": "1d"}},
    )

    assert await is_token_within_budget_for_model(model="paid-model", valid_token=token, llm_router=_router()) is True


@pytest.mark.asyncio
async def test_team_member_limiter_is_not_called_without_member_caps() -> None:
    token: Final = _token(user_id="u1", team_id="t1")
    limiter, redis_cache = await _team_member_limiter()

    within_budget: Final = await is_token_within_budget_for_model(
        model="paid-model",
        valid_token=token,
        llm_router=_router(),
        team_member_model_budget_limiter=limiter,
    )

    assert within_budget is True
    assert redis_cache.async_get_cache.call_args_list == []


@pytest.mark.asyncio
async def test_zero_cost_target_allowed_even_when_over_budget():
    """Refusing a free target would deny a request on spend some other model accrued."""
    token = _token(user_spend=1900.0, user_max_budget=50.0)
    assert await is_token_within_budget_for_model(model="free-model", valid_token=token, llm_router=_router()) is True


@pytest.mark.asyncio
async def test_no_budget_configured_is_always_within_budget():
    token = _token(spend=9999.0, user_spend=9999.0)
    assert await is_token_within_budget_for_model(model="paid-model", valid_token=token, llm_router=_router()) is True


@pytest.mark.asyncio
async def test_team_key_does_not_inherit_personal_budget_by_default(monkeypatch):
    """Mirrors _PROXY_MaxBudgetLimiter: a team key ignores the owner's personal cap."""
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {}, raising=False)
    token = _token(team_id="t1", user_spend=1900.0, user_max_budget=50.0)
    assert await is_token_within_budget_for_model(model="paid-model", valid_token=token, llm_router=_router()) is True


@pytest.mark.asyncio
async def test_team_key_inherits_personal_budget_when_opted_in(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"apply_user_budget_to_team_keys": True}, raising=False)
    token = _token(team_id="t1", user_spend=1900.0, user_max_budget=50.0)
    assert await is_token_within_budget_for_model(model="paid-model", valid_token=token, llm_router=_router()) is False


@pytest.mark.asyncio
async def test_check_is_a_no_op_while_not_enforced():
    request = {"metadata": {"user_api_key_auth": _token(user_spend=1900.0, user_max_budget=50.0)}}
    assert await NOT_ENFORCED(model="paid-model", request_kwargs=request, ctx=_router()._core_ctx) is True


@pytest.mark.asyncio
async def test_request_without_a_key_is_unrestricted():
    assert await ENFORCED(model="paid-model", request_kwargs={}, ctx=_router()._core_ctx) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
async def test_enforced_check_reads_the_key_from_request_metadata(metadata_field: str):
    over = {metadata_field: {"user_api_key_auth": _token(user_spend=1900.0, user_max_budget=50.0)}}
    under = {metadata_field: {"user_api_key_auth": _token(user_spend=1.0, user_max_budget=50.0)}}

    assert await ENFORCED(model="paid-model", request_kwargs=over, ctx=_router()._core_ctx) is False
    assert await ENFORCED(model="paid-model", request_kwargs=under, ctx=_router()._core_ctx) is True


@pytest.mark.asyncio
async def test_router_check_passes_member_limiter_from_request_metadata() -> None:
    member_caps: Final = {"paid-model": {"max_budget": 5.0, "budget_duration": "1d"}}
    token: Final = _token(
        user_id="u1",
        team_id="t1",
        team_member_model_max_budget=member_caps,
    )
    limiter, redis_cache = await _team_member_limiter(spend_by_model={"paid-model": 10.0})
    budget_check: Final = RouterFallbackBudgetCheck(
        is_enforced=lambda: True,
        team_member_model_budget_limiter=limiter,
    )
    request_kwargs: Final = {"metadata": {"user_api_key_auth": token}}

    within_budget: Final = await budget_check(
        model="paid-model",
        request_kwargs=request_kwargs,
        ctx=_router()._core_ctx,
    )

    assert within_budget is False
    assert tuple(call.kwargs["key"] for call in redis_cache.async_get_cache.call_args_list) == (
        _member_budget_spend_keys(("paid-model",))
    )


@pytest.mark.asyncio
async def test_a_stale_low_counter_still_refuses_a_paid_target(monkeypatch):
    """
    The counter can read low (e.g. restored from an older Redis snapshot). Passing the budget makes
    `get_current_spend` verify against authoritative spend instead of trusting that read, so the
    paid target is still refused.
    """
    from litellm.proxy import proxy_server

    seen: list[dict] = []

    async def _stale_counter(**kwargs):
        seen.append(kwargs)
        # a stale-low counter read; the authoritative spend is what the budget must be judged on
        return 0.0 if kwargs.get("max_budget") is None else kwargs["fallback_spend"]

    monkeypatch.setattr(proxy_server, "get_current_spend", _stale_counter, raising=False)
    token = _token(user_spend=1900.0, user_max_budget=50.0)

    assert await is_token_within_budget_for_model(model="paid-model", valid_token=token, llm_router=_router()) is False
    assert [call["max_budget"] for call in seen] == [50.0]


@pytest.mark.asyncio
async def test_check_fails_closed_when_the_spend_lookup_breaks(monkeypatch):
    from litellm.proxy import proxy_server

    async def _boom(**kwargs):
        raise RuntimeError("spend counter unavailable")

    monkeypatch.setattr(proxy_server, "get_current_spend", _boom, raising=False)
    request = {"metadata": {"user_api_key_auth": _token(user_spend=1.0, user_max_budget=50.0)}}

    assert await ENFORCED(model="paid-model", request_kwargs=request, ctx=_router()._core_ctx) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("check", "spend", "expected"),
    [
        (ENFORCED, 1900.0, ("free-model-id", "backup-id")),
        (ENFORCED, 1.0, ("free-model-id", "paid-model-id")),
        (None, 1900.0, ("free-model-id", "paid-model-id")),
    ],
)
async def test_router_checks_the_budget_before_attempting_a_fallback(
    check: RouterFallbackBudgetCheck | None, spend: float, expected: tuple[str, str]
) -> None:
    host: Final = ScriptedHost()
    core: Final = rust_core(
        host,
        model_list=[
            FREE_MODEL,
            PAID_MODEL,
            {
                **FREE_MODEL,
                "model_name": "backup",
                "litellm_params": {**FREE_MODEL["litellm_params"], "model": "ollama/backup"},
                "model_info": {**FREE_MODEL["model_info"], "id": "backup-id"},
            },
        ],
        fallbacks=[{"free-model": ["paid-model", "backup"]}],
        fallback_budget_check=check,
        num_retries=0,
    )
    host.answer("ollama/llama2", rate_limit())
    response: Final = await core.acompletion(
        model="free-model",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"user_api_key_auth": _token(user_spend=spend, user_max_budget=50.0)},
    )
    assert tuple(host.deployments_called("acompletion")) == expected
    assert response._hidden_params["model_id"] == expected[-1]


@pytest.mark.asyncio
async def test_enforcement_is_on_by_default_and_opt_out_restores_the_leak(monkeypatch):
    """
    Leaving the paid fallback unguarded is the budget bypass this module exists to close, so an
    unconfigured proxy has to enforce. `enforce_fallback_budget: false` is the deliberate opt-out.
    """
    from litellm.proxy import proxy_server

    over = {"metadata": {"user_api_key_auth": _token(user_spend=1900.0, user_max_budget=50.0)}}

    monkeypatch.setattr(proxy_server, "general_settings", {}, raising=False)
    assert await router_fallback_budget_check(model="paid-model", request_kwargs=over, ctx=_router()._core_ctx) is False

    monkeypatch.setattr(proxy_server, "general_settings", {"enforce_fallback_budget": False}, raising=False)
    assert await router_fallback_budget_check(model="paid-model", request_kwargs=over, ctx=_router()._core_ctx) is True
