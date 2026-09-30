from typing import Final
from unittest.mock import AsyncMock

import litellm
import pytest

from litellm import Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.fallback_budget import (
    RouterFallbackBudgetCheck,
    is_token_within_budget_for_model,
    router_fallback_budget_check,
)

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
async def test_team_member_budget_refuses_an_exhausted_paid_target() -> None:
    member_caps: Final = {"paid-model": {"max_budget": 5.0, "budget_duration": "1d"}}
    token: Final = _token(
        user_id="u1",
        team_id="t1",
        team_member_model_max_budget=member_caps,
    )
    limiter: Final = AsyncMock()
    limiter.is_team_member_within_model_budget.side_effect = litellm.BudgetExceededError(
        current_cost=10,
        max_budget=5,
    )

    within_budget: Final = await is_token_within_budget_for_model(
        model="paid-model",
        valid_token=token,
        llm_router=_router(),
        team_member_model_budget_limiter=limiter,
    )

    assert within_budget is False
    limiter.is_team_member_within_model_budget.assert_awaited_once_with(
        user_id="u1",
        team_id="t1",
        team_member_model_max_budget=member_caps,
        model="paid-model",
    )


@pytest.mark.asyncio
async def test_team_member_budget_allows_a_paid_target_when_under_cap() -> None:
    member_caps: Final = {"paid-model": {"max_budget": 5.0, "budget_duration": "1d"}}
    token: Final = _token(
        user_id="u1",
        team_id="t1",
        team_member_model_max_budget=member_caps,
    )
    limiter: Final = AsyncMock()
    limiter.is_team_member_within_model_budget.return_value = True

    within_budget: Final = await is_token_within_budget_for_model(
        model="paid-model",
        valid_token=token,
        llm_router=_router(),
        team_member_model_budget_limiter=limiter,
    )

    assert within_budget is True
    limiter.is_team_member_within_model_budget.assert_awaited_once()


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
    limiter: Final = AsyncMock()

    within_budget: Final = await is_token_within_budget_for_model(
        model="paid-model",
        valid_token=token,
        llm_router=_router(),
        team_member_model_budget_limiter=limiter,
    )

    assert within_budget is True
    limiter.is_team_member_within_model_budget.assert_not_awaited()


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
    assert await NOT_ENFORCED(model="paid-model", request_kwargs=request, llm_router=_router()) is True


@pytest.mark.asyncio
async def test_request_without_a_key_is_unrestricted():
    assert await ENFORCED(model="paid-model", request_kwargs={}, llm_router=_router()) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
async def test_enforced_check_reads_the_key_from_request_metadata(metadata_field: str):
    over = {metadata_field: {"user_api_key_auth": _token(user_spend=1900.0, user_max_budget=50.0)}}
    under = {metadata_field: {"user_api_key_auth": _token(user_spend=1.0, user_max_budget=50.0)}}

    assert await ENFORCED(model="paid-model", request_kwargs=over, llm_router=_router()) is False
    assert await ENFORCED(model="paid-model", request_kwargs=under, llm_router=_router()) is True


@pytest.mark.asyncio
async def test_router_check_passes_member_limiter_from_request_metadata() -> None:
    member_caps: Final = {"paid-model": {"max_budget": 5.0, "budget_duration": "1d"}}
    token: Final = _token(
        user_id="u1",
        team_id="t1",
        team_member_model_max_budget=member_caps,
    )
    limiter: Final = AsyncMock()
    limiter.is_team_member_within_model_budget.side_effect = litellm.BudgetExceededError(
        current_cost=10,
        max_budget=5,
    )
    budget_check: Final = RouterFallbackBudgetCheck(
        is_enforced=lambda: True,
        team_member_model_budget_limiter=limiter,
    )
    request_kwargs: Final = {"metadata": {"user_api_key_auth": token}}

    within_budget: Final = await budget_check(
        model="paid-model",
        request_kwargs=request_kwargs,
        llm_router=_router(),
    )

    assert within_budget is False
    limiter.is_team_member_within_model_budget.assert_awaited_once_with(
        user_id="u1",
        team_id="t1",
        team_member_model_max_budget=member_caps,
        model="paid-model",
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

    assert await ENFORCED(model="paid-model", request_kwargs=request, llm_router=_router()) is False


@pytest.mark.asyncio
async def test_router_skips_the_paid_fallback_target_when_over_budget():
    from litellm.router_utils.fallback_event_handlers import _is_fallback_target_within_budget

    router = Router(
        model_list=[FREE_MODEL, PAID_MODEL],
        fallbacks=[{"free-model": ["paid-model"]}],
        fallback_budget_check=ENFORCED,
    )
    over = {"metadata": {"user_api_key_auth": _token(user_spend=1900.0, user_max_budget=50.0)}}
    under = {"metadata": {"user_api_key_auth": _token(user_spend=1.0, user_max_budget=50.0)}}

    assert await _is_fallback_target_within_budget(router, "paid-model", "free-model", over) is False
    assert await _is_fallback_target_within_budget(router, "paid-model", "free-model", under) is True


@pytest.mark.asyncio
async def test_router_without_a_budget_check_attempts_every_fallback():
    from litellm.router_utils.fallback_event_handlers import _is_fallback_target_within_budget

    router = _router()  # fallback_budget_check defaults to None
    over = {"metadata": {"user_api_key_auth": _token(user_spend=1900.0, user_max_budget=50.0)}}

    assert await _is_fallback_target_within_budget(router, "paid-model", "free-model", over) is True


@pytest.mark.asyncio
async def test_enforcement_is_on_by_default_and_opt_out_restores_the_leak(monkeypatch):
    """
    Leaving the paid fallback unguarded is the budget bypass this module exists to close, so an
    unconfigured proxy has to enforce. `enforce_fallback_budget: false` is the deliberate opt-out.
    """
    from litellm.proxy import proxy_server

    over = {"metadata": {"user_api_key_auth": _token(user_spend=1900.0, user_max_budget=50.0)}}

    monkeypatch.setattr(proxy_server, "general_settings", {}, raising=False)
    assert await router_fallback_budget_check(model="paid-model", request_kwargs=over, llm_router=_router()) is False

    monkeypatch.setattr(proxy_server, "general_settings", {"enforce_fallback_budget": False}, raising=False)
    assert await router_fallback_budget_check(model="paid-model", request_kwargs=over, llm_router=_router()) is True
