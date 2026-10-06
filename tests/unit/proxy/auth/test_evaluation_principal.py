"""The shadow-eval creator principal: who it is, what its stamped calls charge, and
whether the request path's own budget owners let it pay."""

from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth import evaluation_principal as module
from litellm.proxy.auth.evaluation_principal import (
    evaluation_principal,
    principal_call_metadata,
    principal_can_pay_for,
)
from litellm.proxy.hooks.model_max_budget_limiter import _PROXY_VirtualKeyModelMaxBudgetLimiter
from litellm.models.user import LiteLLM_UserTable
from litellm.types.proxy.auth.auth_checks import UserNotFoundError

PAID_MODEL: Final = {
    "model_name": "eval-judge",
    "litellm_params": {
        "model": "openai/gpt-4o",
        "api_key": "k",
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000002,
    },
    "model_info": {"id": "eval-judge-id"},
}


def _creator(**overrides) -> LiteLLM_UserTable:
    fields = {"user_id": "eval-admin", "user_role": "proxy_admin", "spend": 0.0, "max_budget": 10.0}
    return LiteLLM_UserTable(**{**fields, **overrides})


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> _PROXY_VirtualKeyModelMaxBudgetLimiter:
    limiter: Final = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())

    async def counter_spend(counter_key: str, fallback_spend: float, max_budget: float | None = None) -> float:
        return fallback_spend

    monkeypatch.setattr(proxy_server, "llm_router", Router(model_list=[PAID_MODEL]), raising=False)
    monkeypatch.setattr(proxy_server, "model_max_budget_limiter", limiter, raising=False)
    monkeypatch.setattr(proxy_server, "get_current_spend", counter_spend, raising=False)
    monkeypatch.setattr(proxy_server, "general_settings", {}, raising=False)
    monkeypatch.setattr(proxy_server, "prisma_client", object(), raising=False)
    return limiter


@pytest.mark.asyncio
async def test_spend_charged_under_the_stamp_exhausts_the_budget_the_gate_reads(proxy, monkeypatch):
    """The workflow end to end: an evaluation call stamped as the creator is charged by the
    real per-model limiter, and the next admission reads that same counter and refuses."""
    monkeypatch.setattr(
        module,
        "get_user_object",
        AsyncMock(
            return_value=_creator(model_max_budget={"eval-judge": {"budget_limit": 0.001, "time_period": "1d"}})
        ),
    )
    principal = await evaluation_principal("eval-admin")
    assert await principal_can_pay_for(principal, ("eval-judge",)) is True

    stamped = principal_call_metadata(principal)
    await proxy.async_log_success_event(
        {
            "standard_logging_object": {
                "call_type": "acompletion",
                "response_cost": 0.002,
                "model": "openai/gpt-4o",
                "model_group": "eval-judge",
                "metadata": {k: stamped.get(k) for k in ("user_api_key_hash", "user_api_key_user_id")},
            },
            "litellm_params": {"metadata": dict(stamped)},
        },
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert stamped["user_api_key_user_id"] == "eval-admin"
    assert await principal_can_pay_for(principal, ("eval-judge",)) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "creator,models,expected",
    [
        (_creator(spend=10.0), ("eval-judge",), False),
        (_creator(spend=10.0), (), True),
        (_creator(spend=9.0), ("eval-judge",), True),
        (_creator(spend=999.0, max_budget=None), ("eval-judge",), True),
    ],
)
async def test_total_budget_gates_every_paid_model(
    proxy: _PROXY_VirtualKeyModelMaxBudgetLimiter,
    monkeypatch: pytest.MonkeyPatch,
    creator: LiteLLM_UserTable,
    models: tuple[str, ...],
    expected: bool,
):
    monkeypatch.setattr(module, "get_user_object", AsyncMock(return_value=creator))
    assert await principal_can_pay_for(await evaluation_principal("eval-admin"), models) is expected


@pytest.mark.asyncio
async def test_a_deleted_creator_withholds_instead_of_becoming_an_unbudgeted_admin(proxy, monkeypatch):
    monkeypatch.setattr(module, "get_user_object", AsyncMock(side_effect=UserNotFoundError(user_id="eval-admin")))

    with pytest.raises(UserNotFoundError):
        await evaluation_principal("eval-admin")


@pytest.mark.asyncio
async def test_a_creator_without_a_user_row_is_the_unbudgeted_master_key_admin(proxy, monkeypatch):
    monkeypatch.setattr(module, "get_user_object", AsyncMock(side_effect=UserNotFoundError(user_id="default_user_id")))

    principal = await evaluation_principal("default_user_id")

    assert (principal.user_id, principal.user_role) == ("default_user_id", LitellmUserRoles.PROXY_ADMIN)
    assert await principal_can_pay_for(principal, ("eval-judge",)) is True
    assert principal_call_metadata(principal)["user_api_key_user_id"] == "default_user_id"


@pytest.mark.asyncio
async def test_an_unreadable_creator_row_propagates_so_the_caller_withholds(proxy, monkeypatch):
    monkeypatch.setattr(module, "get_user_object", AsyncMock(side_effect=RuntimeError("db down")))

    with pytest.raises(RuntimeError, match="db down"):
        await evaluation_principal("eval-admin")
