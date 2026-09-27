"""
Spend tracking in RouterBudgetLimiting.async_log_success_event.

Only chat completions puts custom_llm_provider into litellm_params. The responses,
anthropic_messages, embedding and rerank surfaces leave it unset, which used to make
the callback raise before any spend was recorded, so those budgets never moved.
"""

from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import BudgetConfig


@pytest.fixture
def disable_budget_sync(monkeypatch):
    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        noop,
    )


def _success_kwargs(
    *,
    provider_in_litellm_params: str | None,
    provider_in_payload: str | None,
    call_type: str = "aresponses",
    response_cost: float = 0.25,
    model_id: str = "deployment-1",
) -> dict[str, object]:
    provider_params: Final[dict[str, str]] = (
        {} if provider_in_litellm_params is None else {"custom_llm_provider": provider_in_litellm_params}
    )
    litellm_params: Final[dict[str, str]] = {"model": "openai/gpt-4o", **provider_params}

    return {
        "call_type": call_type,
        "litellm_params": litellm_params,
        "standard_logging_object": {
            "response_cost": response_cost,
            "model_id": model_id,
            "custom_llm_provider": provider_in_payload,
        },
    }


async def _log_success(limiter: RouterBudgetLimiting, kwargs: dict[str, object]) -> None:
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["aresponses", "anthropic_messages", "aembedding", "arerank"])
async def test_provider_spend_tracked_when_litellm_params_omits_provider(disable_budget_sync, call_type):
    """Non-chat surfaces carry the provider only on the standard logging payload."""
    limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={"openai": {"budget_limit": 10.0, "time_period": "1d"}},
    )

    await _log_success(
        limiter,
        _success_kwargs(
            provider_in_litellm_params=None,
            provider_in_payload="openai",
            call_type=call_type,
        ),
    )

    assert await limiter.dual_cache.async_get_cache("provider_spend:openai:1d") == 0.25


@pytest.mark.asyncio
async def test_chat_completions_spend_still_tracked(disable_budget_sync):
    """Chat completions fills in both sources and must keep accumulating."""
    limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={"openai": {"budget_limit": 10.0, "time_period": "1d"}},
    )

    await _log_success(
        limiter,
        _success_kwargs(
            provider_in_litellm_params="openai",
            provider_in_payload="openai",
            call_type="acompletion",
        ),
    )

    assert await limiter.dual_cache.async_get_cache("provider_spend:openai:1d") == 0.25


@pytest.mark.asyncio
async def test_budget_of_other_provider_is_untouched(disable_budget_sync):
    """A provider without its own budget must not bleed into a configured one."""
    limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={"openai": {"budget_limit": 10.0, "time_period": "1d"}},
    )

    await _log_success(
        limiter,
        _success_kwargs(provider_in_litellm_params=None, provider_in_payload="anthropic"),
    )

    assert await limiter.dual_cache.async_get_cache("provider_spend:openai:1d") in (None, 0.0)


@pytest.mark.asyncio
async def test_deployment_budget_tracked_when_provider_is_unresolvable(disable_budget_sync):
    """An unresolvable provider must not abort the deployment and tag budgets that follow it."""
    limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config=None,
        model_list=[
            {
                "model_name": "some-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "max_budget": 10.0,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "deployment-1"},
            }
        ],
    )

    await _log_success(
        limiter,
        _success_kwargs(provider_in_litellm_params=None, provider_in_payload=None),
    )

    assert await limiter.dual_cache.async_get_cache("deployment_spend:deployment-1:1d") == 0.25


def _deployment(model_id: str = "deployment-1") -> dict[str, object]:
    return {
        "model_name": "gpt-4",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"id": model_id},
    }


def _budget(*, limit: float | None, duration: str = "1d") -> BudgetConfig:
    return BudgetConfig(budget_limit=limit, time_period=duration)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["provider", "deployment", "tag"])
async def test_zero_max_budget_blocks_spend(disable_budget_sync, scope: str) -> None:
    """`max_budget=0` means "block all spend", not "unlimited".

    Each of the three guards read `if config.max_budget and ...`, so a configured 0 was falsy
    and the comparison was skipped, leaving that scope unlimited. The provider guard already
    uses `is not None` a few lines above for exactly this reason.
    """
    limiter = RouterBudgetLimiting(dual_cache=DualCache(), provider_budget_config=None)
    deployment = _deployment()

    provider_configs: dict[str, BudgetConfig] = {}
    deployment_configs: dict[str, BudgetConfig] = {}
    deployment_providers: list[str | None] = []
    spend_map: dict[str, float] = {}
    request_tags: list[str] = []

    if scope == "provider":
        limiter.provider_budget_config = {"openai": _budget(limit=0.0)}
        provider_configs = {"openai": _budget(limit=0.0)}
        deployment_providers = ["openai"]
        spend_map = {"provider_spend:openai:1d": 0.5}
    elif scope == "deployment":
        limiter.deployment_budget_config = {"deployment-1": _budget(limit=0.0)}
        deployment_configs = {"deployment-1": _budget(limit=0.0)}
        spend_map = {"deployment_spend:deployment-1:1d": 0.5}
    else:
        limiter.tag_budget_config = {"prod": _budget(limit=0.0)}
        spend_map = {"tag_spend:prod:1d": 0.5}
        request_tags = ["prod"]

    kept, blocked_info = limiter._filter_out_deployments_above_budget(
        potential_deployments=[],
        healthy_deployments=[deployment],
        provider_configs=provider_configs,
        deployment_configs=deployment_configs,
        deployment_providers=deployment_providers,
        spend_map=spend_map,
        request_tags=request_tags,
    )

    assert kept == [], f"{scope} budget of 0 must not keep the deployment"
    assert blocked_info != "", f"{scope} budget of 0 must report why it blocked"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit,spend,expected_kept",
    [(10.0, 0.5, True), (10.0, 10.0, False)],
)
async def test_provider_budget_boundary_is_unchanged(
    disable_budget_sync, limit: float | None, spend: float, expected_kept: bool
) -> None:
    """`is not None` must not change behaviour for non-zero or unset budgets."""
    limiter = RouterBudgetLimiting(dual_cache=DualCache(), provider_budget_config={"openai": _budget(limit=limit)})
    kept, _ = limiter._filter_out_deployments_above_budget(
        potential_deployments=[],
        healthy_deployments=[_deployment()],
        provider_configs={"openai": _budget(limit=limit)},
        deployment_configs={},
        deployment_providers=["openai"],
        spend_map={"provider_spend:openai:1d": spend},
        request_tags=[],
    )

    assert (kept != []) is expected_kept
