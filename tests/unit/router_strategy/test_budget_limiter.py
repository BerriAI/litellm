"""
Spend tracking in RouterBudgetLimiting.async_log_success_event.

Only chat completions puts custom_llm_provider into litellm_params. The responses,
anthropic_messages, embedding and rerank surfaces leave it unset, which used to make
the callback raise before any spend was recorded, so those budgets never moved.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, assert_never

import pytest

from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.router import DeploymentTypedDict
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


_Scope = Literal["provider", "deployment", "tag"]
_DURATION: Final = "1d"
_MODEL_ID: Final = "dep-1"
_DEPLOYMENT: Final[dict[str, object]] = {
    "model_name": "gpt-4",
    "litellm_params": {"model": "openai/gpt-4"},
    "model_info": {"id": _MODEL_ID},
}


def _blocked_debug(scope: _Scope, spend: float, cap: float | None) -> str:
    if scope == "provider":
        return f"Exceeded budget for provider openai: {spend} >= {cap}\n"
    if scope == "deployment":
        return (
            "Exceeded budget for deployment model_name: gpt-4, litellm_params.model: openai/gpt-4, "
            f"model_id: {_MODEL_ID}: {spend} >= {cap}\n"
        )
    return f"Exceeded budget for tag='prod', tag_spend={spend}, tag_budget_limit={cap}\n"


def _apply_budget_filter(
    *,
    provider_budget: Mapping[str, BudgetConfig] | None,
    deployment_budget: Mapping[str, BudgetConfig] | None,
    tag_budget: Mapping[str, BudgetConfig] | None,
    provider_configs: Mapping[str, BudgetConfig],
    deployment_configs: Mapping[str, BudgetConfig],
    deployment_providers: Sequence[str | None],
    request_tags: Sequence[str],
    spend_key: str,
    spend: float,
):
    limiter: Final = RouterBudgetLimiting(dual_cache=DualCache(), provider_budget_config=None)
    limiter.provider_budget_config = provider_budget
    limiter.deployment_budget_config = deployment_budget
    limiter.tag_budget_config = tag_budget
    return limiter._filter_out_deployments_above_budget(
        potential_deployments=[],  # mutable-ok: the filter appends each deployment it keeps
        healthy_deployments=(_DEPLOYMENT,),
        provider_configs=provider_configs,
        deployment_configs=deployment_configs,
        deployment_providers=deployment_providers,
        spend_map=MappingProxyType({spend_key: spend}),
        request_tags=request_tags,
    )


def _filter_one(scope: _Scope, config: BudgetConfig, spend: float):
    match scope:
        case "provider":
            return _apply_budget_filter(
                provider_budget=MappingProxyType({"openai": config}),
                deployment_budget=None,
                tag_budget=None,
                provider_configs=MappingProxyType({"openai": config}),
                deployment_configs=MappingProxyType({}),
                deployment_providers=("openai",),
                request_tags=(),
                spend_key=f"provider_spend:openai:{_DURATION}",
                spend=spend,
            )
        case "deployment":
            return _apply_budget_filter(
                provider_budget=None,
                deployment_budget=MappingProxyType({_MODEL_ID: config}),
                tag_budget=None,
                provider_configs=MappingProxyType({}),
                deployment_configs=MappingProxyType({_MODEL_ID: config}),
                deployment_providers=(),
                request_tags=(),
                spend_key=f"deployment_spend:{_MODEL_ID}:{_DURATION}",
                spend=spend,
            )
        case "tag":
            return _apply_budget_filter(
                provider_budget=None,
                deployment_budget=None,
                tag_budget=MappingProxyType({"prod": config}),
                provider_configs=MappingProxyType({}),
                deployment_configs=MappingProxyType({}),
                deployment_providers=(),
                request_tags=("prod",),
                spend_key=f"tag_spend:prod:{_DURATION}",
                spend=spend,
            )
        case _:
            assert_never(scope)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "max_budget", "spend", "stays"),
    [
        ("provider", None, 50.0, True),
        ("provider", 0.0, 0.0, False),
        ("provider", 0.01, 0.0, True),
        ("provider", 0.01, 0.01, False),
        ("deployment", None, 50.0, True),
        ("deployment", 0.0, 0.0, False),
        ("deployment", 0.01, 0.0, True),
        ("deployment", 0.01, 0.01, False),
        ("tag", None, 50.0, True),
        ("tag", 0.0, 0.0, False),
        ("tag", 0.01, 0.0, True),
        ("tag", 0.01, 0.01, False),
    ],
)
async def test_max_budget_of_zero_blocks_and_none_does_not(
    disable_budget_sync,
    scope: _Scope,
    max_budget: float | None,
    spend: float,
    stays: bool,
) -> None:
    config: Final = BudgetConfig(max_budget=max_budget, budget_duration=_DURATION)
    kept, debug_info = _filter_one(scope, config, spend)
    if stays:
        assert kept == [_DEPLOYMENT]
        assert debug_info == ""
        return
    assert kept == []
    assert debug_info == _blocked_debug(scope, spend, config.max_budget)


@pytest.mark.asyncio
async def test_unset_provider_cap_still_enforces_a_zero_deployment_cap(disable_budget_sync) -> None:
    provider: Final = BudgetConfig(max_budget=None, budget_duration=_DURATION)
    deployment_cap: Final = BudgetConfig(max_budget=0, budget_duration=_DURATION)
    limiter: Final = RouterBudgetLimiting(dual_cache=DualCache(), provider_budget_config={"openai": provider})
    limiter.deployment_budget_config = {_MODEL_ID: deployment_cap}

    kept, debug_info = limiter._filter_out_deployments_above_budget(
        potential_deployments=[],
        healthy_deployments=[_DEPLOYMENT],
        provider_configs={"openai": provider},
        deployment_configs={_MODEL_ID: deployment_cap},
        deployment_providers=["openai"],
        spend_map={},
        request_tags=[],
    )

    assert kept == []
    assert debug_info == _blocked_debug("deployment", 0.0, deployment_cap.max_budget)


def _router_deployment(model_name: str, model: str, model_id: str, *, weight: int = 0) -> DeploymentTypedDict:
    deployment: Final[DeploymentTypedDict] = {
        "model_name": model_name,
        "litellm_params": {"model": model, "api_key": "sk-fake", "weight": weight},
        "model_info": {"id": model_id},
    }
    return deployment


@pytest.mark.asyncio
async def test_router_serves_the_uncapped_sibling_when_provider_max_budget_is_zero(disable_budget_sync) -> None:
    router: Final = Router(
        model_list=[
            _router_deployment("chat", "openai/gpt-4o-mini", "openai-capped", weight=100),
            _router_deployment("chat", "anthropic/claude-haiku-4-5", "anthropic-open"),
            _router_deployment("openai-only", "openai/gpt-4o-mini", "openai-only", weight=100),
        ],
        provider_budget_config={"openai": BudgetConfig(budget_limit=0, time_period="1d")},
        num_retries=0,
    )
    messages: Final = [{"role": "user", "content": "hi"}]

    served: Final = await router.acompletion(model="chat", messages=messages, mock_response="served")

    assert served._hidden_params["model_id"] == "anthropic-open"
    assert served._hidden_params["custom_llm_provider"] == "anthropic"

    with pytest.raises(ValueError, match=r"Exceeded budget for provider openai: 0\.0 >= 0\.0"):
        await router.acompletion(model="openai-only", messages=messages, mock_response="served")
