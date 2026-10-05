"""
Spend tracking in RouterBudgetLimiting.async_log_success_event.

Only chat completions puts custom_llm_provider into litellm_params. The responses,
anthropic_messages, embedding and rerank surfaces leave it unset, which used to make
the callback raise before any spend was recorded, so those budgets never moved.
"""

from typing import Final

import pytest

from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting


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


_OPENAI_DEPLOYMENT: Final = {
    "model_name": "chat",
    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
    "model_info": {"id": "openai-dep"},
}
_ANTHROPIC_DEPLOYMENT: Final = {
    "model_name": "chat",
    "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-fake"},
    "model_info": {"id": "anthropic-dep"},
}


def _openai_deployment_with_budget(cap: float | None) -> dict[str, object]:
    budget: Final = {"budget_duration": "1d"} if cap is None else {"max_budget": cap, "budget_duration": "1d"}
    return {**_OPENAI_DEPLOYMENT, "litellm_params": {**_OPENAI_DEPLOYMENT["litellm_params"], **budget}}


def _limiter_capping_openai(scope: str, cap: float | None, monkeypatch: pytest.MonkeyPatch) -> RouterBudgetLimiting:
    budget: Final = {"time_period": "1d"} if cap is None else {"budget_limit": cap, "time_period": "1d"}
    if scope == "provider":
        return RouterBudgetLimiting(dual_cache=DualCache(), provider_budget_config={"openai": budget})
    if scope == "deployment":
        return RouterBudgetLimiting(
            dual_cache=DualCache(),
            provider_budget_config=None,
            model_list=[_openai_deployment_with_budget(cap), _ANTHROPIC_DEPLOYMENT],
        )
    monkeypatch.setattr("litellm.tag_budget_config", {"prod": budget})
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    return RouterBudgetLimiting(dual_cache=DualCache(), provider_budget_config=None)


async def _record_openai_spend(limiter: RouterBudgetLimiting, cost: float) -> None:
    await _log_success(
        limiter,
        {
            "call_type": "acompletion",
            "metadata": {"tags": ["prod"]},
            "litellm_params": {"model": "openai/gpt-4o", "custom_llm_provider": "openai"},
            "standard_logging_object": {
                "response_cost": cost,
                "model_id": "openai-dep",
                "custom_llm_provider": "openai",
            },
        },
    )


async def _routable_ids(limiter: RouterBudgetLimiting, deployments: list[dict[str, object]]) -> list[str]:
    kept: Final = await limiter.async_filter_deployments(
        model="chat",
        healthy_deployments=deployments,
        messages=None,
        request_kwargs={"metadata": {"tags": ["prod"]}},
    )
    return [deployment["model_info"]["id"] for deployment in kept]


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["provider", "deployment", "tag"])
@pytest.mark.parametrize(
    ("cap", "spend", "openai_routable"),
    [
        (0.0, 0.0, False),
        (0.01, 0.0, True),
        (0.01, 0.009, True),
        (0.01, 0.01, False),
        (None, 50.0, True),
    ],
)
async def test_zero_cap_blocks_and_only_an_unset_cap_is_unlimited(
    disable_budget_sync: None,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    cap: float | None,
    spend: float,
    openai_routable: bool,
) -> None:
    limiter: Final = _limiter_capping_openai(scope, cap, monkeypatch)
    if spend > 0:
        await _record_openai_spend(limiter, spend)
    openai: Final = _openai_deployment_with_budget(cap) if scope == "deployment" else _OPENAI_DEPLOYMENT

    if scope == "tag" and not openai_routable:
        with pytest.raises(ValueError, match=r"Exceeded budget for tag='prod'"):
            await _routable_ids(limiter, [openai, _ANTHROPIC_DEPLOYMENT])
        return

    routable: Final = await _routable_ids(limiter, [openai, _ANTHROPIC_DEPLOYMENT])

    expected: Final = ["openai-dep", "anthropic-dep"] if openai_routable else ["anthropic-dep"]
    assert routable == expected, f"{scope} cap={cap} spend={spend}"


@pytest.mark.asyncio
async def test_router_rejects_requests_to_a_provider_capped_at_zero(disable_budget_sync: None) -> None:
    router: Final = Router(
        model_list=[_OPENAI_DEPLOYMENT],
        provider_budget_config={"openai": {"budget_limit": 0, "time_period": "1d"}},
    )

    with pytest.raises(ValueError, match=r"Exceeded budget for provider openai: 0\.0 >= 0\.0"):
        await router.acompletion(model="chat", messages=[{"role": "user", "content": "hi"}], mock_response="served")


@pytest.mark.asyncio
async def test_router_serves_a_provider_with_a_period_but_no_cap(disable_budget_sync: None) -> None:
    router: Final = Router(
        model_list=[_OPENAI_DEPLOYMENT],
        provider_budget_config={"openai": {"time_period": "1d"}},
    )

    served: Final = await router.acompletion(
        model="chat", messages=[{"role": "user", "content": "hi"}], mock_response="served"
    )

    assert served.choices[0].message.content == "served"
