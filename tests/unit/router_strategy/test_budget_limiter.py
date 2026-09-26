"""
Spend tracking in RouterBudgetLimiting.async_log_success_event.

Only chat completions puts custom_llm_provider into litellm_params. The responses,
anthropic_messages, embedding and rerank surfaces leave it unset, which used to make
the callback raise before any spend was recorded, so those budgets never moved.
"""

import asyncio
from typing import Final

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.router import Router
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.router import RouterErrors
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


_DURATION: Final = "1d"
_MODEL_ID: Final = "dep-1"
_TAG: Final = "prod"
_BUDGET_CASES: Final = (
    (None, 50.0, True),
    (0.0, 0.0, False),
    (0.0, 50.0, False),
    (0.01, 0.0, True),
    (0.01, 0.01, False),
    (0.01, 50.0, False),
)
_BUDGET_CASE_IDS: Final = (
    "unset-cap-stays-routable",
    "zero-cap-blocks-with-no-spend",
    "zero-cap-blocks-recorded-spend",
    "under-cap-stays-routable",
    "spend-equal-to-cap-is-blocked",
    "spend-above-cap-is-blocked",
)


@pytest.fixture
def isolated_callbacks(disable_budget_sync):
    before: Final = tuple(litellm.callbacks) if isinstance(litellm.callbacks, list) else ()
    if isinstance(litellm.callbacks, list):
        litellm.callbacks[:] = [
            callback for callback in litellm.callbacks if not isinstance(callback, RouterBudgetLimiting)
        ]
    yield
    if isinstance(litellm.callbacks, list):
        litellm.callbacks[:] = list(before)


def _budget(max_budget: float | None) -> BudgetConfig:
    return BudgetConfig(max_budget=max_budget, budget_duration=_DURATION)


def _deployment(
    *,
    model: str = "openai/gpt-4",
    model_id: str = _MODEL_ID,
    model_name: str = "gpt-4",
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {"model": model},
        "model_info": {"id": model_id},
    }


def _limiter() -> RouterBudgetLimiting:
    return RouterBudgetLimiting(dual_cache=DualCache(), provider_budget_config=None)


def _filter(
    limiter: RouterBudgetLimiting,
    deployment: dict[str, object],
    *,
    provider_configs: dict[str, BudgetConfig],
    deployment_configs: dict[str, BudgetConfig],
    deployment_providers: list[str | None],
    spend_map: dict[str, float],
    request_tags: list[str],
) -> tuple[list[dict[str, object]], str]:
    return limiter._filter_out_deployments_above_budget(
        potential_deployments=[],
        healthy_deployments=[deployment],
        provider_configs=provider_configs,
        deployment_configs=deployment_configs,
        deployment_providers=deployment_providers,
        spend_map=spend_map,
        request_tags=request_tags,
    )


def _assert_routed(
    kept: list[dict[str, object]],
    debug_info: str,
    deployment: dict[str, object],
    *,
    stays: bool,
    blocked_debug: str,
) -> None:
    if stays:
        assert kept == [deployment]
        assert debug_info == ""
        return
    assert kept == []
    assert debug_info == blocked_debug


async def _drain_background_tasks() -> None:
    current: Final = asyncio.current_task()
    pending: Final = tuple(task for task in asyncio.all_tasks() if task is not current and not task.done())
    if pending:
        await asyncio.gather(*pending)


def _router_model(
    *,
    model_name: str,
    model: str,
    mock_response: str,
    model_id: str,
    max_budget: float | None = None,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": model,
            "mock_response": mock_response,
            "api_key": "sk-fake",
            **({} if max_budget is None else {"max_budget": max_budget, "budget_duration": _DURATION}),
        },
        "model_info": {"id": model_id},
    }


async def _complete(router: Router, model: str, **extra: object) -> str:
    response: Final = await router.acompletion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        num_retries=0,
        **extra,
    )
    content: Final = response.choices[0].message.content
    assert isinstance(content, str)
    return content


@pytest.mark.asyncio
@pytest.mark.parametrize(("max_budget", "spend", "stays"), _BUDGET_CASES, ids=_BUDGET_CASE_IDS)
async def test_provider_max_budget_treats_zero_as_a_cap_and_none_as_uncapped(
    isolated_callbacks,
    max_budget: float | None,
    spend: float,
    stays: bool,
) -> None:
    config: Final = _budget(max_budget)
    deployment: Final = _deployment()
    limiter: Final = _limiter()
    limiter.provider_budget_config = {"openai": config}
    kept, debug_info = _filter(
        limiter,
        deployment,
        provider_configs={"openai": config},
        deployment_configs={},
        deployment_providers=["openai"],
        spend_map={f"provider_spend:openai:{_DURATION}": spend},
        request_tags=[],
    )
    _assert_routed(
        kept,
        debug_info,
        deployment,
        stays=stays,
        blocked_debug=f"Exceeded budget for provider openai: {spend} >= {config.max_budget}\n",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("max_budget", "spend", "stays"), _BUDGET_CASES, ids=_BUDGET_CASE_IDS)
async def test_deployment_max_budget_treats_zero_as_a_cap_and_none_as_uncapped(
    isolated_callbacks,
    max_budget: float | None,
    spend: float,
    stays: bool,
) -> None:
    config: Final = _budget(max_budget)
    deployment: Final = _deployment()
    limiter: Final = _limiter()
    limiter.deployment_budget_config = {_MODEL_ID: config}
    kept, debug_info = _filter(
        limiter,
        deployment,
        provider_configs={},
        deployment_configs={_MODEL_ID: config},
        deployment_providers=[],
        spend_map={f"deployment_spend:{_MODEL_ID}:{_DURATION}": spend},
        request_tags=[],
    )
    _assert_routed(
        kept,
        debug_info,
        deployment,
        stays=stays,
        blocked_debug=(
            "Exceeded budget for deployment model_name: gpt-4, litellm_params.model: openai/gpt-4, "
            f"model_id: {_MODEL_ID}: {spend} >= {config.max_budget}\n"
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("max_budget", "spend", "stays"), _BUDGET_CASES, ids=_BUDGET_CASE_IDS)
async def test_tag_max_budget_treats_zero_as_a_cap_and_none_as_uncapped(
    isolated_callbacks,
    max_budget: float | None,
    spend: float,
    stays: bool,
) -> None:
    config: Final = _budget(max_budget)
    deployment: Final = _deployment()
    limiter: Final = _limiter()
    limiter.tag_budget_config = {_TAG: config}
    kept, debug_info = _filter(
        limiter,
        deployment,
        provider_configs={},
        deployment_configs={},
        deployment_providers=[],
        spend_map={f"tag_spend:{_TAG}:{_DURATION}": spend},
        request_tags=[_TAG],
    )
    _assert_routed(
        kept,
        debug_info,
        deployment,
        stays=stays,
        blocked_debug=f"Exceeded budget for tag='{_TAG}', tag_spend={spend}, tag_budget_limit={config.max_budget}\n",
    )


@pytest.mark.asyncio
async def test_provider_zero_cap_blocks_when_no_spend_has_been_recorded(isolated_callbacks) -> None:
    config: Final = _budget(0)
    deployment: Final = _deployment()
    limiter: Final = _limiter()
    limiter.provider_budget_config = {"openai": config}
    kept, debug_info = _filter(
        limiter,
        deployment,
        provider_configs={"openai": config},
        deployment_configs={},
        deployment_providers=["openai"],
        spend_map={},
        request_tags=[],
    )
    _assert_routed(
        kept,
        debug_info,
        deployment,
        stays=False,
        blocked_debug=f"Exceeded budget for provider openai: 0.0 >= {config.max_budget}\n",
    )


@pytest.mark.asyncio
async def test_unset_provider_cap_still_enforces_a_zero_deployment_cap(isolated_callbacks) -> None:
    provider: Final = _budget(None)
    deployment_cap: Final = _budget(0)
    deployment: Final = _deployment()
    limiter: Final = _limiter()
    limiter.provider_budget_config = {"openai": provider}
    limiter.deployment_budget_config = {_MODEL_ID: deployment_cap}
    kept, debug_info = _filter(
        limiter,
        deployment,
        provider_configs={"openai": provider},
        deployment_configs={_MODEL_ID: deployment_cap},
        deployment_providers=["openai"],
        spend_map={},
        request_tags=[],
    )
    _assert_routed(
        kept,
        debug_info,
        deployment,
        stays=False,
        blocked_debug=(
            "Exceeded budget for deployment model_name: gpt-4, litellm_params.model: openai/gpt-4, "
            f"model_id: {_MODEL_ID}: 0.0 >= {deployment_cap.max_budget}\n"
        ),
    )


@pytest.mark.asyncio
async def test_router_refuses_a_provider_whose_max_budget_is_zero(isolated_callbacks) -> None:
    router: Final = Router(
        model_list=[
            _router_model(model_name="gpt-4", model="openai/gpt-4", mock_response="served", model_id=_MODEL_ID)
        ],
        provider_budget_config={"openai": _budget(0)},
    )
    await _drain_background_tasks()

    with pytest.raises(ValueError) as exc_info:
        await _complete(router, "gpt-4")

    assert str(exc_info.value) == (
        f"{RouterErrors.no_deployments_with_provider_budget_routing.value}: "
        "Exceeded budget for provider openai: 0.0 >= 0.0\n"
    )


@pytest.mark.asyncio
async def test_router_refuses_a_provider_that_already_spent_against_a_zero_cap(isolated_callbacks) -> None:
    router: Final = Router(
        model_list=[
            _router_model(model_name="gpt-4", model="openai/gpt-4", mock_response="served", model_id=_MODEL_ID)
        ],
        provider_budget_config={"openai": _budget(0)},
    )
    await _drain_background_tasks()
    logger: Final = router.router_budget_logger
    assert logger is not None
    await logger.dual_cache.async_set_cache(key=f"provider_spend:openai:{_DURATION}", value=50.0)

    with pytest.raises(ValueError) as exc_info:
        await _complete(router, "gpt-4")

    assert str(exc_info.value) == (
        f"{RouterErrors.no_deployments_with_provider_budget_routing.value}: "
        "Exceeded budget for provider openai: 50.0 >= 0.0\n"
    )


@pytest.mark.asyncio
async def test_router_serves_a_provider_under_its_cap(isolated_callbacks) -> None:
    router: Final = Router(
        model_list=[
            _router_model(model_name="gpt-4", model="openai/gpt-4", mock_response="served", model_id=_MODEL_ID)
        ],
        provider_budget_config={"openai": _budget(0.01)},
    )
    await _drain_background_tasks()

    assert await _complete(router, "gpt-4") == "served"


@pytest.mark.asyncio
async def test_router_serves_when_the_provider_cap_is_unset(isolated_callbacks) -> None:
    router: Final = Router(
        model_list=[
            _router_model(model_name="gpt-4", model="openai/gpt-4", mock_response="served", model_id=_MODEL_ID)
        ],
        provider_budget_config={"openai": _budget(None)},
    )
    await _drain_background_tasks()

    assert await _complete(router, "gpt-4") == "served"


@pytest.mark.asyncio
async def test_router_keeps_an_uncapped_provider_when_another_is_at_zero(isolated_callbacks) -> None:
    router: Final = Router(
        model_list=[
            _router_model(model_name="chat", model="openai/gpt-4", mock_response="from openai", model_id="dep-openai"),
            _router_model(
                model_name="chat",
                model="anthropic/claude-3-5-sonnet-latest",
                mock_response="from anthropic",
                model_id="dep-anthropic",
            ),
        ],
        provider_budget_config={"openai": _budget(0)},
    )
    await _drain_background_tasks()

    served: Final = tuple([await _complete(router, "chat") for _ in range(20)])
    assert served == ("from anthropic",) * 20


@pytest.mark.asyncio
async def test_router_refuses_a_deployment_whose_max_budget_is_zero(isolated_callbacks) -> None:
    router: Final = Router(
        model_list=[
            _router_model(
                model_name="gpt-4",
                model="openai/gpt-4",
                mock_response="served",
                model_id=_MODEL_ID,
                max_budget=0,
            )
        ],
    )
    await _drain_background_tasks()

    with pytest.raises(ValueError) as exc_info:
        await _complete(router, "gpt-4")

    assert str(exc_info.value) == (
        f"{RouterErrors.no_deployments_with_provider_budget_routing.value}: "
        "Exceeded budget for deployment model_name: gpt-4, litellm_params.model: openai/gpt-4, "
        f"model_id: {_MODEL_ID}: 0.0 >= 0.0\n"
    )


@pytest.mark.asyncio
async def test_router_serves_a_deployment_under_its_cap(isolated_callbacks) -> None:
    router: Final = Router(
        model_list=[
            _router_model(
                model_name="gpt-4",
                model="openai/gpt-4",
                mock_response="served",
                model_id=_MODEL_ID,
                max_budget=0.01,
            )
        ],
    )
    await _drain_background_tasks()

    assert await _complete(router, "gpt-4") == "served"


@pytest.mark.asyncio
async def test_router_tag_cap_of_zero_blocks_only_tagged_requests(isolated_callbacks) -> None:
    router: Final = Router(
        model_list=[
            _router_model(model_name="gpt-4", model="openai/gpt-4", mock_response="served", model_id=_MODEL_ID)
        ],
        provider_budget_config={"openai": _budget(1.0)},
    )
    await _drain_background_tasks()
    logger: Final = router.router_budget_logger
    assert logger is not None
    logger.tag_budget_config = {_TAG: _budget(0)}

    assert await _complete(router, "gpt-4") == "served"
    with pytest.raises(ValueError) as exc_info:
        await _complete(router, "gpt-4", metadata={"tags": [_TAG]})

    assert str(exc_info.value) == (
        f"{RouterErrors.no_deployments_with_provider_budget_routing.value}: "
        f"Exceeded budget for tag='{_TAG}', tag_spend=0.0, tag_budget_limit=0.0\n"
    )
