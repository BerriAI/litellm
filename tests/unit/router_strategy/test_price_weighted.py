import asyncio
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Final, Literal, assert_never

import pytest
from pydantic import ConfigDict

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.price_weighted import PriceWeightedLoggingHandler
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import DeploymentTypedDict
from litellm.types.utils import LiteLLMPydanticObjectBase

MODEL_GROUP: Final = "price-weighted-test"
MESSAGES: Final[tuple[AllMessageValues, ...]] = ({"role": "user", "content": "hi"},)


class _SelectedModelInfo(LiteLLMPydanticObjectBase):
    id: str | int | None = None
    model_config = ConfigDict(frozen=True, from_attributes=True, protected_namespaces=())


class _SelectedDeployment(LiteLLMPydanticObjectBase):
    model_info: _SelectedModelInfo | None = None
    model_config = ConfigDict(frozen=True, from_attributes=True, protected_namespaces=())


class _ServerError(Exception):
    status_code: Final = 500


def _deployment(deployment_id: str) -> DeploymentTypedDict:
    return {
        "model_name": MODEL_GROUP,
        "litellm_params": {"model": "openai/gpt-4o-mini"},
        "model_info": {"id": deployment_id},
    }


def _handler(
    prices: Mapping[str, tuple[float | None, float | None]],
    routing_args: Mapping[str, object] | None = None,
    seed: int = 71,
) -> PriceWeightedLoggingHandler:
    def price_lookup(model_id: str, model_name: str) -> tuple[float | None, float | None]:
        return prices.get(model_id, (None, None))

    return PriceWeightedLoggingHandler(
        router_cache=DualCache(),
        price_lookup=price_lookup,
        routing_args=routing_args,
        rng=random.Random(seed),
    )


def _selected_id(deployment: object | None) -> str:
    assert deployment is not None
    selected_deployment: Final = _SelectedDeployment.model_validate(deployment)
    assert selected_deployment.model_info is not None
    deployment_id: Final = selected_deployment.model_info.id
    assert deployment_id is not None
    return str(deployment_id)


def _router_deployment(router: Router) -> object:
    return _SelectedDeployment.model_validate(
        router.get_available_deployment(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # Router selector has no return annotation
            model=MODEL_GROUP
        )
    )


async def _router_response(router: Router) -> object:
    return await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # Router completion kwargs are dynamically typed
        model=MODEL_GROUP,
        messages=list(MESSAGES),
        stream=False,
    )


def _picks(
    handler: PriceWeightedLoggingHandler,
    deployments: Sequence[DeploymentTypedDict],
    count: int,
) -> tuple[str, ...]:
    return tuple(
        _selected_id(
            handler.get_available_deployments(
                model_group=MODEL_GROUP,
                healthy_deployments=deployments,
                messages=MESSAGES,
            )
        )
        for _ in range(count)
    )


async def _async_picks(
    handler: PriceWeightedLoggingHandler,
    deployments: Sequence[DeploymentTypedDict],
    count: int,
) -> tuple[str, ...]:
    responses: Final = await asyncio.gather(
        *(
            handler.async_get_available_deployments(
                model_group=MODEL_GROUP,
                healthy_deployments=deployments,
                messages=MESSAGES,
            )
            for _ in range(count)
        )
    )
    return tuple(_selected_id(response) for response in responses)


def _chi_square(
    picks: Sequence[str],
    deployment_ids: tuple[str, ...],
    expected_weights: tuple[int, ...],
) -> float:
    counts: Final = Counter(picks)
    total: Final = len(picks)
    weight_total: Final = sum(expected_weights)
    return sum(
        (counts.get(deployment_id, 0) - total * weight / weight_total) ** 2 / (total * weight / weight_total)
        for deployment_id, weight in zip(deployment_ids, expected_weights)
    )


def test_price_weighted_routing_matches_inverse_square_prices() -> None:
    deployments: Final = tuple(_deployment(deployment_id) for deployment_id in ("cost-1", "cost-2", "cost-3"))
    prices: Final = {
        "cost-1": (1.0, 1.0),
        "cost-2": (2.0, 2.0),
        "cost-3": (3.0, 3.0),
    }
    handler: Final = _handler(prices)
    picks: Final = _picks(handler, deployments, 10_000)

    assert _chi_square(picks, ("cost-1", "cost-2", "cost-3"), (36, 9, 4)) < 13.82


def test_equal_price_deployments_are_selected_uniformly() -> None:
    deployments: Final = tuple(_deployment(deployment_id) for deployment_id in ("equal-a", "equal-b", "equal-c"))
    handler: Final = _handler({deployment_id: (2.0, 2.0) for deployment_id in ("equal-a", "equal-b", "equal-c")})

    assert _chi_square(_picks(handler, deployments, 9_000), ("equal-a", "equal-b", "equal-c"), (1, 1, 1)) < 13.82


def test_zero_cost_deployment_always_wins() -> None:
    deployments: Final = (_deployment("zero"), _deployment("priced"))
    handler: Final = _handler({"zero": (0.0, 0.0), "priced": (1.0, 1.0)})

    assert frozenset(_picks(handler, deployments, 200)) == frozenset({"zero"})


def test_unknown_price_is_assigned_the_highest_known_cost() -> None:
    deployments: Final = tuple(_deployment(deployment_id) for deployment_id in ("cost-1", "cost-2", "unknown"))
    handler: Final = _handler({"cost-1": (1.0, 1.0), "cost-2": (2.0, 2.0)})

    assert _chi_square(_picks(handler, deployments, 10_000), ("cost-1", "cost-2", "unknown"), (36, 9, 9)) < 13.82


def test_all_unknown_prices_are_selected_uniformly() -> None:
    deployment_ids: Final = ("unknown-a", "unknown-b", "unknown-c")
    deployments: Final = tuple(_deployment(deployment_id) for deployment_id in deployment_ids)
    handler: Final = _handler({})

    assert _chi_square(_picks(handler, deployments, 9_000), deployment_ids, (1, 1, 1)) < 13.82


def test_top_n_one_selects_the_cheapest_stable_deployment() -> None:
    deployment_ids: Final = ("expensive", "cheapest", "middle")
    deployments: Final = tuple(_deployment(deployment_id) for deployment_id in deployment_ids)
    handler: Final = _handler(
        {"expensive": (3.0, 3.0), "cheapest": (1.0, 1.0), "middle": (2.0, 2.0)},
        routing_args={"top_n": 1},
    )

    assert frozenset(_picks(handler, deployments, 200)) == frozenset({"cheapest"})


def test_no_healthy_deployments_returns_none() -> None:
    handler: Final = _handler({})

    assert handler.get_available_deployments(MODEL_GROUP, ()) is None


def _failure(kind: Literal["500", "429", "timeout", "400"]) -> Exception:
    match kind:
        case "500":
            return litellm.InternalServerError(
                message="server error",
                llm_provider="openai",
                model="openai/gpt-4o-mini",
            )
        case "429":
            return litellm.RateLimitError(
                message="rate limited",
                llm_provider="openai",
                model="openai/gpt-4o-mini",
            )
        case "timeout":
            return litellm.Timeout(
                message="timed out",
                llm_provider="openai",
                model="openai/gpt-4o-mini",
            )
        case "400":
            return litellm.APIError(
                status_code=400,
                message="bad request",
                llm_provider="openai",
                model="openai/gpt-4o-mini",
            )
    assert_never(kind)


def _failure_kwargs(deployment_id: str, exception: Exception) -> Mapping[str, object]:
    return {
        "exception": exception,
        "litellm_params": {"model_info": {"id": deployment_id}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False], ids=["sync", "async"])
@pytest.mark.parametrize("failure_kind", ["500", "429", "timeout"])
async def test_retryable_failure_demotes_cheapest_until_health_window_expires(
    sync_mode: bool,
    failure_kind: Literal["500", "429", "timeout"],
) -> None:
    deployments: Final = (_deployment("cheapest"), _deployment("expensive"))
    handler: Final = _handler(
        {"cheapest": (1.0, 1.0), "expensive": (10.0, 10.0)},
        routing_args={"health_window_seconds": 0.2, "top_n": 1},
    )
    failure: Final = _failure(failure_kind)
    kwargs: Final = _failure_kwargs("cheapest", failure)

    if sync_mode:
        handler.log_failure_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
    else:
        await handler.async_log_failure_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
    demoted_picks: Final = (
        _picks(handler, deployments, 200) if sync_mode else await _async_picks(handler, deployments, 200)
    )
    assert frozenset(demoted_picks) == frozenset({"expensive"})

    time.sleep(0.3)

    recovered: Final = (
        _selected_id(handler.get_available_deployments(MODEL_GROUP, deployments))
        if sync_mode
        else _selected_id(await handler.async_get_available_deployments(MODEL_GROUP, deployments))
    )
    assert recovered == "cheapest"


def test_bad_request_does_not_demote_a_deployment() -> None:
    deployments: Final = (_deployment("cheapest"), _deployment("expensive"))
    handler: Final = _handler(
        {"cheapest": (1.0, 1.0), "expensive": (10.0, 10.0)},
        routing_args={"top_n": 1},
    )
    handler.log_failure_event(
        kwargs=_failure_kwargs("cheapest", _failure("400")),
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert frozenset(_picks(handler, deployments, 200)) == frozenset({"cheapest"})


def test_all_demoted_deployments_fall_back_to_the_healthy_candidates() -> None:
    deployments: Final = (_deployment("first"), _deployment("second"))
    handler: Final = _handler({"first": (1.0, 1.0), "second": (2.0, 2.0)})
    for deployment_id in ("first", "second"):
        handler.log_failure_event(
            kwargs=_failure_kwargs(deployment_id, _failure("500")),
            response_obj=None,
            start_time=None,
            end_time=None,
        )

    assert _selected_id(handler.get_available_deployments(MODEL_GROUP, deployments)) in {"first", "second"}


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_success_updates_per_group_output_token_ema(async_mode: bool) -> None:
    deployments: Final = (_deployment("input-cost"), _deployment("output-cost"))
    handler: Final = _handler(
        {"input-cost": (1.0, 0.0), "output-cost": (0.0, 1.0)},
        routing_args={"top_n": 1},
    )
    assert _selected_id(handler.get_available_deployments(MODEL_GROUP, deployments, input="hello")) == "input-cost"
    kwargs: Final = {"litellm_params": {"metadata": {"model_group": MODEL_GROUP}}}
    response: Final = litellm.ModelResponse(
        model="openai/gpt-4o-mini",
        choices=[{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        usage=litellm.Usage(prompt_tokens=1, completion_tokens=0, total_tokens=1),
    )

    if async_mode:
        await handler.async_log_success_event(kwargs, response, None, None)
    else:
        handler.log_success_event(kwargs, response, None, None)
    picked: Final = (
        _selected_id(await handler.async_get_available_deployments(MODEL_GROUP, deployments, input="hello"))
        if async_mode
        else _selected_id(handler.get_available_deployments(MODEL_GROUP, deployments, input="hello"))
    )

    assert picked == "output-cost"


@pytest.mark.asyncio
async def test_router_weighted_failover_demotes_a_failing_cheap_deployment() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "mock_response": _ServerError("server error"),
                },
                "model_info": {"id": "cheap", "input_cost_per_token": 1e-10, "output_cost_per_token": 1e-10},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "mock_response": "ok from expensive",
                },
                "model_info": {"id": "expensive", "input_cost_per_token": 1e-3, "output_cost_per_token": 1e-3},
            },
        ],
        routing_strategy="price-weighted-routing",
        enable_weighted_failover=True,
        routing_strategy_args={"top_n": 1},
        num_retries=0,
    )
    first_response: Final = await _router_response(router)

    assert get_hidden_params_dict(first_response)["model_id"] == "expensive"
    assert _selected_id(_router_deployment(router)) == "expensive"

    next_response: Final = await _router_response(router)

    assert get_hidden_params_dict(next_response)["model_id"] == "expensive"


def test_router_uses_model_info_custom_prices_for_weighted_selection() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": "openai/gpt-4o-mini", "mock_response": "ok"},
                "model_info": {"id": "expensive", "input_cost_per_token": 1e-6, "output_cost_per_token": 1e-6},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": "openai/gpt-4o-mini", "mock_response": "ok"},
                "model_info": {"id": "cheap", "input_cost_per_token": 1e-8, "output_cost_per_token": 1e-8},
            },
        ],
        routing_strategy="price-weighted-routing",
    )
    picks: Final = tuple(_selected_id(_router_deployment(router)) for _ in range(200))

    assert picks.count("cheap") > 160
