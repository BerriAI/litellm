"""
Responses websocket turns on the Rust router core (``async_admit_deployment_turn``): the socket is
opened on one deployment, and the core hands the handler a ``turn_admission`` callable that re-runs
that deployment's admission before every later turn: the io-token turn budget, the deployment
budget limiter, the routing strategy (on that deployment alone) and the pre-call checks.

Which strategy and selector apply is the Rust core's routing context (``routing_context``); the
selectors, limiters and checks are the live Python objects the host keeps.
"""

from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.router_rust_shaping import as_dict

if TYPE_CHECKING:
    from litellm.router_rust_core import RustCore


def turn_admission(
    core: "RustCore", model_group: str, deployment: Mapping[str, object], kwargs: Mapping[str, object]
) -> Callable[[], object]:
    """The ``turn_admission`` the handler calls before each later turn."""
    return partial(admit_deployment_turn, core, model_group, deployment, kwargs)


def _deployment_as_configured_now(core: "RustCore", deployment: Mapping[str, object]) -> Mapping[str, object]:
    model_info: Final = deployment.get("model_info")
    model_id: Final = as_dict(model_info).get("id") if isinstance(model_info, dict) else None  # pyright: ignore[reportUnknownArgumentType]  # the configured model_info dict
    current: Final = core._hosted._core_get_deployment(model_id) if isinstance(model_id, str) else None  # pyright: ignore[reportPrivateUsage]  # the core's host registry
    return deployment if current is None else current.model_dump(exclude_none=True)


def _budget_limiter(core: "RustCore") -> object:
    """The router's deployment-budget callback; the Rust core declines every config that has one."""
    from litellm.router_rust_core import RustCoreUnsupported

    try:
        return core.registry().plugins("budget_limiter")
    except RustCoreUnsupported:
        return None


async def admit_deployment_turn(
    core: "RustCore", model_group: str, opened_on: Mapping[str, object], request_kwargs: Mapping[str, object]
) -> Mapping[str, object]:
    """`async_admit_deployment_turn`."""
    from litellm import exceptions as litellm_exceptions
    from litellm.router_rust_selection import routing_pre_call_checks
    from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import (
        async_admit_io_token_turn,
        without_io_token_limits,
    )
    from litellm.router_utils.pre_call_checks.model_rate_limit_check import ModelRateLimitingCheck
    from litellm.utils import get_utc_datetime

    deployment: Final = _deployment_as_configured_now(core, opened_on)
    rate_limit_check: Final = next(
        (
            callback
            for callback in core._checks.routing_callbacks()  # pyright: ignore[reportPrivateUsage]  # `_routing_callbacks`
            if isinstance(callback, ModelRateLimitingCheck)
        ),
        None,
    )
    io_token_admission: Final[Mapping[str, object]] = (
        MappingProxyType({})
        if rate_limit_check is None
        else await async_admit_io_token_turn(rate_limit_check.dual_cache, deployment, get_utc_datetime())  # pyright: ignore[reportArgumentType]  # the configured deployment dict
    )
    budget_limiter: Final[object] = _budget_limiter(core)
    if budget_limiter is not None:
        try:
            await budget_limiter.async_filter_deployments(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # RouterBudgetLimiting
                model=model_group,
                healthy_deployments=[dict(deployment)],
                messages=None,
                request_kwargs=dict(request_kwargs),
            )
        except ValueError as e:
            raise litellm_exceptions.RateLimitError(
                message=str(e),
                llm_provider="",
                model=model_group,
                category=litellm_exceptions.RateLimitErrorCategory.LITELLM_RATE_LIMIT,
            ) from e
    await _admit_turn_through_routing_strategy(core, model_group, deployment, request_kwargs)
    await routing_pre_call_checks(core._checks.routing_callbacks(), dict(without_io_token_limits(deployment)))  # pyright: ignore[reportPrivateUsage]  # `_routing_callbacks`
    return io_token_admission


async def _admit_turn_through_routing_strategy(
    core: "RustCore", model_group: str, deployment: Mapping[str, object], request_kwargs: Mapping[str, object]
) -> None:
    """`_admit_turn_through_routing_strategy`: the group's strategy picks among this deployment alone."""
    from litellm import exceptions as litellm_exceptions
    from litellm.router_api import RoutingContext
    from litellm.router_rust_selection import pick_async
    from litellm.router_utils.handle_error import async_raise_no_deployment_exception

    routing_kwargs: Final = {key: value for key, value in request_kwargs.items() if key != "litellm_logging_obj"}
    override: Final = routing_kwargs.get("routing_strategy")
    strategy, scope, group = core._handle.routing_context(  # pyright: ignore[reportPrivateUsage]  # the core's Rust handle
        model_group, override if isinstance(override, str) else None
    )
    selectors: Final = core._selectors  # pyright: ignore[reportPrivateUsage]  # the core's strategy selectors
    selector: Final = selectors.selector_for(strategy, scope, group, routing_kwargs)
    if strategy == "simple-shuffle" or selector is None:
        return
    selected: Final = await pick_async(
        strategy, selector, model_group, [dict(deployment)], None, None, dict(routing_kwargs)
    )
    if selected is None:
        no_deployment: Final = await async_raise_no_deployment_exception(
            ctx=RoutingContext(core),  # pyright: ignore[reportArgumentType]  # the error reads registry() and health()
            model=model_group,
            parent_otel_span=None,
        )
        raise litellm_exceptions.RateLimitError(
            message=str(no_deployment),
            llm_provider="",
            model=model_group,
            category=litellm_exceptions.RateLimitErrorCategory.LITELLM_RATE_LIMIT,
        ) from no_deployment
    if selectors.is_override(strategy, selector):
        await selector.async_pre_call_check(dict(deployment), None)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # every selector is a CustomLogger


__all__ = ["admit_deployment_turn", "turn_admission"]
