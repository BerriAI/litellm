"""
Responses websocket turns on the Rust router core (``async_admit_deployment_turn``): the socket is
opened on one deployment, and the core hands the handler a ``turn_admission`` callable that re-runs
that deployment's admission before every later turn: the io-token turn budget, the deployment
budget limiter, the routing strategy (on that deployment alone) and the pre-call checks.

Which strategy and selector apply is the Rust core's routing context (``routing_context``); the
selectors, limiters and checks are the live Python objects the host keeps.
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm._internal_context import request_parent_otel_span
from litellm._logging import verbose_router_logger
from litellm.constants import USAGE_BASED_ROUTING_COUNTED_TOKENS_METADATA_KEY
from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs, without
from litellm.router_rust.invoke.shaping import as_dict
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    count_before_callbacks,
    find_deployment_metadata,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.router_rust.invoke.endpoints import EndpointCore


def turn_admission(
    core: "EndpointCore", model_group: str, deployment: Mapping[str, object], kwargs: Mapping[str, object]
) -> Callable[[], object]:
    """The ``turn_admission`` the handler calls before each later turn."""
    return partial(admit_deployment_turn, core, model_group, deployment, kwargs)


@dataclass(slots=True)
class TurnUsageRecorder:
    core: "EndpointCore"
    model_group: str
    request_kwargs: Mapping[str, object]
    counted_a_turn: bool = field(default=False, init=False)

    async def __call__(self, turn_litellm_params: Mapping[str, object], total_tokens: int) -> None:
        if await record_deployment_turn_usage(
            self.core, self.model_group, self.request_kwargs, turn_litellm_params, total_tokens
        ):
            self.counted_a_turn = True


def turn_usage_recorder(core: "EndpointCore", model_group: str, kwargs: Mapping[str, object]) -> TurnUsageRecorder:
    return TurnUsageRecorder(core, model_group, kwargs)


def turns_counted_their_requests(kwargs: Mapping[str, object]) -> bool:
    recorder: Final = kwargs.get("turn_usage_recorder")
    return isinstance(recorder, TurnUsageRecorder) and recorder.counted_a_turn


async def record_deployment_turn_usage(
    core: "EndpointCore",
    model_group: str,
    request_kwargs: Mapping[str, object],
    turn_litellm_params: Mapping[str, object],
    total_tokens: int,
) -> bool:
    """Count a finished turn early. Returns whether the router's own usage counted its request."""
    deployment_metadata: Final = find_deployment_metadata(turn_litellm_params)
    if deployment_metadata is None:
        return False
    parent_otel_span: Final = get_parent_otel_span_from_kwargs(dict(request_kwargs))
    counted_request: Final = await _counted_early(
        core.router.increment_deployment_usage_before_callbacks(
            deployment_metadata, model_group, total_tokens, parent_otel_span
        )
    )
    _ = await _counted_early(
        _count_usage_based_routing_tokens(
            core, model_group, request_kwargs, deployment_metadata, total_tokens, parent_otel_span
        )
    )
    return counted_request


async def _counted_early(count: Awaitable[bool]) -> bool:
    try:
        return await count
    except Exception as e:  # noqa: BLE001  # a failed early count must not break the turn; its logging counts it
        verbose_router_logger.debug("record_deployment_turn_usage: not counted early. Got - %s", e)
        return False


async def _count_usage_based_routing_tokens(
    core: "EndpointCore",
    model_group: str,
    request_kwargs: Mapping[str, object],
    deployment_metadata: dict[str, object],  # mutable-ok: the success callbacks read the counted tokens off it
    total_tokens: int,
    parent_otel_span: "Span | None",
) -> bool:
    from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2

    routing_kwargs: Final = without(request_kwargs, "litellm_logging_obj")
    override: Final = routing_kwargs.get("routing_strategy")
    strategy, scope, group = core.handle.routing_context(model_group, override if isinstance(override, str) else None)
    selector: Final = core.selectors.selector_for(strategy, scope, group, routing_kwargs)
    model_info: Final = deployment_metadata.get("model_info")
    deployment_id: Final = as_dict(model_info).get("id") if isinstance(model_info, dict) else None  # pyright: ignore[reportUnknownArgumentType]  # the routed model_info dict
    deployment_name: Final = deployment_metadata.get("deployment")
    if (
        not isinstance(selector, LowestTPMLoggingHandler_v2)
        or deployment_id is None
        or not isinstance(deployment_name, str)
    ):
        return False
    with request_parent_otel_span(parent_otel_span):
        await count_before_callbacks(
            deployment_metadata,
            USAGE_BASED_ROUTING_COUNTED_TOKENS_METADATA_KEY,
            total_tokens,
            partial(
                selector.async_increment_tpm,
                deployment_id=str(deployment_id),
                deployment_name=deployment_name,
                tokens=total_tokens,
            ),
        )
    return True


def _deployment_as_configured_now(core: "EndpointCore", deployment: Mapping[str, object]) -> Mapping[str, object]:
    model_info: Final = deployment.get("model_info")
    model_id: Final = as_dict(model_info).get("id") if isinstance(model_info, dict) else None  # pyright: ignore[reportUnknownArgumentType]  # the configured model_info dict
    current: Final = core.configured_deployment(model_id) if isinstance(model_id, str) else None
    return deployment if current is None else current.model_dump(exclude_none=True)


def _budget_limiter(core: "EndpointCore") -> object:
    """The router's deployment-budget callback; the Rust core declines every config that has one."""
    from litellm.router_rust.core import RustCoreUnsupported

    try:
        return core.registry().plugins("budget_limiter")
    except RustCoreUnsupported:
        return None


async def admit_deployment_turn(
    core: "EndpointCore", model_group: str, opened_on: Mapping[str, object], request_kwargs: Mapping[str, object]
) -> Mapping[str, object]:
    """`async_admit_deployment_turn`."""
    from litellm import exceptions as litellm_exceptions
    from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import (
        async_admit_io_token_turn,
        without_io_token_limits,
    )
    from litellm.router_utils.pre_call_checks.model_rate_limit_check import ModelRateLimitingCheck
    from litellm.utils import get_utc_datetime

    deployment: Final = _deployment_as_configured_now(core, opened_on)
    rate_limit_check: Final = next(
        (callback for callback in core.routing_callbacks() if isinstance(callback, ModelRateLimitingCheck)),
        None,
    )
    io_token_admission: Final[Mapping[str, object]] = (
        MappingProxyType({})
        if rate_limit_check is None
        else await async_admit_io_token_turn(rate_limit_check.usage, deployment, get_utc_datetime())  # pyright: ignore[reportArgumentType]  # the configured deployment dict
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
    await core.router.async_routing_strategy_pre_call_checks(
        deployment=dict(without_io_token_limits(deployment)), parent_otel_span=None
    )
    return io_token_admission


async def _admit_turn_through_routing_strategy(
    core: "EndpointCore", model_group: str, deployment: Mapping[str, object], request_kwargs: Mapping[str, object]
) -> None:
    """`_admit_turn_through_routing_strategy`: the group's strategy picks among this deployment alone."""
    from litellm import exceptions as litellm_exceptions
    from litellm.router_api import RoutingContext
    from litellm.router_rust.steps.selection import pick_async
    from litellm.router_utils.handle_error import async_raise_no_deployment_exception

    routing_kwargs: Final = without(request_kwargs, "litellm_logging_obj")
    override: Final = routing_kwargs.get("routing_strategy")
    strategy, scope, group = core.handle.routing_context(model_group, override if isinstance(override, str) else None)
    selectors: Final = core.selectors
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


__all__ = [
    "TurnUsageRecorder",
    "admit_deployment_turn",
    "record_deployment_turn_usage",
    "turn_admission",
    "turn_usage_recorder",
    "turns_counted_their_requests",
]
