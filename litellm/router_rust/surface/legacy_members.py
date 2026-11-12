"""
``RustLegacyMembers``: the old public ``Router`` members outside the entry points, on the Rust
core: the routing-state lifecycle, the routing callbacks' checks, one provider call's accounting
and the pre-routing strategy builders. Each one is the Python router core's, ported (never
imported), over the state the Rust core and its host keep.

Not served, and refused by ``RustCore.__getattr__``: the Python fallback walk
(``async_function_with_fallbacks``, ``function_with_fallbacks``, ``async_function_with_retries``,
``async_function_with_fallbacks_common_utils``) and ``factory_function``. They run a caller's
``original_function``, which picks its own deployment; router-core makes each pick itself and
attempts one host op, named by the endpoint's plan.
"""

import asyncio
import contextlib
import inspect
import traceback
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger
from litellm.constants import ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.add_retry_fallback_headers import response_total_token_count
from litellm.litellm_core_utils.batch_utils import is_batch_retrieve_call_type
from litellm.litellm_core_utils.core_helpers import as_str_mapping, get_parent_otel_span_from_kwargs
from litellm.litellm_core_utils.coroutine_checker import coroutine_checker
from litellm.litellm_core_utils.routing_groups import validate_routing_strategy
from litellm.litellm_core_utils.token_counter import served_openai_models as _served_openai_models
from litellm.redis.stores.router_cooldowns import ROUTER_COOLDOWNS_TARGET
from litellm.redis.stores.router_usage import ROUTER_USAGE_TARGET, RouterUsage, UsageCounter
from litellm.router_api import RoutingContext
from litellm.router_rust.invoke.websocket import admit_deployment_turn
from litellm.router_rust.steps.callbacks import RustOptionalChecks
from litellm.router_rust.steps.prerouting import TIER_PARAMS_NEVER_DROPPED as _TIER_PARAMS_NEVER_DROPPED
from litellm.router_rust.surface.members import RustRegistryMembers
from litellm.router_strategy.auto_router.auto_router_model_naming import (
    GatedAutoRouterCapability,
    capability_limit_violation,
    count_capability_routers,
)
from litellm.router_utils.cooldown_handlers import set_cooldown_deployments
from litellm.router_utils.endpoint_eligibility import endpoint_selection
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import async_io_token_refund_failure
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    count_before_callbacks,
    find_deployment_metadata,
)
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.caching.dual_cache import DualCache
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.router_rust.invoke.endpoints import EndpointCore
    from litellm.types.router import CustomRoutingStrategyBase, Deployment, RoutingGroup

# `RoutingArgs.ttl`: a usage counter lives for its minute.
_USAGE_TTL: Final = 60

# What `initialize_assistants_endpoint` binds.
_ASSISTANTS_ENDPOINTS: Final = (
    "acreate_assistants",
    "adelete_assistant",
    "aget_assistants",
    "acreate_thread",
    "aget_thread",
    "a_add_message",
    "aget_messages",
    "arun_thread",
)

# What `initialize_router_endpoints` binds, with the container file endpoints it reads from
# `litellm/containers/endpoints.json`.
_ROUTER_ENDPOINTS: Final = frozenset(
    {
        "_arealtime",
        "_aresponses_websocket",
        "aadapter_generate_content",
        "aanthropic_messages",
        "acancel_fine_tuning_job",
        "acancel_interaction",
        "acancel_responses",
        "acompact_responses",
        "acreate_agent",
        "acreate_container",
        "acreate_fine_tuning_job",
        "acreate_interaction",
        "acreate_realtime_client_secret",
        "acreate_realtime_transcription_session",
        "acreate_realtime_translation_client_secret",
        "acreate_skill",
        "adecisions",
        "adelete_agent",
        "adelete_container",
        "adelete_interaction",
        "adelete_responses",
        "adelete_skill",
        "aextraction",
        "afile_content",
        "afile_delete",
        "afile_list",
        "agenerate_content",
        "agenerate_content_stream",
        "aget_agent",
        "aget_interaction",
        "aget_responses",
        "aget_skill",
        "aimage_edit",
        "alist_agent_versions",
        "alist_agents",
        "alist_containers",
        "alist_fine_tuning_jobs",
        "alist_input_items",
        "alist_skills",
        "allm_passthrough_route",
        "amoderation",
        "anthropic_messages",
        "aocr",
        "arealtime_calls",
        "arealtime_translation_calls",
        "aresponses",
        "aretrieve_container",
        "aretrieve_fine_tuning_job",
        "asearch",
        "asystemone",
        "avector_store_delete",
        "avector_store_file_content",
        "avector_store_file_create",
        "avector_store_file_delete",
        "avector_store_file_list",
        "avector_store_file_retrieve",
        "avector_store_file_update",
        "avector_store_list",
        "avector_store_retrieve",
        "avector_store_search",
        "avector_store_update",
        "avideo_content",
        "avideo_create_character",
        "avideo_edit",
        "avideo_extension",
        "avideo_generation",
        "avideo_get_character",
        "avideo_list",
        "avideo_remix",
        "avideo_status",
        "cancel_interaction",
        "create_agent",
        "create_container",
        "create_interaction",
        "decisions",
        "delete_agent",
        "delete_container",
        "delete_interaction",
        "extraction",
        "generate_content",
        "generate_content_stream",
        "get_agent",
        "get_interaction",
        "list_agent_versions",
        "list_agents",
        "list_containers",
        "ocr",
        "responses",
        "retrieve_container",
        "search",
        "systemone",
        "vector_store_create",
        "vector_store_delete",
        "vector_store_file_content",
        "vector_store_file_create",
        "vector_store_file_delete",
        "vector_store_file_list",
        "vector_store_file_retrieve",
        "vector_store_file_update",
        "vector_store_list",
        "vector_store_retrieve",
        "vector_store_search",
        "vector_store_update",
        "video_content",
        "video_create_character",
        "video_edit",
        "video_extension",
        "video_generation",
        "video_get_character",
        "video_list",
        "video_remix",
        "video_status",
    }
)


def _dispatch_failure(logging_obj: "Logging | None", error: Exception) -> None:
    """The ``## LOG FAILURE EVENT`` of a routing callback that raised."""
    if logging_obj is not None:
        asyncio.create_task(
            logging_obj.dispatch_failure_handlers(
                exception=error,
                traceback_exception=traceback.format_exc(),
                prefer_async_handlers=True,
            )
        )


class _FailedCallParams(TypedDict):
    """The ``litellm_params`` of a failed call, as the failure callback reads them."""

    metadata: ReadOnly[Mapping[str, object]]
    model_info: ReadOnly[NotRequired[Mapping[str, object] | None]]


async def _deployments_filtered_by(
    callback: CustomLogger,
    model: str,
    deployments: object,
    messages: object,
    request_kwargs: dict[str, object] | None,  # mutable-ok: CustomLogger takes the request dict
    parent_otel_span: "Span | None",
) -> object:
    return await callback.async_filter_deployments(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped deployment dicts
        model=model,
        healthy_deployments=deployments,  # pyright: ignore[reportArgumentType]  # what the last callback returned
        messages=messages,  # pyright: ignore[reportArgumentType]  # the request's messages, as given
        request_kwargs=request_kwargs,
        parent_otel_span=parent_otel_span,
    )


def _metadata_model_id(model_info: object) -> object:
    return model_info.get("id") if isinstance(model_info, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the metadata's model_info dict


class RustLegacyMembers(RustRegistryMembers, RustOptionalChecks):
    """Mixed into ``RustCore``, which supplies its host registry, selectors and checks."""

    # RustCore sets these in its constructor.
    _cache: "DualCache"
    _usage: RouterUsage
    _cooldown_given: object
    _custom_strategy: "CustomRoutingStrategyBase | None"
    # The constructor's argument, or the attribute as assigned.
    auto_router_capability_limit: "Callable[[], int | None] | None"

    TIER_PARAMS_NEVER_DROPPED: Final = _TIER_PARAMS_NEVER_DROPPED

    # -- registry lookups --------------------------------------------------------------------------

    def get_routing_group(self, model_name: str) -> "RoutingGroup | None":
        """The routing group callable as ``model_name`` (``get_routing_group``)."""
        return self._registry_state.get_routing_group(model_name)

    def team_public_model_names_of(self, team_id: str) -> frozenset[str]:
        """The public names ``team_id`` serves its own deployments under (``team_public_model_names_of``)."""
        return self._registry_state.team_public_model_names_of(team_id)

    def served_openai_models(self, model_name: str) -> tuple[str | None, ...]:
        """The OpenAI models ``model_name`` serves (``_served_openai_models``). The Python core keeps
        an ``lru_cache`` over it, emptied with every model list change."""
        return _served_openai_models(self._registry_state._core_deployments_named(model_name) or ())  # pyright: ignore[reportPrivateUsage]  # the core reads its own host registry

    def auto_router_capability_violation(self, capability: GatedAutoRouterCapability) -> str | None:
        """Why one more router claiming ``capability`` cannot join this router, or None when it can
        (``auto_router_capability_violation``); the limit is resolved on every call."""
        resolver: Final = self.auto_router_capability_limit
        limit: Final = resolver() if resolver is not None else None
        others: Final = count_capability_routers(
            (deployment for deployment in self._registry_state._core_model_list if isinstance(deployment, Mapping)),  # pyright: ignore[reportPrivateUsage, reportUnnecessaryIsInstance]  # the live list
            capability=capability,
        )
        return capability_limit_violation(capability=capability, held=others + 1, limit=limit)

    # -- routing state -----------------------------------------------------------------------------

    def retire_routing_state(self) -> None:
        """Stop the background work of the selectors and the budget limiter (``retire_routing_state``)."""
        self._opt.retire_state((*self._sel.callback_selectors(), *self._sel.override_selectors()))

    def routing_strategy_init(self, routing_strategy: object, routing_strategy_args: Mapping[str, object]) -> None:
        """``routing_strategy_init``: the default selector only. As on the Python core the router's
        ``routing_strategy`` setting stays, so picks keep following it; ``apply`` changes it."""
        verbose_router_logger.info("Routing strategy: %s", routing_strategy)
        validate_routing_strategy(routing_strategy)  # pyright: ignore[reportArgumentType]  # the strategy as given
        # `_reset_custom_routing_strategy`.
        self._custom_strategy = None
        self.__dict__.pop("get_available_deployment", None)
        self.__dict__.pop("_core_async_get_available_deployment", None)
        self._sel.init_default(routing_strategy, routing_strategy_args, configured=False)

    def set_optional_pre_call_checks(self, optional_pre_call_checks: Iterable[object] | None) -> None:
        """``set_optional_pre_call_checks``: drop the toggleable checks the list leaves out, then add it."""
        self._opt.set(None if optional_pre_call_checks is None else tuple(optional_pre_call_checks))

    def initialize_assistants_endpoint(self) -> None:
        """``initialize_assistants_endpoint`` binds the endpoints; this core answers them from
        ``__getattr__``, so what is left to do is to drop an assigned replacement."""
        for name in _ASSISTANTS_ENDPOINTS:
            self.__dict__.pop(name, None)

    def initialize_router_endpoints(self) -> None:
        """``initialize_router_endpoints``, as ``initialize_assistants_endpoint`` above."""
        from litellm.containers.endpoint_factory import (
            _generated_endpoints,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # the names the Python core registers
        )

        for name in (*_ROUTER_ENDPOINTS, *_generated_endpoints):
            self.__dict__.pop(name, None)

    # -- pre-routing strategies --------------------------------------------------------------------

    def init_auto_router_deployment(self, deployment: "Deployment") -> None:
        """Register the deployment's auto-router (``init_auto_router_deployment``)."""
        self._registry_state.strategies._init_auto_router(deployment)  # pyright: ignore[reportOptionalMemberAccess, reportPrivateUsage]  # RustCore installs its strategies in its constructor

    def init_complexity_router_deployment(self, deployment: "Deployment") -> None:
        """Register the deployment's complexity-router (``init_complexity_router_deployment``)."""
        self._registry_state.strategies._init_complexity_router(deployment)  # pyright: ignore[reportOptionalMemberAccess, reportPrivateUsage]  # see above

    def init_quality_router_deployment(self, deployment: "Deployment") -> None:
        """Register the deployment's quality-router (``init_quality_router_deployment``)."""
        self._registry_state.strategies._init_quality_router(deployment)  # pyright: ignore[reportOptionalMemberAccess, reportPrivateUsage]  # see above

    def init_adaptive_router_deployment(self, deployment: "Deployment") -> None:
        """Register the deployment's adaptive-router and its post-call hook (``init_adaptive_router_deployment``)."""
        self._registry_state.strategies._init_adaptive_router(deployment)  # pyright: ignore[reportOptionalMemberAccess, reportPrivateUsage]  # see above

    # -- routing callbacks -------------------------------------------------------------------------

    def routing_strategy_pre_call_checks(
        self,
        deployment: dict[str, object],  # mutable-ok: CustomLogger takes the deployment dict
    ) -> None:
        """``routing_strategy_pre_call_checks``: the sync check of each ``CustomLogger`` callback."""
        for callback in self._opt.routing_callbacks():
            if isinstance(callback, CustomLogger):
                callback.pre_call_check(deployment)  # pyright: ignore[reportUnknownMemberType]  # CustomLogger takes an untyped deployment dict

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    async def async_routing_strategy_pre_call_checks(
        self,
        deployment: dict[str, object],  # mutable-ok: CustomLogger takes the deployment dict
        parent_otel_span: "Span | None",
        logging_obj: "Logging | None" = None,
    ) -> None:
        """``async_routing_strategy_pre_call_checks``: a check's rate limit cools the deployment down
        under the Python rules (``set_cooldown_deployments``, cooldown event included)."""
        from litellm import exceptions as litellm_exceptions

        for callback in self._opt.routing_callbacks():
            if isinstance(callback, CustomLogger):
                try:
                    await callback.async_pre_call_check(deployment, parent_otel_span)  # pyright: ignore[reportUnknownMemberType]  # see above
                except litellm_exceptions.RateLimitError as e:
                    _dispatch_failure(logging_obj, e)
                    _ = set_cooldown_deployments(
                        ctx=RoutingContext(self),  # pyright: ignore[reportArgumentType]  # the core's RoutingContext
                        exception_status=e.status_code,
                        original_exception=e,
                        deployment=deployment["model_info"]["id"],  # pyright: ignore[reportIndexIssue, reportUnknownArgumentType]  # read as the Python core reads it
                        time_to_cooldown=self._cooldown_given,  # pyright: ignore[reportArgumentType]  # the value as given, as the Python core stores it
                    )
                    raise
                except Exception as e:
                    _dispatch_failure(logging_obj, e)
                    raise

    async def async_callback_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict[str, object]],  # mutable-ok: CustomLogger deployment-list contract
        messages: object,
        parent_otel_span: "Span | None",
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: CustomLogger takes the request dict
        logging_obj: "Logging | None" = None,
    ) -> object:
        """``async_callback_filter_deployments``: each routing callback narrows the deployments in turn."""
        remaining: object = healthy_deployments  # rebind-ok: narrowed by each callback
        for callback in self._opt.routing_callbacks():
            if isinstance(callback, CustomLogger):
                try:
                    remaining = await _deployments_filtered_by(
                        callback, model, remaining, messages, request_kwargs, parent_otel_span
                    )
                except Exception as e:
                    _dispatch_failure(logging_obj, e)
                    raise
        return remaining

    async def async_admit_deployment_turn(
        self, model_group: str, opened_on: Mapping[str, object], request_kwargs: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Admit a later turn of an open websocket (``async_admit_deployment_turn``)."""
        port: Final[EndpointCore] = self.__dict__["_port"]  # pyright: ignore[reportAny]  # RustCore's own attribute
        return await admit_deployment_turn(port, model_group, opened_on, request_kwargs)

    # -- one provider call -------------------------------------------------------------------------

    async def make_call(
        self,
        original_function: Callable[..., object],
        *args: object,
        **kwargs: object,  # kwargs-ok: the endpoint's request
    ) -> object:
        """``make_call``: call the endpoint, count its usage and stamp the response's headers. A
        failed call hands back the io tokens it reserved."""
        model_group: Final = kwargs.get("model")
        generic_function: Final = kwargs.get("original_generic_function")
        endpoint_function: Final = generic_function if callable(generic_function) else original_function
        try:
            with endpoint_selection(
                getattr(endpoint_function, "__name__", None),
                getattr(endpoint_function, "__module__", None),
                model=model_group,
            ):
                response: object = original_function(*args, **kwargs)  # rebind-ok: awaited below
                if coroutine_checker.is_async_callable(response) or inspect.isawaitable(response):
                    awaitable: Final[Awaitable[object]] = response  # pyright: ignore[reportAssignmentType]  # the endpoint's awaitable
                    response = await awaitable  # rebind-ok: awaited here
        except Exception:
            with contextlib.suppress(Exception):
                await async_io_token_refund_failure(
                    self._cache, kwargs, parent_otel_span=get_parent_otel_span_from_kwargs(kwargs)
                )
            raise
        await self.increment_deployment_usage_for_response(response=response, request_kwargs=kwargs)
        return await self.set_response_headers(
            response=response, model_group=model_group if isinstance(model_group, str) else None, request_kwargs=kwargs
        )

    async def increment_deployment_usage_for_response(
        self, response: object, request_kwargs: Mapping[str, object]
    ) -> None:
        """Count a response's tokens and request against its deployment, once the call returned, and
        note them on the request for the success callback (``increment_deployment_usage_for_response``)."""
        if response is None:
            return
        try:
            deployment_metadata: Final = find_deployment_metadata(request_kwargs)
            model_group: Final = request_kwargs.get("model")
            if deployment_metadata is None or not isinstance(model_group, str):
                return
            _ = await self.increment_deployment_usage_before_callbacks(
                deployment_metadata,
                model_group,
                response_total_token_count(response),
                get_parent_otel_span_from_kwargs(dict(request_kwargs)),
            )
        except Exception as e:  # noqa: BLE001  # as the Python core: usage accounting never fails a served call
            verbose_router_logger.debug(
                "litellm.router.Router::increment_deployment_usage_for_response(): Exception occured - %s", e
            )

    async def increment_deployment_usage_before_callbacks(
        self,
        deployment_metadata: dict[str, object],  # mutable-ok: the success callback reads the counted tokens off it
        model_group: str,
        total_tokens: int,
        parent_otel_span: "Span | None",
    ) -> bool:
        deployment_id: Final = _metadata_model_id(deployment_metadata.get("model_info"))
        if deployment_id is None:
            return False
        deployment_name: Final = deployment_metadata.get("deployment")
        await count_before_callbacks(
            deployment_metadata,
            ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY,
            total_tokens,
            partial(
                self._increment_deployment_usage,
                deployment_id=str(deployment_id),
                deployment_name=deployment_name if isinstance(deployment_name, str) else None,
                model_group=model_group,
                total_tokens=total_tokens,
                parent_otel_span=parent_otel_span,
            ),
        )
        return True

    @with_service_target(ROUTER_USAGE_TARGET)
    async def _increment_deployment_usage(
        self,
        *,
        deployment_id: str,
        deployment_name: str | None,
        model_group: str,
        total_tokens: float,
        parent_otel_span: "Span | None",
    ) -> None:
        """``_increment_deployment_usage`` with its one request."""
        if not self._registry_state.cached_deployment_tracks_usage(deployment_id, model_group):
            return
        current_minute: Final = get_utc_datetime().strftime("%H-%M")
        await self._usage.increment_deployment(deployment_id, deployment_name, current_minute, total_tokens, 1)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_deployment_callback_on_failure(
        self, kwargs: Mapping[str, object], completion_response: object, start_time: object, end_time: object
    ) -> None:
        """Count a failed call as one more request of the minute (``async_deployment_callback_on_failure``)."""
        if is_batch_retrieve_call_type(kwargs.get("call_type")):
            return
        litellm_params: Final[_FailedCallParams] = kwargs["litellm_params"]  # pyright: ignore[reportAssignmentType]  # read as the Python core reads it, a missing key included
        deployment_name: Final = litellm_params["metadata"].get("deployment", None)
        model_group: Final = litellm_params["metadata"].get("model_group", None)
        model_info: Final = litellm_params.get("model_info", {}) or {}
        deployment_id: Final = model_info.get("id", None)
        if model_group is None or deployment_id is None:
            return
        model: Final = None if deployment_name is None else str(deployment_name)
        counter: Final = UsageCounter(str(deployment_id), model, get_utc_datetime().strftime("%H-%M"), "rpm")
        await self._usage.increment(counter=counter, value=1, ttl=_USAGE_TTL)

    # -- the plugin context ------------------------------------------------------------------------

    def _update_kwargs_before_fallbacks(
        self,
        model: str,
        kwargs: dict[str, object],  # mutable-ok: the core updates the request kwargs in place
        metadata_variable_name: str | None = "metadata",
    ) -> None:
        """``_update_kwargs_before_fallbacks``: a trace id and the requested group, in place."""
        _ = kwargs.setdefault("litellm_trace_id", str(uuid.uuid4()))
        alias: Final = model if self._registry_state._core_get_model_from_alias(model=model) else None  # pyright: ignore[reportPrivateUsage]  # the core reads its own host registry
        kwargs.setdefault(metadata_variable_name, {}).update({"model_group": model, "model_group_alias": alias})  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue, reportUnknownMemberType]  # the bucket, as the Python core writes it

    def _params_sent_to_deployment(
        self, request_kwargs: Mapping[str, object], deployment: "Deployment | None"
    ) -> Mapping[str, object]:
        """``_params_sent_to_deployment`` of a ``Deployment``, which ``RoutingContext.deployment_params`` hands in."""
        params: Final = deployment.litellm_params.model_dump(exclude_none=True) if deployment else {}
        configured: Final = as_str_mapping(RoutingContext(self).settings().default_litellm_params)  # pyright: ignore[reportArgumentType, reportUnknownMemberType, reportUnknownArgumentType]  # the core's RoutingContext
        defaults: Final = {k: v for k, v in (configured or {}).items() if v is not None}
        tool_choice: Final = {"tool_choice": params["tool_choice"]} if params.get("tool_choice") is not None else {}
        return MappingProxyType({**params, **defaults, **tool_choice, **request_kwargs})


__all__ = ["RustLegacyMembers"]
