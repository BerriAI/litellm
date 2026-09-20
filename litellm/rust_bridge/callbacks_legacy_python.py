"""The Python half of the legacy callback contract the native call lifecycle drives.

Everything here is named after the `Logging` object and the sync/async callback
registries it fans out to. It expires with that contract.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime
import traceback
import uuid
from collections.abc import Awaitable, Coroutine, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    cast,  # noqa: TID251  # bounded compatibility calls into legacy Python integrations
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import CredentialItem


class MetadataUpdater(Protocol):
    def __call__(
        self,
        result: object,
        logging_obj: Logging,
        model: str | None,
        kwargs: dict[str, object],
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CallSetup:
    logger: Logging
    kwargs: dict[str, object]


def setup(
    call_type: str,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    start_time: datetime.datetime,
    asynchronous: bool,
) -> CallSetup:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.utils import Rules, function_setup

    arguments: Final = {  # mutable-ok: function_setup consumes an owned kwargs dict
        "litellm_call_id": str(uuid.uuid4()),
        **kwargs,
    }
    supplied: Final = arguments.get("litellm_logging_obj")
    if isinstance(supplied, Logging):
        return CallSetup(supplied, arguments)
    logger, prepared = function_setup(call_type, Rules(), start_time, *args, is_async_call=asynchronous, **arguments)
    return CallSetup(logger, prepared)


def check_limits(kwargs: Mapping[str, object]) -> None:
    from litellm import (
        BudgetExceededError,
        _current_cost,  # pyright: ignore[reportPrivateUsage]  # shared SDK budget counter has no public accessor
        max_budget,
        num_retries_per_request,
    )
    from litellm.litellm_core_utils.core_helpers import max_retries_per_request_hit

    if max_budget and _current_cost > max_budget:
        raise BudgetExceededError(current_cost=_current_cost, max_budget=max_budget)
    if max_retries_per_request_hit(kwargs, num_retries_per_request):
        raise RuntimeError("Max retries per request hit!")


def finalize(
    response: object,
    logger: Logging,
    kwargs: dict[str, object],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> None:
    from litellm.litellm_core_utils.llm_response_utils import response_metadata

    model: Final = kwargs.get("model")
    update: Final = cast(  # cast-ok: legacy metadata function accepts concrete kwargs
        MetadataUpdater, response_metadata.update_response_metadata
    )
    update(response, logger, model if isinstance(model, str) else None, kwargs, start_time, end_time)


class LoggingSurface(Protocol):
    def update_from_kwargs(
        self,
        kwargs: dict[str, object],
        litellm_params: dict[str, object] | None = None,
        optional_params: dict[str, object] | None = None,
        model: str | None = None,
        user: str | None = None,
        **additional_params: object,
    ) -> None: ...

    def pre_call(
        self, input: object, api_key: object, model: object = None, additional_args: dict[str, object] = ...
    ) -> object: ...

    def post_call(
        self,
        original_response: object,
        input: object = None,
        api_key: object = None,
        additional_args: dict[str, object] = ...,
    ) -> object: ...

    def handle_sync_success_callbacks_for_async_calls(
        self, result: object, start_time: datetime.datetime, end_time: datetime.datetime, cache_hit: object = None
    ) -> None: ...

    def failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime | None = None,
        end_time: datetime.datetime | None = None,
    ) -> None: ...

    def async_failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime | None = None,
        end_time: datetime.datetime | None = None,
    ) -> Coroutine[object, object, None]: ...

    def success_handler(
        self,
        result: object = None,
        start_time: datetime.datetime | None = None,
        end_time: datetime.datetime | None = None,
        cache_hit: bool | None = None,
        **kwargs: object,
    ) -> None: ...

    def async_success_handler(
        self,
        result: object = None,
        start_time: datetime.datetime | None = None,
        end_time: datetime.datetime | None = None,
        cache_hit: bool | None = None,
        **kwargs: object,
    ) -> Coroutine[object, object, None]: ...


if TYPE_CHECKING:
    _LOGGING_CONFORMS: type[LoggingSurface] = Logging


class LoggingWorker(Protocol):
    def ensure_initialized_and_enqueue(self, async_coroutine: Coroutine[object, object, None]) -> None: ...


class StreamingLogBuilder(Protocol):
    def __call__(
        self,
        *,
        litellm_logging_obj: Logging,
        passthrough_success_handler_obj: object,
        url_route: str,
        request_body: dict[str, object],
        endpoint_type: object,
        start_time: datetime.datetime,
        raw_bytes: list[bytes],
        end_time: datetime.datetime,
    ) -> Coroutine[object, object, None]: ...


class DeploymentHook(Protocol):
    def __call__(self, kwargs: dict[str, object], call_type: str) -> Awaitable[object]: ...


class DeploymentSuccessHook(Protocol):
    def __call__(self, request_data: dict[str, object], response: object, call_type: object) -> Awaitable[object]: ...


class DeploymentFailureHook(Protocol):
    def __call__(self, request_data: Mapping[str, object], exception: Exception, call_type: str) -> Awaitable[None]: ...


def update_logging(
    logger: LoggingSurface,
    kwargs: dict[str, object],
    model: str,
    optional_params: dict[str, object],
    litellm_params: dict[str, object],
    custom_llm_provider: str,
) -> None:
    logger.update_from_kwargs(
        kwargs=kwargs,
        model=model,
        optional_params=optional_params,
        litellm_params=litellm_params,
        custom_llm_provider=custom_llm_provider,
    )


def pre_call(logger: LoggingSurface, input: str, api_key: str | None, additional_args: dict[str, object]) -> None:
    logger.pre_call(input=input, api_key=api_key, additional_args=additional_args)


def post_call(
    logger: LoggingSurface, original_response: str, api_key: str | None, additional_args: dict[str, object]
) -> None:
    logger.post_call(original_response=original_response, api_key=api_key, additional_args=additional_args)


def defers_async_logging(logger: LoggingSurface) -> bool:
    return bool(getattr(logger, "_defer_async_logging", False))


def defer_success(logger: LoggingSurface, pending: object) -> None:
    setattr(logger, "_native_pending_logging", pending)


def sync_success_for_async_call(
    logger: LoggingSurface, response: object, start: datetime.datetime, end: datetime.datetime
) -> None:
    logger.handle_sync_success_callbacks_for_async_calls(result=response, start_time=start, end_time=end)


def failure_handler(
    logger: LoggingSurface, error: Exception, start: datetime.datetime, end: datetime.datetime, asynchronous: bool
) -> Coroutine[object, object, None] | None:
    trace: Final = "".join(traceback.format_exception(error))
    if asynchronous:
        return logger.async_failure_handler(error, trace, start, end)
    logger.failure_handler(error, trace, start, end)
    return None


def submit_success(logger: LoggingSurface, response: object, start: datetime.datetime, end: datetime.datetime) -> None:
    from litellm.litellm_core_utils.litellm_logging import executor

    executor.submit(contextvars.copy_context().run, logger.success_handler, response, start, end)


def async_success_handler(
    logger: LoggingSurface, response: object, start: datetime.datetime, end: datetime.datetime
) -> Coroutine[object, object, None]:
    return logger.async_success_handler(response, start, end)


def enqueue_logging(coroutine: Coroutine[object, object, None]) -> None:
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    worker: Final = cast(  # cast-ok: bounded adapter for the untyped logging worker
        LoggingWorker, GLOBAL_LOGGING_WORKER
    )
    contextvars.copy_context().run(worker.ensure_initialized_and_enqueue, coroutine)


def restore_context(logger: LoggingSurface) -> None:
    from litellm.utils import (
        _restore_correlation_context_if_supported,  # pyright: ignore[reportPrivateUsage]  # the @client wrapper restores the same correlation context
    )

    _restore_correlation_context_if_supported(logger)


def custom_pricing_fields() -> tuple[str, ...]:
    from litellm.types.utils import CustomPricingLiteLLMParams

    return tuple(CustomPricingLiteLLMParams.model_fields)


def is_internal_call() -> bool:
    from litellm._internal_context import is_internal_call as internal

    return internal.get()


def credential_list() -> list[CredentialItem]:
    from litellm import credential_list as credentials

    return credentials


def warn_unknown_credential(name: str, loaded: int) -> None:
    from litellm._logging import verbose_logger

    verbose_logger.warning(
        "litellm_credential_name=%s matched none of the %d loaded credentials; the request runs without it",
        name,
        loaded,
    )


def before_deployment_call(kwargs: dict[str, object], call_type: str) -> Awaitable[object]:
    from litellm import utils

    hook: Final = cast(  # cast-ok: bounded adapter for the untyped deployment hook
        DeploymentHook, utils.async_pre_call_deployment_hook
    )
    return hook(kwargs, call_type)


def after_deployment_success(kwargs: dict[str, object], response: object, call_type: str) -> Awaitable[object]:
    from litellm import utils
    from litellm.types.utils import CallTypes

    hook: Final = cast(  # cast-ok: bounded adapter for the untyped deployment hook
        DeploymentSuccessHook, utils.async_post_call_success_deployment_hook
    )
    return hook(kwargs, response, CallTypes(call_type))


def after_deployment_failure(kwargs: dict[str, object], error: Exception, call_type: str) -> Awaitable[None]:
    from litellm import utils

    hook: Final = cast(  # cast-ok: bounded adapter for the untyped deployment hook
        DeploymentFailureHook, utils.async_post_call_failure_deployment_hook
    )
    return hook(kwargs, error, call_type)


def stream_opened(logger: Logging) -> None:
    logger.stream = True
    logger.model_call_details["stream"] = True


def stream_success(
    logger: Logging,
    url_route: str,
    endpoint_type: str,
    request_body: dict[str, object],
    chunks: list[bytes],
    start: datetime.datetime,
    end: datetime.datetime,
    first_chunk: datetime.datetime | None,
) -> None:
    from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
        GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ,
    )
    from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    if first_chunk is not None:
        logger.completion_start_time = first_chunk
        logger.model_call_details["completion_start_time"] = first_chunk
    build: Final = cast(  # cast-ok: bounded adapter for the untyped pass-through logging builder
        StreamingLogBuilder,
        PassThroughStreamingHandler._route_streaming_logging_to_handler,  # pyright: ignore[reportPrivateUsage]  # the Messages stream iterator bills through the same builder
    )
    coroutine: Final = build(
        litellm_logging_obj=logger,
        passthrough_success_handler_obj=GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ,
        url_route=url_route,
        request_body=request_body,
        endpoint_type=EndpointType(endpoint_type),
        start_time=start,
        raw_bytes=chunks,
        end_time=end,
    )
    if getattr(logger, "_on_deferred_stream_complete", None) is not None:
        logger._deferred_stream_complete_args = (coroutine,)  # pyright: ignore[reportAttributeAccessIssue]  # the proxy's deferred stream release reads this slot
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        from litellm.litellm_core_utils.litellm_logging import executor

        executor.submit(contextvars.copy_context().run, asyncio.run, coroutine)
        return
    enqueue_logging(coroutine)


def stream_failure(
    logger: Logging,
    endpoint_type: str,
    request_body: dict[str, object],
    chunks: list[bytes],
    error: Exception,
) -> Coroutine[object, object, None]:
    from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    return PassThroughStreamingHandler.schedule_stream_failure_logging(
        litellm_logging_obj=logger,
        endpoint_type=EndpointType(endpoint_type),
        request_body=request_body,
        raw_bytes=chunks,
        exception=error,
    )
