"""The Python half of the legacy callback contract the native call lifecycle drives.

Everything here is named after the `Logging` object and the sync/async callback
registries it fans out to. It expires with that contract.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime
import inspect
import json
import sys
import traceback
import uuid
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    cast,  # noqa: TID251  # bounded compatibility calls into legacy Python integrations
)

import httpx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging


class MetadataUpdater(Protocol):
    def __call__(
        self,
        result: object,
        logging_obj: Logging,
        model: str | None,
        kwargs: dict[str, object],
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        cache_hit: bool | None = None,
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

    arguments: Final = {
        "litellm_call_id": str(uuid.uuid4()),
        **kwargs,
    }
    supplied: Final = arguments.get("litellm_logging_obj")
    if isinstance(supplied, Logging):
        return _begin_call(CallSetup(supplied, arguments), asynchronous)
    logger, prepared = function_setup(call_type, Rules(), start_time, *args, is_async_call=asynchronous, **arguments)
    return _begin_call(CallSetup(logger, prepared), asynchronous)


# Loggers whose current native call is async: their sync failure step is followed by the async one
_ASYNC_CALLS: Final[weakref.WeakSet[Logging]] = weakref.WeakSet()


def _begin_call(call_setup: CallSetup, asynchronous: bool) -> CallSetup:
    if asynchronous:
        _ASYNC_CALLS.add(call_setup.logger)
    else:
        _ASYNC_CALLS.discard(call_setup.logger)
    return _claim_budget_reservation(call_setup, asynchronous)


def _claim_budget_reservation(call_setup: CallSetup, asynchronous: bool) -> CallSetup:
    from litellm.litellm_core_utils.core_helpers import bind_budget_reservation_to_callbacks

    if asynchronous and not is_internal_call():
        bind_budget_reservation_to_callbacks(call_setup.logger.litellm_params)
    return call_setup


def finalize(
    response: object,
    logger: Logging,
    kwargs: dict[str, object],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    cache_hit: bool,
) -> None:
    from litellm.litellm_core_utils.llm_response_utils import response_metadata

    model: Final = kwargs.get("model")
    update: Final = cast(  # cast-ok: legacy metadata function accepts concrete kwargs
        MetadataUpdater, response_metadata.update_response_metadata
    )
    if cache_hit:
        from litellm.caching.caching_handler import (
            _set_cached_hidden_param,  # pyright: ignore[reportPrivateUsage]  # the @client cache hit's own hidden params
        )
        from litellm.rust_bridge.host.cache import hit_cache_key, restore_hit

        restore_hit(response, logger)
        update(response, logger, logger.model, kwargs, start_time, end_time, cache_hit=True)
        _set_cached_hidden_param(response, "cache_key", hit_cache_key(logger))
        return
    update(response, logger, model if isinstance(model, str) else None, kwargs, start_time, end_time)
    cache_key: Final = logger.model_call_details.get("cache_key")
    if logger.model_call_details.get("cache_hit") is True and isinstance(cache_key, str):
        from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict

        hidden: Final = get_hidden_params_dict(response, create=True)
        hidden.update({"cache_key": cache_key, "cache_hit": True})


class LoggingSurface(Protocol):
    deferred_success_result: tuple[object, datetime.datetime, datetime.datetime] | None
    enqueue_deferred_logging: Callable[[], None] | None

    @property
    def model_call_details(self) -> Mapping[str, object]: ...

    @property
    def litellm_params(self) -> Mapping[str, object]: ...

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

    def claim_async_success_log(self) -> bool: ...

    def mark_async_success_log_enqueued(self) -> None: ...

    def handle_sync_success_callbacks_for_async_calls(
        self, result: object, start_time: datetime.datetime, end_time: datetime.datetime, cache_hit: bool | None = None
    ) -> None: ...

    def failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime | None = None,
        end_time: datetime.datetime | None = None,
        *,
        for_async_request: bool = False,
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
        *,
        for_async_request: bool = False,
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


def pre_call(
    logger: LoggingSurface,
    input: str | Sequence[Mapping[str, str]],
    api_key: str | None,
    additional_args: dict[str, object],
) -> None:
    logger.pre_call(input=input, api_key=api_key, additional_args=additional_args)


def messages_input(body: Mapping[str, object], kwargs: Mapping[str, object]) -> Sequence[Mapping[str, str]]:
    from litellm.messages import anthropic_messages_handler

    # Python's Messages handler builds its body as messages, max_tokens, model, then its own parameters in
    # signature order and the caller's other keywords in their order, and logs that body as serialized
    order: Final = (
        "messages",
        "max_tokens",
        "model",
        *inspect.signature(anthropic_messages_handler).parameters,
        *kwargs,
        *body,
    )
    return [
        {"role": "user", "content": json.dumps({name: body[name] for name in dict.fromkeys(order) if name in body})}
    ]


def post_call(
    logger: LoggingSurface, original_response: str, api_key: str | None, additional_args: dict[str, object]
) -> None:
    logger.post_call(original_response=original_response, api_key=api_key, additional_args=additional_args)


class _UnreadBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        return iter(())

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in ():
            yield chunk


def wire_headers(headers: Sequence[tuple[str, str]]) -> httpx.Headers:
    return httpx.Headers([(name.encode(), value.encode()) for name, value in headers])


def _sent_request(sent: tuple[str, Mapping[str, str], Mapping[str, object]]) -> httpx.Request:
    url, headers, body = sent
    return httpx.Request("POST", url, headers=list(headers.items()), content=json.dumps(body, default=str).encode())


def record_response(
    logger: Logging,
    status: int,
    headers: Sequence[tuple[str, str]],
    body: str,
    sent: tuple[str, Mapping[str, str], Mapping[str, object]] | None,
) -> None:
    """Stores the provider response the way Python's Messages handler does, for logging to replay."""
    request: Final = None if sent is None else _sent_request(sent)
    streamed: Final = sent is not None and sent[2].get("stream") is True
    response: Final = (
        httpx.Response(status_code=status, stream=_UnreadBody(), request=request)
        if streamed
        else httpx.Response(status_code=status, content=body.encode(), request=request)
    )
    response.headers = wire_headers(headers)
    details: Final = logger.model_call_details  # the call's own logger: recording on it is the point
    details["httpx_response"] = response
    details["response_headers"] = dict(response.headers)


class _ThinkingSignatureRetry(TypedDict):
    kind: ReadOnly[Literal["thinking_signature"]]


class _StrictToolsRetry(TypedDict):
    kind: ReadOnly[Literal["strict_tools"]]
    dropped: ReadOnly[int]


_RETRY_REASONS: Final = TypeAdapter(tuple[_ThinkingSignatureRetry | _StrictToolsRetry, ...])


async def retried_request(
    logger: Logging,
    logged_body: dict[str, object],  # mutable-ok: the retry rewrites the body pre_call logged, as Python's loop does
    body: Mapping[str, object],
    reasons: Sequence[Mapping[str, object]],
) -> None:
    from litellm.llms.anthropic.common_utils import warn_claude_strict_tools_dropped

    if logger.baseline_cache_context is not None:
        await logger.invalidate_baseline_cache_estimate("retried_request")
    for reason in _RETRY_REASONS.validate_python(reasons):
        if reason["kind"] == "strict_tools":
            warn_claude_strict_tools_dropped(reason["dropped"])
    for name in logged_body.keys() - body.keys():
        del logged_body[name]  # rebind-ok: the logged body is the one logging already holds
    logged_body.update({name: value for name, value in body.items() if logged_body.get(name) != value})
    logger.model_call_details.update(logged_body)


def record_api_window(logger: Logging, start: float, end: float) -> None:
    """Records the provider exchange's window the way Python's timed `AsyncHTTPHandler.post` does."""
    from litellm.litellm_core_utils.logging_utils import (
        _set_duration_in_model_call_details,  # pyright: ignore[reportPrivateUsage]  # the timed post's own recorder
    )

    _set_duration_in_model_call_details(
        logging_obj=logger,
        start_time=datetime.datetime.fromtimestamp(start),  # noqa: DTZ006  # the timed post records naive local times
        end_time=datetime.datetime.fromtimestamp(end),  # noqa: DTZ006  # the timed post records naive local times
    )
    _log_api_window(logger, start, end)


def _log_api_window(logger: Logging, start: float, end: float) -> None:
    from litellm._logging import verbose_logger
    from litellm.litellm_core_utils.logging_utils import (
        _get_parent_otel_span_from_logging_obj,  # pyright: ignore[reportPrivateUsage]  # the timed post's own span lookup
        _get_service_logger,  # pyright: ignore[reportPrivateUsage]  # the timed post's own service logger
    )
    from litellm.types.services import ServiceTypes

    parent_otel_span: Final = _get_parent_otel_span_from_logging_obj(logger)
    try:
        service_logger: Final = _get_service_logger()
        _ = asyncio.create_task(
            service_logger.async_service_success_hook(  # pyright: ignore[reportUnknownMemberType]  # the hook's untyped event_metadata dict
                service=ServiceTypes.LITELLM,
                duration=end - start,
                call_type="post <- track_llm_api_timing",
                start_time=start,
                end_time=end,
                parent_otel_span=parent_otel_span,
            )
        )
    except Exception as e:  # noqa: BLE001  # the timed post logs and drops a service logging failure
        verbose_logger.debug("Error in service logging: %s", e)


def times_provider_exchange(client: object, timeout: object) -> bool:
    from litellm.rust_bridge.host import http_transport

    return not http_transport.sends_through_python(client, timeout)


def defers_async_logging(logger: LoggingSurface) -> bool:
    return bool(getattr(logger, "defer_async_logging", False))


def claim_async_success_log(logger: LoggingSurface) -> bool:
    return logger.claim_async_success_log()


def mark_async_success_enqueued(logger: LoggingSurface) -> None:
    logger.mark_async_success_log_enqueued()


def defer_success(
    logger: LoggingSurface,
    pending: Callable[[], None],
    response: object,
    start: datetime.datetime,
    end: datetime.datetime,
) -> None:
    logger.deferred_success_result = (response, start, end)  # rebind-ok: capture the native attempt before dispatch
    logger.enqueue_deferred_logging = pending  # rebind-ok: the proxy releases logging here


def _cache_hit(logger: LoggingSurface) -> Literal[True] | None:
    return True if logger.model_call_details.get("cache_hit") is True else None


def sync_success_for_async_call(
    logger: LoggingSurface, response: object, start: datetime.datetime, end: datetime.datetime
) -> None:
    logger.handle_sync_success_callbacks_for_async_calls(
        result=response,
        start_time=start,
        end_time=end,
        cache_hit=_cache_hit(logger),
    )


def failure_handler(
    logger: LoggingSurface, error: Exception, start: datetime.datetime, end: datetime.datetime, asynchronous: bool
) -> Coroutine[object, object, None] | None:
    from litellm.litellm_core_utils.core_helpers import unbind_budget_reservation_from_callbacks

    trace: Final = "".join(traceback.format_exception(error))
    if asynchronous:
        if not is_internal_call():
            unbind_budget_reservation_from_callbacks(logger.litellm_params)
        return logger.async_failure_handler(error, trace, start, end)
    logger.failure_handler(error, trace, start, end, for_async_request=logger in _ASYNC_CALLS)
    return None


def submit_success(logger: LoggingSurface, response: object, start: datetime.datetime, end: datetime.datetime) -> None:
    from litellm.litellm_core_utils.litellm_logging import executor

    executor.submit(
        contextvars.copy_context().run,
        logger.success_handler,
        response,
        start,
        end,
        cache_hit=_cache_hit(logger),
    )


def async_success_handler(
    logger: LoggingSurface, response: object, start: datetime.datetime, end: datetime.datetime
) -> Coroutine[object, object, None]:
    return logger.async_success_handler(response, start, end, cache_hit=_cache_hit(logger))


def cache_hit_success(logger: Logging, response: object, start: datetime.datetime, end: datetime.datetime) -> None:
    from litellm.caching.caching_handler import LLMCachingHandler

    handler: Final = logger.llm_caching_handler
    if isinstance(handler, LLMCachingHandler):
        handler.async_log_cache_hit_on_callbacks(
            logging_obj=logger, cached_result=response, start_time=start, end_time=end, cache_hit=True
        )


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
    typed: Final = CallTypes(call_type)
    return hook(kwargs, _messages_stream(response) if typed == CallTypes.anthropic_messages else response, typed)


def _messages_stream(response: object) -> object:
    from litellm.rust_bridge.execution import AsyncStream

    if not isinstance(response, AsyncStream):
        return response
    from litellm.llms.anthropic.messages.native.streaming_iterator import (
        AnthropicMessagesStreamHiddenParams,
        AnthropicMessagesStreamingResponse,
    )

    return AnthropicMessagesStreamingResponse(
        completion_stream=cast(  # cast-ok: the native stream yields the provider's SSE bytes
            AsyncIterator[bytes], response
        ),
        hidden_params=cast(  # cast-ok: the stream's stamped hidden params, shared as Python's handler shares them
            AnthropicMessagesStreamHiddenParams,
            response._hidden_params,  # pyright: ignore[reportPrivateUsage]  # the native stream's own hidden params
        ),
    )


def after_deployment_failure(kwargs: dict[str, object], error: Exception, call_type: str) -> Awaitable[None]:
    from litellm import utils

    hook: Final = cast(  # cast-ok: bounded adapter for the untyped deployment hook
        DeploymentFailureHook, utils.async_post_call_failure_deployment_hook
    )
    return hook(kwargs, error, call_type)


_STREAM_HEADERS: Final = TypeAdapter(Mapping[str, str])


class _HiddenParamsCarrier:
    """Stands in for the stream `@client` stamps, so the head becomes its `_hidden_params`."""

    def __init__(self, hidden_params: dict[str, object]) -> None:  # mutable-ok: the stamps land in this dict
        self._hidden_params: object = hidden_params


def stamp_stream_head(
    head: dict[str, object],  # mutable-ok: the stream's hidden params, stamped in place
    logger: Logging,
    kwargs: dict[str, object],  # mutable-ok: `finalize` takes the call kwargs as a dict
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    cache_hit: bool,
) -> None:
    if kwargs.get("stream") is True:
        logger.stream = True
        logger.model_call_details["stream"] = True
    carrier: Final = _HiddenParamsCarrier(head)
    finalize(carrier, logger, kwargs, start_time, end_time, cache_hit)
    stamped: Final = carrier._hidden_params  # pyright: ignore[reportPrivateUsage]  # the carrier's own field
    if stamped is not head and isinstance(stamped, dict):
        head.update(cast("dict[str, object]", stamped))  # cast-ok: update_response_metadata writes a str-keyed dict


def stream_opened(logger: Logging, head: Mapping[str, object], keeps_requested: bool = False) -> None:
    requested: Final = logger.model_call_details.get("stream")
    streamed: Final = requested if keeps_requested and requested else True
    logger.stream = cast(bool, streamed)  # cast-ok: Python's Messages handler logs the stream flag it sent
    logger.model_call_details["stream"] = streamed
    if "httpx_response" in logger.model_call_details:
        return
    logger.model_call_details["response_headers"] = _STREAM_HEADERS.validate_python(
        head.get("additional_headers") or {}
    )


def _cancelling(task: asyncio.Task[object]) -> bool:
    return task.cancelling() > 0 if sys.version_info >= (3, 11) else True


def _proxy_request(logger: LoggingSurface) -> object:
    return logger.litellm_params.get("proxy_server_request")


def stream_success(
    logger: Logging,
    url_route: str,
    endpoint_type: str,
    request_body: dict[str, object],
    chunks: list[bytes],
    start: datetime.datetime,
    end: datetime.datetime,
    first_chunk: datetime.datetime | None,
    stream_teardown: bool,
) -> None:
    from litellm.llms.anthropic.messages.native.streaming_iterator import (
        GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ,
    )
    from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    if first_chunk is not None:
        logger.completion_start_time = first_chunk
        logger.model_call_details["completion_start_time"] = first_chunk
    superseded_error_event: Final = PassThroughStreamingHandler._error_frame_closed_at(  # pyright: ignore[reportPrivateUsage]  # the Messages stream's own close-at-frame check
        EndpointType(endpoint_type), chunks
    )
    if superseded_error_event is not None:
        _log_superseded_stream_failure(logger, endpoint_type, request_body, chunks, superseded_error_event)
        return
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
    if not stream_teardown and getattr(logger, "_on_deferred_stream_complete", None) is not None:
        logger._deferred_stream_complete_args = (coroutine,)  # pyright: ignore[reportAttributeAccessIssue]  # the proxy's deferred stream release reads this slot
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        from litellm.litellm_core_utils.litellm_logging import executor

        executor.submit(contextvars.copy_context().run, asyncio.run, coroutine)
        return
    cancelled: Final = asyncio.current_task()
    if stream_teardown and cancelled is not None and _cancelling(cancelled) and _proxy_request(logger):
        # the proxy's disconnect cleanup runs later in this task and stamps the 499 error_information this row carries
        cancelled.add_done_callback(lambda _: enqueue_logging(coroutine))
        return
    enqueue_logging(coroutine)


def _log_superseded_stream_failure(
    logger: Logging,
    endpoint_type: str,
    request_body: Mapping[str, object],
    chunks: Sequence[bytes],
    error_event: tuple[str, str, int],
) -> None:
    from litellm._logging import verbose_proxy_logger
    from litellm.llms.anthropic.common_utils import anthropic_error_frame_exception
    from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    error_type, message, status_code = error_event
    PassThroughStreamingHandler._record_partial_usage_for_failure(  # pyright: ignore[reportPrivateUsage]  # the superseded stream's own partial usage record
        litellm_logging_obj=logger,
        endpoint_type=EndpointType(endpoint_type),
        request_body=dict(request_body),
        raw_bytes=tuple(chunks),
        stream_context=None,
    )
    try:
        enqueue_logging(
            logger.dispatch_attempt_failure_handlers(  # pyright: ignore[reportArgumentType]  # the dispatch snapshots now, its coroutine never receives a sent value
                anthropic_error_frame_exception(error_type, message, status_code, str(logger.model)),
                traceback.format_exc(),
            )
        )
    except Exception as e:  # noqa: BLE001  # the stream's failure scheduling logs and drops its own failure
        verbose_proxy_logger.error("Error scheduling stream failure logging: %s", e)


def stream_failure(
    logger: Logging,
    endpoint_type: str,
    request_body: dict[str, object],
    chunks: list[bytes],
    error: Exception,
) -> Coroutine[object, object, None]:
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    verbose_proxy_logger.error("Error in chunk_processor: %s", error)
    # Python's stream fails before its success handler records `cache_hit`; only a cache hit
    # stamps it earlier.
    if logger.model_call_details.get("cache_hit") is not True:
        logger.model_call_details.pop("cache_hit", None)
    return PassThroughStreamingHandler.schedule_stream_failure_logging(
        litellm_logging_obj=logger,
        endpoint_type=EndpointType(endpoint_type),
        request_body=request_body,
        raw_bytes=chunks,
        exception=error,
    )
