from __future__ import annotations

import asyncio
import contextlib
import datetime
import errno
import functools
import http
import http.client
import inspect
import ipaddress
import os
import socket
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Final, TypeAlias, TypeVar, cast  # noqa: TID251  # narrows native payloads to public TypedDicts
from urllib.parse import SplitResult, urlsplit

import httpx
from aiohttp.http_exceptions import ContentLengthError, TransferEncodingError
from pydantic import JsonValue, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.caching.caching_handler import LLMCachingHandler
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils import exception_mapping_utils
from litellm.litellm_core_utils.core_helpers import (
    as_str_mapping,
    drops_unsupported_params,
    normalize_drop_params,
    only,
    proxy_request_headers,
    stream_cost_injection_requested,
)
from litellm.litellm_core_utils.exception_mapping_utils import with_deployment_debug_info
from litellm.litellm_core_utils.get_llm_provider_logic import get_provider_affinity_header
from litellm.litellm_core_utils.get_provider_specific_headers import ProviderSpecificHeaderUtils
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.model_latest_aliases import is_model_latest_alias
from litellm.litellm_core_utils.provider_affinity import add_provider_affinity_header
from litellm.litellm_core_utils.reasoning_effort_capability import deployment_reasoning_effort_levels
from litellm.litellm_core_utils.request_timeout_resolver import get_configured_request_timeout
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.anthropic.common_utils import (
    AnthropicError,
    AnthropicModelInfo,
    anthropic_pass_through_api_base,
    strip_provider_specific_fields_from_anthropic_messages,
)
from litellm.llms.anthropic.messages.native import handler
from litellm.llms.anthropic.messages.native import utils as native_utils
from litellm.llms.anthropic.messages.native.handler import (
    _bridge_target as _bridged,  # pyright: ignore[reportPrivateUsage]  # the handler's own bridge target
)
from litellm.llms.anthropic.messages.native.handler import (
    _BridgeTarget,  # pyright: ignore[reportPrivateUsage]  # the handler's own bridge target
)
from litellm.llms.anthropic.messages.native.handler import (
    _native_messages_config as native_messages_config,  # pyright: ignore[reportPrivateUsage]  # the handler's own config lookup
)
from litellm.llms.anthropic.messages.native.handler import (
    _resolve_requested_provider as _requested_provider,  # pyright: ignore[reportPrivateUsage]  # the handler's own provider resolution
)
from litellm.llms.anthropic.messages.native.handler import (
    _sets_own_entra_credential as _own_entra_credential,  # pyright: ignore[reportPrivateUsage]  # the handler's own credential check
)
from litellm.llms.anthropic.messages.native.handler import (
    _with_deployment_output_effort as _deployment_output_effort,  # pyright: ignore[reportPrivateUsage]  # the handler's own effort rounding
)
from litellm.llms.anthropic.messages.native.interceptors import MessagesInterceptor, get_messages_interceptors
from litellm.llms.anthropic.messages.native.utils import normalize_native_messages, prepare_native_messages
from litellm.llms.anthropic.messages.safeguards import (
    SafeguardsEvaluator,
    StreamedSafeguardResults,
    native_route_classifies,
    with_safeguard_results,
)
from litellm.llms.anthropic.messages.utils import is_reasoning_auto_summary_enabled
from litellm.llms.anthropic.wif import get_anthropic_wif_token
from litellm.llms.base_llm.anthropic_messages.transformation import BaseAnthropicMessagesConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # its params dict is untyped
    get_ssl_configuration,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import (
    _merge_messages_header_sources as _merged_headers,  # pyright: ignore[reportPrivateUsage]  # the handler's own header merge
)
from litellm.responses.mcp.tool_precheck import may_name_litellm_mcp_gateway
from litellm.rust_bridge import detectors, execution, failures
from litellm.rust_bridge.execution import AsyncStream
from litellm.rust_bridge.host import cache, callbacks_legacy_python, http_transport
from litellm.rust_bridge.messages.entrypoints import MessagesStreamHead
from litellm.rust_bridge.public_call import NativeCall, known_params, optional_sequence, optional_str
from litellm.secret_managers.main import get_secret_str
from litellm.types.integrations.custom_logger import WEBSEARCH_CONVERTED_STREAM_KEY
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import CallTypes, ProviderSpecificHeader, anthropic_wif_litellm_params
from litellm.utils import clamp_max_tokens

_RaisedT: Final = TypeVar("_RaisedT", bound=Exception)
_METADATA: Final = TypeAdapter(dict[str, object])
_BODY_FIELDS: Final = frozenset(
    {
        "max_tokens",
        "metadata",
        "stop_sequences",
        "stream",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_k",
        "inference_geo",
        "top_p",
        "mcp_servers",
        "context_management",
        "compaction",
        "container",
        "output_format",
        "speed",
        "output_config",
        "cache_control",
        "reasoning_effort",
        "safeguards",
        "service_tier",
        "thread",
    }
)

WORKLOAD_IDENTITY_FIELDS: Final = anthropic_wif_litellm_params


@dataclass(frozen=True, slots=True)
class MessagesSettings:
    drop_params: bool
    reasoning_auto_summary: bool
    additional_drop_params: Sequence[str]
    request_timeout: float | None
    drops_unsupported_params: bool
    model: str | None = None
    mock_response: Mapping[str, object] | None = None
    stream_cost_model: str | None = None
    extra_headers: Mapping[str, object] | None = None
    mcp_gateway: bool = False
    ascii_response_headers: bool = False
    joined_reads: bool = False
    ascii_request_headers: bool = True


def response(value: Mapping[str, object]) -> AnthropicMessagesResponse:
    return cast(  # cast-ok: AnthropicMessagesResponse is a TypedDict over the normalized native payload
        AnthropicMessagesResponse,
        dict(value),
    )


def inferred_provider(model: str, api_base: str | None) -> str | None:
    """The provider Python's handler resolves for a call that names none, or None when the model
    carries no provider `get_llm_provider` knows."""
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=model, api_base=api_base)
    except litellm.BadRequestError:
        return None
    return provider


def modify_max_tokens(
    model: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],  # mutable-ok: the call's keyword view, rewritten in place as `@client` does
) -> None:
    unclamped: Final = kwargs.get("max_tokens")
    clamp_max_tokens(model, args, kwargs, CallTypes.anthropic_messages.value)
    if kwargs.get("max_tokens") != unclamped:
        cache.remember_unclamped(kwargs.get("litellm_logging_obj"), unclamped)


def _anthropic_metadata(params: Mapping[str, object]) -> Mapping[str, object]:
    metadata: Final = as_str_mapping(params.get("metadata"))
    if metadata is None:
        return params
    return {**params, "metadata": only(metadata, "user_id")}


def _body_fields(request: Mapping[str, object]) -> Mapping[str, object]:
    params: Final = _anthropic_metadata(
        known_params(_BODY_FIELDS.difference(handler.additional_drop_paths(request)), request)
    )
    if deployment_reasoning_effort_levels(request.get("model_info")) is None:
        return params
    return _with_deployment_effort(params, request)


def body_params(request: Mapping[str, object], asynchronous: bool, hooked: bool = False) -> Mapping[str, object]:
    fields: Final = _body_fields(request)
    if not detectors.cache_control_injection(NativeCall(args=(), kwargs=request, base={})):
        return {**fields, "messages": request["messages"]} if hooked else fields
    messages, system, tools = _prompt_cached(request, asynchronous, hooked)
    return {
        **fields,
        "messages": messages,
        **({"system": system} if "system" in fields else {}),
        **({"tools": tools} if tools is not None and "tools" in fields else {}),
    }


def _with_deployment_effort(params: Mapping[str, object], request: Mapping[str, object]) -> Mapping[str, object]:
    """Python's handler rounds the requested effort to the levels the deployment declares in
    `model_info.reasoning_effort_levels`, the one place a deployment, not the model, decides."""
    explicit_provider: Final = optional_str(request.get("custom_llm_provider"))
    try:
        model, provider, _, _ = litellm.get_llm_provider(
            model=str(request["model"]), custom_llm_provider=explicit_provider
        )
    except litellm.BadRequestError:
        return params
    return _deployment_output_effort(
        params,
        model=model,
        custom_llm_provider=provider,
        kwargs=request,
        drop_params=litellm.drop_params is True or request.get("drop_params") is True,
    )


def cached_function() -> Callable[..., object]:
    return cast(  # cast-ok: the @client cache read names and checks the legacy handler it wraps
        "Callable[..., object]",
        handler.anthropic_messages,  # noqa: TID251  # the @client cache read names and checks the legacy handler it wraps
    )


_ENTRY: Final = cast(  # cast-ok: only the untyped handler's signature is read
    Callable[..., object],
    handler.anthropic_messages,  # noqa: TID251  # reads the handler's signature, never calls it
)
_HANDLER: Final = cast(  # cast-ok: only the untyped handler's signature is read
    Callable[..., object],
    handler.anthropic_messages_handler,  # noqa: TID251  # reads the handler's signature, never calls it
)
_ENTRY_PARAMS: Final = frozenset(inspect.signature(_ENTRY).parameters) - {"kwargs"}
_MAIN_NATIVE_ERROR_PROVIDERS: Final = frozenset({"anthropic"})
_HANDLER_PARAMS: Final = frozenset(inspect.signature(_HANDLER).parameters) - {"kwargs"}
_JsonBlocks: TypeAlias = list[dict[str, JsonValue]]
_Prepared: TypeAlias = tuple[_JsonBlocks, str | _JsonBlocks | None, Sequence[object] | None]
_SecretsRead: TypeAlias = list[tuple[str, str | None]]


def _prompt_cached(request: Mapping[str, object], asynchronous: bool, hooked: bool) -> _Prepared:
    model: Final = str(request["model"])
    provider: Final = optional_str(request.get("custom_llm_provider"))
    api_base: Final = optional_str(request.get("api_base"))
    tools: Final = cast(_JsonBlocks | None, request.get("tools"))  # cast-ok: handed on unchecked, as the handler does
    messages: Final = cast(_JsonBlocks, request["messages"])  # cast-ok: handed on unchecked, as the handler does
    system: Final = cast(  # cast-ok: handed on unchecked, as the handler does
        str | _JsonBlocks | None, request.get("system")
    )
    if not asynchronous:
        return _handler_prepared(
            request,
            messages,
            system,
            model=model,
            provider=provider,
            tools=tools,
            api_base=api_base,
            presanitized=False,
        )
    entry_kwargs: Final = {name: value for name, value in request.items() if name not in _ENTRY_PARAMS}
    if hooked:
        return _handler_prepared(
            entry_kwargs,
            messages,
            system,
            model=model,
            provider=handler.provider_after_pre_request_hooks(model, provider, entry_kwargs),
            tools=tools,
            api_base=api_base,
            presanitized=True,
        )
    entry_messages, entry_system = prepare_native_messages(
        messages, system, entry_kwargs, model=model, custom_llm_provider=provider, tools=tools, api_base=api_base
    )
    return _handler_prepared(
        entry_kwargs,
        entry_messages,
        entry_system,
        model=model,
        provider=handler.provider_after_pre_request_hooks(model, provider, entry_kwargs),
        tools=tools,
        api_base=api_base,
        presanitized=True,
    )


def _handler_prepared(
    call: Mapping[str, object],
    messages: _JsonBlocks,
    system: str | _JsonBlocks | None,
    *,
    model: str,
    provider: str | None,
    tools: _JsonBlocks | None,
    api_base: str | None,
    presanitized: bool,
) -> _Prepared:
    handler_kwargs: Final = {name: value for name, value in call.items() if name not in _HANDLER_PARAMS}
    prepared_messages, prepared_system = prepare_native_messages(
        messages,
        system,
        handler_kwargs,
        model=model,
        custom_llm_provider=provider,
        tools=tools,
        api_base=api_base,
        presanitized=presanitized,
    )
    return prepared_messages, prepared_system, handler.tools_marked_for_cache_points(tools, handler_kwargs)


_execute_pre_request_hooks: Final = cast(  # cast-ok: the handler's untyped pre-request hook loop
    Callable[..., Awaitable[dict[str, object]]],
    handler._execute_pre_request_hooks,  # pyright: ignore[reportPrivateUsage]  # the handler's own pre-request hook loop
)
_HOOKED_PARAMS: Final = (
    "tools",
    "stream",
    "metadata",
    "stop_sequences",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "top_k",
    "top_p",
)


def runs_pre_request_hooks() -> bool:
    return detectors.overridden("async_pre_request_hook", litellm.callbacks) or any(
        isinstance(callback, CustomLogger) and "async_pre_request_hook" in vars(callback)
        for callback in litellm.callbacks
    )


def _unsupported(provider: str | None, model: str, unsupported: str) -> litellm.UnsupportedParamsError:
    return litellm.UnsupportedParamsError(
        message=(
            f"provider {provider} on /v1/messages runs only on the Rust path, which does not support {unsupported}"
        ),
        llm_provider=provider,
        model=model,
    )


async def pre_request_hooks(request: Mapping[str, object]) -> Mapping[str, object]:
    model: Final = str(request["model"])
    provider: Final = optional_str(request.get("custom_llm_provider"))
    tools: Final = cast(_JsonBlocks | None, request.get("tools"))  # cast-ok: handed on unchecked, as the handler does
    kwargs: Final = {name: value for name, value in request.items() if name not in _ENTRY_PARAMS}
    messages, system = prepare_native_messages(
        cast(_JsonBlocks, request["messages"]),  # cast-ok: handed on unchecked, as the handler does
        cast(str | _JsonBlocks | None, request.get("system")),  # cast-ok: handed on unchecked, as the handler does
        kwargs,
        model=model,
        custom_llm_provider=provider,
        tools=tools,
        api_base=optional_str(request.get("api_base")),
    )
    named: Final = {**{name: request.get(name) for name in _ENTRY_PARAMS}, "messages": messages, "system": system}
    hooked: Final = await _execute_pre_request_hooks(
        model=model,
        messages=messages,
        tools=tools,
        stream=cast(bool | None, named["stream"]),  # cast-ok: handed on unchecked, as the handler does
        custom_llm_provider=provider,
        tool_choice=named["tool_choice"],
        **kwargs,
    )
    for name in _HOOKED_PARAMS:
        named[name] = hooked.pop(name, named[name])
    hooked_provider: Final = handler.provider_after_pre_request_hooks(model, provider, hooked)
    hooked.pop("litellm_params", None)
    kwargs.update(hooked)
    requested_provider: Final = handler.provider_after_pre_request_hooks(model, provider, {})
    functools.partial(
        _HANDLER,
        **{name: value for name, value in named.items() if name in _HANDLER_PARAMS - {"custom_llm_provider"}},
        custom_llm_provider=hooked_provider,
        _litellm_messages_presanitized=True,
        **kwargs,
    )
    _python_handler_params({**named, **kwargs}, hooked_provider)
    if hooked_provider != requested_provider:
        raise _unsupported(requested_provider, model, f"a pre-request hook routing the call to {hooked_provider}")
    return {**named, **kwargs}


def _intercepted_provider(request: Mapping[str, object]) -> str | None:
    return handler.provider_after_pre_request_hooks(
        str(request["model"]), optional_str(request.get("custom_llm_provider")), {}
    )


def has_interceptors() -> bool:
    return bool(get_messages_interceptors())


def short_circuits() -> bool:
    from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger

    stock: Final[object] = WebSearchInterceptionLogger.try_short_circuit_search  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the logger annotates bare dicts
    return any(
        isinstance(callback, WebSearchInterceptionLogger) and type(callback).try_short_circuit_search is not stock  # pyright: ignore[reportUnknownMemberType]  # the logger annotates bare dicts
        for callback in litellm.callbacks
    )


def interceptor(request: Mapping[str, object]) -> MessagesInterceptor | None:
    tools: Final = cast(_JsonBlocks | None, request.get("tools"))  # cast-ok: handed on unchecked, as the handler does
    provider: Final = _intercepted_provider(request)
    return next((owner for owner in get_messages_interceptors() if owner.can_handle(tools, provider)), None)


def _entry_messages(
    request: Mapping[str, object],
    kwargs: dict[str, object],  # mutable-ok: the call's own kwargs, rewritten in place by the handler's normalizers
    hooked: bool,
) -> tuple[_JsonBlocks, str | _JsonBlocks | None]:
    messages: Final = cast(_JsonBlocks, request["messages"])  # cast-ok: handed on unchecked, as the handler does
    system: Final = cast(  # cast-ok: handed on unchecked, as the handler does
        str | _JsonBlocks | None, request.get("system")
    )
    if hooked:
        return messages, system
    return prepare_native_messages(
        messages,
        system,
        kwargs,
        model=str(request["model"]),
        custom_llm_provider=optional_str(request.get("custom_llm_provider")),
        tools=cast(_JsonBlocks | None, request.get("tools")),  # cast-ok: handed on unchecked, as the handler does
        api_base=optional_str(request.get("api_base")),
    )


def short_circuit(request: Mapping[str, object], stream: bool, hooked: bool) -> Awaitable[object]:
    kwargs: Final = {name: value for name, value in request.items() if name not in _ENTRY_PARAMS}
    messages, _ = _entry_messages(request, kwargs, hooked)
    return handler._try_websearch_short_circuit(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]  # the handler's own short-circuit, which annotates bare dicts
        model=str(request["model"]),
        messages=messages,
        tools=cast(_JsonBlocks | None, request.get("tools")),  # cast-ok: handed on unchecked, as the handler does
        custom_llm_provider=_intercepted_provider(request),
        stream=stream or bool(kwargs.get(WEBSEARCH_CONVERTED_STREAM_KEY, False)),
        kwargs={**kwargs, "metadata": request.get("metadata")},
    )


def answer(owner: MessagesInterceptor, request: Mapping[str, object], stream: bool, hooked: bool) -> Awaitable[object]:
    model: Final = str(request["model"])
    tools: Final = cast(_JsonBlocks | None, request.get("tools"))  # cast-ok: handed on unchecked, as the handler does
    kwargs: Final = {name: value for name, value in request.items() if name not in _ENTRY_PARAMS}
    messages, system = _entry_messages(request, kwargs, hooked)
    handle: Final = cast(  # cast-ok: the interceptor answers with whatever the caller receives
        Callable[..., Awaitable[object]], owner.handle
    )
    return handle(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream or kwargs.get("_websearch_interception_converted_stream", False),
        max_tokens=cast(int, request["max_tokens"]),  # cast-ok: admission requires an int max_tokens
        custom_llm_provider=_intercepted_provider(request),
        api_key=request.get("api_key"),
        api_base=request.get("api_base"),
        metadata=request.get("metadata"),
        stop_sequences=request.get("stop_sequences"),
        system=system,
        temperature=request.get("temperature"),
        thinking=request.get("thinking"),
        tool_choice=request.get("tool_choice"),
        top_k=request.get("top_k"),
        top_p=request.get("top_p"),
        **kwargs,
    )


def stream_hidden_params(headers: Sequence[tuple[str, str]]) -> Mapping[str, object]:
    from litellm.llms.anthropic.messages.native.streaming_iterator import (
        anthropic_messages_stream_hidden_params,
    )

    return MessagesStreamHead(anthropic_messages_stream_hidden_params(callbacks_legacy_python.wire_headers(headers)))


async def _unrequested_stream_delivered(
    stream: object,
    kwargs: dict[str, object],  # mutable-ok: the call's own kwargs, as `@client` hands them on
) -> object:
    try:
        delivered: Final = await callbacks_legacy_python.after_deployment_success(
            kwargs, stream, CallTypes.anthropic_messages.value
        )
        function: Final = cached_function()
        caching: Final = LLMCachingHandler(
            original_function=function,
            request_kwargs=kwargs,
            start_time=datetime.datetime.now(),  # noqa: DTZ005  # @client's naive start time
        )
        store: Final = cast(  # cast-ok: the handler leaves its original_function untyped
            Callable[..., Awaitable[None]], caching.async_set_cache
        )
        await store(result=delivered, original_function=function, kwargs=kwargs)
    except BaseException:
        from litellm.llms.anthropic.messages.native.streaming_iterator import aclose_if_supported

        await aclose_if_supported(stream)
        raise
    return delivered


def python_stream_types(head: object, kwargs: dict[str, object]) -> None:  # mutable-ok: the call's own kwargs
    """Has the head hand back the stream types Python returns: its streaming response, the cached replay a hit
    returns with the metadata `@client` stamps on it, or the cache writer a miss is wrapped in."""
    logger: Final = kwargs.get("litellm_logging_obj")
    if not isinstance(head, MessagesStreamHead):
        return
    head.python_types = True  # rebind-ok: the head carries it back
    if not isinstance(logger, LiteLLMLoggingObj):
        return
    hit: Final = cache.stream_hit(logger)
    if hit is None:
        head.cache_writer = functools.partial(cache.stream_writer, logger)  # rebind-ok: the head carries it back
        return
    served: Final = datetime.datetime.now(datetime.timezone.utc).astimezone().replace(tzinfo=None)
    callbacks_legacy_python.finalize(hit, logger, kwargs, logger.start_time, served, cache_hit=True)
    head.cached = hit  # rebind-ok: the head carries the cached stream back


def unrequested_stream(
    head: object,
    kwargs: dict[str, object],  # mutable-ok: the call's own kwargs, as `@client` hands them on
) -> None:
    if not isinstance(head, MessagesStreamHead):
        return
    deliver: Final = functools.partial(_unrequested_stream_delivered, kwargs=kwargs)
    head.deliver = deliver  # rebind-ok: the head carries the delivery back


_UPSTREAM_ARGS: Final = TypeAdapter(tuple[int, str])
_UPSTREAM_HEADERS: Final = TypeAdapter(list[tuple[str, str]])
_BODY_CUT: Final = TypeAdapter(tuple[int, int | None, bool])
_ERROR_CLASS: TypeAlias = Callable[..., Exception]
_validate_metadata: Final = cast(  # cast-ok: the handler's untyped metadata validator
    Callable[[object], object], handler.validate_anthropic_api_metadata
)
_mock_response: Final = cast(  # cast-ok: the handler's mock builder takes untyped messages
    Callable[..., Mapping[str, object]], native_utils.mock_response
)
_exception_type: Final = cast(  # cast-ok: the legacy public exception mapper is untyped
    Callable[..., Exception], exception_mapping_utils.exception_type
)
_BODY_READ_TIMEOUT: Final = "Timeout on reading data from socket"  # aiohttp's SocketTimeoutError text
_UNREACHABLE: Final = "could not reach the provider: "
_UNSENDABLE_HEADER: Final = "invalid request: failed to parse header value"
_AIOHTTP_UNSENDABLE_HEADER: Final = "Forbidden control character detected in headers. Potential header injection attack."  # aiohttp's `_safe_header` text
_TOO_MANY_REDIRECTS: Final = "upstream network error: error following redirect: too many redirects"
_SEND_FAILED: Final = "upstream network error: error sending request: client error (SendRequest): "
_DNS_FAILURE: Final = "failed to lookup address information: "
_HTTPX_DNS_ERRNO: Final = {
    "Name or service not known": socket.EAI_NONAME,
    "Temporary failure in name resolution": socket.EAI_AGAIN,
}
_TRANSPORT_TIMEOUTS: Final[Mapping[object, type[httpx.TimeoutException]]] = {
    "Connection": httpx.ConnectTimeout,
    "Read": httpx.ReadTimeout,
}
_MISSING_KEY: Final = (
    "Missing Anthropic API Key - A call is being made to anthropic but no key is set "
    "either in the environment variables or via params. Please set `ANTHROPIC_API_KEY` "
    "or `ANTHROPIC_AUTH_TOKEN` in your environment vars"
)


def _rejection(error: Exception, request: Mapping[str, object], request_provider: str, asynchronous: bool) -> Exception:
    """The error Python's handler raises for a request it refuses before sending it."""
    metadata: Final = request.get("metadata")
    if isinstance(metadata, dict):
        try:
            _validate_metadata(_METADATA.validate_python(metadata))
        except ValidationError as invalid:
            return invalid
    provider_rejection: Final = getattr(error, "messages_provider_rejection", None)
    if isinstance(provider_rejection, str):
        _log_resolved_call(request, request_provider)
        _flag_azure_ai_call(request, request_provider, asynchronous)
        return _public_provider_error(
            _config_error(400, provider_rejection, (), request, request_provider), error, request, request_provider
        )
    if getattr(error, "messages_unsupported_param", False):
        _flag_azure_ai_call(request, request_provider, asynchronous)
        return litellm.UnsupportedParamsError(message=str(error))
    return litellm.BadRequestError(
        message=str(error),
        model=str(request["model"]).removeprefix(f"{request_provider}/"),
        llm_provider=request_provider,
    )


def _timeout_text(phase: str, request: Mapping[str, object]) -> str:
    """`litellm.Timeout`'s text as Python's HTTP client raises it for the same timeout."""
    seconds: Final = request.get("timeout")
    logging_obj: Final = request.get("litellm_logging_obj")
    start_time: Final = getattr(logging_obj, "start_time", None)
    time_taken: Final = (
        round(
            (datetime.datetime.now() - start_time).total_seconds(),  # noqa: DTZ005  # the logger's start_time is naive local time
            3,
        )
        if isinstance(start_time, datetime.datetime)
        else None
    )
    return str(
        litellm.Timeout(
            message=f"{phase} timed out. Timeout passed={seconds}, time taken={time_taken} seconds",
            model="default-model-name",
            llm_provider="litellm-httpx-handler",
        )
    )


def _body_cut(received: int, expected: int | None, reset: bool) -> httpx.TransportError:
    if not http_transport.uses_aiohttp_transport():
        if reset:
            return httpx.ReadError("")
        detail: Final = (
            "incomplete chunked read" if expected is None else f"received {received} bytes, expected {expected}"
        )
        return httpx.RemoteProtocolError(f"peer closed connection without sending complete message body ({detail})")
    payload: Final = (
        TransferEncodingError("Not enough data to satisfy transfer length header.")
        if expected is None
        else ContentLengthError(
            f"Not enough data to satisfy content length header (received {received} of {expected} bytes)."
        )
    )
    text: Final = f"Response payload is not completed: {payload!r}"
    if not reset:
        return httpx.ReadError(text)
    return httpx.ReadError(f"{text}. {ConnectionResetError(errno.ECONNRESET, os.strerror(errno.ECONNRESET))!r}")


def _sent_base(request: Mapping[str, object], request_provider: str, secrets_read: _SecretsRead) -> str | None:
    """The base the call was sent to: azure_ai's falls back to the `AZURE_API_BASE` the host read
    to send it, and anthropic's to its default."""
    api_base: Final = request.get("api_base")
    if request_provider == "anthropic":
        return api_base if isinstance(api_base, str) else anthropic_pass_through_api_base()
    if request_provider == "azure_ai" and not api_base:
        return dict(secrets_read).get("AZURE_API_BASE")
    return api_base if isinstance(api_base, str) else None


def _connect_failure(
    native: str, request: Mapping[str, object], request_provider: str, secrets_read: _SecretsRead
) -> str | None:
    base: Final = _sent_base(request, request_provider, secrets_read)
    if base is None:
        return None
    refused: Final = "tcp connect error: Connection refused" in native
    dns_reason: Final = native.rsplit(_DNS_FAILURE, 1)[1] if _DNS_FAILURE in native else None
    tls_eof: Final = native.endswith("tls handshake eof")
    if not http_transport.uses_aiohttp_transport():
        if refused:
            return "All connection attempts failed"
        dns_errno: Final = None if dns_reason is None else _HTTPX_DNS_ERRNO.get(dns_reason)
        if dns_errno is not None:
            return f"[Errno {dns_errno}] {dns_reason}"
        return "" if tls_eof else None
    url: Final = httpx.URL(base)
    port: Final = url.port or (443 if url.scheme == "https" else 80)
    if refused:
        try:
            address: Final = ipaddress.ip_address(url.host)
        except ValueError:
            return None
        peer: Final = (str(address), port) if address.version == 4 else (str(address), port, 0, 0)
        return _aiohttp_connect_failure(url, port, request, f"Connect call failed {peer}")
    if dns_reason is not None:
        return _aiohttp_connect_failure(url, port, request, dns_reason)
    if tls_eof:
        return _aiohttp_connect_failure(url, port, request, None)
    return None


def _aiohttp_connect_failure(url: httpx.URL, port: int, request: Mapping[str, object], reason: str | None) -> str:
    ssl_verify: Final = request.get("ssl_verify")
    ssl_config: Final = get_ssl_configuration(ssl_verify if isinstance(ssl_verify, bool | str) else None)
    return f"Cannot connect to host {url.host}:{port} ssl:{'default' if ssl_config is True else ssl_config} [{reason}]"


def _transport_failure(
    error: Exception, request: Mapping[str, object], request_provider: str, secrets_read: _SecretsRead
) -> str | None:
    """The text Python's HTTP client raises for a call that failed before any response arrived."""
    body_cut: Final = getattr(error, "messages_body_cut", None)
    if body_cut is not None:
        return str(_body_cut(*_BODY_CUT.validate_python(body_cut)))
    native: Final = str(error)
    if native == _UNSENDABLE_HEADER and request_provider not in _MAIN_NATIVE_ERROR_PROVIDERS:
        # httpx names the header value in its text, which may be a credential.
        return _AIOHTTP_UNSENDABLE_HEADER if http_transport.uses_aiohttp_transport() else native
    if native == _TOO_MANY_REDIRECTS:
        return "Exceeded maximum allowed redirects."
    if native.startswith(_SEND_FAILED):
        reset: Final = native.endswith(f"Connection reset by peer (os error {errno.ECONNRESET})")
        closed: Final = native.endswith("connection closed before message completed")
        if http_transport.uses_aiohttp_transport():
            if reset:
                return str(ConnectionResetError(errno.ECONNRESET, os.strerror(errno.ECONNRESET)))
            return "Server disconnected" if closed else native
        if reset:
            return ""
        return "Server disconnected without sending a response." if closed else native
    if not native.startswith(_UNREACHABLE):
        return None
    connect: Final = _connect_failure(native, request, request_provider, secrets_read)
    return native if connect is None else connect


def _upstream_failure(
    error: Exception, request: Mapping[str, object]
) -> tuple[int, str, Sequence[tuple[str, str]]] | None:
    """The status, text and headers Python's HTTP client reports for the same failure."""
    timeout_phase: Final = getattr(error, "messages_timeout", None)
    if isinstance(timeout_phase, str):
        return 408, _timeout_text(timeout_phase, request), ()
    try:
        status, text = _UPSTREAM_ARGS.validate_python(error.args)
        headers: Final = _UPSTREAM_HEADERS.validate_python(getattr(error, "headers", None))
    except ValidationError:
        return None
    if not headers:
        return status, text, headers
    # Python's `MaskedHTTPStatusError` rebuilds the response from its decoded body without the
    # wire's content-encoding and content-length, so httpx states a length only for a non-empty body.
    rebuilt: Final = httpx.Response(
        status,
        content=text.encode(),
        headers=[
            (name, value) for name, value in headers if name.lower() not in ("content-encoding", "content-length")
        ],
    )
    return status, text, tuple(rebuilt.headers.items())


def _provider_error(
    error: Exception, request: Mapping[str, object], request_provider: str, secrets_read: _SecretsRead
) -> Exception | None:
    """The provider exception Python's `_handle_error` builds for the same failure."""
    transport: Final = _transport_failure(error, request, request_provider, secrets_read)
    failure: Final = _upstream_failure(error, request) if transport is None else (500, transport, ())
    if failure is None:
        return None
    status, text, headers = failure
    provider_error: Final = _config_error(status, text, headers, request, request_provider)
    if transport is not None and isinstance(provider_error, BaseLLMException):
        # Python's `_handle_error` gives a failure without a response the status 500.
        provider_error.status_code_is_synthesized = True
    return provider_error


def _config_error(
    status: int,
    text: str,
    headers: Sequence[tuple[str, str]],
    request: Mapping[str, object],
    request_provider: str,
) -> Exception:
    api_base: Final = request.get("api_base")
    config: Final = native_messages_config(
        str(request["model"]).removeprefix(f"{request_provider}/"),
        request_provider,
        request.get("model_info"),
        api_base if isinstance(api_base, str) else None,
    )
    if config is None:
        return AnthropicError(status_code=status, message=text, headers=httpx.Headers(headers))
    return cast(_ERROR_CLASS, config.get_error_class)(  # cast-ok: the config's untyped headers parameter
        error_message=text, status_code=status, headers=dict(headers)
    )


def _mapped_kwargs(
    request: Mapping[str, object], request_provider: str
) -> dict[str, object]:  # mutable-ok: the legacy mapper takes a dict
    """What the exception mapping reads for the debug info. A provider mapped as Python's handler maps it
    reads the entry's `**kwargs`, so the named `api_base` adds no API Base line. Anthropic, by the owner's
    ruling, keeps main's native mapping of the whole call, which names the `api_base`."""
    if request_provider in _MAIN_NATIVE_ERROR_PROVIDERS:
        return dict(request)
    return {name: value for name, value in request.items() if name not in _ENTRY_PARAMS}


def _public_provider_error(
    provider_error: Exception, error: Exception, request: Mapping[str, object], request_provider: str
) -> Exception:
    try:
        return _exception_type(
            model=str(request["model"]),
            custom_llm_provider=request_provider,
            original_exception=provider_error,
            extra_kwargs=_mapped_kwargs(request, request_provider),
        )
    except Exception as mapped:  # noqa: BLE001  # exception_type raises the mapped public error
        mapped.__context__ = error
        return mapped


def _log_resolved_call(request: Mapping[str, object], request_provider: str) -> None:
    """Tells the logger the resolved call, as Python's handler does before its provider config
    transforms the request."""
    logger: Final = request.get("litellm_logging_obj")
    if not isinstance(logger, LiteLLMLoggingObj):
        return
    callbacks_legacy_python.update_logging(
        logger,
        kwargs={name: value for name, value in request.items() if name != "metadata"},
        model=str(request["model"]).removeprefix(f"{request_provider}/"),
        optional_params=dict(_body_fields(request)),
        litellm_params={"model_info": request.get("model_info"), **_body_fields(request)},
        custom_llm_provider=request_provider,
    )


def _with_deployment_debug_info(error: Exception, request: Mapping[str, object], request_provider: str) -> Exception:
    logger: Final = request.get("litellm_logging_obj")
    if not isinstance(logger, LiteLLMLoggingObj):
        return error
    details: Final = logger.model_call_details
    provider: Final = details.get("custom_llm_provider")
    return with_deployment_debug_info(
        error,
        logger.model,
        provider if isinstance(provider, str) and provider else request_provider,
        details.get("litellm_params"),
    )


def _raised(error: _RaisedT) -> _RaisedT:
    """`error` raised through a Python frame, so its failure log carries a traceback as the one Python raises does."""
    try:
        raise error
    except Exception:  # noqa: BLE001  # catches only the exception it just raised
        return error


def _raised_like_python(error: _RaisedT, request_provider: str) -> _RaisedT:
    """`error` raised through a Python frame, as Python's handler raises it. Anthropic, by the owner's ruling,
    keeps main's native route, which hands the exception back unraised, without a traceback."""
    return error if request_provider in _MAIN_NATIVE_ERROR_PROVIDERS else _raised(error)


def map_failure(
    error: Exception,
    request: Mapping[str, object],
    request_provider: str,
    asynchronous: bool = False,
    secrets_read: _SecretsRead | None = None,
) -> Exception:
    """Raises what Python's Messages handler raises for the same failure. `secrets_read` are the
    secrets the host read to send the call, such as azure_ai's `AZURE_API_BASE`."""
    if getattr(error, "messages_timeout", None) == "Body":
        # Python reads a stream's body outside its HTTP client's error mapping, so the transport's
        # own timeout reaches the caller unmapped.
        return _with_deployment_debug_info(
            httpx.ReadTimeout(_BODY_READ_TIMEOUT if http_transport.uses_aiohttp_transport() else ""),
            request,
            request_provider,
        )
    body_cut: Final = getattr(error, "messages_body_cut", None)
    if body_cut is not None and request.get("stream") is True:
        # Python reads a stream's body outside its HTTP client's error mapping, so a body the
        # connection cut short reaches the caller as its transport raised it.
        return _with_deployment_debug_info(_body_cut(*_BODY_CUT.validate_python(body_cut)), request, request_provider)
    transport_timeout: Final = _TRANSPORT_TIMEOUTS.get(getattr(error, "messages_timeout", None))
    if transport_timeout is not None:
        # Python's `litellm.Timeout` is raised while handling httpx's own timeout, which the proxy's
        # failure spend reads to bill a read timeout's prompt.
        chained: Final = transport_timeout(str(error))
        error.__context__ = chained  # rebind-ok: Python chains the transport timeout as the exception context
    if getattr(error, "messages_late_rejection", False):
        _log_resolved_call(request, request_provider)
    if getattr(error, "messages_python_raised", False):
        # Python's request transform raises this builtin exception unmapped, e.g. the TypeError
        # of comparing a string max_tokens while capping the thinking budget.
        _flag_messages_call(request.get("litellm_logging_obj"), asynchronous)
        return _raised_like_python(error, request_provider)
    if getattr(error, "messages_request_error", False):
        return _rejection(error, request, request_provider, asynchronous)
    if getattr(error, "messages_missing_credential", False):
        return _raised_like_python(
            litellm.AuthenticationError(
                message=_MISSING_KEY,
                llm_provider=request_provider,
                model=str(request["model"]).removeprefix(f"{request_provider}/"),
            ),
            request_provider,
        )
    provider_error: Final = _provider_error(error, request, request_provider, secrets_read or [])
    if provider_error is None:
        return failures.map_native_failure(
            error,
            str(request["model"]),
            request_provider,
            _mapped_kwargs(request, request_provider),
            optional_str(request.get("api_base")),
        )
    return _public_provider_error(provider_error, error, request, request_provider)


def _drop_params(kwargs: Mapping[str, object]) -> bool:
    return bool(litellm.drop_params) or normalize_drop_params(kwargs.get("drop_params")) is True


def _python_handler_params(request: Mapping[str, object], custom_llm_provider: object) -> GenericLiteLLMParams:
    return GenericLiteLLMParams(
        **{name: value for name, value in request.items() if name not in _HANDLER_PARAMS},
        api_key=request.get("api_key"),
        api_base=request.get("api_base"),
        custom_llm_provider=custom_llm_provider,
    )


def _resolved_provider(request: Mapping[str, object]) -> str | None:
    return optional_str(request.get("custom_llm_provider")) or inferred_provider(
        str(request["model"]), optional_str(request.get("api_base"))
    )


def failure_provider(request: Mapping[str, object]) -> str:
    return _resolved_provider(request) or "anthropic"


def _scope_provider_specific_header(request: Mapping[str, object]) -> None:
    scoped: Final = request.get("provider_specific_header")
    if scoped is None:
        return
    ProviderSpecificHeaderUtils.get_provider_specific_headers(
        provider_specific_header=cast(  # cast-ok: the handler hands the caller's value on unvalidated
            ProviderSpecificHeader | None, scoped
        ),
        custom_llm_provider=_resolved_provider(request),
        native_anthropic_messages=True,
    )


def _python_messages_url(api_base: str | None) -> str:
    """`AnthropicMessagesConfig.get_complete_url`, untrimmed, whose `urlsplit` raises a bare
    ValueError for a base it cannot split (``Invalid IPv6 URL``)."""
    return ensure_api_route(AnthropicModelInfo.get_api_base(api_base) or "https://api.anthropic.com", "v1/messages")


def _raise_as_python_does(request: Mapping[str, object], asynchronous: bool) -> None:
    iter(cast(Iterable[object], request.get("messages")))  # cast-ok: iterated unchecked, as Python's handler does
    tools: Final = request.get("tools")
    if tools is not None and not isinstance(tools, list | tuple) and has_interceptors():
        interceptor(request)
    _python_handler_params(request, request.get("custom_llm_provider"))
    # Python's handler scopes the caller's provider_specific_header before it builds any header, so
    # a value of the wrong shape raises its own error (a str: "'str' object has no attribute 'get'"),
    # and then builds the URL, unmapped.
    logger: Final = request.get("litellm_logging_obj")
    try:
        _scope_provider_specific_header(request)
    except Exception:
        _flag_messages_call(logger, asynchronous)
        raise
    api_base: Final = request.get("api_base")
    if not isinstance(api_base, str) or _resolved_provider(request) != "anthropic":
        return
    try:
        _python_messages_url(api_base)
    except ValueError:
        _log_resolved_call(request, "anthropic")
        _flag_messages_call(logger, asynchronous)
        raise


def settings(
    kwargs: Mapping[str, object], request: Mapping[str, object] | None = None, asynchronous: bool = False
) -> dict[str, object]:
    if request is not None:
        _raise_as_python_does(request, asynchronous)
    aiohttp: Final = http_transport.uses_aiohttp_transport()
    configured: Final = MessagesSettings(
        drop_params=_drop_params(kwargs),
        reasoning_auto_summary=is_reasoning_auto_summary_enabled(),
        additional_drop_params=handler.additional_drop_paths(kwargs),
        request_timeout=get_configured_request_timeout(),
        drops_unsupported_params=drops_unsupported_params(kwargs.get("drop_params")),
        ascii_response_headers=aiohttp,
        joined_reads=aiohttp,
    )
    return asdict(
        configured if request is None else _with_call_settings(configured, request, kwargs.get("litellm_logging_obj"))
    )


def _flag_messages_call(logger: object, asynchronous: bool) -> None:
    if not isinstance(logger, LiteLLMLoggingObj):
        return
    logger.model_call_details.setdefault("litellm_params", {})[CallTypes.aanthropic_messages.value] = asynchronous


def _flag_azure_ai_call(request: Mapping[str, object], request_provider: str, asynchronous: bool) -> None:
    """Python's handler flags the call before it gets the credential or refuses a param. By the owner's ruling
    anthropic keeps main's unflagged failure."""
    if request_provider == "azure_ai":
        _flag_messages_call(request.get("litellm_logging_obj"), asynchronous)


@dataclass(frozen=True, slots=True)
class _Answered:
    value: object

    def send(self, _value: object) -> execution.Complete:
        return execution.Complete(self.value)

    def throw(self, error: BaseException) -> execution.Step:
        raise error

    def close(self) -> None:
        return None


def sync_answer(request: Mapping[str, object]) -> _Answered | None:
    logger: Final = request.get("litellm_logging_obj")
    mock: Final = _mock(request, logger, asynchronous=False)
    if mock is not None:
        _flag_messages_call(logger, asynchronous=False)
        return _Answered(response(mock))
    served: Final = gateway(request, asynchronous=False)
    return None if served is None else _Answered(served)


def _handler_params(request: Mapping[str, object]) -> GenericLiteLLMParams:
    return GenericLiteLLMParams(
        **{name: value for name, value in request.items() if name not in _HANDLER_PARAMS},
        api_key=optional_str(request.get("api_key")),
        api_base=optional_str(request.get("api_base")),
        custom_llm_provider=optional_str(request.get("custom_llm_provider")),
    )


def _requested_model(request: Mapping[str, object]) -> str:
    model, _, _, _ = _requested_provider(
        str(request["model"]), optional_str(request.get("custom_llm_provider")), _handler_params(request)
    )
    return model


def _model_without_latest_alias(request: Mapping[str, object], model: str, custom_llm_provider: str | None) -> str:
    if not is_model_latest_alias(model, custom_llm_provider):
        return model
    resolved, provider, _, _ = _requested_provider(model, custom_llm_provider, _handler_params(request))
    prefix, separator, _ = model.partition("/")
    return f"{prefix}/{resolved}" if separator and prefix == provider else resolved


def _mock(request: Mapping[str, object], logger: object, asynchronous: bool) -> Mapping[str, object] | None:
    mock: Final = request.get("mock_response")
    if not mock or not isinstance(mock, str):
        return None
    _validate_metadata(request.get("metadata"))
    messages: Final = request.get("messages")
    max_tokens: Final = request.get("max_tokens")
    model: Final = _requested_model(request)
    try:
        return dict(
            _mock_response(
                model=model,
                messages=messages if isinstance(messages, list) else [],
                max_tokens=max_tokens if isinstance(max_tokens, int) else 0,
                mock_response=mock,
            )
        )
    except Exception:
        _flag_messages_call(logger, asynchronous)
        raise


def _stream_cost_model(request: Mapping[str, object]) -> str | None:
    if request.get("stream") is not True or not stream_cost_injection_requested(proxy_request_headers(request)):
        return None
    return _requested_model(request)


def _affinity_extra_headers(
    request: Mapping[str, object], model: str, explicit_provider: str | None
) -> Mapping[str, object] | None:
    api_base: Final = optional_str(request.get("api_base"))
    anthropic_header: Final = get_provider_affinity_header(AnthropicModelInfo.get_api_base(api_base))
    other_header: Final = get_provider_affinity_header(api_base)
    if anthropic_header is None and other_header is None:
        return None
    provider: Final = explicit_provider or inferred_provider(model, api_base)
    default_header: Final = anthropic_header if provider == "anthropic" else other_header
    if default_header is None:
        return None
    headers: Final = as_str_mapping(request.get("headers"))
    extra_headers: Final = as_str_mapping(request.get("extra_headers"))
    merged: Final = _merged_headers(
        headers,
        extra_headers,
        ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header=cast(  # cast-ok: the handler hands the caller's value on unvalidated
                ProviderSpecificHeader | None, request.get("provider_specific_header")
            ),
            custom_llm_provider=provider,
            native_anthropic_messages=True,
        ),
    )
    added: Final = add_provider_affinity_header(merged, _handler_params(request), default_header)
    affinity: Final = {name: value for name, value in added.items() if name not in merged}
    if not affinity:
        return None
    return dict(_merged_headers(headers, extra_headers, affinity))


def _with_call_settings(
    configured: MessagesSettings, request: Mapping[str, object], logger: object
) -> MessagesSettings:
    explicit_provider: Final = optional_str(request.get("custom_llm_provider"))
    model: Final = _model_without_latest_alias(request, str(request["model"]), explicit_provider)
    mock: Final = _mock(request, logger, asynchronous=True)
    if mock is not None:
        return replace(configured, model=model, mock_response=mock)
    return replace(
        configured,
        model=model,
        stream_cost_model=_stream_cost_model(request),
        extra_headers=_affinity_extra_headers(request, model, explicit_provider),
        mcp_gateway=_takes_mcp_gateway(request),
    )


def _takes_mcp_gateway(request: Mapping[str, object]) -> bool:
    tools: Final = optional_sequence(request.get("tools"))
    return not request.get("_skip_mcp_handler", False) and bool(tools) and may_name_litellm_mcp_gateway(tools)


MessagesResult: TypeAlias = AnthropicMessagesResponse | Iterator[bytes] | AsyncIterator[object]


def gateway(
    request: Mapping[str, object], asynchronous: bool, hooked: bool = False
) -> Coroutine[object, object, MessagesResult] | None:
    if not _takes_mcp_gateway(request):
        return None
    model: Final = str(request["model"])
    provider: Final = optional_str(request.get("custom_llm_provider"))
    api_base: Final = optional_str(request.get("api_base"))
    tools: Final = cast(_JsonBlocks, request.get("tools"))  # cast-ok: handed on unchecked, as the handler does
    messages: Final = cast(_JsonBlocks, request["messages"])  # cast-ok: handed on unchecked, as the handler does
    system: Final = cast(  # cast-ok: handed on unchecked, as the handler does
        str | _JsonBlocks | None, request.get("system")
    )
    if not asynchronous:
        return _gateway(request, messages, system, model=model, provider=provider, tools=tools, presanitized=False)
    entry_kwargs: Final = {name: value for name, value in request.items() if name not in _ENTRY_PARAMS}
    entry_messages, entry_system = (
        (messages, system)
        if hooked
        else prepare_native_messages(
            messages, system, entry_kwargs, model=model, custom_llm_provider=provider, tools=tools, api_base=api_base
        )
    )
    return _gateway(
        {**request, **entry_kwargs, "is_async": True},
        entry_messages,
        entry_system,
        model=model,
        provider=handler.provider_after_pre_request_hooks(model, provider, entry_kwargs),
        tools=tools,
        presanitized=True,
    )


def _gateway(
    call: Mapping[str, object],
    messages: _JsonBlocks,
    system: str | _JsonBlocks | None,
    *,
    model: str,
    provider: str | None,
    tools: _JsonBlocks,
    presanitized: bool,
) -> Coroutine[object, object, MessagesResult]:
    kwargs: Final = {name: value for name, value in call.items() if name not in _HANDLER_PARAMS}
    api_key: Final = optional_str(call.get("api_key"))
    api_base: Final = optional_str(call.get("api_base"))
    prepared_messages, prepared_system = prepare_native_messages(
        messages,
        system,
        kwargs,
        model=model,
        custom_llm_provider=provider,
        tools=tools,
        api_base=api_base,
        presanitized=presanitized,
    )
    metadata: Final = _validate_metadata(call.get("metadata"))
    is_async: Final = kwargs.pop("is_async", False)
    logging_obj: Final = kwargs.get("litellm_logging_obj")
    litellm_params: Final = GenericLiteLLMParams(
        **kwargs, api_key=api_key, api_base=api_base, custom_llm_provider=provider
    )
    resolved_model, resolved_provider, dynamic_api_key, dynamic_api_base = _requested_provider(
        model, provider, litellm_params
    )
    if isinstance(logging_obj, LiteLLMLoggingObj):
        logging_obj.model_call_details["agentic_loop_params"] = {
            "model": model,
            "custom_llm_provider": resolved_provider,
        }
        logging_obj.model_call_details.setdefault("litellm_params", {})[CallTypes.aanthropic_messages.value] = is_async
        if kwargs.get("_websearch_interception_converted_stream", False):
            logging_obj.model_call_details["websearch_interception_converted_stream"] = True
    bridge: Final = _bridged(
        _BridgeTarget(model=model, api_key=api_key, api_base=api_base),
        _BridgeTarget(
            model=resolved_model,
            api_key=None if _own_entra_credential(litellm_params) else dynamic_api_key,
            api_base=dynamic_api_base,
        ),
        resolved_provider,
    )
    kwargs.pop("_skip_mcp_handler", False)
    from litellm.llms.anthropic.messages.native.mcp_handler import anthropic_messages_with_mcp

    return anthropic_messages_with_mcp(
        max_tokens=cast(int, call["max_tokens"]),  # cast-ok: handed on unchecked, as the handler does
        messages=prepared_messages,
        model=resolved_model,
        metadata=metadata,
        stop_sequences=call.get("stop_sequences"),
        stream=call.get("stream", False),
        system=prepared_system,
        temperature=call.get("temperature"),
        thinking=call.get("thinking"),
        tool_choice=call.get("tool_choice"),
        tools=tools,
        top_k=call.get("top_k"),
        top_p=call.get("top_p"),
        container=call.get("container"),
        api_key=bridge.api_key,
        api_base=bridge.api_base,
        client=call.get("client"),
        custom_llm_provider=resolved_provider,
        **kwargs,
    )


def inject_stream_cost(frames: bytes, model: str, request: Mapping[str, object]) -> object:
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    logger: Final = request.get("litellm_logging_obj")
    logging_obj: Final = logger if isinstance(logger, LiteLLMLoggingObj) else None
    return ProxyBaseLLMRequestProcessing.process_chunk_with_cost_injection(
        frames,
        model,
        logging_obj,
        request_headers=proxy_request_headers(
            as_str_mapping(logging_obj.model_call_details.get("litellm_params")) if logging_obj else None
        ),
    )


def azure_ad_token(arguments: Mapping[str, object]) -> str:
    from litellm.llms.azure.common_utils import get_azure_ad_token

    return get_azure_ad_token(_handler_params(arguments)) or ""


def map_token_failure(
    error: Exception, request: Mapping[str, object], request_provider: str, asynchronous: bool = False
) -> Exception:
    """Python's handler maps a litellm exception raised while it gets the call's credential, and lets any other
    exception through as raised."""
    _flag_azure_ai_call(request, request_provider, asynchronous)
    if not isinstance(error, BaseLLMException):
        return error
    return _public_provider_error(error, error, request, request_provider)


def azure_ad_token_failure(error: Exception) -> Exception:
    """What `get_azure_ad_token` logs and raises when its token provider fails with `error`."""
    from litellm.llms.azure.common_utils import azure_ad_token_failure as python_failure

    verbose_logger.error("Error calling Azure AD token provider: %s", error)
    failure: Final = python_failure(error)
    failure.__cause__ = error
    return _raised(failure)


def non_string_azure_ad_token(token: object) -> TypeError:
    """What `get_azure_ad_token` logs and raises when its token provider returns `token`, not a string."""
    verbose_logger.error("Azure AD token provider returned non-string value: %s", type(token))
    return _raised(TypeError(f"Azure AD token must be a string, got {type(token)}"))


def entra_token_failure(reason: str) -> Exception:
    """What `get_azure_ad_token` raises when azure-identity's `ClientSecretCredential` fails for `reason`."""
    from azure.core.exceptions import ClientAuthenticationError

    return azure_ad_token_failure(ClientAuthenticationError(message=f"Authentication failed: {reason}"))


def requests_proxy_unreachable(host: str, port: int, code: int) -> str:
    """What `requests` reports when it cannot open a connection to its proxy."""
    from urllib3.connection import HTTPSConnection
    from urllib3.exceptions import NewConnectionError, ProxyError

    refused: Final = NewConnectionError(
        HTTPSConnection(host, port), f"Failed to establish a new connection: [Errno {code}] {os.strerror(code)}"
    )
    return str(ProxyError(_PROXY_UNREACHABLE, refused))


def requests_proxy_unresolved(host: str, port: int, code: int, lookup: str) -> str:
    """What `requests` reports when the name of its proxy does not resolve."""
    from urllib3.connection import HTTPSConnection
    from urllib3.exceptions import NameResolutionError, ProxyError

    unresolved: Final = NameResolutionError(host, HTTPSConnection(host, port), socket.gaierror(code, lookup))
    return str(ProxyError(_PROXY_UNREACHABLE, unresolved))


def requests_proxy_authentication_required() -> str:
    """What `requests` reports when its proxy answers the tunnel with 407."""
    from urllib3.exceptions import ProxyError

    status: Final = http.HTTPStatus.PROXY_AUTHENTICATION_REQUIRED
    return str(ProxyError(_PROXY_UNREACHABLE, OSError(f"Tunnel connection failed: {status.value} {status.phrase}")))


def requests_proxy_disconnected() -> str:
    """What `requests` reports when the tunnel through its proxy closes before the token response."""
    from urllib3.exceptions import ProxyError

    return str(ProxyError(_PROXY_UNREACHABLE, http.client.RemoteDisconnected(_REMOTE_CLOSED)))


def requests_connection_aborted(code: int | None) -> str:
    """What `requests` reports when the connection is reset (errno `code`) or closed (None) before an answer."""
    from urllib3.exceptions import ProtocolError

    cause: Final = http.client.RemoteDisconnected(_REMOTE_CLOSED) if code is None else OSError(code, os.strerror(code))
    return str(ProtocolError("Connection aborted.", cause))


def requests_tls_eof() -> str:
    """What `requests` reports when the connection ends inside the TLS handshake, as this build's `ssl` words it."""
    incoming: Final = ssl.MemoryBIO()
    handshake: Final = ssl.create_default_context().wrap_bio(incoming, ssl.MemoryBIO(), server_hostname="eof.invalid")
    with contextlib.suppress(ssl.SSLWantReadError):
        handshake.do_handshake()
    incoming.write_eof()
    try:
        handshake.do_handshake()
    except ssl.SSLError as eof:
        return str(eof)
    raise AssertionError("a handshake over a closed connection completed")


def requests_certificate_rejected(reason: str) -> str:
    """What `requests` reports when OpenSSL rejects the server certificate for `reason`."""
    site: Final = requests_tls_eof().rsplit(" ", 1)[-1]
    return f"[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: {reason} {site}"


def requests_read_timeout(host: str, port: int) -> str:
    """What `requests` reports when azure-identity's default read timeout passes with no answer."""
    from azure.core.configuration import ConnectionConfiguration
    from urllib3 import HTTPSConnectionPool
    from urllib3.exceptions import ReadTimeoutError

    seconds: Final = ConnectionConfiguration().read_timeout
    return str(ReadTimeoutError(HTTPSConnectionPool(host, port), None, f"Read timed out. (read timeout={seconds})"))


_PROXY_UNREACHABLE: Final = "Unable to connect to proxy"
_REMOTE_CLOSED: Final = "Remote end closed connection without response"


def entra_token_response_failure(status: int, headers: Sequence[tuple[str, str]], body: bytes) -> Exception | None:
    """What `get_azure_ad_token` raises when the token endpoint gave a native client secret flow this
    final answer, read by azure-identity's own `ClientSecretCredential` replaying it."""
    from azure.core.exceptions import ClientAuthenticationError

    from litellm.rust_bridge.host.routes.entra_replay import replayed_credential

    with replayed_credential(status, tuple(headers), body) as credential:
        try:
            credential.get_token(_ENTRA_SCOPE)
        except ClientAuthenticationError as error:
            return azure_ad_token_failure(error)
    return None


_ENTRA_SCOPE: Final = "https://cognitiveservices.azure.com/.default"


def workload_identity_token(params: Mapping[str, object], api_base: str | None) -> str | None:
    litellm_params: Final = GenericLiteLLMParams(**params)
    return get_anthropic_wif_token(dict(litellm_params), api_base, "")


class RawTransportFailure(Exception):
    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


@dataclass(frozen=True, slots=True)
class Sent:
    status: int
    headers: Sequence[tuple[str, str]]
    text: str | None = None
    chunks: AsyncIterator[bytes] | None = None
    rejection: httpx.HTTPStatusError | None = None


def _unsendable_api_base(api_base: object) -> bool:
    """A caller api_base the native URL parser rejects or reads otherwise than httpx and aiohttp
    (no http(s) scheme, whitespace, control or non-ASCII characters, credentials, no host or a port
    Python cannot read): Python's client decides what such a URL does, from a relative-URL error to
    a request on the host it finds."""
    if not isinstance(api_base, str) or not api_base:
        return False
    return not (
        api_base.isascii()
        and api_base.isprintable()
        and " " not in api_base
        and api_base.lower().startswith(("http://", "https://"))
        and _plain_authority(api_base)
    )


def _plain_authority(api_base: str) -> bool:
    try:
        parts: Final = urlsplit(api_base)
    except ValueError:
        return True
    return bool(parts.hostname) and parts.username is None and _readable_port(parts)


def _readable_port(parts: SplitResult) -> bool:
    try:
        return parts.port is None or parts.port >= 0
    except ValueError:
        return False


def _sent_api_base(request: Mapping[str, object]) -> tuple[object, _SecretsRead]:
    """The base Python's URL builder reads, and the secrets read to find it: azure_ai falls back to
    `AZURE_API_BASE`, read once, which the native route then reuses."""
    api_base: Final = request.get("api_base")
    if api_base or "model" not in request or _resolved_provider(request) != "azure_ai":
        return api_base, []
    environment: Final = get_secret_str("AZURE_API_BASE")
    return environment, [("AZURE_API_BASE", environment)]


def transport_handler(request: Mapping[str, object]) -> tuple[AsyncHTTPHandler | None, _SecretsRead]:
    """The client the call sends through (None for the native one), and the secrets read to choose
    it, which the native route reuses rather than reading them again."""
    client: Final = request.get("client")
    if isinstance(client, AsyncHTTPHandler):
        return client, []
    if http_transport.sends_through_python(client, request.get("timeout")):
        return get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC), []
    api_base, read = _sent_api_base(request)
    if not _unsendable_api_base(api_base):
        return None, read
    return get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC), read


def decoding_failure(message: str, request: Mapping[str, object], request_provider: str) -> Exception:
    error: Final = httpx.DecodingError(message)
    if request.get("stream") is True:
        # Python's stream iterator attaches the deployment's debug info. Anthropic, by the owner's ruling,
        # keeps main's native route, which raises the decoding error bare.
        if request_provider in _MAIN_NATIVE_ERROR_PROVIDERS:
            return error
        return _with_deployment_debug_info(error, request, request_provider)
    return map_transport_failure(error, request, request_provider)


def unencodable_header(value: bytes, received: bool) -> Exception:
    text: Final = value.decode("utf-8", "surrogateescape" if received else "surrogatepass")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as unencodable:
        return unencodable
    return ValueError(text)


def _sent_url(request: Mapping[str, object], url: str) -> str:
    """The native URL builder trims the caller's api_base; Python sends it as given."""
    api_base: Final = request.get("api_base")
    if (
        not isinstance(api_base, str)
        or api_base == api_base.strip()
        or not _unsendable_api_base(api_base)
        or _resolved_provider(request) != "anthropic"
    ):
        return url
    try:
        return _python_messages_url(api_base) if url == _python_messages_url(api_base.strip()) else url
    except ValueError:  # a base urlsplit cannot split keeps the URL the native builder made
        return url


async def transport_send(
    handler: AsyncHTTPHandler,
    request: Mapping[str, object],
    url: str,
    headers: Sequence[tuple[str, str]],
    body: bytes,
    stream: bool,
) -> Sent:
    try:
        response: Final = await handler.post(  # pyright: ignore[reportUnknownMemberType]  # the handler's untyped dict params
            url=_sent_url(request, url),
            headers=dict(headers),
            data=body,
            stream=stream,
            logging_obj=cast(  # cast-ok: the call's logger, handed on as Python's handler hands it
                "LiteLLMLoggingObj | None", request.get("litellm_logging_obj")
            ),
            timeout=cast(  # cast-ok: the call's timeout as the handler resolved it for the post
                "float | httpx.Timeout | None", request.get("timeout")
            ),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as rejection:
        try:
            text: Final = rejection.response.text
        except Exception as unreadable:  # noqa: BLE001  # Python's retry check raises it from its except block, unmapped
            raise RawTransportFailure(unreadable) from None
        return Sent(rejection.response.status_code, rejection.response.headers.multi_items(), text, None, rejection)
    if stream:
        return Sent(response.status_code, response.headers.multi_items(), chunks=response.aiter_bytes())
    return Sent(response.status_code, response.headers.multi_items(), text=response.text)


def map_transport_failure(error: Exception, request: Mapping[str, object], request_provider: str) -> Exception:
    api_base: Final = request.get("api_base")
    config: Final = native_messages_config(
        str(request["model"]).removeprefix(f"{request_provider}/"),
        request_provider,
        request.get("model_info"),
        api_base if isinstance(api_base, str) else None,
    )
    try:
        handler.base_llm_http_handler._handle_error(e=error, provider_config=config)  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # the handler's own error mapping, which always raises
    except Exception as provider_error:  # noqa: BLE001  # _handle_error raises the provider exception it builds
        provider_error.__context__ = error
        if not isinstance(provider_error, BaseLLMException):
            return provider_error
        try:
            return _exception_type(
                model=str(request["model"]),
                custom_llm_provider=request_provider,
                original_exception=provider_error,
                extra_kwargs=_mapped_kwargs(request, request_provider),
            )
        except Exception as mapped:  # noqa: BLE001  # exception_type raises the mapped public error
            mapped.__context__ = provider_error
            return mapped
    return error


_AGENTIC_UNSENT_PARAMS: Final = frozenset({"model", "messages", "max_tokens"})
_HANDLER_POPPED: Final = frozenset(
    {"is_async", "_litellm_messages_presanitized", "_skip_mcp_handler", "cache_control_injection_points"}
)


def _logger(request: Mapping[str, object]) -> LiteLLMLoggingObj | None:
    logger: Final = request.get("litellm_logging_obj")
    return logger if isinstance(logger, LiteLLMLoggingObj) else None


def runs_agentic_hooks(request: Mapping[str, object]) -> bool:
    logger: Final = _logger(request)
    if logger is None:
        return False
    return BaseLLMHTTPHandler._has_agentic_completion_hook(logger) or bool(  # pyright: ignore[reportPrivateUsage]  # the handler's own agentic gate
        request.get(WEBSEARCH_CONVERTED_STREAM_KEY, False)
    )


def record_agentic_loop(request: Mapping[str, object], custom_llm_provider: str) -> None:
    logger: Final = _logger(request)
    if logger is None:
        return
    logger.model_call_details["agentic_loop_params"] = {
        "model": request["model"],
        "custom_llm_provider": custom_llm_provider,
    }
    if request.get(WEBSEARCH_CONVERTED_STREAM_KEY, False):
        logger.model_call_details["websearch_interception_converted_stream"] = True


@dataclass(frozen=True, slots=True)
class _AgenticCall:
    model: str
    messages: list[dict[str, object]]  # mutable-ok: the handler's agentic hooks take a message list
    config: BaseAnthropicMessagesConfig
    params: dict[str, object]  # mutable-ok: the handler's agentic hooks take a params dict
    logger: LiteLLMLoggingObj
    provider: str
    kwargs: dict[str, object]  # mutable-ok: the handler's agentic hooks take a kwargs dict


def _agentic_params(
    sent: Mapping[str, object], requested: Mapping[str, object], request: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: the handler's agentic hooks take a params dict
    params: Final = {name: value for name, value in sent.items() if name not in _AGENTIC_UNSENT_PARAMS}
    if "tools" not in params:
        return params
    caller_tools: Final = request.get("tools")
    requested_tools: Final = requested.get("tools")
    return {**params, "tools": caller_tools if caller_tools == requested_tools else requested_tools}


def _agentic_call(
    request: Mapping[str, object],
    sent: Mapping[str, object],
    model: str,
    custom_llm_provider: str,
    requested: Mapping[str, object],
    hooked: bool,
) -> _AgenticCall:
    logger: Final = _logger(request)
    api_base: Final = optional_str(request.get("api_base"))
    config: Final = native_messages_config(model, custom_llm_provider, request.get("model_info"), api_base)
    if logger is None or config is None:
        raise ValueError("an agentic Messages call needs its logger and provider config")
    kwargs: Final = {
        name: value for name, value in request.items() if name not in _HANDLER_PARAMS and name not in _HANDLER_POPPED
    }
    messages, _, _ = _prompt_cached(request, asynchronous=True, hooked=hooked)
    return _AgenticCall(
        model=model,
        messages=cast(  # cast-ok: the handler hands its prepared messages to the hooks as dicts
            list[dict[str, object]], strip_provider_specific_fields_from_anthropic_messages(messages)
        ),
        config=config,
        params=_agentic_params(sent, requested, request),
        logger=logger,
        provider=custom_llm_provider,
        kwargs=BaseLLMHTTPHandler._agentic_hook_kwargs(  # pyright: ignore[reportPrivateUsage]  # the handler's own follow-up kwargs
            kwargs=kwargs, api_key=optional_str(request.get("api_key")), api_base=api_base
        ),
    )


async def _failed_deployment(request: Mapping[str, object], error: Exception) -> None:
    from litellm.utils import async_post_call_failure_deployment_hook

    try:
        await async_post_call_failure_deployment_hook(
            request_data=request, exception=error, call_type=CallTypes.anthropic_messages.value
        )
    except asyncio.CancelledError:
        raise
    except BaseException:  # noqa: BLE001, S110  # a failing hook never replaces the agentic loop's own failure, as in `@client`
        pass


async def agentic_response(
    response: object,
    request: Mapping[str, object],
    sent: Mapping[str, object],
    model: str,
    custom_llm_provider: str,
    requested: Mapping[str, object],
    hooked: bool,
) -> object:
    try:
        call: Final = _agentic_call(request, sent, model, custom_llm_provider, requested, hooked)
        initial: Final = cast(AnthropicMessagesResponse, response)  # cast-ok: the public response the handler finishes
        return await handler.base_llm_http_handler._finalize_anthropic_messages_response(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]  # the handler's own agentic finish, typed with bare dicts
            initial_response=initial,
            model=call.model,
            messages=call.messages,
            anthropic_messages_provider_config=call.config,
            anthropic_messages_optional_request_params=call.params,
            logging_obj=call.logger,
            custom_llm_provider=call.provider,
            kwargs=call.kwargs,
        )
    except Exception as error:
        await _failed_deployment(request, error)
        raise


def _agentic_stream(stream: AsyncStream, call: _AgenticCall, held: frozenset[str]) -> object:
    from litellm.llms.anthropic.messages.native.agentic_streaming_iterator import AgenticAnthropicStreamingIterator
    from litellm.llms.anthropic.messages.native.streaming_iterator import (
        AnthropicMessagesStreamHiddenParams,
        AnthropicMessagesStreamingResponse,
    )

    return AnthropicMessagesStreamingResponse(
        completion_stream=AgenticAnthropicStreamingIterator(
            completion_stream=stream,
            http_handler=handler.base_llm_http_handler,
            model=call.model,
            messages=call.messages,
            anthropic_messages_provider_config=call.config,
            anthropic_messages_optional_request_params=call.params,
            logging_obj=call.logger,
            custom_llm_provider=call.provider,
            kwargs=call.kwargs,
            hold_back=bool(held),
            server_fulfilled_tool_names=held,
        ),
        hidden_params=cast(  # cast-ok: the stream's stamped hidden params, shared as Python's handler shares them
            AnthropicMessagesStreamHiddenParams,
            stream._hidden_params,  # pyright: ignore[reportPrivateUsage]  # the native stream's own hidden params
        ),
    )


def agentic_stream(
    head: object,
    request: Mapping[str, object],
    sent: Mapping[str, object],
    model: str,
    custom_llm_provider: str,
    requested: Mapping[str, object],
    hooked: bool,
) -> None:
    logger: Final = _logger(request)
    if (
        not isinstance(head, MessagesStreamHead)
        or logger is None
        or not BaseLLMHTTPHandler._has_agentic_completion_hook(logger)  # pyright: ignore[reportPrivateUsage]  # the handler's own agentic gate
    ):
        return
    call: Final = _agentic_call(request, sent, model, custom_llm_provider, requested, hooked)
    held: Final = BaseLLMHTTPHandler._server_fulfilled_tools_in_request(  # pyright: ignore[reportPrivateUsage]  # the handler's own hold-back gate
        logging_obj=logger, tools=call.params.get("tools")
    )
    wrap: Final = functools.partial(_agentic_stream, call=call, held=held)
    head.agentic = wrap  # rebind-ok: the head carries the stream wrapper back


class MessagesSafeguards:
    def __init__(self, request: Mapping[str, object]) -> None:
        self._request: Final = request
        self._frames: _StampedFrames | None = None
        self._results: object = None

    async def _evaluator(self) -> SafeguardsEvaluator | None:
        model, provider = _handler_model(self._request)
        if not await handler.proxy_classifies_tool_uses(model, provider, self._request):
            return None
        messages: Final = normalize_native_messages(optional_sequence(self._request.get("messages")) or ())
        return await asyncio.to_thread(
            handler._native_safeguards_evaluator,  # pyright: ignore[reportPrivateUsage]  # the handler's own evaluator build
            self._request,
            messages,
        )

    async def results(self, value: Mapping[str, object]) -> object:
        evaluator: Final = await self._evaluator()
        if evaluator is None:
            return None
        unstamped: Final = response(value)
        stamped: Final = await with_safeguard_results(unstamped, evaluator)
        if stamped is unstamped:
            return None
        self._results = stamped.get("safeguard_results")
        return self._results

    def published(self, public: Mapping[str, object]) -> Mapping[str, object]:
        return public if self._results is None else {**public, "safeguard_results": self._results}

    async def open_stream(self) -> bool:
        evaluator: Final = await self._evaluator()
        if evaluator is None:
            return False
        self._frames = _StampedFrames(StreamedSafeguardResults(evaluator))
        return True

    async def frames(self, chunk: bytes) -> Sequence[bytes]:
        return await self._open_frames().feed(chunk)

    async def flush(self) -> Sequence[bytes]:
        return await self._open_frames().flush()

    def _open_frames(self) -> _StampedFrames:
        if self._frames is None:
            raise RuntimeError("safeguards stream frames before the stream opened")
        return self._frames


class _StampedFrames:
    def __init__(self, safeguard_results: StreamedSafeguardResults) -> None:
        self._safeguard_results: Final = safeguard_results
        self._open_frame: Final = bytearray()

    async def feed(self, chunk: bytes) -> Sequence[bytes]:
        from litellm.llms.anthropic.messages.native.streaming_iterator import (
            _SSE_FRAME_END,  # pyright: ignore[reportPrivateUsage]  # the handler's own SSE framing
            _SSE_TERMINATOR_OVERLAP,  # pyright: ignore[reportPrivateUsage]  # the handler's own SSE framing
            _ends_a_frame,  # pyright: ignore[reportPrivateUsage]  # the handler's own SSE framing
            _stamped_sse_frame,  # pyright: ignore[reportPrivateUsage]  # the handler's own SSE framing
        )

        scan_from: Final = max(len(self._open_frame) - _SSE_TERMINATOR_OVERLAP, 0)
        self._open_frame.extend(chunk)
        if not _ends_a_frame(self._open_frame, scan_from):
            return ()
        *frames, unterminated = _SSE_FRAME_END.split(self._open_frame)
        self._open_frame[:] = unterminated
        return tuple([bytes(await _stamped_sse_frame(frame, self._safeguard_results)) for frame in frames])

    async def flush(self) -> Sequence[bytes]:
        if not self._open_frame:
            return ()
        from litellm.llms.anthropic.messages.native.streaming_iterator import (
            _stamped_sse_frame,  # pyright: ignore[reportPrivateUsage]  # the handler's own SSE framing
        )

        return (bytes(await _stamped_sse_frame(bytes(self._open_frame), self._safeguard_results)),)


def _handler_model(request: Mapping[str, object]) -> tuple[str, str | None]:
    explicit_provider: Final = optional_str(request.get("custom_llm_provider"))
    try:
        model, provider, _, _ = litellm.get_llm_provider(
            model=str(request["model"]), custom_llm_provider=explicit_provider
        )
    except litellm.BadRequestError:
        return str(request["model"]), explicit_provider
    return model, provider


def safeguards(request: Mapping[str, object]) -> MessagesSafeguards | None:
    if "safeguards" not in request or not native_route_classifies(str(request["model"]), request["safeguards"]):
        return None
    return MessagesSafeguards(request)
