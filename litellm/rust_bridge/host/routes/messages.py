from __future__ import annotations

import datetime
import errno
import ipaddress
import os
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final, TypeAlias, cast  # noqa: TID251  # narrows native payloads to public TypedDicts

import httpx
from aiohttp.http_exceptions import ContentLengthError, TransferEncodingError
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils import exception_mapping_utils
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.reasoning_effort_capability import deployment_reasoning_effort_levels
from litellm.litellm_core_utils.request_timeout_resolver import get_configured_request_timeout
from litellm.llms.anthropic.common_utils import AnthropicError, anthropic_pass_through_api_base
from litellm.llms.anthropic.messages.native import handler
from litellm.llms.anthropic.messages.native.handler import (
    _native_messages_config as native_messages_config,  # pyright: ignore[reportPrivateUsage]  # the handler's own config lookup
)
from litellm.llms.anthropic.messages.native.handler import (
    _with_deployment_output_effort as _deployment_output_effort,  # pyright: ignore[reportPrivateUsage]  # the handler's own effort rounding
)
from litellm.llms.anthropic.messages.utils import is_reasoning_auto_summary_enabled
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import get_ssl_configuration
from litellm.rust_bridge import failures
from litellm.rust_bridge.host import callbacks_legacy_python, http_transport
from litellm.rust_bridge.public_call import known_params, optional_str
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

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
    }
)


@dataclass(frozen=True, slots=True)
class MessagesSettings:
    drop_params: bool
    reasoning_auto_summary: bool
    additional_drop_params: Sequence[str]
    request_timeout: float | None


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


def body_params(request: Mapping[str, object]) -> Mapping[str, object]:
    params: Final = known_params(_BODY_FIELDS.difference(handler.additional_drop_paths(request)), request)
    if deployment_reasoning_effort_levels(request.get("model_info")) is None:
        return params
    return _with_deployment_effort(params, request)


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


def stream_hidden_params(headers: Sequence[tuple[str, str]]) -> Mapping[str, object]:
    from litellm.llms.anthropic.messages.native.streaming_iterator import (
        anthropic_messages_stream_hidden_params,
    )

    return anthropic_messages_stream_hidden_params(httpx.Headers(list(headers)))


_UPSTREAM_ARGS: Final = TypeAdapter(tuple[int, str])
_UPSTREAM_HEADERS: Final = TypeAdapter(list[tuple[str, str]])
_BODY_CUT: Final = TypeAdapter(tuple[int, int | None, bool])
_ERROR_CLASS: TypeAlias = Callable[..., Exception]
_validate_metadata: Final = cast(  # cast-ok: the handler's untyped metadata validator
    Callable[[dict[str, object]], object], handler.validate_anthropic_api_metadata
)
_exception_type: Final = cast(  # cast-ok: the legacy public exception mapper is untyped
    Callable[..., Exception], exception_mapping_utils.exception_type
)
_BODY_READ_TIMEOUT: Final = "Timeout on reading data from socket"  # aiohttp's SocketTimeoutError text
_UNREACHABLE: Final = "could not reach the provider: "
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


def _rejection(error: Exception, request: Mapping[str, object], request_provider: str) -> Exception:
    """The error Python's handler raises for a request it refuses before sending it."""
    metadata: Final = request.get("metadata")
    if isinstance(metadata, dict):
        try:
            _validate_metadata(_METADATA.validate_python(metadata))
        except ValidationError as invalid:
            return invalid
    if getattr(error, "messages_unsupported_param", False):
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


def _connect_failure(native: str, request: Mapping[str, object], request_provider: str) -> str | None:
    api_base: Final = request.get("api_base")
    if not isinstance(api_base, str) and request_provider != "anthropic":
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
    url: Final = httpx.URL(api_base if isinstance(api_base, str) else anthropic_pass_through_api_base())
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


def _transport_failure(error: Exception, request: Mapping[str, object], request_provider: str) -> str | None:
    """The text Python's HTTP client raises for a call that failed before any response arrived."""
    native: Final = str(error)
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
    connect: Final = _connect_failure(native, request, request_provider)
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


def _provider_error(error: Exception, request: Mapping[str, object], request_provider: str) -> Exception | None:
    """The provider exception Python's `_handle_error` builds for the same failure."""
    transport: Final = _transport_failure(error, request, request_provider)
    failure: Final = _upstream_failure(error, request) if transport is None else (500, transport, ())
    if failure is None:
        return None
    status, text, headers = failure
    api_base: Final = request.get("api_base")
    config: Final = native_messages_config(
        str(request["model"]).removeprefix(f"{request_provider}/"),
        request_provider,
        request.get("model_info"),
        api_base if isinstance(api_base, str) else None,
    )
    provider_error: Final = (
        cast(_ERROR_CLASS, config.get_error_class)(  # cast-ok: the config's untyped headers parameter
            error_message=text, status_code=status, headers=dict(headers)
        )
        if config is not None
        else AnthropicError(status_code=status, message=text, headers=httpx.Headers(headers))
    )
    if transport is not None and isinstance(provider_error, BaseLLMException):
        # Python's `_handle_error` gives a failure without a response the status 500.
        provider_error.status_code_is_synthesized = True
    return provider_error


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
        optional_params=dict(body_params(request)),
        litellm_params={"model_info": request.get("model_info"), **body_params(request)},
        custom_llm_provider=request_provider,
    )


def map_failure(error: Exception, request: Mapping[str, object], request_provider: str) -> Exception:
    """Raises what Python's Messages handler raises for the same failure."""
    if getattr(error, "messages_timeout", None) == "Body":
        # Python reads a stream's body outside its HTTP client's error mapping, so the transport's
        # own timeout reaches the caller unmapped.
        return httpx.ReadTimeout(_BODY_READ_TIMEOUT)
    body_cut: Final = getattr(error, "messages_body_cut", None)
    if body_cut is not None:
        # Python reads a stream's body outside its HTTP client's error mapping, so a body the
        # connection cut short reaches the caller as its transport raised it.
        return _body_cut(*_BODY_CUT.validate_python(body_cut))
    transport_timeout: Final = _TRANSPORT_TIMEOUTS.get(getattr(error, "messages_timeout", None))
    if transport_timeout is not None:
        # Python's `litellm.Timeout` is raised while handling httpx's own timeout, which the proxy's
        # failure spend reads to bill a read timeout's prompt.
        chained: Final = transport_timeout(str(error))
        error.__context__ = chained  # rebind-ok: Python chains the transport timeout as the exception context
    if getattr(error, "messages_late_rejection", False):
        _log_resolved_call(request, request_provider)
    if getattr(error, "messages_request_error", False):
        return _rejection(error, request, request_provider)
    if getattr(error, "messages_missing_credential", False):
        return litellm.AuthenticationError(
            message=_MISSING_KEY,
            llm_provider=request_provider,
            model=str(request["model"]).removeprefix(f"{request_provider}/"),
        )
    provider_error: Final = _provider_error(error, request, request_provider)
    if provider_error is None:
        return failures.map_native_failure(
            error, str(request["model"]), request_provider, request, optional_str(request.get("api_base"))
        )
    try:
        return _exception_type(
            model=str(request["model"]),
            custom_llm_provider=request_provider,
            original_exception=provider_error,
            extra_kwargs=dict(request),
        )
    except Exception as mapped:  # noqa: BLE001  # exception_type raises the mapped public error
        mapped.__context__ = error
        return mapped


def _drop_params(kwargs: Mapping[str, object]) -> bool:
    return bool(litellm.drop_params) or normalize_drop_params(kwargs.get("drop_params")) is True


def settings(kwargs: Mapping[str, object]) -> dict[str, object]:
    return asdict(
        MessagesSettings(
            drop_params=_drop_params(kwargs),
            reasoning_auto_summary=is_reasoning_auto_summary_enabled(),
            additional_drop_params=handler.additional_drop_paths(kwargs),
            request_timeout=get_configured_request_timeout(),
        )
    )
