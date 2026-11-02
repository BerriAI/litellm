from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final, TypeAlias, cast  # noqa: TID251  # narrows the normalized native payload to the public TypedDict

import httpx
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils import exception_mapping_utils
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.reasoning_effort_capability import deployment_reasoning_effort_levels
from litellm.litellm_core_utils.request_timeout_resolver import get_configured_request_timeout
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.messages.native import handler
from litellm.llms.anthropic.messages.native.handler import (
    _native_messages_config as native_messages_config,  # pyright: ignore[reportPrivateUsage]  # the handler's own config lookup
)
from litellm.llms.anthropic.messages.native.handler import (
    _with_deployment_output_effort as _deployment_output_effort,  # pyright: ignore[reportPrivateUsage]  # the handler's own effort rounding
)
from litellm.llms.anthropic.messages.utils import is_reasoning_auto_summary_enabled
from litellm.rust_bridge import failures
from litellm.rust_bridge.host import callbacks_legacy_python
from litellm.rust_bridge.public_call import known_params, optional_str
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

_DROP_PATHS: Final = TypeAdapter(list[object])
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
    params: Final = known_params(_BODY_FIELDS, request)
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
_ERROR_CLASS: TypeAlias = Callable[..., Exception]
_validate_metadata: Final = cast(  # cast-ok: the handler's untyped metadata validator
    Callable[[dict[str, object]], object], handler.validate_anthropic_api_metadata
)
_exception_type: Final = cast(  # cast-ok: the legacy public exception mapper is untyped
    Callable[..., Exception], exception_mapping_utils.exception_type
)
_BODY_READ_TIMEOUT: Final = "Timeout on reading data from socket"  # aiohttp's SocketTimeoutError text
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
        round((datetime.datetime.now() - start_time).total_seconds(), 3)
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


def _provider_error(error: Exception, request: Mapping[str, object], request_provider: str) -> Exception | None:
    """The provider exception Python's `_handle_error` builds for the same failure."""
    timeout_phase: Final = getattr(error, "messages_timeout", None)
    if isinstance(timeout_phase, str):
        status, text, headers = 408, _timeout_text(timeout_phase, request), []
    else:
        try:
            status, text = _UPSTREAM_ARGS.validate_python(error.args)
            headers = _UPSTREAM_HEADERS.validate_python(getattr(error, "headers", None))
        except ValidationError:
            return None
    api_base: Final = request.get("api_base")
    config: Final = native_messages_config(
        str(request["model"]).removeprefix(f"{request_provider}/"),
        request_provider,
        request.get("model_info"),
        api_base if isinstance(api_base, str) else None,
    )
    return (
        cast(_ERROR_CLASS, config.get_error_class)(  # cast-ok: the config's untyped headers parameter
            error_message=text, status_code=status, headers=dict(headers)
        )
        if config is not None
        else AnthropicError(status_code=status, message=text, headers=httpx.Headers(headers))
    )


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


def _additional_drop_params(kwargs: Mapping[str, object]) -> tuple[str, ...]:
    try:
        configured: Final = _DROP_PATHS.validate_python(kwargs.get("additional_drop_params"))
    except ValidationError:
        return ()
    return tuple(path for path in configured if isinstance(path, str))


def settings(kwargs: Mapping[str, object]) -> dict[str, object]:
    return asdict(
        MessagesSettings(
            drop_params=_drop_params(kwargs),
            reasoning_auto_summary=is_reasoning_auto_summary_enabled(),
            additional_drop_params=_additional_drop_params(kwargs),
            request_timeout=get_configured_request_timeout(),
        )
    )
