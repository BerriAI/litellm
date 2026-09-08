"""Thin Python wrapper for the native Rust chat completions bridge.

The Rust core owns the conversation translation, the provider call, and the
response normalization for the subset of `/chat/completions` requests it
accepts. This module only marshals inputs and hands the normalized result to
LiteLLM's existing `ModelResponse` builder.

``None`` means the provider was never called, so the caller is free to serve the
request on the Python path. A failure after the call was issued raises instead:
retrying it there would bill the customer for the same work twice.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    cast,  # noqa: TID251  # native callables require runtime signature narrowing
)

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.exceptions import APIError
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.bedrock.request_metadata import bedrock_request_metadata_is_owned
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.loader import get_native_bridge
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

# Providers whose `/chat/completions` deployments the Rust core can serve. A
# provider outside this set never reaches the bridge.
RUST_CHAT_COMPLETIONS_PROVIDERS: Final = frozenset({"anthropic", "bedrock"})

# `litellm_params` values are `object`, so validate the one this module reads
# rather than narrowing an unparameterized `Mapping` and typing the result Any.
_LITELLM_METADATA_ADAPTER: Final = TypeAdapter(Mapping[str, object])

RUST_RESPONSE_HEADER: Final = "x-litellm-rust"


class RustChatCompletions(Protocol):
    def __call__(self, arguments: dict[str, object]) -> ModelResponse:
        raise NotImplementedError


class RustAchatCompletions(Protocol):
    def __call__(self, arguments: dict[str, object]) -> Awaitable[ModelResponse]:
        raise NotImplementedError


class LegacyRustChatCompletions(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
    ) -> Mapping[str, object]: ...


class LegacyRustAchatCompletions(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
    ) -> Awaitable[Mapping[str, object]]: ...


class RustChatCompletionsDecline(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        custom_llm_provider: str | None,
    ) -> str | None:
        raise NotImplementedError


class ResponseObserver(Protocol):
    def __call__(self, rust_response: Mapping[str, object], /) -> None:
        raise NotImplementedError


def response_logger(
    *,
    logging_obj: LiteLLMLoggingObj,
    messages: Sequence[object],
    api_key: str,
    additional_args: Mapping[str, object],
) -> ResponseObserver:
    def log(rust_response: Mapping[str, object], /) -> None:
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=json.dumps(rust_response),
            additional_args=additional_args,
        )

    return log


def _uses_argument_bag(call: object) -> bool:
    try:
        parameters: Final = inspect.signature(call).parameters.values()
    except (TypeError, ValueError):
        return True
    return not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass(slots=True)
class _RustChatCompletionsState:
    chat_completions: RustChatCompletions | LegacyRustChatCompletions | None = None
    achat_completions: RustAchatCompletions | LegacyRustAchatCompletions | None = None
    decline: RustChatCompletionsDecline | None = None


_STATE: Final[_RustChatCompletionsState] = _RustChatCompletionsState()


def set_rust_chat_completions(
    *,
    chat_completions: RustChatCompletions | LegacyRustChatCompletions | None | _Unset = _UNSET,
    achat_completions: RustAchatCompletions | LegacyRustAchatCompletions | None | _Unset = _UNSET,
    decline: RustChatCompletionsDecline | None | _Unset = _UNSET,
) -> None:
    """Inject the native callables, so tests can supply a double instead of
    patching module attributes."""
    if not isinstance(chat_completions, _Unset):
        _STATE.chat_completions = chat_completions
    if not isinstance(achat_completions, _Unset):
        _STATE.achat_completions = achat_completions
    if not isinstance(decline, _Unset):
        _STATE.decline = decline


def load_rust_chat_completions() -> RustChatCompletions | LegacyRustChatCompletions | None:
    if _STATE.chat_completions is not None:
        return _STATE.chat_completions
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    loaded: RustChatCompletions | None = getattr(native_bridge, "chat_completions", None)
    return loaded


def load_rust_achat_completions() -> RustAchatCompletions | LegacyRustAchatCompletions | None:
    if _STATE.achat_completions is not None:
        return _STATE.achat_completions
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    loaded: RustAchatCompletions | None = getattr(native_bridge, "achat_completions", None)
    return loaded


def _load_rust_decline() -> RustChatCompletionsDecline | None:
    if _STATE.decline is not None:
        return _STATE.decline
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    loaded: RustChatCompletionsDecline | None = getattr(native_bridge, "chat_completions_decline", None)
    return loaded


def _anthropic_user_id_reaches_the_body(litellm_params: Mapping[str, object] | None) -> bool:
    metadata: Final = litellm_params.get("metadata") if litellm_params is not None else None
    try:
        entries: Final = _LITELLM_METADATA_ADAPTER.validate_python(metadata)
    except ValidationError:
        return False
    return entries.get("user_id") is not None


def _litellm_metadata_reaches_the_provider(
    custom_llm_provider: str | None, litellm_params: Mapping[str, object] | None
) -> bool:
    """Whether the Python transform would promote proxy-owned attribution into the
    provider request, below this gate and inside the function the Rust route replaces.

    `AnthropicConfig.transform_request` promotes a valid `metadata["user_id"]`
    into the Messages body, so the core never sees the key and would send the
    request to Anthropic with the abuse-detection attribution missing.

    `AmazonConverseConfig` resolves proxy-owned `requestMetadata` onto the
    Converse body whenever the operator armed `bedrock_request_metadata_fields`.
    Owning that field also means evicting a caller-supplied one, which the core
    cannot do either, so ownership alone is the condition rather than whether
    anything resolved.

    Deliberately a superset of Python's condition in both cases: declining a
    request Python would not have attributed anyway costs only the Rust path,
    while missing one loses the attribution silently.
    """
    match custom_llm_provider:
        case "anthropic":
            return _anthropic_user_id_reaches_the_body(litellm_params)
        case "bedrock":
            return bedrock_request_metadata_is_owned()
        case _:
            return False


def rust_chat_completions_accepts(
    *,
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object],
    custom_llm_provider: str | None,
    litellm_params: Mapping[str, object] | None,
    stream: object,
) -> bool:
    """Whether the Rust path will serve this request.

    Asked before the caller commits to either path, so pre-call logging is
    emitted exactly once, on whichever path actually runs. The core's own
    capability gate answers the second half; it resolves no credentials and
    performs no I/O.
    """
    if custom_llm_provider not in RUST_CHAT_COMPLETIONS_PROVIDERS:
        return False
    if stream:
        return False
    if not rust_enabled():
        return False
    if _litellm_metadata_reaches_the_provider(custom_llm_provider, litellm_params):
        verbose_logger.debug("Rust chat completions declined (litellm metadata user_id); using the Python path")
        return False
    decline: Final = _load_rust_decline()
    if decline is None:
        return False
    try:
        reason: Final = decline(
            model=model,
            messages=messages,
            optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
        )
    except Exception as rust_error:  # noqa: BLE001  # rollout-safety fallback: any Rust bridge failure must fall back to the Python path
        verbose_logger.debug(
            "Rust chat completions gate raised %s; staying on the Python path",
            type(rust_error).__name__,
        )
        return False
    if reason is not None:
        verbose_logger.debug("Rust chat completions declined (%s); using the Python path", reason)
        return False
    return True


def _rust_bridge_exceptions() -> tuple[type[BaseException], type[BaseException]] | None:
    """`(declined, upstream_failed)` from the native module, or None when absent."""
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    declined: Final = getattr(native_bridge, "RustBridgeDeclined", None)
    upstream: Final = getattr(native_bridge, "RustUpstreamError", None)
    if declined is None or upstream is None:
        return None
    return declined, upstream


def _reraise_or_decline(
    rust_error: BaseException,
    *,
    model: str,
    custom_llm_provider: str | None,
) -> None:
    """Re-raise a failure the provider already saw, or return so the caller declines.

    A request that never reached the provider is safe to serve on the Python
    path. One that did is not: the provider has already done the work, so a
    second attempt bills for it twice. Those surface as an `APIError` carrying
    the upstream status, which LiteLLM's exception mapping already understands.
    """
    exceptions: Final = _rust_bridge_exceptions()
    if exceptions is None:
        verbose_logger.debug(
            "Rust chat completions bridge raised %s; falling back to Python path",
            type(rust_error).__name__,
        )
        return
    declined, upstream_failed = exceptions
    if isinstance(rust_error, upstream_failed):
        args: Final = rust_error.args
        status: Final = args[0] if args else 0
        message: Final = args[1] if len(args) > 1 else ""
        raise APIError(
            status_code=int(status) or 500,
            message=f"litellm rust chat completions: {message}",
            llm_provider=custom_llm_provider or "",
            model=model,
        )
    if not isinstance(rust_error, declined):
        raise rust_error
    verbose_logger.debug(
        "Rust chat completions declined before calling the provider (%s); using the Python path",
        rust_error,
    )


def build_model_response(
    rust_response: Mapping[str, object],
    model_response: ModelResponse,
) -> ModelResponse:
    built: Final = convert_to_model_response_object(
        response_object=dict(rust_response),  # mutable-ok: the converter takes a real dict and rewrites it
        model_response_object=model_response,
        hidden_params={"additional_headers": {RUST_RESPONSE_HEADER: "true"}},  # mutable-ok: rewritten by the converter
    )
    if not isinstance(built, ModelResponse):
        raise TypeError(f"expected a ModelResponse from the rust path, got {type(built).__name__}")
    return built


def chat_completions(
    *,
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object],
    model_response: ModelResponse,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    arguments: dict[str, object] | None = None,
    logging_api_key: str | None = None,
    on_response: ResponseObserver | None = None,
) -> ModelResponse | None:
    rust_chat_completions: Final = load_rust_chat_completions()
    if rust_chat_completions is None:
        return None
    try:
        if _STATE.chat_completions is not None and _uses_argument_bag(rust_chat_completions):
            argument_bag_call: Final = cast(  # cast-ok: signature inspection selected the argument-bag callable
                RustChatCompletions, rust_chat_completions
            )
            rust_result: Final = argument_bag_call(
                _arguments(
                    arguments,
                    model,
                    messages,
                    optional_params,
                    model_response,
                    api_key,
                    api_base,
                    custom_llm_provider,
                    extra_headers,
                    timeout,
                    logging_api_key,
                )
            )
            return rust_result
        if _STATE.chat_completions is not None:
            legacy: Final = cast(  # cast-ok: signature inspection selected the legacy injected callable
                LegacyRustChatCompletions, rust_chat_completions
            )
            rust_response: Final = legacy(
                model=model,
                messages=messages,
                optional_params=optional_params,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
            )
            if on_response is not None:
                on_response(rust_response)
            return build_model_response(rust_response, model_response)
        native_call: Final = cast(RustChatCompletions, rust_chat_completions)  # cast-ok: native ABI uses argument bag
        return native_call(
            _arguments(
                arguments,
                model,
                messages,
                optional_params,
                model_response,
                api_key,
                api_base,
                custom_llm_provider,
                extra_headers,
                timeout,
                logging_api_key,
            )
        )
    except Exception as rust_error:  # noqa: BLE001  # rollout safety: the helper re-raises anything the provider already saw
        _reraise_or_decline(rust_error, model=model, custom_llm_provider=custom_llm_provider)
        return None
    raise AssertionError("unreachable")


async def achat_completions(
    *,
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object],
    model_response: ModelResponse,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    arguments: dict[str, object] | None = None,
    logging_api_key: str | None = None,
    on_response: ResponseObserver | None = None,
) -> ModelResponse | None:
    rust_achat_completions: Final = load_rust_achat_completions()
    if rust_achat_completions is None:
        return None
    try:
        if _STATE.achat_completions is not None and _uses_argument_bag(rust_achat_completions):
            argument_bag_call: Final = cast(  # cast-ok: signature inspection selected the argument-bag callable
                RustAchatCompletions, rust_achat_completions
            )
            rust_result: Final = await argument_bag_call(
                _arguments(
                    arguments,
                    model,
                    messages,
                    optional_params,
                    model_response,
                    api_key,
                    api_base,
                    custom_llm_provider,
                    extra_headers,
                    timeout,
                    logging_api_key,
                )
            )
            return rust_result
        if _STATE.achat_completions is not None:
            legacy: Final = cast(  # cast-ok: signature inspection selected the legacy injected callable
                LegacyRustAchatCompletions, rust_achat_completions
            )
            rust_response: Final = await legacy(
                model=model,
                messages=messages,
                optional_params=optional_params,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
            )
            if on_response is not None:
                on_response(rust_response)
            return build_model_response(rust_response, model_response)
        native_call: Final = cast(  # cast-ok: native ABI uses argument bag
            RustAchatCompletions, rust_achat_completions
        )
        return await native_call(
            _arguments(
                arguments,
                model,
                messages,
                optional_params,
                model_response,
                api_key,
                api_base,
                custom_llm_provider,
                extra_headers,
                timeout,
                logging_api_key,
            )
        )
    except Exception as rust_error:  # noqa: BLE001  # rollout safety: the helper re-raises anything the provider already saw
        _reraise_or_decline(rust_error, model=model, custom_llm_provider=custom_llm_provider)
        return None
    raise AssertionError("unreachable")


async def achat_completions_or_fallback(
    *,
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object],
    model_response: ModelResponse,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    python_fallback: Callable[[], Awaitable[object]],
    arguments: dict[str, object] | None = None,
    logging_api_key: str | None = None,
    on_response: ResponseObserver | None = None,
) -> object:
    """Await the Rust path, falling back to the caller's own Python path when
    the bridge is unavailable or the call fails.

    The caller supplies the fallback, so the bridge stays free of provider
    dispatch. This exists because a caller that dispatches asynchronously has
    already returned a coroutine by the time a Rust failure surfaces, and so
    cannot fall back on its own.
    """
    response: Final = await achat_completions(
        model=model,
        messages=messages,
        optional_params=optional_params,
        model_response=model_response,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        timeout=timeout,
        arguments=arguments,
        logging_api_key=logging_api_key,
        on_response=on_response,
    )
    if response is not None:
        return response
    return await python_fallback()


def initialize_logging(arguments: dict[str, object], asynchronous: bool) -> object:
    from litellm.rust_bridge._lifecycle import initialize_logging as initialize_lifecycle_logging

    return initialize_lifecycle_logging(arguments, asynchronous, "completion")


def _arguments(
    arguments: dict[str, object] | None,
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object],
    model_response: ModelResponse,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    logging_api_key: str | None,
) -> dict[str, object]:
    return {
        **(arguments or {}),
        "model": model,
        "messages": messages,
        "optional_params": optional_params,
        "model_response": model_response,
        "api_key": api_key,
        "api_base": api_base,
        "custom_llm_provider": custom_llm_provider,
        "extra_headers": extra_headers,
        "timeout_seconds": timeout_to_seconds(timeout),
        "logging_api_key": logging_api_key if logging_api_key is not None else api_key or "",
    }
