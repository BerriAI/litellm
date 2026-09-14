"""Native chat completions bindings.

The Rust core owns the conversation translation, the provider call, and the
response normalization for the subset of `/chat/completions` requests it
accepts. This module only marshals inputs and hands the normalized result to
LiteLLM's existing `ModelResponse` builder.

``None`` means the provider was never called, so the caller is free to serve the
request on the Python path. A failure after the call was issued raises instead:
retrying it there would bill the customer for the same work twice.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final, cast  # noqa: TID251  # native callables are validated at load time

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.exceptions import APIError
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.bedrock.request_metadata import bedrock_request_metadata_is_owned
from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset, native_exception_types
from litellm.rust_bridge.chat_completions.types import (
    ResponseObserver,
    RustAchatCompletions,
    RustChatCompletions,
    RustChatCompletionsDecline,
)
from litellm.rust_bridge.configuration import RouteName
from litellm.rust_bridge.route import NativeRoute
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.utils import ModelResponse

# Providers whose `/chat/completions` deployments the Rust core can serve. A
# provider outside this set never reaches the bridge.
RUST_CHAT_COMPLETIONS_PROVIDERS: Final = frozenset({"anthropic", "bedrock"})

# `litellm_params` values are `object`, so validate the one this module reads
# rather than narrowing an unparameterized `Mapping` and typing the result Any.
_LITELLM_METADATA_ADAPTER: Final = TypeAdapter(Mapping[str, object])

RUST_RESPONSE_HEADER: Final = "x-litellm-rust"


ROUTE: Final = NativeRoute(RouteName.CHAT_COMPLETIONS)


def _as_chat(value: object) -> RustChatCompletions | None:
    return cast(RustChatCompletions, value) if callable(value) else None


def _as_achat(value: object) -> RustAchatCompletions | None:
    return cast(RustAchatCompletions, value) if callable(value) else None


def _as_decline(value: object) -> RustChatCompletionsDecline | None:
    return cast(RustChatCompletionsDecline, value) if callable(value) else None


_CHAT: Final = ROUTE.bind("chat_completions", validate=_as_chat)
_ACHAT: Final = ROUTE.bind("achat_completions", validate=_as_achat)
_DECLINE: Final = ROUTE.bind("chat_completions_decline", validate=_as_decline)


def set_rust_chat_completions(
    *,
    chat_completions: RustChatCompletions | None | BindingUnset = BINDING_UNSET,
    achat_completions: RustAchatCompletions | None | BindingUnset = BINDING_UNSET,
    decline: RustChatCompletionsDecline | None | BindingUnset = BINDING_UNSET,
) -> None:
    """Inject the native callables, so tests can supply a double instead of
    patching module attributes."""
    _CHAT.configure(chat_completions)
    _ACHAT.configure(achat_completions)
    _DECLINE.configure(decline)


def load_rust_chat_completions() -> RustChatCompletions | None:
    return ROUTE.select(_CHAT)


def load_rust_achat_completions() -> RustAchatCompletions | None:
    return ROUTE.select(_ACHAT)


def _load_rust_decline() -> RustChatCompletionsDecline | None:
    return ROUTE.select(_DECLINE)


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
    if not ROUTE.enabled():
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
    return native_exception_types()


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


def _build_model_response(
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
    on_response: ResponseObserver,
) -> ModelResponse | None:
    rust_chat_completions: Final = load_rust_chat_completions()
    if rust_chat_completions is None:
        return None
    try:
        rust_response: Final = rust_chat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    except Exception as rust_error:  # noqa: BLE001  # rollout safety: the helper re-raises anything the provider already saw
        _reraise_or_decline(rust_error, model=model, custom_llm_provider=custom_llm_provider)
        return None
    on_response(rust_response)
    return _build_model_response(rust_response, model_response)


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
    on_response: ResponseObserver,
) -> ModelResponse | None:
    rust_achat_completions: Final = load_rust_achat_completions()
    if rust_achat_completions is None:
        return None
    try:
        rust_response: Final = await rust_achat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    except Exception as rust_error:  # noqa: BLE001  # rollout safety: the helper re-raises anything the provider already saw
        _reraise_or_decline(rust_error, model=model, custom_llm_provider=custom_llm_provider)
        return None
    on_response(rust_response)
    return _build_model_response(rust_response, model_response)


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
    on_response: ResponseObserver,
    python_fallback: Callable[[], Awaitable[object]],
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
        on_response=on_response,
    )
    if response is not None:
        return response
    return await python_fallback()
