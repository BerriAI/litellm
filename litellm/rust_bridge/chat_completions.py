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

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol, cast

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.bedrock.request_metadata import bedrock_request_metadata_is_owned
from litellm.rust_bridge.bindings import UNSET, NativeBinding, Unset
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    CoreEngine,
    ExecutionResult,
    FallbackMode,
    RustDeclined,
    RustHandled,
    aattempt,
    ainvoke,
    attempt,
    execution_hidden_params,
    identity,
    invoke,
)
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


class RustChatCompletions(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
    ) -> Mapping[str, object]:
        raise NotImplementedError


class RustAchatCompletions(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
    ) -> Awaitable[Mapping[str, object]]:
        raise NotImplementedError


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
    """Invoked with the payload the core returned, on success only.

    Lets the caller emit its own `post_call` on whichever path served the
    request. Both entry points call it, so the synchronous and asynchronous
    paths cannot drift apart the way the pre_call suppression once did.
    """

    def __call__(self, rust_response: Mapping[str, object], /) -> None:
        raise NotImplementedError


def response_logger(
    *,
    logging_obj: LiteLLMLoggingObj,
    messages: Sequence[object],
    api_key: str,
    additional_args: Mapping[str, object],
) -> ResponseObserver:
    """A `ResponseObserver` that emits the caller's `post_call` for a Rust-served
    request.

    The core owns the provider call, so the Python transform that normally
    raises this event never runs; without it every `post_call` callback goes
    silent on a Rust-served request and `original_response` stays unset. The
    payload is the core's normalized response rather than the provider's wire
    body, which is the closest thing that crosses the bridge.
    """

    def log(rust_response: Mapping[str, object], /) -> None:
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=json.dumps(rust_response),
            additional_args=additional_args,
        )

    return log


_CHAT_COMPLETIONS: Final = NativeBinding[RustChatCompletions]("chat_completions")
_ACHAT_COMPLETIONS: Final = NativeBinding[RustAchatCompletions]("achat_completions")
_DECLINE: Final = NativeBinding[RustChatCompletionsDecline]("chat_completions_decline")


def set_rust_chat_completions(
    *,
    chat_completions: RustChatCompletions | None | Unset = UNSET,
    achat_completions: RustAchatCompletions | None | Unset = UNSET,
    decline: RustChatCompletionsDecline | None | Unset = UNSET,
) -> None:
    """Inject the native callables, so tests can supply a double instead of
    patching module attributes."""
    _CHAT_COMPLETIONS.update(chat_completions)
    _ACHAT_COMPLETIONS.update(achat_completions)
    _DECLINE.update(decline)


def load_rust_chat_completions() -> RustChatCompletions | None:
    return _CHAT_COMPLETIONS.load()


def load_rust_achat_completions() -> RustAchatCompletions | None:
    return _ACHAT_COMPLETIONS.load()


def _load_rust_decline() -> RustChatCompletionsDecline | None:
    return _DECLINE.load()


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
    request_override: Final = litellm_params.get("rust") if litellm_params is not None else None
    if not rust_enabled(request_override=request_override if isinstance(request_override, bool) else None):
        return False
    if _litellm_metadata_reaches_the_provider(custom_llm_provider, litellm_params):
        verbose_logger.debug("Rust chat completions declined (litellm metadata user_id); using the Python path")
        return False
    decline: Final = _load_rust_decline()
    if decline is None:
        return False
    gate_result: Final = attempt(
        native_call=lambda: decline(
            model=model,
            messages=messages,
            optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
        ),
        adapt=identity,
        context=BridgeErrorContext(
            route="chat completions capability check",
            provider=custom_llm_provider or "",
            model=model,
        ),
    )
    if isinstance(gate_result, RustDeclined):
        verbose_logger.debug(
            "Rust chat completions declined (%s); using the Python path",
            gate_result.reason,
        )
        return False
    if not isinstance(gate_result, RustHandled):
        return False
    reason: Final = gate_result.value
    if reason is not None:
        verbose_logger.debug("Rust chat completions declined (%s); using the Python path", reason)
        return False
    return True


def _build_model_response(
    rust_response: Mapping[str, object],
    model_response: ModelResponse,
) -> ModelResponse:
    built: Final = convert_to_model_response_object(
        response_object=dict(rust_response),  # mutable-ok: the converter takes a real dict and rewrites it
        model_response_object=model_response,
        hidden_params=execution_hidden_params(None, CoreEngine.RUST),  # mutable-ok: rewritten by the converter
    )
    if not isinstance(built, ModelResponse):
        raise TypeError(f"expected a ModelResponse from the rust path, got {type(built).__name__}")
    return built


def _unwrap_execution(result: ExecutionResult[object]) -> object:
    value: Final = result.value
    if isinstance(value, ModelResponse):
        raw_hidden_params: Final = cast(
            object,
            value._hidden_params,  # pyright: ignore[reportPrivateUsage]  # ModelResponse has no public metadata getter
        )
        hidden_params: Final = (
            cast(Mapping[str, object], raw_hidden_params)  # cast-ok: guarded by the mapping check
            if isinstance(raw_hidden_params, Mapping)
            else None
        )
        value._hidden_params = execution_hidden_params(  # pyright: ignore[reportPrivateUsage]  # ModelResponse has no public metadata setter
            hidden_params, result.source
        )
    return value


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
    native_call: Final = (
        None
        if rust_chat_completions is None
        else lambda: rust_chat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    )

    def adapt(rust_response: Mapping[str, object]) -> ModelResponse:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    result: Final = attempt(
        native_call=native_call,
        adapt=adapt,
        context=BridgeErrorContext(route="chat completions", provider=custom_llm_provider or "", model=model),
    )
    return result.value if isinstance(result, RustHandled) else None


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
    native_call: Final = (
        None
        if rust_achat_completions is None
        else lambda: rust_achat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    )

    def adapt(rust_response: Mapping[str, object]) -> ModelResponse:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    result: Final = await aattempt(
        native_call=native_call,
        adapt=adapt,
        context=BridgeErrorContext(route="chat completions", provider=custom_llm_provider or "", model=model),
    )
    return result.value if isinstance(result, RustHandled) else None


def chat_completions_or_fallback(
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
    python_fallback: Callable[[], object],
) -> object:
    rust_chat_completions: Final = load_rust_chat_completions()
    native_call: Final = (
        None
        if rust_chat_completions is None
        else lambda: rust_chat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    )

    def adapt(rust_response: Mapping[str, object]) -> object:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    return _unwrap_execution(
        invoke(
            native_call=native_call,
            fallback=python_fallback,
            adapt=adapt,
            mode=FallbackMode.PYTHON,
            context=BridgeErrorContext(route="chat completions", provider=custom_llm_provider or "", model=model),
        )
    )


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
    rust_achat_completions: Final = load_rust_achat_completions()
    native_call: Final = (
        None
        if rust_achat_completions is None
        else lambda: rust_achat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    )

    def adapt(rust_response: Mapping[str, object]) -> object:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    return _unwrap_execution(
        await ainvoke(
            native_call=native_call,
            fallback=python_fallback,
            adapt=adapt,
            mode=FallbackMode.PYTHON,
            context=BridgeErrorContext(route="chat completions", provider=custom_llm_provider or "", model=model),
        )
    )
