"""Thin Python wrapper for the native Rust chat completions bridge.

The Rust core owns the conversation translation, the provider call, and the
response normalization for the subset of `/chat/completions` requests it
accepts. This module only marshals inputs and hands the normalized result to
LiteLLM's existing `ModelResponse` builder.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.bedrock.request_metadata import bedrock_request_metadata_is_owned
from litellm.rust_bridge.bindings import UNCHANGED, NativeBinding, Unchanged
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.protocols import (
    RustAchatCompletions,
    RustChatCompletions,
    RustChatCompletionsDecline,
)
from litellm.rust_bridge.request import (
    NativeAnthropicOptions,
    NativeBedrockOptions,
    NativeChatCompletionsRequest,
    NativeRequestCapabilities,
    NativeRequestContext,
    NativeRequestOptions,
    PreparedNativeCall,
    call_native,
    with_capabilities,
)
from litellm.rust_bridge.runtime import DispatchResult, aattempt, attempt
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


_CHAT: Final[NativeBinding[RustChatCompletions]] = NativeBinding(lambda native: native.chat_completions)
_ACHAT: Final[NativeBinding[RustAchatCompletions]] = NativeBinding(lambda native: native.achat_completions)
_CHAT_PREFLIGHT: Final[NativeBinding[RustChatCompletionsDecline]] = NativeBinding(
    lambda native: native.chat_completions_decline
)


def set_rust_chat_completions(
    *,
    chat_completions: RustChatCompletions | None | Unchanged = UNCHANGED,
    achat_completions: RustAchatCompletions | None | Unchanged = UNCHANGED,
    decline: RustChatCompletionsDecline | None | Unchanged = UNCHANGED,
) -> None:
    """Inject the native callables, so tests can supply a double instead of
    patching module attributes."""
    if not isinstance(chat_completions, Unchanged):
        if chat_completions is None:
            _CHAT.reset()
        else:
            _CHAT.override(chat_completions)
    if not isinstance(achat_completions, Unchanged):
        if achat_completions is None:
            _ACHAT.reset()
        else:
            _ACHAT.override(achat_completions)
    if not isinstance(decline, Unchanged):
        if decline is None:
            _CHAT_PREFLIGHT.reset()
        else:
            _CHAT_PREFLIGHT.override(decline)


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
    if _litellm_metadata_reaches_the_provider(custom_llm_provider, litellm_params):
        verbose_logger.debug("Rust chat completions declined (litellm metadata user_id); using the Python path")
        return False
    if not rust_enabled():
        return False
    decline: Final = _CHAT_PREFLIGHT.load()
    if decline is None:
        return False
    try:
        reason: Final = decline(
            model=model,
            messages=messages,
            optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
        )
    except Exception as error:  # noqa: BLE001  # capability checks perform no provider I/O
        verbose_logger.debug("Native chat acceptance check failed: %s", error)
        return False
    if reason is not None:
        verbose_logger.debug("Native chat request is ineligible: %s", reason)
    return reason is None


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
    bedrock: NativeBedrockOptions | None = None,
    anthropic: NativeAnthropicOptions | None = None,
    stream: bool = False,
    has_custom_client: bool = False,
    eligible: bool = True,
    context: NativeRequestContext | None = None,
) -> DispatchResult[ModelResponse]:
    def adapt(rust_response: Mapping[str, object]) -> ModelResponse:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    def call(
        native: RustChatCompletions, prepared: PreparedNativeCall[NativeChatCompletionsRequest]
    ) -> Mapping[str, object]:
        return call_native(native, prepared)

    return attempt(
        load=_CHAT.load,
        enabled=rust_enabled(),
        eligible=eligible,
        prepare=lambda: PreparedNativeCall(
            request=NativeChatCompletionsRequest(model=model, messages=messages, optional_params=optional_params),
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
                bedrock=bedrock,
                anthropic=anthropic,
            ),
            context=with_capabilities(
                context or NativeRequestContext(),
                NativeRequestCapabilities(
                    execution_mode="sync",
                    stream=stream,
                    has_custom_client=has_custom_client,
                ),
            ),
        ),
        call=call,
        adapt=adapt,
    )


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
    bedrock: NativeBedrockOptions | None = None,
    anthropic: NativeAnthropicOptions | None = None,
    stream: bool = False,
    has_custom_client: bool = False,
    eligible: bool = True,
    context: NativeRequestContext | None = None,
) -> DispatchResult[ModelResponse]:
    def adapt(rust_response: Mapping[str, object]) -> ModelResponse:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    async def call(
        native: RustAchatCompletions,
        prepared: PreparedNativeCall[NativeChatCompletionsRequest],
    ) -> Mapping[str, object]:
        return await call_native(native, prepared)

    return await aattempt(
        load=_ACHAT.load,
        enabled=rust_enabled(),
        eligible=eligible,
        prepare=lambda: PreparedNativeCall(
            request=NativeChatCompletionsRequest(model=model, messages=messages, optional_params=optional_params),
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
                bedrock=bedrock,
                anthropic=anthropic,
            ),
            context=with_capabilities(
                context or NativeRequestContext(),
                NativeRequestCapabilities(
                    execution_mode="async",
                    stream=stream,
                    has_custom_client=has_custom_client,
                ),
            ),
        ),
        call=call,
        adapt=adapt,
    )
