from __future__ import annotations

from typing import Final

import httpx

from litellm.rust_bridge.bindings import UNCHANGED, NativeBinding, Unchanged
from litellm.rust_bridge.protocols import RustAtranscription, RustTranscription
from litellm.rust_bridge.request import (
    NativeRequestCapabilities,
    NativeRequestContext,
    NativeRequestOptions,
    NativeTranscriptionRequest,
    PreparedNativeCall,
    bedrock_options,
    call_native,
    with_capabilities,
)
from litellm.rust_bridge.runtime import DispatchResult, aattempt, attempt, identity
from litellm.rust_bridge.timeouts import timeout_to_seconds

_TRANSCRIPTION: Final[NativeBinding[RustTranscription]] = NativeBinding(lambda native: native.transcription)
_ATRANSCRIPTION: Final[NativeBinding[RustAtranscription]] = NativeBinding(lambda native: native.atranscription)


def configure_rust_transcription(
    *,
    transcription: RustTranscription | None | Unchanged = UNCHANGED,
    atranscription: RustAtranscription | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(transcription, Unchanged):
        if transcription is None:
            _TRANSCRIPTION.reset()
        else:
            _TRANSCRIPTION.override(transcription)
    if not isinstance(atranscription, Unchanged):
        if atranscription is None:
            _ATRANSCRIPTION.reset()
        else:
            _ATRANSCRIPTION.override(atranscription)


def load_rust_transcription() -> RustTranscription | None:
    return _TRANSCRIPTION.load()


def load_rust_atranscription() -> RustAtranscription | None:
    return _ATRANSCRIPTION.load()


def transcription(
    *,
    model: str,
    audio: object,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: float | httpx.Timeout | None,
    stream: bool = False,
    has_custom_client: bool = False,
    input_source_kind: str | None = None,
    context: NativeRequestContext | None = None,
) -> DispatchResult[dict[str, object]]:
    return attempt(
        load=_TRANSCRIPTION.load,
        enabled=True,
        eligible=True,
        prepare=lambda: PreparedNativeCall(
            request=NativeTranscriptionRequest(model=model, audio=audio, optional_params=optional_params),
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
                bedrock=bedrock_options(optional_params),
            ),
            context=with_capabilities(
                context or NativeRequestContext(),
                NativeRequestCapabilities(
                    execution_mode="sync",
                    stream=stream,
                    has_custom_client=has_custom_client,
                    input_source_kind=input_source_kind,
                ),
            ),
        ),
        call=call_native,
        adapt=identity,
    )


async def atranscription(
    *,
    model: str,
    audio: object,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: float | httpx.Timeout | None,
    stream: bool = False,
    has_custom_client: bool = False,
    input_source_kind: str | None = None,
    context: NativeRequestContext | None = None,
) -> DispatchResult[dict[str, object]]:
    return await aattempt(
        load=_ATRANSCRIPTION.load,
        enabled=True,
        eligible=True,
        prepare=lambda: PreparedNativeCall(
            request=NativeTranscriptionRequest(model=model, audio=audio, optional_params=optional_params),
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
                bedrock=bedrock_options(optional_params),
            ),
            context=with_capabilities(
                context or NativeRequestContext(),
                NativeRequestCapabilities(
                    execution_mode="async",
                    stream=stream,
                    has_custom_client=has_custom_client,
                    input_source_kind=input_source_kind,
                ),
            ),
        ),
        call=call_native,
        adapt=identity,
    )
