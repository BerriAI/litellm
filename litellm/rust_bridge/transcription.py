from __future__ import annotations

from typing import Final

import httpx

from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.protocols import RustAtranscription, RustTranscription
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointDispatch,
    always_enabled,
    async_none,
    identity,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds

_TRANSCRIPTION: Final[EndpointDispatch[RustTranscription, RustAtranscription]] = EndpointDispatch.native(
    route="audio transcription",
    sync=lambda native: native.transcription,
    asynchronous=lambda native: native.atranscription,
    enabled=always_enabled,
)


def configure_rust_transcription(
    enabled: bool = True,
    *,
    transcription: RustTranscription | None | Unchanged = UNCHANGED,
    atranscription: RustAtranscription | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(transcription, Unchanged):
        if transcription is None:
            _TRANSCRIPTION.sync.reset()
        else:
            _TRANSCRIPTION.sync.override(transcription)
    if not isinstance(atranscription, Unchanged):
        if atranscription is None:
            _TRANSCRIPTION.asynchronous.reset()
        else:
            _TRANSCRIPTION.asynchronous.override(atranscription)


def load_rust_transcription() -> RustTranscription | None:
    return _TRANSCRIPTION.sync.load()


def load_rust_atranscription() -> RustAtranscription | None:
    return _TRANSCRIPTION.asynchronous.load()


def transcription(
    *,
    model: str,
    audio: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: float | httpx.Timeout | None,
) -> dict[str, object] | None:
    return _TRANSCRIPTION.invoke(
        prepare=lambda: timeout_to_seconds(timeout),
        call=lambda rust_transcription, timeout_seconds: rust_transcription(
            model=model,
            audio=audio,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=timeout_seconds,
        ),
        fallback=lambda: None,
        adapt=identity,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )


async def atranscription(
    *,
    model: str,
    audio: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: float | httpx.Timeout | None,
) -> dict[str, object] | None:
    return await _TRANSCRIPTION.ainvoke(
        prepare=lambda: timeout_to_seconds(timeout),
        call=lambda rust_atranscription, timeout_seconds: rust_atranscription(
            model=model,
            audio=audio,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=timeout_seconds,
        ),
        fallback=async_none,
        adapt=identity,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )
