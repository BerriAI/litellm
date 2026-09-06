from __future__ import annotations

from typing import Final

import httpx

from litellm.rust_bridge.bindings import UNCHANGED, NativeBinding, Unchanged
from litellm.rust_bridge.protocols import RustAtranscription, RustTranscription
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
    audio: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: float | httpx.Timeout | None,
) -> DispatchResult[dict[str, object]]:
    return attempt(
        load=_TRANSCRIPTION.load,
        enabled=True,
        eligible=True,
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
        adapt=identity,
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
) -> DispatchResult[dict[str, object]]:
    return await aattempt(
        load=_ATRANSCRIPTION.load,
        enabled=True,
        eligible=True,
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
        adapt=identity,
    )
