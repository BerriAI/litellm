from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol

import httpx

from litellm.rust_bridge.runtime import (
    UNSET,
    BridgeErrorContext,
    FallbackMode,
    NativeBinding,
    Unset,
    ainvoke,
    async_none,
    identity,
    invoke,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds


class RustTranscription(Protocol):
    def __call__(
        self,
        model: str,
        audio: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        raise NotImplementedError


class RustAtranscription(Protocol):
    def __call__(
        self,
        model: str,
        audio: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError


_TRANSCRIPTION: Final = NativeBinding[RustTranscription]("transcription")
_ATRANSCRIPTION: Final = NativeBinding[RustAtranscription]("atranscription")


def configure_rust_transcription(
    enabled: bool = True,
    *,
    transcription: RustTranscription | None | Unset = UNSET,
    atranscription: RustAtranscription | None | Unset = UNSET,
) -> None:
    _ = enabled
    _TRANSCRIPTION.update(transcription)
    _ATRANSCRIPTION.update(atranscription)


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
) -> dict[str, object] | None:
    rust_transcription: Final = load_rust_transcription()
    native_call: Final = (
        None
        if rust_transcription is None
        else lambda: rust_transcription(
            model=model,
            audio=audio,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    )
    result: Final = invoke(
        native_call=native_call,
        fallback=lambda: None,
        adapt=identity,
        mode=FallbackMode.RUST_REQUIRED,
        context=BridgeErrorContext(route="audio transcription", provider=custom_llm_provider or "", model=model),
    )
    return result.value


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
    rust_atranscription: Final = load_rust_atranscription()
    native_call: Final = (
        None
        if rust_atranscription is None
        else lambda: rust_atranscription(
            model=model,
            audio=audio,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    )
    result: Final = await ainvoke(
        native_call=native_call,
        fallback=async_none,
        adapt=identity,
        mode=FallbackMode.RUST_REQUIRED,
        context=BridgeErrorContext(route="audio transcription", provider=custom_llm_provider or "", model=model),
    )
    return result.value
