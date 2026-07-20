from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Awaitable, Final, Protocol, Union, cast

import httpx

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


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass
class _RustTranscriptionState:
    enabled: bool
    transcription: RustTranscription | None = None
    atranscription: RustAtranscription | None = None


_STATE = _RustTranscriptionState(
    enabled=os.getenv("LITELLM_USE_RUST_TRANSCRIPTION", "").strip().lower() in {"1", "true", "yes", "on"}
)


def configure_rust_transcription(
    enabled: bool = True,
    *,
    transcription: RustTranscription | None | _Unset = _UNSET,
    atranscription: RustAtranscription | None | _Unset = _UNSET,
) -> None:
    _STATE.enabled = enabled
    if not isinstance(transcription, _Unset):
        _STATE.transcription = transcription
    if not isinstance(atranscription, _Unset):
        _STATE.atranscription = atranscription


def rust_transcription_enabled() -> bool:
    return _STATE.enabled


def load_rust_transcription() -> RustTranscription | None:
    if _STATE.transcription is not None:
        return _STATE.transcription
    from litellm.rust_bridge import get_native_bridge

    native_bridge = get_native_bridge()
    return (
        None
        if native_bridge is None
        else cast(  # cast-ok: native extension protocol is runtime-defined
            RustTranscription, getattr(native_bridge, "transcription", None)
        )
    )


def load_rust_atranscription() -> RustAtranscription | None:
    if _STATE.atranscription is not None:
        return _STATE.atranscription
    from litellm.rust_bridge import get_native_bridge

    native_bridge = get_native_bridge()
    return (
        None
        if native_bridge is None
        else cast(  # cast-ok: native extension protocol is runtime-defined
            RustAtranscription, getattr(native_bridge, "atranscription", None)
        )
    )


def transcription(
    *,
    model: str,
    audio: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: Union[float, httpx.Timeout] | None,
) -> dict[str, object] | None:
    rust_transcription = load_rust_transcription()
    if rust_transcription is None:
        return None
    return rust_transcription(
        model=model,
        audio=audio,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        optional_params=optional_params,
        timeout_seconds=timeout_to_seconds(timeout),
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
    timeout: Union[float, httpx.Timeout] | None,
) -> dict[str, object] | None:
    rust_atranscription = load_rust_atranscription()
    if rust_atranscription is None:
        return None
    return await rust_atranscription(
        model=model,
        audio=audio,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        optional_params=optional_params,
        timeout_seconds=timeout_to_seconds(timeout),
    )
