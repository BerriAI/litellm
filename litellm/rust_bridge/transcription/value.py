from __future__ import annotations

from typing import (
    Final,
    cast,  # noqa: TID251  # native callable signatures are checked by bridge contract tests
)

import httpx

from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset
from litellm.rust_bridge.configuration import RouteName
from litellm.rust_bridge.route import NativeRoute
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.rust_bridge.transcription.types import RustAtranscription, RustTranscription

ROUTE: Final = NativeRoute(RouteName.TRANSCRIPTION)


def _as_transcription(value: object) -> RustTranscription | None:
    return cast(RustTranscription, value) if callable(value) else None  # cast-ok: validated callable native binding


def _as_atranscription(value: object) -> RustAtranscription | None:
    return cast(RustAtranscription, value) if callable(value) else None  # cast-ok: validated callable native binding


_TRANSCRIPTION: Final = ROUTE.bind("transcription", validate=_as_transcription)
_ATRANSCRIPTION: Final = ROUTE.bind("atranscription", validate=_as_atranscription)


def configure_rust_transcription(
    *,
    transcription: RustTranscription | None | BindingUnset = BINDING_UNSET,
    atranscription: RustAtranscription | None | BindingUnset = BINDING_UNSET,
) -> None:
    _TRANSCRIPTION.configure(transcription)
    _ATRANSCRIPTION.configure(atranscription)


def load_rust_transcription() -> RustTranscription | None:
    return ROUTE.select(_TRANSCRIPTION)


def load_rust_atranscription() -> RustAtranscription | None:
    return ROUTE.select(_ATRANSCRIPTION)


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
    timeout: float | httpx.Timeout | None,
) -> dict[str, object] | None:
    rust_atranscription: Final = load_rust_atranscription()
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
