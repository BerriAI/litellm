from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import (
    Final,
    cast,  # noqa: TID251  # native callable signatures are checked by bridge contract tests
)

import httpx

from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset
from litellm.rust_bridge.configuration import CapabilityContext
from litellm.rust_bridge.runtime import BridgeErrorContext, ainvoke, invoke
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.rust_bridge.transcription.definition import COMPONENT
from litellm.rust_bridge.transcription.types import RustAtranscription, RustTranscription


def _as_transcription(value: object) -> RustTranscription | None:
    return cast(RustTranscription, value) if callable(value) else None  # cast-ok: validated callable native binding


def _as_atranscription(value: object) -> RustAtranscription | None:
    return cast(RustAtranscription, value) if callable(value) else None  # cast-ok: validated callable native binding


_TRANSCRIPTION: Final = COMPONENT.bind("transcription", validate=_as_transcription)
_ATRANSCRIPTION: Final = COMPONENT.bind("atranscription", validate=_as_atranscription)


def configure_rust_transcription(
    *,
    transcription: RustTranscription | None | BindingUnset = BINDING_UNSET,
    atranscription: RustAtranscription | None | BindingUnset = BINDING_UNSET,
) -> None:
    _TRANSCRIPTION.configure(transcription)
    _ATRANSCRIPTION.configure(atranscription)


def load_rust_transcription(*, context: CapabilityContext) -> RustTranscription | None:
    return COMPONENT.resolve(context).select(_TRANSCRIPTION)


def load_rust_atranscription(*, context: CapabilityContext) -> RustAtranscription | None:
    return COMPONENT.resolve(context).select(_ATRANSCRIPTION)


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
    python_fallback: Callable[[], dict[str, object]] | None,
) -> dict[str, object]:
    execution: Final = COMPONENT.resolve(CapabilityContext(provider=custom_llm_provider or "", model=model))
    rust_transcription: Final = execution.select(_TRANSCRIPTION)
    return invoke(
        execution=execution,
        native_call=(
            lambda: rust_transcription(
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
        if rust_transcription is not None
        else None,
        python_fallback=python_fallback,
        adapt=lambda response: response,
        context=BridgeErrorContext(route=COMPONENT.name.value, provider=custom_llm_provider or "", model=model),
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
    python_fallback: Callable[[], Awaitable[dict[str, object]]] | None,
) -> dict[str, object]:
    execution: Final = COMPONENT.resolve(CapabilityContext(provider=custom_llm_provider or "", model=model))
    rust_atranscription: Final = execution.select(_ATRANSCRIPTION)

    async def adapt(response: dict[str, object]) -> dict[str, object]:
        return response

    return await ainvoke(
        execution=execution,
        native_call=(
            lambda: rust_atranscription(
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
        if rust_atranscription is not None
        else None,
        python_fallback=python_fallback,
        adapt=adapt,
        context=BridgeErrorContext(route=COMPONENT.name.value, provider=custom_llm_provider or "", model=model),
    )
