from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # runtime typing constructs

import httpx

from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.callbacks import OneShotCallbackHandle
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointDispatch,
    always_enabled,
    identity,
)  # cast-ok: generic classmethod cannot preserve the route Protocol parameters
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
        callback_adapter: OneShotCallbackHandle | None,
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
        callback_adapter: OneShotCallbackHandle | None,
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError


_TRANSCRIPTION: Final = cast(  # cast-ok: generic classmethod loses the route Protocol parameters
    EndpointDispatch[RustTranscription, RustAtranscription],
    EndpointDispatch.native(
        route="transcription",
        sync="transcription",
        asynchronous="atranscription",
        enabled=always_enabled,
    ),
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
    callback_adapter: OneShotCallbackHandle | None = None,
) -> dict[str, object]:
    return _TRANSCRIPTION.require(
        call=lambda rust_transcription: rust_transcription(
            model=model,
            audio=audio,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=timeout_to_seconds(timeout),
            callback_adapter=callback_adapter,
        ),
        adapt=identity,
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
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
    callback_adapter: OneShotCallbackHandle | None = None,
) -> dict[str, object]:
    return await _TRANSCRIPTION.arequire(
        call=lambda rust_atranscription: rust_atranscription(
            model=model,
            audio=audio,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=timeout_to_seconds(timeout),
            callback_adapter=callback_adapter,
        ),
        adapt=identity,
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )
