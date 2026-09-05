from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class NativeTranscriptionRequest:
    model: str
    audio: dict[str, object]
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str | None
    extra_headers: dict[str, object] | None
    optional_params: dict[str, object]
    timeout: float | httpx.Timeout | None
    callback_adapter: OneShotCallbackHandle | None = None


_TRANSCRIPTION: Final = cast(  # cast-ok: generic classmethod loses the route Protocol parameters
    EndpointDispatch[RustTranscription, RustAtranscription],
    EndpointDispatch.native(
        route="transcription",
        sync="transcription",
        asynchronous="atranscription",
        enabled=always_enabled,
    ),
)


def set_rust_transcription(
    *,
    sync: RustTranscription | None | Unchanged = UNCHANGED,
    asynchronous: RustAtranscription | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(sync, Unchanged):
        if sync is None:
            _TRANSCRIPTION.sync.reset()
        else:
            _TRANSCRIPTION.sync.override(sync)
    if not isinstance(asynchronous, Unchanged):
        if asynchronous is None:
            _TRANSCRIPTION.asynchronous.reset()
        else:
            _TRANSCRIPTION.asynchronous.override(asynchronous)


def dispatch_transcription(
    *,
    prepare: Callable[[], NativeTranscriptionRequest],
    model: str,
    provider: str,
) -> dict[str, object]:
    return _TRANSCRIPTION.require(
        prepare=prepare,
        call=_call_transcription,
        adapt=identity,
        context=BridgeErrorContext(provider=provider, model=model),
    )


async def adispatch_transcription(
    *,
    prepare: Callable[[], NativeTranscriptionRequest],
    model: str,
    provider: str,
) -> dict[str, object]:
    return await _TRANSCRIPTION.arequire(
        prepare=prepare,
        call=_call_atranscription,
        adapt=identity,
        context=BridgeErrorContext(provider=provider, model=model),
    )


def _call_transcription(
    native: RustTranscription,
    request: NativeTranscriptionRequest,
) -> dict[str, object]:
    return native(
        model=request.model,
        audio=request.audio,
        api_key=request.api_key,
        api_base=request.api_base,
        custom_llm_provider=request.custom_llm_provider,
        extra_headers=request.extra_headers,
        optional_params=request.optional_params,
        timeout_seconds=timeout_to_seconds(request.timeout),
        callback_adapter=request.callback_adapter,
    )


def _call_atranscription(
    native: RustAtranscription,
    request: NativeTranscriptionRequest,
) -> Awaitable[dict[str, object]]:
    return native(
        model=request.model,
        audio=request.audio,
        api_key=request.api_key,
        api_base=request.api_base,
        custom_llm_provider=request.custom_llm_provider,
        extra_headers=request.extra_headers,
        optional_params=request.optional_params,
        timeout_seconds=timeout_to_seconds(request.timeout),
        callback_adapter=request.callback_adapter,
    )
