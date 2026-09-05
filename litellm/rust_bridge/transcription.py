from __future__ import annotations

import base64
import json
from collections.abc import Callable, Coroutine
from contextlib import nullcontext
from dataclasses import dataclass
from io import IOBase
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.protocols import RustAtranscription, RustRouteDecline, RustTranscription
from litellm.rust_bridge.request import (
    NativePreCallDetails,
    NativeRequestCapabilities,
    NativeRequestContext,
    NativeRequestOptions,
    NativeTranscriptionRequest,
    PreparedNativeCall,
    bedrock_options,
    call_native,
    request_context,
    with_capabilities,
)
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointBinding,
    EndpointDispatch,
    PythonFallback,
    always_enabled,
    assess_route,
    async_none,
    identity,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import FileTypes, TranscriptionResponse

_TRANSCRIPTION: Final[EndpointDispatch[RustTranscription, RustAtranscription]] = EndpointDispatch.native(
    route="transcription",
    sync=lambda native: native.transcription,
    asynchronous=lambda native: native.atranscription,
    enabled=always_enabled,
)


_PREFLIGHT: Final[EndpointBinding[RustRouteDecline]] = EndpointBinding.native(
    route="transcription",
    select=lambda native: native.transcription_decline,
    enabled=always_enabled,
)


def configure_rust_transcription(
    *,
    transcription: RustTranscription | None | Unchanged = UNCHANGED,
    atranscription: RustAtranscription | None | Unchanged = UNCHANGED,
    decline: RustRouteDecline | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(decline, Unchanged):
        if decline is None:
            _PREFLIGHT.reset()
        else:
            _PREFLIGHT.override(decline)
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
) -> dict[str, object] | None:
    return _TRANSCRIPTION.invoke(
        prepare=lambda: PreparedNativeCall(
            NativeTranscriptionRequest(
                model=model,
                audio=audio,
                optional_params=optional_params,
            ),
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
        preflight=lambda: assess_route(_PREFLIGHT, model, custom_llm_provider or ""),
        fallback=lambda: None,
        adapt=identity,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
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
) -> dict[str, object] | None:
    return await _TRANSCRIPTION.ainvoke(
        prepare=lambda: PreparedNativeCall(
            NativeTranscriptionRequest(
                model=model,
                audio=audio,
                optional_params=optional_params,
            ),
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
        preflight=lambda: assess_route(_PREFLIGHT, model, custom_llm_provider or ""),
        fallback=async_none,
        adapt=identity,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )


TranscriptionResult = TranscriptionResponse | Coroutine[object, object, TranscriptionResponse]


def _input_source_kind(file: FileTypes) -> str:
    content: Final = file[1] if isinstance(file, tuple) else file
    if isinstance(content, (bytes, bytearray, memoryview)):
        return "bytes"
    if isinstance(content, IOBase):
        return "file"
    if isinstance(content, str):
        return "path"
    return "opaque"


@dataclass
class _TranscriptionOperation:
    model: str
    provider: str
    file: FileTypes
    api_key: str | None
    api_base: str | None
    headers: dict[str, object] | None
    optional_params: dict[str, object]
    timeout: float | httpx.Timeout | None
    logging: Logging
    python: Callable[[FileTypes], TranscriptionResult]
    asynchronous: bool = False
    has_custom_client: bool = False
    fallback_file: FileTypes | None = None
    logged: bool = False

    def prepare(self) -> PreparedNativeCall[NativeTranscriptionRequest]:
        key: Final = (
            self.api_key
            or litellm.api_key
            or TypeAdapter(str | None).validate_python(getattr(litellm, f"{self.provider}_key", None))
            or get_secret_str(f"{self.provider.upper()}_API_KEY")
        )
        base: Final = (
            self.api_base
            or litellm.api_base
            or get_secret_str(f"{self.provider.upper()}_BASE_URL")
            or get_secret_str(f"{self.provider.upper()}_API_BASE")
        )
        content: Final = self.file[1] if isinstance(self.file, tuple) else self.file
        position: Final = content.tell() if isinstance(content, IOBase) and content.seekable() else None
        try:
            processed: Final = process_audio_file(self.file)
        finally:
            if position is not None and isinstance(content, IOBase):
                content.seek(position)
        self.fallback_file = (processed.filename, processed.file_content, processed.content_type)
        audio: Final = TypeAdapter(dict[str, object]).validate_python(
            MappingProxyType(
                {
                    "data": base64.b64encode(processed.file_content).decode("ascii"),
                    "format": processed.filename.rsplit(".", 1)[-1].lower() if "." in processed.filename else "wav",
                    "filename": processed.filename,
                }
            )
        )
        log_details: Final[NativePreCallDetails] = {
            "api_base": base or "",
            "headers": self.headers,
            "complete_input_dict": {"model": self.model, **self.optional_params},
        }
        self.logging.pre_call(input="audio transcription", api_key=key, additional_args=log_details)
        self.logged = True
        return PreparedNativeCall(
            NativeTranscriptionRequest(
                model=self.model,
                audio=audio,
                optional_params=self.optional_params,
            ),
            options=NativeRequestOptions(
                api_key=key,
                api_base=base,
                custom_llm_provider=self.provider,
                extra_headers=self.headers,
                timeout_seconds=timeout_to_seconds(self.timeout),
                bedrock=bedrock_options(self.optional_params),
            ),
            context=request_context(
                logging_obj=self.logging,
                request_model=self.logging.model,
                litellm_params=self.logging.litellm_params,
                capabilities=NativeRequestCapabilities(
                    execution_mode="async" if self.asynchronous else "sync",
                    stream=self.optional_params.get("stream") is True,
                    has_custom_client=self.has_custom_client,
                    input_source_kind=_input_source_kind(self.file),
                ),
            ),
        )

    def fallback(self) -> TranscriptionResult:
        with self.logging.suppress_next_pre_call() if self.logged else nullcontext():
            return self.python(self.fallback_file if self.fallback_file is not None else self.file)

    async def afallback(self) -> TranscriptionResponse:
        with self.logging.suppress_next_pre_call() if self.logged else nullcontext():
            result: Final = self.python(self.fallback_file if self.fallback_file is not None else self.file)
            return await result if isinstance(result, Coroutine) else result

    def adapt(self, response: dict[str, object]) -> TranscriptionResponse:
        text: Final = TypeAdapter(str).validate_python(response["text"])
        parsed: Final = TranscriptionResponse(text=text)
        self.logging.post_call(
            input="audio transcription", api_key=self.api_key, original_response=json.dumps(response)
        )
        return parsed


def dispatch_transcription(
    *,
    model: str,
    provider: str,
    file: FileTypes,
    api_key: str | None,
    api_base: str | None,
    headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: float | httpx.Timeout | None,
    logging: Logging,
    asynchronous: bool,
    has_custom_client: bool,
    fallback: Callable[[FileTypes], TranscriptionResult],
) -> TranscriptionResult:
    operation: Final = _TranscriptionOperation(
        model,
        provider,
        file,
        api_key,
        api_base,
        headers,
        optional_params,
        timeout,
        logging,
        fallback,
        asynchronous,
        has_custom_client,
    )

    def preflight() -> PythonFallback | None:
        return assess_route(
            _PREFLIGHT,
            model,
            provider,
            stream=optional_params.get("stream") is True,
            has_custom_client=has_custom_client,
        )

    error_context: Final = BridgeErrorContext(provider=provider, model=model)
    if provider == "bedrock":
        if asynchronous:
            return _TRANSCRIPTION.arequire(
                prepare=operation.prepare,
                call=call_native,
                adapt=operation.adapt,
                error_context=error_context,
                preflight=preflight,
            )
        return _TRANSCRIPTION.require(
            prepare=operation.prepare,
            call=call_native,
            adapt=operation.adapt,
            error_context=error_context,
            preflight=preflight,
        )
    if asynchronous:
        return _TRANSCRIPTION.ainvoke(
            prepare=operation.prepare,
            call=call_native,
            adapt=operation.adapt,
            fallback=operation.afallback,
            error_context=error_context,
            eligible=rust_enabled(),
            preflight=preflight,
        )
    return _TRANSCRIPTION.invoke(
        prepare=operation.prepare,
        call=call_native,
        adapt=operation.adapt,
        fallback=operation.fallback,
        error_context=error_context,
        eligible=rust_enabled(),
        preflight=preflight,
    )
