import importlib

import pytest

import litellm
from litellm.llms.bedrock.audio_transcription import BedrockAudioTranscriptionRustDispatch
from litellm.rust_bridge.request import NativeRequestContext, NativeRequestOptions, NativeTranscriptionRequest
from litellm.rust_bridge.runtime import Handled

rust_bridge = importlib.import_module("litellm.rust_bridge.transcription")


class SyncBridge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.contexts: list[NativeRequestContext] = []

    def __call__(
        self,
        request: NativeTranscriptionRequest,
        *,
        options: NativeRequestOptions,
        context: NativeRequestContext,
    ) -> dict[str, object]:
        self.calls.append(
            {"model": request.model, "audio": request.audio, "optional_params": request.optional_params}
        )
        self.contexts.append(context)
        return {"text": "hello"}


class AsyncBridge:
    async def __call__(
        self,
        request: NativeTranscriptionRequest,
        *,
        options: NativeRequestOptions,
        context: NativeRequestContext,
    ) -> dict[str, object]:
        return {"text": "async"}


def test_enabled_sync_bridge_receives_audio() -> None:
    bridge = SyncBridge()
    rust_bridge.configure_rust_transcription(transcription=bridge)
    result = rust_bridge.transcription(
        model="mistral.voxtral-mini-3b-2507",
        audio={"data": "AQI=", "format": "wav", "filename": "audio.wav"},
        api_key=None,
        api_base=None,
        custom_llm_provider="bedrock",
        extra_headers=None,
        optional_params={"temperature": 0},
        timeout=5.0,
        stream=True,
        has_custom_client=True,
        input_source_kind="file",
    )
    assert isinstance(result, Handled)
    assert result.value == {"text": "hello"}
    assert bridge.calls[0]["audio"] == {"data": "AQI=", "format": "wav", "filename": "audio.wav"}
    assert bridge.contexts[0].capabilities.execution_mode == "sync"
    assert bridge.contexts[0].capabilities.stream is True
    assert bridge.contexts[0].capabilities.has_custom_client is True
    assert bridge.contexts[0].capabilities.input_source_kind == "file"


@pytest.mark.asyncio
async def test_enabled_async_bridge() -> None:
    rust_bridge.configure_rust_transcription(atranscription=AsyncBridge())
    result = await rust_bridge.atranscription(
        model="mistral.voxtral-mini-3b-2507",
        audio={"data": "AQI=", "format": "wav", "filename": "audio.wav"},
        api_key=None,
        api_base=None,
        custom_llm_provider="bedrock",
        extra_headers=None,
        optional_params={},
        timeout=None,
    )
    assert result == Handled({"text": "async"})


def test_loader_returns_none_without_native_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    rust_bridge.configure_rust_transcription(transcription=None, atranscription=None)
    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: None)
    assert rust_bridge.load_rust_transcription() is None
    assert rust_bridge.load_rust_atranscription() is None


def test_dispatch_sync_path_requires_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    rust_bridge.configure_rust_transcription(transcription=None)
    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: None)

    with pytest.raises(RuntimeError, match="bridge is unavailable"):
        BedrockAudioTranscriptionRustDispatch().audio_transcriptions(
            model="bedrock/mistral.voxtral-mini-3b-2507",
            audio_file=("audio.wav", b"audio", "audio/wav"),
            api_key=None,
            api_base=None,
            custom_llm_provider="bedrock",
            extra_headers=None,
            optional_params={},
            timeout=5,
        )


@pytest.mark.asyncio
async def test_dispatch_async_path_requires_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    rust_bridge.configure_rust_transcription(atranscription=None)
    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: None)

    with pytest.raises(RuntimeError, match="bridge is unavailable"):
        await BedrockAudioTranscriptionRustDispatch().async_audio_transcriptions(
            model="bedrock/mistral.voxtral-mini-3b-2507",
            audio_file=("audio.wav", b"audio", "audio/wav"),
            api_key=None,
            api_base=None,
            custom_llm_provider="bedrock",
            extra_headers=None,
            optional_params={},
            timeout=5,
        )


def test_bedrock_transcription_uses_rust_only_path() -> None:
    rust_bridge.configure_rust_transcription(
        transcription=lambda *_args, **_: {"text": "rust"},
        atranscription=None,
    )
    try:
        response = litellm.transcription(
            model="bedrock/mistral.voxtral-mini-3b-2507",
            file=("audio.wav", b"audio", "audio/wav"),
        )
    finally:
        rust_bridge.configure_rust_transcription(transcription=None, atranscription=None)

    assert response.text == "rust"


@pytest.mark.asyncio
async def test_bedrock_atranscription_uses_rust_only_path() -> None:
    async def rust_response(*_args: object, **_: object) -> dict[str, object]:
        return {"text": "rust"}

    rust_bridge.configure_rust_transcription(transcription=None, atranscription=rust_response)
    try:
        response = await litellm.atranscription(
            model="bedrock/mistral.voxtral-mini-3b-2507",
            file=("audio.wav", b"audio", "audio/wav"),
        )
    finally:
        rust_bridge.configure_rust_transcription(transcription=None, atranscription=None)

    assert response.text == "rust"
