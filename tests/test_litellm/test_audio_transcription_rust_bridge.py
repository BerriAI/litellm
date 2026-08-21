import importlib

import pytest

import litellm
from litellm.llms.bedrock.audio_transcription import BedrockAudioTranscriptionRustDispatch

rust_bridge = importlib.import_module("litellm.rust_bridge.transcription")


class SyncBridge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
        self.calls.append({"model": model, "audio": audio, "optional_params": optional_params})
        return {"text": "hello"}


class AsyncBridge:
    async def __call__(
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
        return {"text": "async"}


def test_enabled_sync_bridge_receives_audio() -> None:
    bridge = SyncBridge()
    rust_bridge.configure_rust_transcription(True, transcription=bridge)
    result = rust_bridge.transcription(
        model="mistral.voxtral-mini-3b-2507",
        audio={"data": "AQI=", "format": "wav", "filename": "audio.wav"},
        api_key=None,
        api_base=None,
        custom_llm_provider="bedrock",
        extra_headers=None,
        optional_params={"temperature": 0},
        timeout=5.0,
    )
    assert result == {"text": "hello"}
    assert bridge.calls[0]["audio"] == {"data": "AQI=", "format": "wav", "filename": "audio.wav"}


@pytest.mark.asyncio
async def test_enabled_async_bridge() -> None:
    rust_bridge.configure_rust_transcription(True, atranscription=AsyncBridge())
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
    assert result == {"text": "async"}


def test_loader_returns_none_without_native_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    rust_bridge.configure_rust_transcription(transcription=None, atranscription=None)
    monkeypatch.setattr("litellm.rust_bridge.get_native_bridge", lambda: None)
    assert rust_bridge.load_rust_transcription() is None
    assert rust_bridge.load_rust_atranscription() is None


def test_dispatch_sync_path_requires_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_bridge, "transcription", lambda **_: None)

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
    async def unavailable(**_: object) -> None:
        return None

    monkeypatch.setattr(rust_bridge, "atranscription", unavailable)

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
        transcription=lambda **_: {"text": "rust"},
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
    async def rust_response(**_: object) -> dict[str, object]:
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
