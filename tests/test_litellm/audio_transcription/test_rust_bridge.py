import importlib

import pytest

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


def test_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    rust_bridge.configure_rust_transcription(False, transcription=None)
    assert rust_bridge.rust_transcription_enabled() is False


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
    rust_bridge.configure_rust_transcription(True, transcription=None, atranscription=None)
    monkeypatch.setattr("litellm.rust_bridge.get_native_bridge", lambda: None)
    assert rust_bridge.load_rust_transcription() is None
    assert rust_bridge.load_rust_atranscription() is None
