import importlib
from collections.abc import Iterator
from types import ModuleType
from typing import Final

import pytest

import litellm
from litellm.exceptions import APIError
from litellm.llms.bedrock.audio_transcription import BedrockAudioTranscriptionRustDispatch
from litellm.rust_bridge import configuration
from litellm.rust_bridge.errors import RustRouteDeclinedError, RustRouteUnavailableError

rust_bridge = importlib.import_module("litellm.rust_bridge.transcription")


@pytest.fixture(autouse=True)
def reset_bridge() -> Iterator[None]:
    configuration.reset_rust_configuration()
    rust_bridge.configure_rust_transcription(transcription=None, atranscription=None)
    yield
    configuration.reset_rust_configuration()
    rust_bridge.configure_rust_transcription(transcription=None, atranscription=None)


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


@pytest.mark.parametrize("enabled", (False, True))
def test_enabled_sync_bridge_receives_audio(enabled: bool) -> None:
    configuration.rust(enabled)
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
        python_fallback=None,
    )
    assert result == {"text": "hello"}
    assert bridge.calls[0]["audio"] == {"data": "AQI=", "format": "wav", "filename": "audio.wav"}


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (False, True))
async def test_enabled_async_bridge(enabled: bool) -> None:
    configuration.rust(enabled)
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
        python_fallback=None,
    )
    assert result == {"text": "async"}


def test_loader_returns_none_without_native_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    rust_bridge.configure_rust_transcription(transcription=None, atranscription=None)
    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: None)
    assert (
        rust_bridge.load_rust_transcription(context=configuration.CapabilityContext(provider="openai", model="test"))
        is None
    )
    assert (
        rust_bridge.load_rust_atranscription(context=configuration.CapabilityContext(provider="openai", model="test"))
        is None
    )


def test_dispatch_sync_path_requires_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: None)

    with pytest.raises(RustRouteUnavailableError, match="bridge is unavailable"):
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
    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: None)

    with pytest.raises(RustRouteUnavailableError, match="bridge is unavailable"):
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


class RustBridgeDeclined(Exception):
    pass


class RustUpstreamError(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected", "message"),
    (
        (RustBridgeDeclined("unsupported model"), RustRouteDeclinedError, "declined the request: unsupported model"),
        (RustUpstreamError(429, "rate limited"), APIError, "rate limited"),
    ),
)
async def test_bedrock_transcription_errors_never_fall_back(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: type[Exception], message: str
) -> None:
    native: Final = ModuleType("native")
    setattr(native, "RustBridgeDeclined", RustBridgeDeclined)
    setattr(native, "RustUpstreamError", RustUpstreamError)
    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: native)

    def fail(**_: object) -> dict[str, object]:
        raise error

    async def afail(**_: object) -> dict[str, object]:
        raise error

    rust_bridge.configure_rust_transcription(transcription=fail, atranscription=afail)
    with pytest.raises(expected, match=message):
        rust_bridge.transcription(
            model="model",
            audio={},
            api_key=None,
            api_base=None,
            custom_llm_provider="bedrock",
            extra_headers=None,
            optional_params={},
            timeout=None,
            python_fallback=None,
        )
    with pytest.raises(expected, match=message):
        await rust_bridge.atranscription(
            model="model",
            audio={},
            api_key=None,
            api_base=None,
            custom_llm_provider="bedrock",
            extra_headers=None,
            optional_params={},
            timeout=None,
            python_fallback=None,
        )


@pytest.mark.asyncio
async def test_python_transcription_skips_rust_when_enabled() -> None:
    configuration.rust(True)

    def unexpected(**_: object) -> dict[str, object]:
        pytest.fail("Python provider must not call Rust")

    async def aunexpected(**_: object) -> dict[str, object]:
        pytest.fail("Python provider must not call Rust")

    rust_bridge.configure_rust_transcription(transcription=unexpected, atranscription=aunexpected)
    assert rust_bridge.transcription(
        model="model",
        audio={},
        api_key=None,
        api_base=None,
        custom_llm_provider="openai",
        extra_headers=None,
        optional_params={},
        timeout=None,
        python_fallback=lambda: {"text": "python"},
    ) == {"text": "python"}

    async def python_fallback() -> dict[str, object]:
        return {"text": "python"}

    assert await rust_bridge.atranscription(
        model="model",
        audio={},
        api_key=None,
        api_base=None,
        custom_llm_provider="openai",
        extra_headers=None,
        optional_params={},
        timeout=None,
        python_fallback=python_fallback,
    ) == {"text": "python"}
