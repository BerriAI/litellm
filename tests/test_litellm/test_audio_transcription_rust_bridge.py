from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.llms.bedrock.audio_transcription import BedrockAudioTranscriptionRustDispatch
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge.transcription import NATIVE_ATRANSCRIPTION, NATIVE_TRANSCRIPTION

MODEL: Final = "bedrock/mistral.voxtral-mini-3b-2507"
AUDIO_FILE: Final = ("audio.wav", b"audio", "audio/wav")


class RustBridgeDeclined(Exception):
    pass


class RustUpstreamError(Exception):
    pass


@pytest.fixture(autouse=True)
def isolated_bridge(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    native: Final = SimpleNamespace(RustBridgeDeclined=RustBridgeDeclined, RustUpstreamError=RustUpstreamError)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    NATIVE_TRANSCRIPTION.reset()
    NATIVE_ATRANSCRIPTION.reset()
    configuration.reset_rust_configuration()


class SyncBridge:
    def __init__(self, effect: BaseException | None = None) -> None:
        self._effect: Final = effect
        self.calls: tuple[dict[str, object], ...] = ()

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
        self.calls = (
            *self.calls,
            {"model": model, "audio": audio, "provider": custom_llm_provider, "timeout": timeout_seconds},
        )
        if self._effect is not None:
            raise self._effect
        return {"text": "rust"}


class AsyncBridge:
    def __init__(self) -> None:
        self.calls: tuple[str, ...] = ()

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
        self.calls = (*self.calls, model)
        return {"text": "async rust"}


def dispatch_sync() -> litellm.TranscriptionResponse:
    return BedrockAudioTranscriptionRustDispatch().audio_transcriptions(
        model=MODEL,
        audio_file=AUDIO_FILE,
        api_key=None,
        api_base=None,
        custom_llm_provider="bedrock",
        extra_headers=None,
        optional_params={"temperature": 0},
        timeout=5,
    )


def test_dispatch_marshals_audio_into_rust_call() -> None:
    bridge: Final = SyncBridge()
    NATIVE_TRANSCRIPTION.override(bridge)

    response: Final = dispatch_sync()

    assert response.text == "rust"
    assert bridge.calls == (
        {
            "model": MODEL,
            "audio": {"data": "YXVkaW8=", "format": "wav", "filename": "audio.wav"},
            "provider": "bedrock",
            "timeout": 5.0,
        },
    )


@pytest.mark.parametrize("disable", ("process", "environment"))
def test_bedrock_transcription_ignores_optional_rust_switches(disable: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if disable == "process":
        litellm.rust(False)
    else:
        monkeypatch.setenv("LITELLM_RUST", "0")
    bridge: Final = SyncBridge()
    NATIVE_TRANSCRIPTION.override(bridge)

    assert dispatch_sync().text == "rust"
    assert len(bridge.calls) == 1


def test_missing_native_binding_raises_without_python_fallback() -> None:
    NATIVE_TRANSCRIPTION.override(None)

    with pytest.raises(RuntimeError, match="bridge is unavailable"):
        dispatch_sync()


def test_admission_decline_raises_for_required_route() -> None:
    NATIVE_TRANSCRIPTION.override(SyncBridge(RustBridgeDeclined("unsupported format")))

    with pytest.raises(RuntimeError, match="declined the request: unsupported format"):
        dispatch_sync()


def test_upstream_error_maps_to_api_error() -> None:
    NATIVE_TRANSCRIPTION.override(SyncBridge(RustUpstreamError(503, "bedrock down")))

    with pytest.raises(litellm.APIError, match="bedrock down") as raised:
        dispatch_sync()
    assert raised.value.status_code == 503


def test_bedrock_transcription_dispatches_to_rust_from_sdk_entrypoint() -> None:
    bridge: Final = SyncBridge()
    NATIVE_TRANSCRIPTION.override(bridge)

    response: Final = litellm.transcription(model=MODEL, file=AUDIO_FILE)

    assert isinstance(response, litellm.TranscriptionResponse)
    assert response.text == "rust"
    assert bridge.calls[0]["model"] == MODEL.removeprefix("bedrock/")


@pytest.mark.asyncio
async def test_bedrock_atranscription_dispatches_to_rust_from_sdk_entrypoint() -> None:
    bridge: Final = AsyncBridge()
    NATIVE_ATRANSCRIPTION.override(bridge)

    response: Final = await litellm.atranscription(model=MODEL, file=AUDIO_FILE)

    assert response.text == "async rust"
    assert bridge.calls == (MODEL.removeprefix("bedrock/"),)


@pytest.mark.asyncio
async def test_async_missing_native_binding_raises_without_python_fallback() -> None:
    NATIVE_ATRANSCRIPTION.override(None)

    with pytest.raises(RuntimeError, match="bridge is unavailable"):
        await BedrockAudioTranscriptionRustDispatch().async_audio_transcriptions(
            model=MODEL,
            audio_file=AUDIO_FILE,
            api_key=None,
            api_base=None,
            custom_llm_provider="bedrock",
            extra_headers=None,
            optional_params={},
            timeout=None,
        )
