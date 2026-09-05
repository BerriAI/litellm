import importlib
from collections.abc import Generator

import pytest

import litellm

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
        callback_adapter: object | None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "audio": audio,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "optional_params": optional_params,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"text": "hello"}


class AsyncBridge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
        callback_adapter: object | None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "audio": audio,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "optional_params": optional_params,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"text": "async"}


@pytest.fixture(autouse=True)
def reset_transcription_bridge() -> Generator[None]:
    rust_bridge._TRANSCRIPTION.reset()
    yield
    rust_bridge._TRANSCRIPTION.reset()


def test_enabled_sync_bridge_receives_audio() -> None:
    bridge = SyncBridge()
    rust_bridge.set_rust_transcription(sync=bridge)
    result = rust_bridge.dispatch_transcription(
        prepare=lambda: rust_bridge.NativeTranscriptionRequest(
            model="mistral.voxtral-mini-3b-2507",
            audio={"data": "AQI=", "format": "wav", "filename": "audio.wav"},
            api_key=None,
            api_base=None,
            custom_llm_provider="bedrock",
            extra_headers=None,
            optional_params={"temperature": 0},
            timeout=5.0,
        ),
        model="mistral.voxtral-mini-3b-2507",
        provider="bedrock",
    )
    assert result == {"text": "hello"}
    assert bridge.calls == [
        {
            "model": "mistral.voxtral-mini-3b-2507",
            "audio": {"data": "AQI=", "format": "wav", "filename": "audio.wav"},
            "api_key": None,
            "api_base": None,
            "custom_llm_provider": "bedrock",
            "extra_headers": None,
            "optional_params": {"temperature": 0},
            "timeout_seconds": 5.0,
        }
    ]


@pytest.mark.asyncio
async def test_enabled_async_bridge() -> None:
    bridge = AsyncBridge()
    rust_bridge.set_rust_transcription(asynchronous=bridge)
    result = await rust_bridge.adispatch_transcription(
        prepare=lambda: rust_bridge.NativeTranscriptionRequest(
            model="mistral.voxtral-mini-3b-2507",
            audio={"data": "AQI=", "format": "wav", "filename": "audio.wav"},
            api_key=None,
            api_base=None,
            custom_llm_provider="bedrock",
            extra_headers=None,
            optional_params={},
            timeout=None,
        ),
        model="mistral.voxtral-mini-3b-2507",
        provider="bedrock",
    )
    assert result == {"text": "async"}
    assert bridge.calls == [
        {
            "model": "mistral.voxtral-mini-3b-2507",
            "audio": {"data": "AQI=", "format": "wav", "filename": "audio.wav"},
            "api_key": None,
            "api_base": None,
            "custom_llm_provider": "bedrock",
            "extra_headers": None,
            "optional_params": {},
            "timeout_seconds": None,
        }
    ]


def test_bedrock_transcription_uses_rust_only_path() -> None:
    rust_bridge.set_rust_transcription(
        sync=lambda **_: {"text": "rust"},
        asynchronous=None,
    )
    try:
        response = litellm.transcription(
            model="bedrock/mistral.voxtral-mini-3b-2507",
            file=("audio.wav", b"audio", "audio/wav"),
        )
    finally:
        rust_bridge.set_rust_transcription(sync=None, asynchronous=None)

    assert response.text == "rust"


@pytest.mark.asyncio
async def test_bedrock_atranscription_uses_rust_only_path() -> None:
    async def rust_response(**_: object) -> dict[str, object]:
        return {"text": "rust"}

    rust_bridge.set_rust_transcription(sync=None, asynchronous=rust_response)
    try:
        response = await litellm.atranscription(
            model="bedrock/mistral.voxtral-mini-3b-2507",
            file=("audio.wav", b"audio", "audio/wav"),
        )
    finally:
        rust_bridge.set_rust_transcription(sync=None, asynchronous=None)

    assert response.text == "rust"
