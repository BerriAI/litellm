import importlib

import pytest

import litellm
from litellm.llms.bedrock.audio_transcription import BedrockAudioTranscriptionRustDispatch
from litellm.rust_bridge.request import NativeRequestContext, NativeTranscriptionRequest

rust_bridge = importlib.import_module("litellm.rust_bridge.transcription")


@pytest.fixture(autouse=True)
def native_transcription_setup():
    rust_bridge.configure_rust_transcription(
        transcription=None, atranscription=None, decline=lambda model, custom_llm_provider, **features: None
    )
    yield
    rust_bridge.configure_rust_transcription(transcription=None, atranscription=None, decline=None)


class SyncBridge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        request: NativeTranscriptionRequest,
        *,
        context: NativeRequestContext,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": request.model,
                "audio": request.audio,
                "optional_params": {**request.optional_params, **(request.options.provider_connection or {})},
            }
        )
        return {"text": "hello"}


class AsyncBridge:
    async def __call__(
        self,
        request: NativeTranscriptionRequest,
        *,
        context: NativeRequestContext,
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
    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: None)
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
        transcription=lambda request, *, context: {"text": "rust"},
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
    async def rust_response(request: NativeTranscriptionRequest, *, context: NativeRequestContext) -> dict[str, object]:
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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_public_transcription_discovers_provider_outside_bedrock(asynchronous):
    litellm.rust(True)
    native = SyncBridge()
    rust_bridge.configure_rust_transcription(transcription=native, atranscription=AsyncBridge())
    kwargs = {"model": "openai/whisper-1", "file": ("audio.wav", b"audio", "audio/wav"), "api_key": "key"}
    result = await litellm.atranscription(**kwargs) if asynchronous else litellm.transcription(**kwargs)
    assert result.text == ("async" if asynchronous else "hello")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_public_bedrock_transcription_does_not_read_audio_when_binding_missing(asynchronous):
    from io import BytesIO

    file = BytesIO(b"audio")
    file.seek(2)
    rust_bridge._TRANSCRIPTION.sync.override(None)
    rust_bridge._TRANSCRIPTION.asynchronous.override(None)
    kwargs = {"model": "bedrock/mistral.voxtral-mini-3b-2507", "file": file}

    async def run():
        return await litellm.atranscription(**kwargs) if asynchronous else litellm.transcription(**kwargs)

    with pytest.raises((RuntimeError, litellm.APIConnectionError), match="unavailable"):
        await run()
    assert file.tell() == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["declined", "missing_preflight", "error", "malformed", "cancelled"])
@pytest.mark.asyncio
async def test_public_transcription_fallback_preserves_audio(monkeypatch, asynchronous, outcome):
    import asyncio
    from io import BytesIO
    from types import SimpleNamespace

    from litellm.rust_bridge import bindings
    from litellm.types.utils import TranscriptionResponse

    class Declined(Exception):
        pass

    class Upstream(Exception):
        pass

    monkeypatch.setattr(
        bindings, "get_native_bridge", lambda: SimpleNamespace(RustBridgeDeclined=Declined, RustUpstreamError=Upstream)
    )
    python_calls = []

    def python(**kwargs):
        python_calls.append(kwargs)
        kwargs["logging_obj"].pre_call(input="audio", api_key="key", additional_args={})
        return TranscriptionResponse(text="python")

    monkeypatch.setattr(
        importlib.import_module("litellm.main"),
        "openai_audio_transcriptions",
        SimpleNamespace(audio_transcriptions=python),
    )

    def native(request, *, context, callback_adapter=None):
        if outcome == "declined":
            raise Declined("unsupported audio")
        if outcome == "error":
            raise RuntimeError("provider failed")
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        return {}

    async def anative(request, *, context, callback_adapter=None):
        return native(request, context=context)

    litellm.rust(True)
    rust_bridge.configure_rust_transcription(transcription=native, atranscription=anative)
    if outcome == "missing_preflight":
        rust_bridge._PREFLIGHT.override(None)
    file = BytesIO(b"prefix-audio")
    file.seek(7)

    async def run():
        kwargs = {"model": "openai/whisper-1", "file": file, "api_key": "key", "num_retries": 0}
        return await litellm.atranscription(**kwargs) if asynchronous else litellm.transcription(**kwargs)

    if outcome in {"declined", "missing_preflight"}:
        assert (await run()).text == "python"
        assert len(python_calls) == 1
        if outcome == "declined":
            assert python_calls[0]["audio_file"][1] == b"audio"
        else:
            assert python_calls[0]["audio_file"] is file
    else:
        with pytest.raises(
            asyncio.CancelledError if outcome == "cancelled" else (RuntimeError, KeyError, litellm.APIConnectionError)
        ):
            await run()
        assert python_calls == []
    assert file.tell() == 7


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_public_transcription_preserves_sdk_credentials(monkeypatch, asynchronous):
    monkeypatch.setattr(litellm, "api_key", "sdk-key")
    monkeypatch.setattr(litellm, "api_base", "https://example.test/v1")
    requests = []

    def native(request, *, context, callback_adapter=None):
        requests.append(request)
        return {"text": "hello"}

    async def anative(request, *, context, callback_adapter=None):
        return native(request, context=context)

    litellm.rust(True)
    rust_bridge.configure_rust_transcription(transcription=native, atranscription=anative)
    kwargs = {"model": "openai/whisper-1", "file": ("audio.wav", b"audio", "audio/wav")}
    result = await litellm.atranscription(**kwargs) if asynchronous else litellm.transcription(**kwargs)
    assert result.text == "hello"
    assert len(requests) == 1
    assert requests[0].options.api_key == "sdk-key"
    assert requests[0].options.api_base == "https://example.test/v1"
