from __future__ import annotations

from collections.abc import Generator
from typing import Final

import pytest

import litellm
from litellm.llms.bedrock.audio_transcription import BedrockAudioTranscriptionRustDispatch
from litellm.rust_bridge import configuration
from litellm.rust_bridge._native import RustBridgeDeclined, RustUpstreamError
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.transcription.native import NATIVE_ATRANSCRIPTION, NATIVE_TRANSCRIPTION
from litellm.types.utils import TranscriptionUsageTokensObject

MODEL: Final = "bedrock/mistral.voxtral-mini-3b-2507"
AUDIO_FILE: Final = ("audio.wav", b"audio", "audio/wav")
TRANSCRIPTION_FIELDS: Final = frozenset(
    {
        "model",
        "audio",
        "api_key",
        "api_base",
        "custom_llm_provider",
        "extra_headers",
        "optional_params",
        "timeout_seconds",
    }
)


@pytest.fixture(autouse=True)
def isolated_bridge(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    NATIVE_TRANSCRIPTION.reset()
    NATIVE_ATRANSCRIPTION.reset()
    configuration.reset_rust_configuration()


class SyncBridge:
    def __init__(self, effect: BaseException | None = None) -> None:
        self._effect: Final = effect
        self.calls: tuple[NativeCall, ...] = ()

    def __call__(self, call: NativeCall) -> dict[str, object]:
        self.calls = (*self.calls, call)
        if self._effect is not None:
            raise self._effect
        return {"text": "rust"}


class AsyncBridge:
    def __init__(self) -> None:
        self.calls: tuple[NativeCall, ...] = ()

    async def __call__(self, call: NativeCall) -> dict[str, object]:
        self.calls = (*self.calls, call)
        return {"text": "async rust"}


class UsageBridge:
    def __call__(self, call: NativeCall) -> dict[str, object]:
        return {
            "text": "rust",
            "usage": {"type": "tokens", "input_tokens": 390, "output_tokens": 60, "total_tokens": 450},
        }


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

    expected: Final = {
        "model": MODEL,
        "audio": {"data": "YXVkaW8=", "format": "wav", "filename": "audio.wav"},
        "api_key": None,
        "api_base": None,
        "custom_llm_provider": "bedrock",
        "extra_headers": None,
        "optional_params": {"temperature": 0},
        "timeout_seconds": 5.0,
    }
    assert response.text == "rust"
    assert bridge.calls == (NativeCall(args=(), kwargs=expected, base={}),)


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


@pytest.mark.asyncio
async def test_async_dispatch_marshals_audio_into_rust_call() -> None:
    bridge: Final = AsyncBridge()
    NATIVE_ATRANSCRIPTION.override(bridge)

    response: Final = await BedrockAudioTranscriptionRustDispatch().async_audio_transcriptions(
        model=MODEL,
        audio_file=AUDIO_FILE,
        api_key=None,
        api_base=None,
        custom_llm_provider="bedrock",
        extra_headers=None,
        optional_params={"temperature": 0},
        timeout=5,
    )

    expected: Final = {
        "model": MODEL,
        "audio": {"data": "YXVkaW8=", "format": "wav", "filename": "audio.wav"},
        "api_key": None,
        "api_base": None,
        "custom_llm_provider": "bedrock",
        "extra_headers": None,
        "optional_params": {"temperature": 0},
        "timeout_seconds": 5.0,
    }
    assert response.text == "async rust"
    assert bridge.calls == (NativeCall(args=(), kwargs=expected, base={}),)


def test_bedrock_transcription_dispatches_to_rust_from_sdk_entrypoint() -> None:
    bridge: Final = SyncBridge()
    NATIVE_TRANSCRIPTION.override(bridge)

    response: Final = litellm.transcription(model=MODEL, file=AUDIO_FILE)

    assert isinstance(response, litellm.TranscriptionResponse)
    assert response.text == "rust"
    assert bridge.calls[0].resolved["model"] == MODEL.removeprefix("bedrock/")
    assert frozenset(bridge.calls[0].resolved) == TRANSCRIPTION_FIELDS


@pytest.mark.asyncio
async def test_bedrock_atranscription_dispatches_to_rust_from_sdk_entrypoint() -> None:
    bridge: Final = AsyncBridge()
    NATIVE_ATRANSCRIPTION.override(bridge)

    response: Final = await litellm.atranscription(model=MODEL, file=AUDIO_FILE)

    assert response.text == "async rust"
    assert tuple(call.resolved["model"] for call in bridge.calls) == (MODEL.removeprefix("bedrock/"),)
    assert frozenset(bridge.calls[0].resolved) == TRANSCRIPTION_FIELDS


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


@pytest.mark.parametrize(
    ("region_kwargs", "region_env", "priced_model"),
    [
        ({}, {}, MODEL),
        ({"aws_region_name": "ap-southeast-2"}, {}, "bedrock/ap-southeast-2/mistral.voxtral-mini-3b-2507"),
        ({}, {"AWS_REGION_NAME": "eu-west-2"}, "bedrock/eu-west-2/mistral.voxtral-mini-3b-2507"),
        ({}, {"AWS_REGION": "ap-northeast-1"}, "bedrock/ap-northeast-1/mistral.voxtral-mini-3b-2507"),
        (
            {"aws_region_name": "ap-southeast-2"},
            {"AWS_REGION_NAME": "eu-west-2"},
            "bedrock/ap-southeast-2/mistral.voxtral-mini-3b-2507",
        ),
    ],
)
def test_bedrock_transcription_bills_the_converse_token_usage(
    region_kwargs: dict[str, str],
    region_env: dict[str, str],
    priced_model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("AWS_REGION_NAME", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    for name, value in region_env.items():
        monkeypatch.setenv(name, value)
    NATIVE_TRANSCRIPTION.override(UsageBridge())

    response: Final = litellm.transcription(model=MODEL, file=AUDIO_FILE, **region_kwargs)

    model_info: Final = litellm.get_model_info(priced_model)
    expected_cost: Final = 390 * (model_info.get("input_cost_per_token") or 0.0) + 60 * (
        model_info.get("output_cost_per_token") or 0.0
    )
    assert response.usage == TranscriptionUsageTokensObject(
        type="tokens", input_tokens=390, output_tokens=60, total_tokens=450
    )
    assert expected_cost > 0
    assert litellm.completion_cost(
        completion_response=response, model=MODEL, call_type="transcription"
    ) == pytest.approx(expected_cost)


def test_bedrock_transcription_forwards_language_and_text_formats_to_rust() -> None:
    bridge: Final = SyncBridge()
    NATIVE_TRANSCRIPTION.override(bridge)

    litellm.transcription(model=MODEL, file=AUDIO_FILE, language="fr", response_format="text", prompt="Names")

    forwarded: Final = bridge.calls[0].resolved["optional_params"]
    assert {key: forwarded.get(key) for key in ("language", "response_format", "prompt")} == {
        "language": "fr",
        "response_format": "text",
        "prompt": "Names",
    }


@pytest.mark.parametrize(("sent", "expected"), [("0.2", 0.2), ("0", 0.0), (0.5, 0.5), (1, 1.0)])
def test_bedrock_transcription_forwards_form_temperature_as_a_number(sent: object, expected: float) -> None:
    bridge: Final = SyncBridge()
    NATIVE_TRANSCRIPTION.override(bridge)

    litellm.transcription(model=MODEL, file=AUDIO_FILE, temperature=sent)

    forwarded: Final = bridge.calls[0].resolved["optional_params"]["temperature"]
    assert (type(forwarded), forwarded) == (float, expected)


@pytest.mark.parametrize(
    "unsupported",
    [
        {"response_format": "srt"},
        {"response_format": "verbose_json"},
        {"timestamp_granularities": ["word"]},
        {"temperature": "warm"},
        {"temperature": "inf"},
        {"temperature": True},
    ],
)
def test_bedrock_transcription_rejects_params_it_cannot_honor(unsupported: dict[str, object]) -> None:
    bridge: Final = SyncBridge()
    NATIVE_TRANSCRIPTION.override(bridge)

    with pytest.raises(litellm.BadRequestError) as raised:
        litellm.transcription(model=MODEL, file=AUDIO_FILE, language="fr", **unsupported)

    assert raised.value.status_code == 400
    assert next(iter(unsupported)) in str(raised.value)
    assert bridge.calls == ()

    dropped: Final = litellm.transcription(model=MODEL, file=AUDIO_FILE, language="fr", drop_params=True, **unsupported)

    assert dropped.text == "rust"
    forwarded: Final = bridge.calls[0].resolved["optional_params"]
    assert forwarded["language"] == "fr"
    assert forwarded.keys().isdisjoint(unsupported)
