from __future__ import annotations

import base64
import io
import json
import wave
from typing import Final

import pytest

from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from .....shared.tracing.steps import Engine, pipeline_issues, pipeline_steps, step
from ..execution import RouteFixture, RouteSpec, collect_trace

STEPS: Final = (
    step("transcription", r"main\.py:\d+ a?transcription$", "audio_transcription"),
    step("prepare_provider_call", rust="prepare_audio_transcription_provider_call"),
    step("get_non_default_params", r"get_non_default_transcription_params$"),
    step("map_params", r"get_optional_params_transcription$", "map_transcription_params"),
    step("get_provider_config", r"ProviderConfigManager\.get_provider_audio_transcription_config$", "provider_config"),
    step("supported_params", rust="supported_transcription_params"),
    step("transform_request", rust="transform_transcription_request"),
    step(
        "execute_provider_call",
        r"BedrockAudioTranscriptionRustDispatch\.(?:async_)?audio_transcriptions$",
        "execute_audio_transcription_provider_call",
    ),
    step("transform_response", rust="transform_transcription_response"),
    step("http_request", rust="http_request"),
)
EDGES: Final = (
    ("transcription", "map_params"),
    ("map_params", "execute_provider_call"),
    ("prepare_provider_call", "transform_request"),
    ("supported_params", "transform_request"),
    ("transform_request", "http_request"),
    ("execute_provider_call", "http_request"),
    ("http_request", "transform_response"),
)


def _audio_bytes() -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00" * 1600)
        return buffer.getvalue()


def _fixture(engine: Engine) -> RouteFixture:
    credentials: Final = {
        "aws_access_key_id": "test-access",
        "aws_secret_access_key": "test-secret",
        "aws_region_name": "us-east-1",
    }
    audio: Final = _audio_bytes()
    payload: Final = (
        {"audio": {"data": base64.b64encode(audio).decode(), "format": "wav"}, "optional_params": credentials}
        if engine == "rust"
        else {"file": ("sample.wav", audio, "audio/wav"), **credentials}
    )
    response: Final = json.dumps(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
        }
    ).encode()
    return RouteFixture(
        kwargs={"model": "bedrock/mistral.voxtral-mini-3b-2507", **payload},
        provider_response=RecordedHttpResponse.from_bytes(
            200, (HttpHeader(name="content-type", value="application/json"),), response
        ),
    )


SPEC: Final = RouteSpec(
    "transcription", ("transcription", "atranscription"), ("transcription", "atranscription"), _fixture
)


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
def test_trace_parity(asynchronous: bool) -> None:
    python: Final = pipeline_steps("python", collect_trace(SPEC, "python", asynchronous=asynchronous), STEPS)
    rust: Final = pipeline_steps("rust", collect_trace(SPEC, "rust", asynchronous=asynchronous), STEPS)

    assert pipeline_issues("python", python, STEPS, EDGES) == ()
    assert pipeline_issues("rust", rust, STEPS, EDGES) == ()
