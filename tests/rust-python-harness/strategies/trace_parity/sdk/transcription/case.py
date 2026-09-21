from __future__ import annotations

import base64
import io
import json
import wave
from typing import Final

from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from .....shared.tracing.steps import Engine, mapping
from ...models import RouteFixture, RouteSpec, TraceScenario, TraceSuite

MAPPINGS: Final = (
    mapping(rust_span="prepare_audio_transcription_provider_call"),
    mapping(span="get_non_default_params", python_frame=r"get_non_default_transcription_params$"),
    mapping(rust_span="map_transcription_params", python_frame=r"get_optional_params_transcription$"),
    mapping(
        span="python_provider_config",
        python_frame=r"ProviderConfigManager\.get_provider_audio_transcription_config$",
    ),
    mapping(rust_span="provider_config"),
    mapping(rust_span="supported_transcription_params"),
    mapping(rust_span="transform_transcription_request"),
    mapping(
        rust_span="execute_audio_transcription_provider_call",
        python_frame=r"BedrockAudioTranscriptionRustDispatch\.(?:async_)?audio_transcriptions$",
    ),
    mapping(rust_span="transform_transcription_response"),
    mapping(rust_span="http_request"),
)

SYNC_MAPPINGS: Final = (
    mapping(rust_span="audio_transcription", python_frame=r"main\.py:\d+ transcription$"),
    *MAPPINGS,
)
ASYNC_MAPPINGS: Final = (
    mapping(rust_span="audio_transcription", python_frame=r"main\.py:\d+ atranscription$"),
    mapping(span="python_transcription_wrapper", python_frame=r"main\.py:\d+ transcription$"),
    *MAPPINGS[:2],
    mapping(span="python_map_transcription_params", python_frame=r"get_optional_params_transcription$"),
    mapping(rust_span="map_transcription_params"),
    *MAPPINGS[3:],
)


def _audio_bytes() -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00" * 1600)
        return buffer.getvalue()


def _fixture(engine: Engine, _base_url: str) -> RouteFixture:
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
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                200, (HttpHeader(name="content-type", value="application/json"),), response
            ),
        ),
    )


SPEC: Final = RouteSpec(
    "transcription",
    ("transcription", "atranscription"),
    ("transcription", "atranscription"),
    _fixture,
)
TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(
            name="bedrock",
            fixture=_fixture,
            mappings=MAPPINGS,
            sync_mappings=SYNC_MAPPINGS,
            async_mappings=ASYNC_MAPPINGS,
        ),
    ),
)
