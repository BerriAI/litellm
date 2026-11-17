from __future__ import annotations

import base64
import io
import json
import wave
from typing import Final

from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from ...models import RouteFixture, RouteSpec, TraceScenario, TraceSuite


def _audio_bytes() -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00" * 1600)
        return buffer.getvalue()


def _fixture(_base_url: str) -> RouteFixture:
    credentials: Final = {
        "aws_access_key_id": "test-access",
        "aws_secret_access_key": "test-secret",
        "aws_region_name": "us-east-1",
    }
    audio: Final = _audio_bytes()
    payload: Final = {"file": ("sample.wav", audio, "audio/wav"), **credentials}
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
    _fixture,
)
TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(
            name="sync-bedrock",
            fixture=_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-bedrock",
            fixture=_fixture,
            asynchronous=True,
        ),
    ),
)
