"""Live e2e: POST /v1/audio/transcriptions turns speech into text (vendor §9.7 / LIT-4778).

Registers an OpenAI speech-to-text deployment at runtime and uploads a spoken
weather question (the realtime suite's 24kHz WAV fixture) as multipart, asserting
the returned transcript is non-empty and mentions the word it was asked about.
Also pins missing file/model negatives.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import Success, UnknownApiError, unwrap
from endpoints_client import EndpointsClient, TranscriptionForm, TranscriptionResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

WEATHER_WAV = (
    Path(__file__).resolve().parent / "realtime" / "fixtures" / "weather_question_24k.wav"
)


class _OptionalTranscriptionForm(BaseModel):
    model: str | None = None
    response_format: str = "json"


def _register(
    endpoints_client: EndpointsClient, resources: ResourceManager
) -> tuple[str, str]:
    model = f"e2e-transcribe-{unique_marker()}"
    model_id = endpoints_client.create_model(
        model,
        LiteLLMParamsBody(
            model="openai/gpt-4o-mini-transcribe", api_key="os.environ/OPENAI_API_KEY"
        ),
    )
    resources.defer(lambda: endpoints_client.delete_model(model_id))
    return model, resources.key()


class TestAudioTranscriptions:
    @pytest.mark.covers("llm.audio_transcriptions.openai.basic.nonstream.works")
    def test_audio_transcriptions_returns_text(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register(endpoints_client, resources)
        result = unwrap(
            endpoints_client.transcribe(
                key, model, filename=WEATHER_WAV.name, content=WEATHER_WAV.read_bytes()
            )
        )
        text = result.text.strip()
        assert text, "/audio/transcriptions returned an empty transcript"
        assert "weather" in text.lower(), (
            f"transcript of a spoken weather question does not mention weather: {text!r}"
        )

    @pytest.mark.covers("llm.audio_transcriptions.openai.input_validation.nonstream.works")
    def test_missing_file_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register(endpoints_client, resources)
        result = endpoints_client.proxy.transport.upload(
            "/v1/audio/transcriptions",
            headers=endpoints_client.proxy.transport.bearer(key),
            form=TranscriptionForm(model=model),
            filename="empty.wav",
            content=b"",
            file_content_type="audio/wav",
            response_type=TranscriptionResult,
        )
        match result:
            case Success():
                pytest.fail("empty audio file must not succeed as a transcript")
            case UnknownApiError(status_code=status):
                assert status in range(400, 600), f"unexpected {status}"
            case _:
                return

    @pytest.mark.covers("llm.audio_transcriptions.openai.input_validation.nonstream.works")
    def test_missing_model_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        _, key = _register(endpoints_client, resources)
        result = endpoints_client.proxy.transport.upload(
            "/v1/audio/transcriptions",
            headers=endpoints_client.proxy.transport.bearer(key),
            form=_OptionalTranscriptionForm(),
            filename=WEATHER_WAV.name,
            content=WEATHER_WAV.read_bytes(),
            file_content_type="audio/wav",
            response_type=TranscriptionResult,
        )
        match result:
            case Success():
                pytest.fail("transcription without model must not succeed")
            case UnknownApiError(status_code=status):
                assert status in range(400, 600), f"unexpected {status}"
            case _:
                return
