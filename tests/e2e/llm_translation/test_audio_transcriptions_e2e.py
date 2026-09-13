"""Live e2e: POST /v1/audio/transcriptions turns speech into text (vendor §9.7 / LIT-4778).

Registers an OpenAI speech-to-text deployment at runtime and uploads a spoken
weather question (the realtime suite's 24kHz WAV fixture) as multipart, asserting
the returned transcript is non-empty and mentions the word it was asked about.
Also pins missing file/model negatives. A model-less request comes back as one of
two 400s depending on whether any wildcard deployment happens to be registered on
the shared proxy, so the assertion accepts either phrasing and holds both to naming
the model as the problem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import UnknownApiError, unwrap
from endpoints_client import EndpointsClient, TranscriptionForm, TranscriptionResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

WEATHER_WAV = (
    Path(__file__).resolve().parent / "realtime" / "fixtures" / "weather_question_24k.wav"
)

MISSING_MODEL_PHRASES: Final = ("model=none", "invalid model", "model is required")


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
            case UnknownApiError(status_code=400, body=body):
                assert "OpenAIException" in body, (
                    f"the rejection must relay the provider's own error rather than a "
                    f"litellm-internal failure that hides why the upload was refused: {body[:300]}"
                )
                assert "invalid_request_error" in body, (
                    f"an unusable upload must be typed as a client input error: {body[:300]}"
                )
            case other:
                pytest.fail(f"empty audio expected a file-specific 400, got {other!r}")

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
            case UnknownApiError(status_code=400, body=body):
                lowered: Final = body.lower()
                assert any(phrase in lowered for phrase in MISSING_MODEL_PHRASES), (
                    f"missing model error must name the model as the problem: {body[:300]}"
                )
            case other:
                pytest.fail(f"missing model expected a model-specific 400, got {other!r}")
