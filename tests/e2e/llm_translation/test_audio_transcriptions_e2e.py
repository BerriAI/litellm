"""Live e2e: POST /v1/audio/transcriptions turns speech into text (vendor §9.7 / LIT-4778).

Registers an OpenAI speech-to-text deployment at runtime and uploads a spoken
weather question (the realtime suite's 24kHz WAV fixture) through the real
OpenAI SDK (LIT-4577), asserting the returned transcript is non-empty and
mentions the word it was asked about. Also pins missing file/model negatives on
the shared multipart transport, since the SDK refuses to send them. A model-less
request comes back as one of two 400s depending on whether any wildcard
deployment happens to be registered on the shared proxy, so the assertion
accepts either phrasing and holds both to naming the model as the problem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import UnknownApiError, unwrap
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from pydantic import BaseModel, Field
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

WEATHER_WAV = (
    Path(__file__).resolve().parent / "realtime" / "fixtures" / "weather_question_24k.wav"
)

MISSING_MODEL_PHRASES: Final = ("model=none", "invalid model", "model is required")


class _OptionalTranscriptionForm(BaseModel):
    model: str | None = None
    response_format: str = "json"


class _TranscriptionResult(BaseModel):
    text: str = ""


def _register(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    model = f"e2e-transcribe-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model="openai/gpt-4o-mini-transcribe", api_key="os.environ/OPENAI_API_KEY"
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


class TestAudioTranscriptions:
    @pytest.mark.covers("llm.audio_transcriptions.openai.basic.nonstream.works")
    def test_audio_transcriptions_returns_text(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources)
        client = sdk.openai(key)

        transcription = client.audio.transcriptions.create(
            model=model, file=(WEATHER_WAV.name, WEATHER_WAV.read_bytes(), "audio/wav")
        )
        text = transcription.text.strip()
        assert text, "/audio/transcriptions returned an empty transcript"
        assert "weather" in text.lower(), (
            f"transcript of a spoken weather question does not mention weather: {text!r}"
        )

    @pytest.mark.covers("llm.audio_transcriptions.openai.input_validation.nonstream.works")
    def test_missing_file_returns_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.upload(
            "/v1/audio/transcriptions",
            headers=proxy.transport.bearer(key),
            form=_OptionalTranscriptionForm(model=model),
            filename="empty.wav",
            content=b"",
            file_content_type="audio/wav",
            response_type=_TranscriptionResult,
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
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        _, key = _register(proxy, resources)
        result = proxy.transport.upload(
            "/v1/audio/transcriptions",
            headers=proxy.transport.bearer(key),
            form=_OptionalTranscriptionForm(),
            filename=WEATHER_WAV.name,
            content=WEATHER_WAV.read_bytes(),
            file_content_type="audio/wav",
            response_type=_TranscriptionResult,
        )
        match result:
            case UnknownApiError(status_code=400, body=body):
                lowered: Final = body.lower()
                assert any(phrase in lowered for phrase in MISSING_MODEL_PHRASES), (
                    f"missing model error must name the model as the problem: {body[:300]}"
                )
            case other:
                pytest.fail(f"missing model expected a model-specific 400, got {other!r}")


class _WhisperForm(BaseModel):
    model: str
    response_format: str
    timestamp_granularities: str | None = Field(default=None, serialization_alias="timestamp_granularities[]")


class _TranscriptWord(BaseModel):
    word: str
    start: float
    end: float


class _VerboseTranscription(BaseModel):
    text: str
    words: list[_TranscriptWord] = []


class TestWhisperTranscriptionFormats:
    def _upload[R: BaseModel](
        self, proxy: ProxyClient, resources: ResourceManager, form: _WhisperForm, response_type: type[R]
    ) -> R:
        model_id = proxy.create_model(
            form.model, LiteLLMParamsBody(model="openai/whisper-1", api_key="os.environ/OPENAI_API_KEY")
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        return unwrap(
            proxy.transport.upload(
                "/v1/audio/transcriptions",
                headers=proxy.transport.bearer(resources.key()),
                form=form,
                filename=WEATHER_WAV.name,
                content=WEATHER_WAV.read_bytes(),
                file_content_type="audio/wav",
                response_type=response_type,
            )
        )

    def test_vtt_format_returns_webvtt_transcript(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        form = _WhisperForm(model=f"e2e-whisper-vtt-{unique_marker()}", response_format="vtt")
        transcript = self._upload(proxy, resources, form, _TranscriptionResult)
        assert transcript.text.lstrip().startswith("WEBVTT"), f"vtt transcript is not WebVTT: {transcript.text[:200]!r}"
        assert "weather" in transcript.text.lower(), f"vtt transcript lost the spoken words: {transcript.text!r}"

    def test_verbose_json_returns_word_timestamps(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        form = _WhisperForm(
            model=f"e2e-whisper-verbose-{unique_marker()}",
            response_format="verbose_json",
            timestamp_granularities="word",
        )
        transcript = self._upload(proxy, resources, form, _VerboseTranscription)
        assert "weather" in transcript.text.lower(), f"verbose transcript lost the spoken words: {transcript.text!r}"
        assert transcript.words, f"word timestamps were requested but none came back: {transcript!r}"
        assert all(word.start <= word.end for word in transcript.words), (
            f"word timings out of order: {transcript.words}"
        )
