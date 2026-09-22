"""Live e2e: the OTel v2 Langfuse generation carries output for every non-chat endpoint (LIT-8309).

With LITELLM_OTEL_V2=true the proxy exports one generation per request to the
team's Langfuse destination. Chat, Responses, embeddings and OCR already fill
its output; this file pins the remaining five families. Each test registers a
real OpenAI deployment, drives the endpoint through the shared transport, then
reads the generation back from Langfuse and asserts its output reflects what
the caller received: the completion text, the transcript, the moderation
verdict, and for images and speech a bounded summary that never carries the
raw base64 or audio bytes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from logging_client import LangfuseCreds, LangfuseObservation, LoggingClient, load_langfuse_creds
from models import LiteLLMParamsBody
from pydantic import BaseModel

pytestmark = [pytest.mark.e2e, pytest.mark.otel_v2]

WEATHER_WAV: Final = (
    Path(__file__).resolve().parent.parent / "llm_translation" / "realtime" / "fixtures" / "weather_question_24k.wav"
)
BOUNDED_OUTPUT_CHARS: Final = 1024


class _CompletionBody(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 8
    n: int = 1


class _CompletionChoice(BaseModel):
    text: str = ""


class _CompletionResponse(BaseModel):
    choices: list[_CompletionChoice] = []


class _ImageBody(BaseModel):
    model: str
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    quality: str = "low"


class _ImageDatum(BaseModel):
    url: str | None = None
    b64_json: str | None = None


class _ImageResponse(BaseModel):
    data: list[_ImageDatum] = []


class _SpeechBody(BaseModel):
    model: str
    input: str
    voice: str = "alloy"


class _TranscriptionForm(BaseModel):
    model: str


class _TranscriptionResponse(BaseModel):
    text: str = ""


class _ModerationBody(BaseModel):
    model: str
    input: str


class _ModerationResult(BaseModel):
    flagged: bool


class _ModerationResponse(BaseModel):
    results: list[_ModerationResult] = []


@pytest.fixture(scope="session")
def langfuse_creds() -> LangfuseCreds:
    return load_langfuse_creds()


def _langfuse_key(
    client: LoggingClient, creds: LangfuseCreds, resources: ResourceManager, params: LiteLLMParamsBody
) -> tuple[str, str, str]:
    """A model registered for this run plus a key on a team whose Langfuse callback is `creds`."""
    model: Final = f"e2e-otel-out-{unique_marker()}"
    model_id: Final = client.proxy.create_model(model, params)
    resources.defer(lambda: client.proxy.delete_model(model_id))
    team_id: Final = client.create_team(f"otel-out-team-{unique_marker()}", models=[model])
    resources.defer(lambda: client.delete_team(team_id))
    client.add_team_langfuse_callback(team_id, creds)
    alias: Final = f"otel-out-key-{unique_marker()}"
    key: Final = client.key_with_alias(alias, models=[model], team_id=team_id)
    resources.defer(lambda: client.delete_key(key))
    return model, key, alias


def _generation(client: LoggingClient, creds: LangfuseCreds, *, alias: str, started: datetime) -> LangfuseObservation:
    since: Final = (started - timedelta(seconds=5)).isoformat()
    observation: Final = client.poll_langfuse_generation(creds, key_alias=alias, from_start_time=since)
    assert observation is not None, f"no OTel v2 generation reached Langfuse for key alias {alias!r}"
    return observation


def _output_blob(observation: LangfuseObservation) -> str:
    blob: Final = json.dumps(observation.output, default=str)
    assert observation.output not in (None, "", [], {}), f"generation output is empty: {observation!r}"
    return blob


def _openai(model: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=model, api_key="os.environ/OPENAI_API_KEY")


class TestOtelV2LangfuseGenerationOutput:
    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["completions"])
    def test_completions_output_is_the_completion_text(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        model, key, alias = _langfuse_key(client, langfuse_creds, resources, _openai("openai/gpt-3.5-turbo-instruct"))
        started: Final = datetime.now(timezone.utc)
        response: Final = unwrap(
            client.proxy.transport.post(
                "/v1/completions",
                headers=client.proxy.transport.bearer(key),
                json=_CompletionBody(model=model, prompt=f"Repeat exactly: {unique_marker()}", n=2),
                response_type=_CompletionResponse,
            )
        )
        texts: Final = tuple(choice.text.strip() for choice in response.choices)
        assert len(texts) == 2 and all(texts), f"/v1/completions returned no text: {response!r}"

        blob: Final = _output_blob(_generation(client, langfuse_creds, alias=alias, started=started))
        assert all(text in blob for text in texts), f"generation output lacks the completion texts {texts!r}: {blob}"

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["images_generations"])
    def test_images_output_is_a_bounded_summary_without_base64(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        model, key, alias = _langfuse_key(client, langfuse_creds, resources, _openai("openai/gpt-image-1-mini"))
        started: Final = datetime.now(timezone.utc)
        response: Final = unwrap(
            client.proxy.transport.post(
                "/v1/images/generations",
                headers=client.proxy.transport.bearer(key),
                json=_ImageBody(model=model, prompt=f"a plain red square {unique_marker()}"),
                response_type=_ImageResponse,
                timeout=180.0,
            )
        )
        assert response.data, f"/v1/images/generations returned no data: {response!r}"
        encoded: Final = response.data[0].b64_json or ""
        assert encoded, f"expected a b64_json image from gpt-image-1-mini: {response.data[0].url!r}"

        blob: Final = _output_blob(_generation(client, langfuse_creds, alias=alias, started=started))
        assert len(blob) <= BOUNDED_OUTPUT_CHARS, f"image generation output is not bounded ({len(blob)} chars)"
        assert encoded[:64] not in blob, "image generation output leaks the raw base64 payload"
        assert re.search(r"\d+ bytes", blob), f"image generation output lacks the encoded size: {blob}"

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["audio_speech"])
    def test_speech_output_is_a_bounded_summary_without_audio_bytes(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        model, key, alias = _langfuse_key(client, langfuse_creds, resources, _openai("openai/gpt-4o-mini-tts"))
        started: Final = datetime.now(timezone.utc)
        audio: Final = client.proxy.transport.stream_binary(
            "/v1/audio/speech",
            headers=client.proxy.transport.bearer(key),
            json=_SpeechBody(model=model, input=f"hello {unique_marker()}"),
        )
        assert audio.ok and audio.total_bytes > 0, f"/v1/audio/speech returned no audio: {audio!r}"

        blob: Final = _output_blob(_generation(client, langfuse_creds, alias=alias, started=started))
        assert len(blob) <= BOUNDED_OUTPUT_CHARS, f"speech output is not bounded ({len(blob)} chars)"
        assert re.search(r"\d+ bytes", blob), f"speech output lacks the audio size: {blob}"

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["audio_transcriptions"])
    def test_transcription_output_is_the_transcript(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        model, key, alias = _langfuse_key(client, langfuse_creds, resources, _openai("openai/gpt-4o-mini-transcribe"))
        started: Final = datetime.now(timezone.utc)
        response: Final = unwrap(
            client.proxy.transport.upload(
                "/v1/audio/transcriptions",
                headers=client.proxy.transport.bearer(key),
                form=_TranscriptionForm(model=model),
                filename=WEATHER_WAV.name,
                content=WEATHER_WAV.read_bytes(),
                file_content_type="audio/wav",
                response_type=_TranscriptionResponse,
            )
        )
        transcript: Final = response.text.strip()
        assert "weather" in transcript.lower(), f"transcript does not mention the weather: {transcript!r}"

        blob: Final = _output_blob(_generation(client, langfuse_creds, alias=alias, started=started))
        assert transcript in blob, f"generation output lacks the transcript {transcript!r}: {blob}"

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["moderations"])
    def test_moderations_output_is_the_verdict(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        model, key, alias = _langfuse_key(client, langfuse_creds, resources, _openai("openai/omni-moderation-latest"))
        started: Final = datetime.now(timezone.utc)
        response: Final = unwrap(
            client.proxy.transport.post(
                "/v1/moderations",
                headers=client.proxy.transport.bearer(key),
                json=_ModerationBody(model=model, input=f"I will find you and hurt you badly {unique_marker()}"),
                response_type=_ModerationResponse,
            )
        )
        assert response.results and response.results[0].flagged, f"expected a flagged moderation: {response!r}"

        blob: Final = _output_blob(_generation(client, langfuse_creds, alias=alias, started=started))
        assert "flagged" in blob, f"generation output lacks the moderation verdict: {blob}"
