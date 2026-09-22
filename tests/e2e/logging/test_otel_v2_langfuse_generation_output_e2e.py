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

import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from logging_client import LangfuseCreds, LangfuseObservation, LoggingClient, load_langfuse_creds
from models import (
    CompletionBody,
    CompletionResponse,
    ImageGenerationBody,
    ImageGenerationResponse,
    LiteLLMParamsBody,
    ModerationBody,
    ModerationResponse,
    SpeechBody,
    TranscriptionForm,
    TranscriptionResponse,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

pytestmark = [pytest.mark.e2e, pytest.mark.otel_v2]

WEATHER_WAV: Final = (
    Path(__file__).resolve().parent.parent / "llm_translation" / "realtime" / "fixtures" / "weather_question_24k.wav"
)
BOUNDED_OUTPUT_CHARS: Final = 1024


class _OutputMessage(BaseModel):
    """One assistant message of the Langfuse generation output; only the text is read."""

    content: str = ""


_OUTPUT_MESSAGES: Final = TypeAdapter(list[_OutputMessage])


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


def _output_text(observation: LangfuseObservation) -> str:
    assert observation.output not in (None, "", [], {}), f"generation output is empty: {observation!r}"
    try:
        messages: Final = _OUTPUT_MESSAGES.validate_python(observation.output)
    except ValidationError:
        pytest.fail(f"generation output is not a list of assistant messages: {observation!r}")
    assert messages, f"generation output is empty: {observation!r}"
    return "\n".join(message.content for message in messages)


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
                json=CompletionBody(model=model, prompt=f"Repeat exactly: {unique_marker()}", n=2),
                response_type=CompletionResponse,
            )
        )
        texts: Final = tuple(choice.text.strip() for choice in response.choices)
        assert len(texts) == 2 and all(texts), f"/v1/completions returned no text: {response!r}"

        output: Final = _output_text(_generation(client, langfuse_creds, alias=alias, started=started))
        assert all(text in output for text in texts), (
            f"generation output lacks the completion texts {texts!r}: {output!r}"
        )

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
                json=ImageGenerationBody(model=model, prompt=f"a plain red square {unique_marker()}"),
                response_type=ImageGenerationResponse,
                timeout=180.0,
            )
        )
        assert response.data, f"/v1/images/generations returned no data: {response!r}"
        encoded: Final = response.data[0].b64_json or ""
        assert encoded, f"expected a b64_json image from gpt-image-1-mini: {response.data[0].url!r}"
        image_bytes: Final = len(base64.b64decode(encoded))

        output: Final = _output_text(_generation(client, langfuse_creds, alias=alias, started=started))
        assert len(output) <= BOUNDED_OUTPUT_CHARS, f"image generation output is not bounded ({len(output)} chars)"
        assert encoded[:64] not in output, "image generation output leaks the raw base64 payload"
        assert output == f"b64_json image ({image_bytes} bytes)", (
            f"image generation output does not report the {image_bytes} decoded bytes: {output!r}"
        )

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["audio_speech"])
    def test_speech_output_is_a_bounded_summary_without_audio_bytes(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        model, key, alias = _langfuse_key(client, langfuse_creds, resources, _openai("openai/gpt-4o-mini-tts"))
        started: Final = datetime.now(timezone.utc)
        audio: Final = client.proxy.transport.stream_binary(
            "/v1/audio/speech",
            headers=client.proxy.transport.bearer(key),
            json=SpeechBody(model=model, input=f"hello {unique_marker()}"),
        )
        assert audio.ok and audio.total_bytes > 0, f"/v1/audio/speech returned no audio: {audio!r}"

        output: Final = _output_text(_generation(client, langfuse_creds, alias=alias, started=started))
        assert len(output) <= BOUNDED_OUTPUT_CHARS, f"speech output is not bounded ({len(output)} chars)"
        assert output.endswith(f" ({audio.total_bytes} bytes)"), (
            f"speech output does not report the {audio.total_bytes} audio bytes the caller received: {output!r}"
        )

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
                form=TranscriptionForm(model=model),
                filename=WEATHER_WAV.name,
                content=WEATHER_WAV.read_bytes(),
                file_content_type="audio/wav",
                response_type=TranscriptionResponse,
            )
        )
        transcript: Final = response.text.strip()
        assert transcript, f"/v1/audio/transcriptions returned no text: {response!r}"

        output: Final = _output_text(_generation(client, langfuse_creds, alias=alias, started=started))
        assert transcript in output, f"generation output lacks the transcript {transcript!r}: {output!r}"

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
                json=ModerationBody(model=model, input=f"I will find you and hurt you badly {unique_marker()}"),
                response_type=ModerationResponse,
            )
        )
        assert response.results, f"/v1/moderations returned no results: {response!r}"
        verdict: Final = "flagged: " if response.results[0].flagged else "not flagged"

        output: Final = _output_text(_generation(client, langfuse_creds, alias=alias, started=started))
        assert output.startswith(verdict), (
            f"generation output does not carry the moderation verdict {verdict!r}: {output!r}"
        )
