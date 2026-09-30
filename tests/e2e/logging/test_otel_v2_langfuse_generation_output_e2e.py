"""Live e2e: the OTel v2 Langfuse generation carries input and output for every non-chat endpoint.

With LITELLM_OTEL_V2=true the proxy exports one generation per request to the
team's Langfuse destination. Chat, Responses and embeddings already fill both
panels; this file pins the other families. LIT-8309 covers the output of
completions, images, speech, transcription and moderation; LIT-8326 covers the
rerank and search output and the OCR, image-edit and search input, which used
to read the `default-message-value` placeholder. Each test registers a real
deployment, drives the endpoint through the shared transport, then reads the
generation back from Langfuse and asserts it reflects what the caller sent and
received: the completion text, the transcript, the moderation verdict, the
ranked rerank indices and scores, the search results, the OCR document URL,
the edit prompt, and for images, speech and uploaded documents a bounded
summary that never carries the raw base64 or audio bytes.
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
    ImageEditForm,
    ImageGenerationBody,
    ImageGenerationResponse,
    LiteLLMParamsBody,
    ModerationBody,
    ModerationResponse,
    OcrBody,
    OcrDocument,
    OcrForm,
    OcrResponse,
    RerankBody,
    SearchBody,
    SearchResponse,
    SearchToolBody,
    SearchToolCreateBody,
    SearchToolLiteLLMParamsBody,
    SpeechBody,
    TranscriptionForm,
    TranscriptionResponse,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

pytestmark = [pytest.mark.e2e, pytest.mark.otel_v2]

WEATHER_WAV: Final = (
    Path(__file__).resolve().parent.parent / "llm_translation" / "realtime" / "fixtures" / "weather_question_24k.wav"
)
DUMMY_PDF: Final = Path(__file__).resolve().parent.parent.parent / "llm_translation" / "fixtures" / "dummy.pdf"
DUMMY_PDF_URL: Final = (
    "https://cdn.jsdelivr.net/gh/BerriAI/litellm"
    "@d769e81c90d453240c61fc572cdb27fae06a89d0"
    "/tests/llm_translation/fixtures/dummy.pdf"
)
RED_SQUARE_PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PMQ0AAAwDoPo3"
    "3UrYvQQckD4XAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AYHLAMpT0sIcNbcEAAAAAElFTkSuQmCC"
)
BOUNDED_OUTPUT_CHARS: Final = 1024
PLACEHOLDER_INPUT: Final = "default-message-value"


class _OutputMessage(BaseModel):
    """One assistant message of the Langfuse generation output; only the text is read."""

    content: str = ""


class _InputMessage(BaseModel):
    """One user message of the Langfuse generation input; only the text is read."""

    content: str = ""


_OUTPUT_MESSAGES: Final = TypeAdapter(list[_OutputMessage])
_INPUT_MESSAGES: Final = TypeAdapter(list[_InputMessage])


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


def _input_text(observation: LangfuseObservation) -> str:
    assert observation.input not in (None, "", [], {}), f"generation input is empty: {observation!r}"
    try:
        messages: Final = _INPUT_MESSAGES.validate_python(observation.input)
    except ValidationError:
        pytest.fail(f"generation input is not a list of user messages: {observation!r}")
    assert messages, f"generation input is empty: {observation!r}"
    text: Final = "\n".join(message.content for message in messages)
    assert text != PLACEHOLDER_INPUT, f"generation input is the placeholder, not the request: {observation!r}"
    return text


def _openai(model: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=model, api_key="os.environ/OPENAI_API_KEY")


def _mistral_ocr() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model="mistral/mistral-ocr-latest", api_key="os.environ/MISTRAL_API_KEY")


def _langfuse_search_tool(
    client: LoggingClient, creds: LangfuseCreds, resources: ResourceManager
) -> tuple[str, str, str]:
    """A keyless DuckDuckGo search tool registered for this run, plus a key on a team whose Langfuse callback is
    `creds`."""
    tool: Final = f"e2e-otel-search-{unique_marker()}"
    tool_id: Final = client.proxy.create_search_tool(
        SearchToolCreateBody(
            search_tool=SearchToolBody(
                search_tool_name=tool,
                litellm_params=SearchToolLiteLLMParamsBody(search_provider="duckduckgo"),
            )
        )
    )
    resources.defer(lambda: client.proxy.delete_search_tool(tool_id))
    team_id: Final = client.create_team(f"otel-search-team-{unique_marker()}", models=[tool])
    resources.defer(lambda: client.delete_team(team_id))
    client.add_team_langfuse_callback(team_id, creds)
    alias: Final = f"otel-search-key-{unique_marker()}"
    key: Final = client.key_with_alias(alias, models=[tool], team_id=team_id)
    resources.defer(lambda: client.delete_key(key))
    return tool, key, alias


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

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["rerank"])
    def test_rerank_output_is_the_ranked_indices_and_scores(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        model, key, alias = _langfuse_key(
            client,
            langfuse_creds,
            resources,
            LiteLLMParamsBody(model="cohere/rerank-v4.0-fast", api_key="os.environ/COHERE_API_KEY"),
        )
        query: Final = f"What is the capital of France? {unique_marker()}"
        started: Final = datetime.now(timezone.utc)
        response: Final = unwrap(
            client.proxy.rerank(
                key,
                RerankBody(
                    model=model,
                    query=query,
                    documents=["Paris is the capital of France.", "Berlin is in Germany.", "Bananas are yellow."],
                    top_n=2,
                ),
            )
        )
        ranked: Final = tuple(f"[{item.index}] {item.relevance_score}" for item in response.results)
        assert len(ranked) == 2 and all(item.index is not None for item in response.results), (
            f"/v1/rerank returned no ranked results: {response!r}"
        )

        generation: Final = _generation(client, langfuse_creds, alias=alias, started=started)
        assert _input_text(generation) == query
        output: Final = _output_text(generation)
        assert output == "\n\n".join(ranked), f"generation output is not the ranked indices and scores: {output!r}"

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["ocr"])
    def test_ocr_input_is_the_document_url(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        model, key, alias = _langfuse_key(client, langfuse_creds, resources, _mistral_ocr())
        started: Final = datetime.now(timezone.utc)
        response: Final = unwrap(
            client.proxy.ocr(
                key, OcrBody(model=model, document=OcrDocument(type="document_url", document_url=DUMMY_PDF_URL))
            )
        )
        assert response.pages and response.pages[0].markdown, f"/v1/ocr returned no page markdown: {response!r}"

        generation: Final = _generation(client, langfuse_creds, alias=alias, started=started)
        assert _input_text(generation) == DUMMY_PDF_URL
        assert response.pages[0].markdown in _output_text(generation)

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["ocr"])
    def test_ocr_upload_input_is_a_bounded_document_summary_without_base64(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        model, key, alias = _langfuse_key(client, langfuse_creds, resources, _mistral_ocr())
        pdf: Final = DUMMY_PDF.read_bytes()
        started: Final = datetime.now(timezone.utc)
        response: Final = unwrap(
            client.proxy.transport.upload(
                "/v1/ocr",
                headers=client.proxy.transport.bearer(key),
                form=OcrForm(model=model),
                filename=DUMMY_PDF.name,
                content=pdf,
                file_content_type="application/pdf",
                response_type=OcrResponse,
            )
        )
        assert response.pages and response.pages[0].markdown, f"/v1/ocr returned no page markdown: {response!r}"

        text: Final = _input_text(_generation(client, langfuse_creds, alias=alias, started=started))
        encoded: Final = base64.b64encode(pdf).decode()
        assert text == f"data:application/pdf;base64 ({len(encoded)} chars)", (
            f"OCR upload input is not the bounded document summary: {text!r}"
        )

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["images_edits"])
    def test_image_edit_input_is_the_edit_prompt(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        model, key, alias = _langfuse_key(client, langfuse_creds, resources, _openai("openai/gpt-image-1-mini"))
        prompt: Final = f"make the square blue {unique_marker()}"
        started: Final = datetime.now(timezone.utc)
        response: Final = unwrap(
            client.proxy.transport.upload(
                "/v1/images/edits",
                headers=client.proxy.transport.bearer(key),
                form=ImageEditForm(model=model, prompt=prompt),
                filename="red_square.png",
                content=RED_SQUARE_PNG,
                file_content_type="image/png",
                file_field="image",
                response_type=ImageGenerationResponse,
                timeout=180.0,
            )
        )
        assert response.data and (response.data[0].b64_json or response.data[0].url), (
            f"/v1/images/edits returned no image: {response!r}"
        )

        generation: Final = _generation(client, langfuse_creds, alias=alias, started=started)
        assert _input_text(generation) == prompt
        assert _output_text(generation).startswith("b64_json image (")

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["search"])
    def test_search_input_is_the_query_and_output_the_results(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        tool, key, alias = _langfuse_search_tool(client, langfuse_creds, resources)
        query: Final = "Eiffel Tower"
        started: Final = datetime.now(timezone.utc)
        response: Final = unwrap(
            client.proxy.transport.post(
                f"/v1/search/{tool}",
                headers=client.proxy.transport.bearer(key),
                json=SearchBody(query=query, max_results=2),
                response_type=SearchResponse,
            )
        )
        assert response.results and all(result.url for result in response.results), (
            f"/v1/search returned no results with a url: {response!r}"
        )

        generation: Final = _generation(client, langfuse_creds, alias=alias, started=started)
        assert _input_text(generation) == query
        output: Final = _output_text(generation)
        assert output == "\n\n".join(
            "\n".join(part for part in (result.title, result.url, result.snippet) if part)
            for result in response.results
        ), f"generation output is not the search results the caller received: {output!r}"
