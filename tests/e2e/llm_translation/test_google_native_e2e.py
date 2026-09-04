from __future__ import annotations

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from endpoints_client import EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

UPSTREAM_MODEL = "gemini/gemini-2.5-flash"


class _StreamPart(BaseModel):
    text: str | None = None


class _StreamContent(BaseModel):
    parts: tuple[_StreamPart, ...] = ()


class _StreamCandidate(BaseModel):
    content: _StreamContent | None = None


class _StreamEvent(BaseModel):
    candidates: tuple[_StreamCandidate, ...] = ()


def _managed_deployment(client: EndpointsClient, resources: ResourceManager) -> str:
    model = f"e2e-google-native-{unique_marker()}"
    model_id = client.create_model(
        model,
        LiteLLMParamsBody(model=UPSTREAM_MODEL, api_key="os.environ/GEMINI_API_KEY"),
    )
    resources.defer(lambda: client.delete_model(model_id))
    return model


def _streamed_text(result: StreamingResponse) -> str:
    return "".join(
        part.text
        for event in result.stream_events
        for candidate in _StreamEvent.model_validate_json(event).candidates
        for part in (candidate.content.parts if candidate.content else ())
        if part.text
    )


class TestGoogleNativeGenerateContent:
    @pytest.mark.covers("llm.google_native.gemini.basic.nonstream.cost_logged")
    def test_generate_content_returns_response_cost_header(
        self,
        endpoints_client: EndpointsClient,
        resources: ResourceManager,
        scoped_key: str,
    ) -> None:
        model = _managed_deployment(endpoints_client, resources)

        result = endpoints_client.generate_content(
            scoped_key, model, f"Reply with the single word ok. {unique_marker()}"
        )

        require_successful_call(result)
        assert result.call_id, "generateContent must stamp x-litellm-call-id"
        assert result.response_cost is not None, (
            "generateContent returned no x-litellm-response-cost header; "
            "google-native traffic cannot be reconciled against spend without it"
        )
        assert result.response_cost > 0, f"x-litellm-response-cost must be a real cost, got {result.response_cost}"

    @pytest.mark.covers("llm.google_native.gemini.basic.stream.works")
    def test_stream_generate_content_frames_sse_the_way_google_sdks_expect(
        self,
        endpoints_client: EndpointsClient,
        resources: ResourceManager,
        scoped_key: str,
    ) -> None:
        model = _managed_deployment(endpoints_client, resources)

        result = endpoints_client.generate_content(
            scoped_key,
            model,
            f"Count from one to five, one number per line. {unique_marker()}",
            stream=True,
        )

        require_successful_call(result)
        assert result.is_streaming, f"expected text/event-stream, got content-type {result.content_type!r}"
        assert result.stream_error is None, f"stream carried an error: {result.stream_error}"
        assert result.stream_events, f"stream delivered no data events (chunks={result.chunks})"

        doubled = tuple(event for event in result.stream_events if event.lstrip().startswith("data:"))
        assert not doubled, (
            f"{len(doubled)} event(s) carry a second data: prefix, so the proxy re-wrapped "
            f"already-framed SSE; first offender: {doubled[0][:120]!r}"
        )
        leaked = tuple(event for event in result.stream_events if event.startswith("b'"))
        assert not leaked, f"event serialized as a Python bytes literal instead of text: {leaked[0][:120]!r}"
        assert _streamed_text(result).strip(), "stream delivered events but no candidate text"
        assert not result.stream_done, (
            "google-native stream emitted the OpenAI [DONE] sentinel; Google never sends it "
            "and the Vertex Java SDK rejects the stream when it appears"
        )
