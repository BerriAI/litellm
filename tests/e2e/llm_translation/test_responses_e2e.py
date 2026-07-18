"""Live e2e: POST /v1/responses returns a real completion.

Registers an OpenAI deployment at runtime, drives the Responses API through the
gateway, and asserts output text came back. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import (
    EndpointsClient,
    ResponsesCompletedEvent,
    ResponsesOutputTextDeltaEvent,
    ResponsesResult,
)
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e


class TestResponses:
    @pytest.mark.covers("llm.responses.openai.basic.nonstream.works")
    def test_responses_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-responses-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses(key, model, "reply with one word")
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        assert parsed.text.strip(), f"/responses returned no output text: {result.body[:300]}"

    @pytest.mark.covers("llm.responses.openai.basic.stream.works")
    def test_responses_streaming_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-responses-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses(key, model, "reply with one word", stream=True)
        require_successful_call(result)
        parsed_events = tuple(
            parsed
            for event in result.stream_events
            if (parsed := _parse_stream_event(event)) is not None
        )
        delta_events = tuple(
            event for event in parsed_events if isinstance(event, ResponsesOutputTextDeltaEvent)
        )
        completed_events = tuple(
            event for event in parsed_events if isinstance(event, ResponsesCompletedEvent)
        )

        assert any(event.delta for event in delta_events), "responses stream returned no text deltas"
        assert completed_events and isinstance(parsed_events[-1], ResponsesCompletedEvent), (
            "responses stream did not terminate with response.completed"
        )


def _parse_stream_event(
    event: str,
) -> ResponsesOutputTextDeltaEvent | ResponsesCompletedEvent | None:
    try:
        return ResponsesOutputTextDeltaEvent.model_validate_json(event)
    except ValidationError:
        try:
            return ResponsesCompletedEvent.model_validate_json(event)
        except ValidationError:
            return None
