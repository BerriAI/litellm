"""
Tests for Gemini Interactions API transformation.

Covers credential leak prevention changes:
- validate_environment sets x-goog-api-key header
- get_complete_url excludes API key from URL
- get/delete/cancel interaction request URLs exclude API key
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.interactions.litellm_responses_transformation.streaming_iterator import (
    LiteLLMResponsesInteractionsStreamingIterator,
)
from litellm.llms.gemini.interactions.transformation import (
    GoogleAIStudioInteractionsConfig,
)
from litellm.types.llms.openai import (
    ContentPartAddedEvent,
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
)
from litellm.types.router import GenericLiteLLMParams

_PATCH_GET_API_KEY = "litellm.llms.gemini.common_utils.GeminiModelInfo.get_api_key"


@pytest.fixture
def config():
    return GoogleAIStudioInteractionsConfig()


class TestValidateEnvironment:
    def test_sets_x_goog_api_key_header(self, config):
        litellm_params = GenericLiteLLMParams(api_key="test-api-key-123")

        headers = config.validate_environment(
            headers={},
            model="gemini-2.5-flash",
            litellm_params=litellm_params,
        )

        assert headers["x-goog-api-key"] == "test-api-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_no_api_key_skips_header(self, config):
        litellm_params = GenericLiteLLMParams(api_key=None)

        with patch(_PATCH_GET_API_KEY, return_value=None):
            headers = config.validate_environment(
                headers={},
                model="gemini-2.5-flash",
                litellm_params=litellm_params,
            )

        assert "x-goog-api-key" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_no_litellm_params_skips_header(self, config):
        headers = config.validate_environment(
            headers={},
            model="gemini-2.5-flash",
            litellm_params=None,
        )

        assert "x-goog-api-key" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_preserves_existing_headers(self, config):
        litellm_params = GenericLiteLLMParams(api_key="test-key")

        headers = config.validate_environment(
            headers={"X-Custom": "value"},
            model="gemini-2.5-flash",
            litellm_params=litellm_params,
        )

        assert headers["X-Custom"] == "value"
        assert headers["x-goog-api-key"] == "test-key"


class TestGetCompleteUrl:
    def test_url_excludes_api_key(self, config):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url = config.get_complete_url(
                api_base=None,
                model="gemini-2.5-flash",
                litellm_params={"api_key": "secret-key"},
            )

        assert "key=" not in url
        assert "secret-key" not in url
        assert url.endswith("/interactions")

    def test_stream_url_has_alt_sse_only(self, config):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url = config.get_complete_url(
                api_base=None,
                model="gemini-2.5-flash",
                litellm_params={"api_key": "secret-key"},
                stream=True,
            )

        assert "key=" not in url
        assert "secret-key" not in url
        assert "alt=sse" in url

    def test_raises_without_api_key(self, config):
        with patch(_PATCH_GET_API_KEY, return_value=None):
            with pytest.raises(ValueError, match="Google API key is required"):
                config.get_complete_url(
                    api_base=None,
                    model="gemini-2.5-flash",
                    litellm_params={"api_key": None},
                )


class TestTransformRequest:
    def test_passes_environment_to_request_body(self, config):
        request_body = config.transform_request(
            model=None,
            agent="my-custom-slides-agent",
            input=[{"type": "text", "text": "Create a 5-slide presentation about AI trends."}],
            optional_params={
                "environment": "remote",
                "stream": False,
            },
            litellm_params=GenericLiteLLMParams(api_key="test-api-key"),
            headers={},
        )

        assert request_body["agent"] == "my-custom-slides-agent"
        assert request_body["environment"] == "remote"
        assert request_body["stream"] is False
        assert request_body["input"] == [
            {"type": "text", "text": "Create a 5-slide presentation about AI trends."}
        ]

    def test_passes_environment_object_to_request_body(self, config):
        environment_config = {
            "type": "remote",
            "sources": [{"type": "gcs", "uri": "gs://bucket/skills.zip"}],
            "network": {"egress": "allow_all"},
        }
        request_body = config.transform_request(
            model=None,
            agent="waverunner",
            input="What is 2 + 2?",
            optional_params={"environment": environment_config},
            litellm_params=GenericLiteLLMParams(api_key="test-api-key"),
            headers={},
        )

        assert request_body["environment"] == environment_config

    def test_passes_existing_environment_id_to_request_body(self, config):
        env_id = "env-abc123"
        request_body = config.transform_request(
            model=None,
            agent="my-custom-slides-agent",
            input="Continue the presentation.",
            optional_params={"environment": env_id},
            litellm_params=GenericLiteLLMParams(api_key="test-api-key"),
            headers={},
        )

        assert request_body["environment"] == env_id
class TestStreamingIterator:
    def _make_iterator(self) -> LiteLLMResponsesInteractionsStreamingIterator:
        return LiteLLMResponsesInteractionsStreamingIterator(
            model="gpt-5.4",
            litellm_custom_stream_wrapper=MagicMock(),
            request_input="hi",
            optional_params={},
        )

    def _make_text_delta(
        self, text: str, item_id: str = "item_1"
    ) -> OutputTextDeltaEvent:
        event = MagicMock(spec=OutputTextDeltaEvent)
        event.delta = text
        event.item_id = item_id
        return event

    def _make_part_added(self, item_id: str = "item_1") -> ContentPartAddedEvent:
        event = MagicMock(spec=ContentPartAddedEvent)
        event.item_id = item_id
        return event

    def _make_response_created(self) -> ResponseCreatedEvent:
        event = MagicMock(spec=ResponseCreatedEvent)
        event.response = MagicMock(id="resp_123")
        return event

    def test_content_delta_includes_type_field(self):
        """content.delta events must carry delta.type='text' so the UI can display them."""
        it = self._make_iterator()
        it.sent_interaction_start = True
        it.sent_content_start = True

        chunk = it._transform_responses_chunk_to_interactions_chunk(
            self._make_text_delta("Hello")
        )

        assert chunk is not None
        assert chunk.event_type == "content.delta"
        assert chunk.delta == {"type": "text", "text": "Hello"}

    def test_response_part_added_emits_content_start(self):
        """ContentPartAddedEvent (arrives before text deltas) should emit content.start
        so the first OutputTextDeltaEvent immediately emits content.delta without dropping text.
        """
        it = self._make_iterator()
        it.sent_interaction_start = True

        chunk = it._transform_responses_chunk_to_interactions_chunk(
            self._make_part_added()
        )

        assert chunk is not None
        assert chunk.event_type == "content.start"
        assert it.sent_content_start is True

    def test_first_text_delta_not_dropped_when_part_added_seen(self):
        """After ContentPartAddedEvent, the first text delta must yield content.delta
        (not content.start), preserving the token text."""
        it = self._make_iterator()
        it.sent_interaction_start = True
        it._transform_responses_chunk_to_interactions_chunk(self._make_part_added())

        chunk = it._transform_responses_chunk_to_interactions_chunk(
            self._make_text_delta("Hello")
        )

        assert chunk is not None
        assert chunk.event_type == "content.delta"
        assert chunk.delta is not None
        assert chunk.delta.get("text") == "Hello"

    def test_part_added_emits_interaction_start_fallback_when_not_sent(self):
        """If ContentPartAddedEvent arrives before any ResponseCreatedEvent,
        the iterator must emit interaction.start before content.start to honor
        the documented event ordering contract."""
        it = self._make_iterator()

        chunk = it._transform_responses_chunk_to_interactions_chunk(
            self._make_part_added(item_id="item_42")
        )

        assert chunk is not None
        assert chunk.event_type == "interaction.start"
        assert chunk.id == "item_42"
        assert chunk.status == "in_progress"
        assert chunk.model == "gpt-5.4"
        assert it.sent_interaction_start is True
        assert it.sent_content_start is False

    def test_part_added_returns_none_when_already_started(self):
        """A second ContentPartAddedEvent (after content.start was already emitted)
        should be a no-op so we don't re-emit content.start."""
        it = self._make_iterator()
        it.sent_interaction_start = True
        it.sent_content_start = True

        chunk = it._transform_responses_chunk_to_interactions_chunk(
            self._make_part_added()
        )

        assert chunk is None

    def test_part_added_without_item_id_falls_back_to_self_id(self):
        """When ContentPartAddedEvent has no item_id and we emit the interaction.start
        fallback, the id must default to an interaction_<id(self)> string."""
        it = self._make_iterator()
        event = MagicMock(spec=ContentPartAddedEvent)
        event.item_id = None

        chunk = it._transform_responses_chunk_to_interactions_chunk(event)

        assert chunk is not None
        assert chunk.event_type == "interaction.start"
        assert chunk.id == f"interaction_{id(it)}"

    def test_first_text_delta_not_dropped_when_no_prior_start_events(self):
        """When OutputTextDeltaEvent arrives before any ResponseCreatedEvent or
        ContentPartAddedEvent, the iterator must emit interaction.start *and*
        immediately follow with a content.start that carries this delta's text,
        so the first token is never silently dropped from the stream."""
        events = [
            self._make_text_delta("Hello"),
            self._make_text_delta(" World"),
        ]
        wrapper = MagicMock()
        wrapper.__iter__ = lambda self: iter(events)
        wrapper.__next__ = lambda self, _it=iter(events): next(_it)
        it = LiteLLMResponsesInteractionsStreamingIterator(
            model="gpt-5.4",
            litellm_custom_stream_wrapper=wrapper,
            request_input="hi",
            optional_params={},
        )

        first = it._transform_responses_chunk_to_interactions_chunk(events[0])
        assert first is not None
        assert first.event_type == "interaction.start"
        assert it.sent_interaction_start is True
        assert it.sent_content_start is True
        assert len(it._pending_events) == 1
        pending = it._pending_events[0]
        assert pending.event_type == "content.start"
        assert pending.delta == {"type": "text", "text": "Hello"}

        second = it._transform_responses_chunk_to_interactions_chunk(events[1])
        assert second is not None
        assert second.event_type == "content.delta"
        assert second.delta == {"type": "text", "text": " World"}


class TestTransformRequest:
    def test_stream_param_included_in_request_body(self, config):
        """When stream=True is in optional_params, the request body must include it
        so the proxy forwards the SSE streaming flag to Google's backend."""
        body = config.transform_request(
            model="gemini-2.5-flash",
            agent=None,
            input="Hello",
            optional_params={"stream": True},
            litellm_params=GenericLiteLLMParams(api_key="test-key"),
            headers={},
        )

        assert body.get("stream") is True
        assert body.get("input") == "Hello"

    def test_stream_false_not_included_when_absent(self, config):
        body = config.transform_request(
            model="gemini-2.5-flash",
            agent=None,
            input="Hello",
            optional_params={},
            litellm_params=GenericLiteLLMParams(api_key="test-key"),
            headers={},
        )

        assert "stream" not in body


class TestInteractionOperationUrls:
    """Test that get/delete/cancel interaction URLs exclude API key."""

    @pytest.mark.parametrize(
        "method_name,interaction_id,expected_suffix",
        [
            ("transform_get_interaction_request", "interaction-123", "interaction-123"),
            (
                "transform_delete_interaction_request",
                "interaction-456",
                "interaction-456",
            ),
            (
                "transform_cancel_interaction_request",
                "interaction-789",
                "interaction-789:cancel",
            ),
        ],
    )
    def test_url_excludes_key(
        self, config, method_name, interaction_id, expected_suffix
    ):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url, params = getattr(config, method_name)(
                interaction_id=interaction_id,
                api_base="https://generativelanguage.googleapis.com",
                litellm_params=GenericLiteLLMParams(api_key="secret-key"),
                headers={},
            )

        assert "key=" not in url
        assert "secret-key" not in url
        assert expected_suffix in url

    def test_interaction_id_is_encoded_as_one_path_segment(self, config):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url, params = config.transform_cancel_interaction_request(
                interaction_id="../../interactions/other?x=1#frag",
                api_base="https://generativelanguage.googleapis.com",
                litellm_params=GenericLiteLLMParams(api_key="secret-key"),
                headers={},
            )

        assert (
            url
            == "https://generativelanguage.googleapis.com/v1beta/interactions/..%2F..%2Finteractions%2Fother%3Fx%3D1%23frag:cancel"
        )
        assert params == {}

    def test_get_interaction_raises_without_key(self, config):
        with patch(_PATCH_GET_API_KEY, return_value=None):
            with pytest.raises(ValueError, match="Google API key is required"):
                config.transform_get_interaction_request(
                    interaction_id="interaction-123",
                    api_base="https://generativelanguage.googleapis.com",
                    litellm_params=GenericLiteLLMParams(api_key=None),
                    headers={},
                )
