"""
Tests for Gemini Interactions API transformation.

Covers:
- validate_environment: x-goog-api-key header, Api-Revision schema selection
- get_complete_url: API key excluded from URL
- get/delete/cancel interaction request URLs
- transform_request: response_mime_type coalescing, image_config migration
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.interactions.litellm_responses_transformation.streaming_iterator import (
    LiteLLMResponsesInteractionsStreamingIterator,
)
from litellm.llms.gemini.interactions.transformation import (
    GoogleAIStudioInteractionsConfig,
)
from litellm.types.llms.openai import (
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

    def test_api_revision_new_schema_by_default(self, config):
        # Default: use_legacy_interactions_schema=False → new steps schema
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = False
            headers = config.validate_environment(
                headers={}, model="gemini-2.5-flash", litellm_params=None
            )
            assert headers["Api-Revision"] == "2026-05-20"
        finally:
            litellm.use_legacy_interactions_schema = original

    def test_api_revision_legacy_schema_when_flag_set(self, config):
        # Flag on → legacy outputs schema until June 8, 2026
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = True
            headers = config.validate_environment(
                headers={}, model="gemini-2.5-flash", litellm_params=None
            )
            assert headers["Api-Revision"] == "2026-05-07"
        finally:
            litellm.use_legacy_interactions_schema = original


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
            input=[
                {
                    "type": "text",
                    "text": "Create a 5-slide presentation about AI trends.",
                }
            ],
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


class TestStreamingIterator:
    def _make_iterator(
        self, use_legacy: bool = False
    ) -> LiteLLMResponsesInteractionsStreamingIterator:
        original = litellm.use_legacy_interactions_schema
        litellm.use_legacy_interactions_schema = use_legacy
        try:
            return LiteLLMResponsesInteractionsStreamingIterator(
                model="gpt-5.4",
                litellm_custom_stream_wrapper=MagicMock(),
                request_input="hi",
                optional_params={},
            )
        finally:
            litellm.use_legacy_interactions_schema = original

    def _make_text_delta(
        self, text: str, item_id: str = "item_1"
    ) -> OutputTextDeltaEvent:
        event = MagicMock(spec=OutputTextDeltaEvent)
        event.delta = text
        event.item_id = item_id
        return event

    def _make_response_created(self) -> ResponseCreatedEvent:
        event = MagicMock(spec=ResponseCreatedEvent)
        event.response = MagicMock(id="resp_123")
        return event

    def test_step_delta_includes_type_field(self):
        """step.delta events must carry delta.type='text' so the UI can display them."""
        it = self._make_iterator(use_legacy=False)
        it.sent_interaction_start = True
        it.sent_content_start = True

        chunk = it._transform_responses_chunk_to_interactions_chunk(
            self._make_text_delta("Hello")
        )

        assert chunk is not None
        assert chunk.event_type == "step.delta"
        assert chunk.delta == {"type": "text", "text": "Hello"}

    def test_content_delta_legacy_schema(self):
        """Legacy schema emits content.delta with type and text fields."""
        it = self._make_iterator(use_legacy=True)
        it.sent_interaction_start = True
        it.sent_content_start = True

        chunk = it._transform_responses_chunk_to_interactions_chunk(
            self._make_text_delta("Hello")
        )

        assert chunk is not None
        assert chunk.event_type == "content.delta"
        assert chunk.delta == {"type": "text", "text": "Hello"}

    def test_response_created_emits_interaction_created(self):
        it = self._make_iterator(use_legacy=False)

        chunk = it._transform_responses_chunk_to_interactions_chunk(
            self._make_response_created()
        )

        assert chunk is not None
        assert chunk.event_type == "interaction.created"
        assert chunk.id == "resp_123"
        assert it.sent_interaction_start is True

    def test_response_created_emits_interaction_start_legacy(self):
        it = self._make_iterator(use_legacy=True)

        chunk = it._transform_responses_chunk_to_interactions_chunk(
            self._make_response_created()
        )

        assert chunk is not None
        assert chunk.event_type == "interaction.start"
        assert chunk.id == "resp_123"

    def test_text_delta_sequence_new_schema(self):
        """First chunk yields created + step.start + step.delta; later chunks yield step.delta."""
        it = self._make_iterator(use_legacy=False)

        first_events = it._events_for_chunk(self._make_text_delta("Hello"))
        assert [e.event_type for e in first_events] == [
            "interaction.created",
            "step.start",
            "step.delta",
        ]
        assert first_events[-1].delta == {"type": "text", "text": "Hello"}
        assert it.sent_interaction_start is True
        assert it.sent_content_start is True

        second_events = it._events_for_chunk(self._make_text_delta(" World"))
        assert [e.event_type for e in second_events] == ["step.delta"]
        assert second_events[0].delta == {"type": "text", "text": " World"}

        third_events = it._events_for_chunk(self._make_text_delta("!"))
        assert [e.event_type for e in third_events] == ["step.delta"]
        assert third_events[0].delta == {"type": "text", "text": "!"}

    def test_text_delta_sequence_legacy_schema(self):
        """Legacy: first chunk yields interaction.start + content.start + content.delta."""
        it = self._make_iterator(use_legacy=True)

        first_events = it._events_for_chunk(self._make_text_delta("Hello"))
        assert [e.event_type for e in first_events] == [
            "interaction.start",
            "content.start",
            "content.delta",
        ]
        assert first_events[-1].delta == {"type": "text", "text": "Hello"}

        second_events = it._events_for_chunk(self._make_text_delta(" World"))
        assert [e.event_type for e in second_events] == ["content.delta"]
        assert second_events[0].delta == {"type": "text", "text": " World"}

    def test_first_text_delta_without_item_id_uses_fallback_id(self):
        it = self._make_iterator(use_legacy=False)
        event = self._make_text_delta("Hi")
        event.item_id = None

        events = it._events_for_chunk(event)

        assert events[0].event_type == "interaction.created"
        assert events[0].id == f"interaction_{id(it)}"

    def test_first_text_delta_emits_text_via_compat_shim(self):
        """The legacy single-chunk shim must surface the synthetic events AND the delta."""
        it = self._make_iterator(use_legacy=False)

        first = it._transform_responses_chunk_to_interactions_chunk(
            self._make_text_delta("Hello")
        )
        assert first is not None
        assert first.event_type == "interaction.created"

        second = it.__next__() if it._pending_events else None
        assert second is not None
        assert second.event_type == "step.start"

        third = it.__next__() if it._pending_events else None
        assert third is not None
        assert third.event_type == "step.delta"
        assert third.delta == {"type": "text", "text": "Hello"}

    def test_response_created_then_text_delta_emits_step_start_and_delta(self):
        """Realistic flow: response.created arrives first, then text delta."""
        it = self._make_iterator(use_legacy=False)

        first = it._events_for_chunk(self._make_response_created())
        assert [e.event_type for e in first] == ["interaction.created"]

        second = it._events_for_chunk(self._make_text_delta("Hello"))
        assert [e.event_type for e in second] == ["step.start", "step.delta"]
        assert second[-1].delta == {"type": "text", "text": "Hello"}

    def test_no_text_token_is_dropped_during_streaming(self):
        """Concatenated step.delta payloads must equal the upstream text."""
        it = self._make_iterator(use_legacy=False)

        chunks = ["Hello", " ", "world", "!"]
        emitted_text = ""
        for c in chunks:
            for ev in it._events_for_chunk(self._make_text_delta(c)):
                if ev.event_type == "step.delta":
                    assert ev.delta is not None
                    emitted_text += ev.delta["text"]

        assert emitted_text == "Hello world!"

    def test_stop_iteration_fallback_emits_completion_event(self):
        """If upstream ends without ResponseCompletedEvent, terminal events still flow."""
        from unittest.mock import MagicMock

        text_event = self._make_text_delta("hi")
        sync_iter = MagicMock()
        sync_iter.__iter__ = lambda self: self
        sync_iter.__next__ = MagicMock(side_effect=[text_event, StopIteration])

        original = litellm.use_legacy_interactions_schema
        litellm.use_legacy_interactions_schema = False
        try:
            it = LiteLLMResponsesInteractionsStreamingIterator(
                model="gpt-5.4",
                litellm_custom_stream_wrapper=sync_iter,
                request_input="hi",
                optional_params={},
            )
        finally:
            litellm.use_legacy_interactions_schema = original

        emitted: list = []
        try:
            while True:
                emitted.append(next(it))
        except StopIteration:
            pass

        event_types = [e.event_type for e in emitted]
        assert event_types == [
            "interaction.created",
            "step.start",
            "step.delta",
            "step.stop",
            "interaction.completed",
        ]
        terminal = emitted[-1]
        assert terminal.steps == [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "hi"}],
            }
        ]
        # EOF-flushed terminal event must carry the same id as interaction.created.
        assert terminal.id == emitted[0].id == "item_1"

    def test_response_completed_emits_stop_then_completion(self):
        """ResponseCompletedEvent expands into step.stop + interaction.completed."""
        from unittest.mock import MagicMock

        text_event = self._make_text_delta("hi")
        completed = MagicMock(spec=ResponseCompletedEvent)
        completed.response = MagicMock(id="resp_999")

        sync_iter = MagicMock()
        sync_iter.__iter__ = lambda self: self
        sync_iter.__next__ = MagicMock(side_effect=[text_event, completed])

        original = litellm.use_legacy_interactions_schema
        litellm.use_legacy_interactions_schema = False
        try:
            it = LiteLLMResponsesInteractionsStreamingIterator(
                model="gpt-5.4",
                litellm_custom_stream_wrapper=sync_iter,
                request_input="hi",
                optional_params={},
            )
        finally:
            litellm.use_legacy_interactions_schema = original

        emitted: list = []
        try:
            while True:
                emitted.append(next(it))
        except StopIteration:
            pass

        event_types = [e.event_type for e in emitted]
        assert event_types == [
            "interaction.created",
            "step.start",
            "step.delta",
            "step.stop",
            "interaction.completed",
        ]
        # StopIteration fallback path must NOT add a duplicate completion event.
        assert event_types.count("interaction.completed") == 1
        # When the stream starts directly with a text delta (no preceding
        # response.created), the terminal events must reuse the id derived from
        # the first chunk's item_id rather than switching to response.id, so
        # consumers can correlate the start and completion events by id.
        assert emitted[0].id == "item_1"
        assert emitted[-1].id == "item_1"


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


class TestTransformRequestSchemaCoalescing:
    """Test new-schema request coalescing (Api-Revision: 2026-05-20)."""

    def test_response_mime_type_folded_into_response_format(self, config):
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = False
            body = config.transform_request(
                model="gemini/gemini-2.5-flash",
                agent=None,
                input="summarise",
                optional_params={
                    "response_mime_type": "application/json",
                    "response_format": {"type": "object", "properties": {}},
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
        finally:
            litellm.use_legacy_interactions_schema = original

        # response_mime_type must not appear as a top-level body key
        assert "response_mime_type" not in body
        rf = body["response_format"]
        assert rf["type"] == "text"
        assert rf["mime_type"] == "application/json"
        assert "schema" in rf

    def test_image_config_moved_to_response_format(self, config):
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = False
            body = config.transform_request(
                model="gemini/gemini-2.5-flash",
                agent=None,
                input="draw a sunset",
                optional_params={
                    "generation_config": {
                        "temperature": 0.7,
                        "image_config": {"aspect_ratio": "1:1", "image_size": "1K"},
                    }
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
        finally:
            litellm.use_legacy_interactions_schema = original

        # image_config removed from generation_config
        assert "image_config" not in body.get("generation_config", {})
        # moved into response_format with type=image
        rf = body["response_format"]
        assert rf["type"] == "image"
        assert rf["aspect_ratio"] == "1:1"

    def test_response_mime_type_skipped_when_response_format_is_list(self, config):
        """Lists are already polymorphic; do not wrap them into schema."""
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = False
            rf_list = [
                {"type": "text", "mime_type": "application/json"},
                {"type": "image", "aspect_ratio": "1:1"},
            ]
            body = config.transform_request(
                model="gemini/gemini-2.5-flash",
                agent=None,
                input="multimodal",
                optional_params={
                    "response_format": rf_list,
                    "response_mime_type": "application/json",
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
        finally:
            litellm.use_legacy_interactions_schema = original

        assert body["response_format"] == rf_list
        assert "response_mime_type" not in body

    def test_image_config_appended_to_response_format_list_without_mutating_input(
        self, config
    ):
        """When response_format is already a list, image_config must not mutate optional_params."""
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = False
            text_rf = {"type": "text", "mime_type": "application/json"}
            optional_params = {
                "response_format": [text_rf],
                "generation_config": {
                    "image_config": {"aspect_ratio": "16:9", "image_size": "2K"},
                },
            }
            original_rf = optional_params["response_format"]

            body = config.transform_request(
                model="gemini/gemini-2.5-flash",
                agent=None,
                input="draw and summarise",
                optional_params=optional_params,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

            assert optional_params["response_format"] is original_rf
            assert len(optional_params["response_format"]) == 1
            assert body["response_format"] == [
                text_rf,
                {"type": "image", "aspect_ratio": "16:9", "image_size": "2K"},
            ]

            # Retry must not append a second image entry into the caller's list.
            body_retry = config.transform_request(
                model="gemini/gemini-2.5-flash",
                agent=None,
                input="draw and summarise",
                optional_params=optional_params,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
            assert len(optional_params["response_format"]) == 1
            assert body_retry["response_format"] == body["response_format"]
        finally:
            litellm.use_legacy_interactions_schema = original

    def test_legacy_schema_passes_fields_unchanged(self, config):
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = True
            body = config.transform_request(
                model="gemini/gemini-2.5-flash",
                agent=None,
                input="hello",
                optional_params={
                    "response_mime_type": "application/json",
                    "generation_config": {"image_config": {"aspect_ratio": "16:9"}},
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
        finally:
            litellm.use_legacy_interactions_schema = original

        assert body["response_mime_type"] == "application/json"
        assert body["generation_config"]["image_config"]["aspect_ratio"] == "16:9"
