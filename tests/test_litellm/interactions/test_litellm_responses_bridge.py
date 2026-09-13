"""
Tests for LiteLLM Responses bridge provider.

Inherits from BaseInteractionsTest to run the same test suite against
the litellm_responses bridge provider, which calls litellm.responses() internally.
"""

import os

from litellm.interactions.litellm_responses_transformation.transformation import (
    LiteLLMResponsesInteractionsConfig,
)
from litellm.types.interactions import Turn
from tests.test_litellm.interactions.base_interactions_test import (
    BaseInteractionsTest,
)


class TestLiteLLMResponsesBridge(BaseInteractionsTest):
    """Test LiteLLM Responses bridge using the base test suite."""

    def get_model(self) -> str:
        """Return the model string for the bridge provider.

        The bridge provider uses litellm.responses() internally, so we can
        use any model that litellm.responses() supports (e.g., gpt-4o).
        """
        return "gpt-4o"

    def get_api_key(self) -> str:
        """Return the OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY", "")


class TestBridgeInputTransformation:
    """Regression tests for translating Interactions input into Responses API input.

    The bridge used to pass Google content parts through raw ({"type": "text"}),
    which the Responses API rejects with a 400, and it dropped the role encoded
    in step types and in the legacy "model" turn role.
    """

    def test_step_input_maps_roles_and_content_types(self):
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [
                {"type": "user_input", "content": [{"type": "text", "text": "I like apples."}]},
                {"type": "model_output", "content": [{"type": "text", "text": "I like oranges."}]},
                {"type": "user_input", "content": [{"type": "text", "text": "What did you say?"}]},
            ]
        )
        assert transformed == [
            {"role": "user", "content": [{"type": "input_text", "text": "I like apples."}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "I like oranges."}]},
            {"role": "user", "content": [{"type": "input_text", "text": "What did you say?"}]},
        ]

    def test_legacy_turn_input_maps_model_role_to_assistant(self):
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [
                {"role": "user", "content": [{"type": "text", "text": "I like apples."}]},
                {"role": "model", "content": [{"type": "text", "text": "I like oranges."}]},
            ]
        )
        assert transformed == [
            {"role": "user", "content": [{"type": "input_text", "text": "I like apples."}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "I like oranges."}]},
        ]

    def test_turn_pydantic_model_with_string_content(self):
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [Turn(role="model", content="I like oranges.")]
        )
        assert transformed == [
            {"role": "assistant", "content": [{"type": "output_text", "text": "I like oranges."}]}
        ]

    def test_string_input_passes_through(self):
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input("Hello")
        assert transformed == "Hello"

    def test_content_list_input_becomes_single_user_message(self):
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [{"type": "text", "text": "Hello"}, "world"]
        )
        assert transformed == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": "world"},
                ],
            }
        ]

    def test_non_text_content_passes_through_unchanged(self):
        image_part = {"type": "image", "data": "base64data", "mime_type": "image/png"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [{"type": "user_input", "content": [image_part]}]
        )
        assert transformed == [{"role": "user", "content": [image_part]}]


class TestBridgeResponseFormat:
    """The bridge used to drop response_format and response_mime_type entirely, so a caller
    who asked the Interactions API for JSON silently got free text back from the Responses API.
    """

    def test_json_schema_response_format_becomes_text_format(self):
        schema = {
            "type": "object",
            "properties": {"greeting": {"type": "string"}, "score": {"type": "number"}},
            "required": ["greeting", "score"],
        }
        request = LiteLLMResponsesInteractionsConfig.transform_interactions_request_to_responses_request(
            model="gemini-3.5-flash",
            input="Say hello and give a score of 1.",
            optional_params={
                "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
            },
        )
        assert request["text"] == {
            "format": {
                "type": "json_schema",
                "name": "response_schema",
                "schema": schema,
                "strict": False,
            }
        }

    def test_legacy_schema_and_response_mime_type_become_text_format(self):
        schema = {"type": "object", "properties": {"greeting": {"type": "string"}}}
        request = LiteLLMResponsesInteractionsConfig.transform_interactions_request_to_responses_request(
            model="gemini-3.5-flash",
            input="Say hello.",
            optional_params={"response_format": schema, "response_mime_type": "application/json"},
        )
        assert request["text"]["format"]["type"] == "json_schema"
        assert request["text"]["format"]["schema"] == schema

    def test_json_mime_type_without_schema_becomes_json_object(self):
        request = LiteLLMResponsesInteractionsConfig.transform_interactions_request_to_responses_request(
            model="gemini-3.5-flash",
            input="Say hello.",
            optional_params={"response_mime_type": "application/json"},
        )
        assert request["text"] == {"format": {"type": "json_object"}}

    def test_image_response_format_entry_is_skipped(self):
        schema = {"type": "object", "properties": {"caption": {"type": "string"}}}
        request = LiteLLMResponsesInteractionsConfig.transform_interactions_request_to_responses_request(
            model="gemini-3.5-flash",
            input="Describe this.",
            optional_params={
                "response_format": [
                    {"type": "image", "aspect_ratio": "1:1"},
                    {"type": "text", "mime_type": "application/json", "schema": schema},
                ],
            },
        )
        assert request["text"]["format"]["schema"] == schema

    def test_image_only_response_format_leaves_text_unset(self):
        request = LiteLLMResponsesInteractionsConfig.transform_interactions_request_to_responses_request(
            model="gemini-3.5-flash",
            input="Draw a cat.",
            optional_params={"response_format": [{"type": "image", "aspect_ratio": "1:1"}]},
        )
        assert "text" not in request

    def test_non_json_mime_type_leaves_text_unset(self):
        request = LiteLLMResponsesInteractionsConfig.transform_interactions_request_to_responses_request(
            model="gemini-3.5-flash",
            input="Say hello.",
            optional_params={"response_format": {"type": "text", "mime_type": "text/plain"}},
        )
        assert "text" not in request

    def test_unrecognized_entry_type_is_not_read_as_a_bare_schema(self):
        request = LiteLLMResponsesInteractionsConfig.transform_interactions_request_to_responses_request(
            model="gemini-3.5-flash",
            input="Say hello.",
            optional_params={"response_format": {"type": "audio", "mime_type": "audio/wav"}},
        )
        assert "text" not in request

    def test_request_without_response_format_is_unchanged(self):
        request = LiteLLMResponsesInteractionsConfig.transform_interactions_request_to_responses_request(
            model="gemini-3.5-flash",
            input="Say hello.",
            optional_params={},
        )
        assert "text" not in request
