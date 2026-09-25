"""
Tests for LiteLLM Responses bridge provider.

Inherits from BaseInteractionsTest to run the same test suite against
the litellm_responses bridge provider, which calls litellm.responses() internally.
"""

import base64
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

    def test_image_content_with_mime_type_is_transformed_to_input_image(self):
        image_part = {"type": "image", "data": "base64data", "mime_type": "image/png"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [{"type": "user_input", "content": [image_part]}]
        )
        assert transformed == [
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:image/png;base64,base64data"}],
            }
        ]

    def test_image_content_with_uri_is_transformed_to_input_image(self):
        image_part = {"type": "image", "uri": "https://example.com/cat.jpg"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [{"type": "user_input", "content": [image_part]}]
        )
        assert transformed == [
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": "https://example.com/cat.jpg"}],
            }
        ]

    def test_image_content_missing_mime_type_is_sniffed_from_data(self):
        png_signature_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"rest-of-file").decode()
        image_part = {"type": "image", "data": png_signature_b64}
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [{"type": "user_input", "content": [image_part]}]
        )
        assert transformed == [
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": f"data:image/png;base64,{png_signature_b64}"}],
            }
        ]

    def test_image_content_missing_mime_type_and_unrecognized_data_defaults_to_octet_stream(self):
        image_part = {"type": "image", "data": "not-a-real-image-signature"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [{"type": "user_input", "content": [image_part]}]
        )
        assert transformed[0]["content"][0]["image_url"].startswith("data:application/octet-stream;base64,")

    def test_image_content_webp_signature_is_sniffed_from_data(self):
        webp_signature_b64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WEBPrest-of-file").decode()
        image_part = {"type": "image", "data": webp_signature_b64}
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [{"type": "user_input", "content": [image_part]}]
        )
        assert transformed == [
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": f"data:image/webp;base64,{webp_signature_b64}"}],
            }
        ]

    def test_image_content_with_undecodable_data_defaults_to_octet_stream(self):
        image_part = {"type": "image", "data": "a" + "!" * 23}
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [{"type": "user_input", "content": [image_part]}]
        )
        assert transformed[0]["content"][0]["image_url"].startswith("data:application/octet-stream;base64,")

    def test_image_content_with_too_short_data_defaults_to_octet_stream(self):
        image_part = {"type": "image", "data": "ab"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [{"type": "user_input", "content": [image_part]}]
        )
        assert transformed == [
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:application/octet-stream;base64,ab"}],
            }
        ]

    def test_image_content_without_data_or_uri_passes_through_unchanged(self):
        image_part = {"type": "image", "mime_type": "image/png"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [{"type": "user_input", "content": [image_part]}]
        )
        assert transformed == [{"role": "user", "content": [image_part]}]

    def test_unrecognized_content_type_passes_through_unchanged(self):
        other_part = {"type": "document", "data": "base64data", "mime_type": "application/pdf"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            [{"type": "user_input", "content": [other_part]}]
        )
        assert transformed == [{"role": "user", "content": [other_part]}]
