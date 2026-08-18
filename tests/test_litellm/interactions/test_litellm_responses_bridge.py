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
