"""
Tests for LiteLLM Responses bridge provider.

Inherits from BaseInteractionsTest to run the same test suite against
the litellm_responses bridge provider, which calls litellm.responses() internally.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Final

import pytest

from litellm.interactions.litellm_responses_transformation.transformation import (
    LiteLLMResponsesInteractionsConfig,
)
from litellm.types.interactions import Turn
from litellm.types.llms.openai import ResponsesAPIResponse


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


def _responses_api_response(status: str, usage: Mapping[str, int] | None) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_123",
        created_at=1700000000,
        model="gpt-4o",
        object="response",
        status=status,
        output=[
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello there", "annotations": []}],
            }
        ],
        usage=usage,
    )


class TestBridgeResponseTransformation:
    @pytest.mark.parametrize(
        ("status", "usage", "model", "expected_model", "expected_usage"),
        [
            (
                "completed",
                {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
                None,
                "gpt-4o",
                {"total_input_tokens": 3, "total_output_tokens": 4},
            ),
            ("in_progress", None, "bridge-model", "bridge-model", None),
        ],
    )
    def test_responses_response_becomes_interaction(
        self,
        status: str,
        usage: Mapping[str, int] | None,
        model: str | None,
        expected_model: str,
        expected_usage: Mapping[str, int] | None,
    ):
        interaction: Final = LiteLLMResponsesInteractionsConfig.transform_responses_response_to_interactions_response(
            _responses_api_response(status, usage), model
        )

        assert interaction.id == "resp_123"
        assert interaction.object == "interaction"
        assert interaction.status == status
        assert interaction.model == expected_model
        assert interaction.outputs == [{"type": "text", "text": "Hello there"}]
        assert interaction.steps == [{"type": "model_output", "content": [{"type": "text", "text": "Hello there"}]}]
        assert interaction.usage == expected_usage
        assert interaction.created is not None
        assert datetime.fromisoformat(interaction.created).timestamp() == 1700000000
        assert interaction.updated == interaction.created
