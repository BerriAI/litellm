"""
Regression tests for Responses API bridge ID prefixes.

The Chat Completions -> Responses bridge must generate Responses-compatible
IDs (resp_*, msg_*) instead of reusing Chat Completions IDs (chatcmpl-*).
Reusing chatcmpl-* IDs causes OpenAI to reject the request when bridged
output is later sent back as Responses input.

Regression test for https://github.com/BerriAI/litellm/issues/27333
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    Usage,
)


def _make_chat_completion_response(**overrides) -> ModelResponse:
    defaults = dict(
        id="chatcmpl-dfa2da3a-1586-4ff7-b64e-f59c692a5d11",
        created=1717000000,
        model="claude-3-5-sonnet-20241022",
        object="chat.completion",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content="Hello"),
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    defaults.update(overrides)
    return ModelResponse(**defaults)


class TestResponseIdPrefixes:
    """Bridged Responses output must use resp_*/msg_* ID prefixes."""

    def test_response_id_uses_resp_prefix(self):
        """Top-level response ID must start with resp_, not chatcmpl-."""
        chat_response = _make_chat_completion_response()

        result = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="test",
            responses_api_request={},
            chat_completion_response=chat_response,
        )

        assert result.id.startswith(
            "resp_"
        ), f"Expected resp_* prefix, got: {result.id}"
        assert not result.id.startswith("chatcmpl-")

    def test_message_output_id_uses_msg_prefix(self):
        """Message output item ID must start with msg_, not chatcmpl-."""
        chat_response = _make_chat_completion_response()

        result = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="test",
            responses_api_request={},
            chat_completion_response=chat_response,
        )

        message_items = [
            item for item in result.output if getattr(item, "type", None) == "message"
        ]
        assert len(message_items) > 0

        for item in message_items:
            assert item.id.startswith("msg_"), f"Expected msg_* prefix, got: {item.id}"
            assert not item.id.startswith("chatcmpl-")

    def test_response_and_message_ids_are_distinct(self):
        """Response ID and message item ID must not be the same value."""
        chat_response = _make_chat_completion_response()

        result = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="test",
            responses_api_request={},
            chat_completion_response=chat_response,
        )

        message_items = [
            item for item in result.output if getattr(item, "type", None) == "message"
        ]
        for item in message_items:
            assert result.id != item.id

    def test_dict_input_also_gets_correct_prefixes(self):
        """When chat_completion_response is passed as a dict, IDs still get correct prefixes."""
        chat_response = _make_chat_completion_response()

        result = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="test",
            responses_api_request={},
            chat_completion_response=dict(chat_response),
        )

        assert result.id.startswith("resp_")
        message_items = [
            item for item in result.output if getattr(item, "type", None) == "message"
        ]
        for item in message_items:
            assert item.id.startswith("msg_")
