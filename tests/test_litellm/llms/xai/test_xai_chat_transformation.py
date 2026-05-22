import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ModelResponse,
    Usage,
)


class TestXAIReasoningTokenFolding:
    """``_fold_reasoning_tokens_into_completion`` re-aligns xAI Usage to the OpenAI invariant."""

    @staticmethod
    def _make_response(
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        reasoning_tokens: int = 0,
    ) -> ModelResponse:
        details = (
            CompletionTokensDetailsWrapper(reasoning_tokens=reasoning_tokens)
            if reasoning_tokens
            else None
        )
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            completion_tokens_details=details,
        )
        response = ModelResponse()
        setattr(response, "usage", usage)
        return response

    def test_should_fold_when_total_explained_by_reasoning_gap(self):
        # xAI live shape: 14 + 10 + 312 == 336.
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=10,
            total_tokens=336,
            reasoning_tokens=312,
        )

        XAIChatConfig._fold_reasoning_tokens_into_completion(response)

        usage = response.usage
        assert usage.completion_tokens == 322
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_should_not_fold_when_already_normalised(self):
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=322,
            total_tokens=336,
            reasoning_tokens=312,
        )

        XAIChatConfig._fold_reasoning_tokens_into_completion(response)

        assert response.usage.completion_tokens == 322

    def test_should_skip_when_no_reasoning_tokens(self):
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=10,
            total_tokens=24,
            reasoning_tokens=0,
        )

        XAIChatConfig._fold_reasoning_tokens_into_completion(response)

        assert response.usage.completion_tokens == 10

    def test_should_skip_when_gap_does_not_match_reasoning(self):
        # Refuse to fold if xAI accounting changes (gap != reasoning_tokens).
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=10,
            total_tokens=999,
            reasoning_tokens=312,
        )

        XAIChatConfig._fold_reasoning_tokens_into_completion(response)

        assert response.usage.completion_tokens == 10
        assert response.usage.total_tokens == 999


class TestXAIParallelToolCalls:
    """Test suite for XAI parallel tool calls functionality."""

    def test_get_supported_openai_params_includes_parallel_tool_calls(self):
        """Test that parallel_tool_calls is in supported parameters."""
        config = XAIChatConfig()
        supported_params = config.get_supported_openai_params("xai/grok-4.20")
        assert "parallel_tool_calls" in supported_params

    def test_transform_request_preserves_parallel_tool_calls(self):
        """Test that transform_request preserves parallel_tool_calls parameter."""
        config = XAIChatConfig()

        messages = [{"role": "user", "content": "What's the weather like?"}]
        optional_params = {"parallel_tool_calls": True}

        result = config.transform_request(
            model="xai/grok-4.20",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result.get("parallel_tool_calls") is True
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
