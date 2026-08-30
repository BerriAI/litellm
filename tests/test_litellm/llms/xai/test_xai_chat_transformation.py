

import pytest

import litellm
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


class TestXAIUsageNormalization:
    def test_preserves_reasoning_tokens_in_total_usage(self):
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=200)

        XAIChatConfig._normalize_openai_compatible_usage_totals(usage)

        assert usage.total_tokens == 200

    def test_preserves_reasoning_tokens_in_streaming_usage(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 200}

        XAIChatConfig._normalize_openai_compatible_usage_totals(usage)

        assert usage["total_tokens"] == 200


class TestXAIChatWebSearchBilling:
    _TOOL_DETAILS = {
        "web_search_calls": 3,
        "x_search_calls": 0,
        "code_interpreter_calls": 0,
        "file_search_calls": 0,
        "mcp_calls": 0,
        "document_search_calls": 0,
    }

    @staticmethod
    def _response_with_usage() -> ModelResponse:
        response = ModelResponse(model="grok-4")
        setattr(
            response,
            "usage",
            Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )
        return response

    def test_enhance_copies_details_and_mirrors_web_search_requests(self):
        response = self._response_with_usage()

        XAIChatConfig()._enhance_usage_with_xai_web_search_fields(
            response,
            {"usage": {"server_side_tool_usage_details": self._TOOL_DETAILS}},
        )

        usage = response.usage
        assert getattr(usage, "server_side_tool_usage_details") == self._TOOL_DETAILS
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.web_search_requests == 3

    def test_enhance_noop_without_details(self):
        response = self._response_with_usage()

        XAIChatConfig()._enhance_usage_with_xai_web_search_fields(
            response, {"usage": {"prompt_tokens": 100}}
        )

        assert response.usage.prompt_tokens_details is None
        assert getattr(response.usage, "server_side_tool_usage_details", None) is None

    def test_completion_cost_bills_chat_web_search_calls(self):
        billed = self._response_with_usage()
        XAIChatConfig()._enhance_usage_with_xai_web_search_fields(
            billed,
            {"usage": {"server_side_tool_usage_details": self._TOOL_DETAILS}},
        )

        with_search = litellm.completion_cost(
            completion_response=billed, model="xai/grok-4", custom_llm_provider="xai"
        )
        without_search = litellm.completion_cost(
            completion_response=self._response_with_usage(),
            model="xai/grok-4",
            custom_llm_provider="xai",
        )

        assert with_search - without_search == pytest.approx(3 * 5.0 / 1000.0)
