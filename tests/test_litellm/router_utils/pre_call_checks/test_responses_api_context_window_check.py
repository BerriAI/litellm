from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest

from litellm import Router
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def _router_with_optional_max_input_tokens(max_input_tokens: Optional[int]) -> Router:
    model_info: dict[str, Any] = {"id": "deployment-1"}
    if max_input_tokens is not None:
        model_info["max_input_tokens"] = max_input_tokens

    return Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "sk-test",
                },
                "model_info": model_info,
            },
        ],
        enable_pre_call_checks=True,
    )


class TestResponsesAPIPreCallCheckIntegration:
    """
    The Responses API passes `input` instead of `messages`. Without translating
    `input` into `messages`, `_pre_call_checks` silently skips context window
    checks for every Responses API call. These tests cover that translation.
    """

    @pytest.mark.asyncio
    async def test_responses_api_rejected_by_max_input_tokens(self):
        """
        When a Responses API request exceeds max_input_tokens, the router
        should raise ContextWindowExceededError before sending the request.
        """
        with (
            patch("litellm.aresponses", new_callable=AsyncMock) as mock_aresponses,
            patch("litellm.token_counter", return_value=101),
            patch(
                "litellm.router.LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages",
                wraps=LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages,
            ) as mock_converter,
        ):
            router = _router_with_optional_max_input_tokens(max_input_tokens=100)

            with pytest.raises(Exception) as exc_info:
                await router.aresponses(
                    model="test-model",
                    input="large input",
                    instructions="You are a helpful assistant.",
                )

            assert "Context Window" in str(exc_info.value) or "context window" in str(exc_info.value).lower()
            mock_converter.assert_called_once()
            mock_aresponses.assert_not_called()

    @pytest.mark.asyncio
    async def test_responses_api_skips_conversion_without_max_input_tokens(self):
        """
        If no deployment resolves a max_input_tokens value, the router should
        not convert Responses API input just for pre-call checks.

        max_input_tokens can be resolved via litellm's model cost map (not
        only a static model_info override), so get_router_model_info is
        mocked directly to simulate "no limit" regardless of the deployment's
        underlying model.
        """
        with (
            patch("litellm.aresponses", new_callable=AsyncMock) as mock_aresponses,
            patch(
                "litellm.router.LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages"
            ) as mock_converter,
            patch("litellm.token_counter") as mock_token_counter,
        ):
            expected_response = {"id": "resp-test"}
            mock_aresponses.return_value = expected_response
            router = _router_with_optional_max_input_tokens(max_input_tokens=None)

            with patch.object(router, "get_router_model_info", return_value={}):
                result = await router.aresponses(
                    model="test-model",
                    input="short input",
                )

            assert result == expected_response
            mock_converter.assert_not_called()
            mock_token_counter.assert_not_called()
            mock_aresponses.assert_called_once()

    @pytest.mark.asyncio
    async def test_responses_api_allowed_within_max_input_tokens(self):
        """
        When a Responses API request is within max_input_tokens, the router
        should proceed to call the original function.
        """
        with (
            patch("litellm.aresponses", new_callable=AsyncMock) as mock_aresponses,
            patch("litellm.token_counter", return_value=50),
            patch(
                "litellm.router.LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages",
                wraps=LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages,
            ) as mock_converter,
        ):
            expected_response = {"id": "resp-test"}
            mock_aresponses.return_value = expected_response
            router = _router_with_optional_max_input_tokens(max_input_tokens=100)

            result = await router.aresponses(
                model="test-model",
                input="short input",
                instructions="You are a helpful assistant.",
            )

            assert result == expected_response
            mock_converter.assert_called_once()
            mock_aresponses.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
