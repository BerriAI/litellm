"""
Test that OpenAITextCompletion.acompletion uses the optimized async http client.

Related issue: https://github.com/BerriAI/litellm/issues/17676
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_acompletion_uses_optimized_http_client():
    """
    Test that OpenAITextCompletion.acompletion uses BaseOpenAILLM._get_async_http_client()
    instead of litellm.aclient_session directly.
    """
    from litellm.llms.openai.completion.handler import OpenAITextCompletion
    from litellm.llms.openai.common_utils import BaseOpenAILLM

    mock_http_client = MagicMock()
    mock_async_openai = AsyncMock()
    mock_async_openai.completions.with_raw_response.create = AsyncMock(
        return_value=MagicMock(
            parse=MagicMock(
                return_value=MagicMock(
                    model_dump=MagicMock(
                        return_value={
                            "id": "test-id",
                            "object": "text_completion",
                            "created": 1234567890,
                            "model": "gpt-3.5-turbo-instruct",
                            "choices": [
                                {
                                    "text": "test response",
                                    "index": 0,
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 5,
                                "completion_tokens": 10,
                                "total_tokens": 15,
                            },
                        }
                    )
                )
            )
        )
    )

    with patch.object(
        BaseOpenAILLM, "_get_async_http_client", return_value=mock_http_client
    ) as mock_get_client:
        with patch(
            "litellm.llms.openai.completion.handler.AsyncOpenAI",
            return_value=mock_async_openai,
        ) as mock_openai_class:
            handler = OpenAITextCompletion()
            logging_obj = MagicMock()
            logging_obj.post_call = MagicMock()

            await handler.acompletion(
                logging_obj=logging_obj,
                api_base="https://api.openai.com/v1",
                data={"prompt": "test", "model": "gpt-3.5-turbo-instruct"},
                headers={},
                model_response=MagicMock(),
                api_key="test-key",
                model="gpt-3.5-turbo-instruct",
                timeout=30.0,
                max_retries=2,
            )

            # Verify _get_async_http_client was called
            mock_get_client.assert_called_once()

            # Verify AsyncOpenAI was initialized with the http_client from _get_async_http_client
            mock_openai_class.assert_called_once()
            call_kwargs = mock_openai_class.call_args.kwargs
            assert call_kwargs["http_client"] == mock_http_client
