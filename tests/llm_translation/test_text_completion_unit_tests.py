import json
from datetime import datetime
from unittest.mock import AsyncMock
import pytest
import httpx
from respx import MockRouter
from unittest.mock import patch, MagicMock


import litellm
from litellm.types.utils import TextCompletionResponse


def test_convert_dict_to_text_completion_response():
    input_dict = {
        "id": "cmpl-ALVLPJgRkqpTomotoOMi3j0cAaL4L",
        "choices": [
            {
                "finish_reason": "length",
                "index": 0,
                "logprobs": {
                    "text_offset": [0, 5],
                    "token_logprobs": [None, -12.203847],
                    "tokens": ["hello", " crisp"],
                    "top_logprobs": [None, {",": -2.1568563}],
                },
                "text": "hello crisp",
            }
        ],
        "created": 1729688739,
        "model": "davinci-002",
        "object": "text_completion",
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 1,
            "total_tokens": 2,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }

    response = TextCompletionResponse(**input_dict)

    assert response.id == "cmpl-ALVLPJgRkqpTomotoOMi3j0cAaL4L"
    assert len(response.choices) == 1
    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].index == 0
    assert response.choices[0].text == "hello crisp"
    assert response.created == 1729688739
    assert response.model == "davinci-002"
    assert response.object == "text_completion"
    assert response.system_fingerprint is None
    assert response.usage.completion_tokens == 1
    assert response.usage.prompt_tokens == 1
    assert response.usage.total_tokens == 2
    assert response.usage.completion_tokens_details is None
    assert response.usage.prompt_tokens_details is None

    # Test logprobs
    assert response.choices[0].logprobs.text_offset == [0, 5]
    assert response.choices[0].logprobs.token_logprobs == [None, -12.203847]
    assert response.choices[0].logprobs.tokens == ["hello", " crisp"]
    assert response.choices[0].logprobs.top_logprobs == [None, {",": -2.1568563}]




@pytest.mark.asyncio
async def test_acompletion_uses_optimized_http_client():
    """
    Test that OpenAITextCompletion.acompletion uses BaseOpenAILLM._get_async_http_client()
    instead of litellm.aclient_session directly.

    Related issue: https://github.com/BerriAI/litellm/issues/17676
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
