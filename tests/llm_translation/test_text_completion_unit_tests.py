import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock
import pytest
import httpx
from respx import MockRouter
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

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


@pytest.mark.skip(
    reason="need to migrate huggingface to support httpx client being passed in"
)
@pytest.mark.asyncio
@pytest.mark.respx
async def test_huggingface_text_completion_logprobs():
    """Test text completion with Hugging Face, focusing on logprobs structure"""
    litellm.set_verbose = True
    litellm.disable_aiohttp_transport = True # since this uses respx, we need to set use_aiohttp_transport to False
    from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler

    mock_response = [
        {
            "generated_text": ",\n\nI have a question...",  # truncated for brevity
            "details": {
                "finish_reason": "length",
                "generated_tokens": 100,
                "seed": None,
                "prefill": [],
                "tokens": [
                    {"id": 28725, "text": ",", "logprob": -1.7626953, "special": False},
                    {"id": 13, "text": "\n", "logprob": -1.7314453, "special": False},
                ],
            },
        }
    ]

    return_val = AsyncMock()

    return_val.json.return_value = mock_response

    client = AsyncHTTPHandler()
    with patch.object(client, "post", return_value=return_val) as mock_post:
        response = await litellm.atext_completion(
            model="huggingface/mistralai/Mistral-7B-Instruct-v0.3",
            prompt="good morning",
            client=client,
        )

        # Verify the request
        mock_post.assert_called_once()
        request_body = json.loads(mock_post.call_args.kwargs["data"])
        assert request_body == {
            "inputs": "good morning",
            "parameters": {"details": True, "return_full_text": False},
            "stream": False,
        }

        print("response=", response)

        # Verify response structure
        assert isinstance(response, TextCompletionResponse)
        assert response.object == "text_completion"
        assert response.model == "mistralai/Mistral-7B-v0.1"

        # Verify logprobs structure
        choice = response.choices[0]
        assert choice.finish_reason == "length"
        assert choice.index == 0
        assert isinstance(choice.logprobs.tokens, list)
        assert isinstance(choice.logprobs.token_logprobs, list)
        assert isinstance(choice.logprobs.text_offset, list)
        assert isinstance(choice.logprobs.top_logprobs, list)
        assert choice.logprobs.tokens == [",", "\n"]
        assert choice.logprobs.token_logprobs == [-1.7626953, -1.7314453]
        assert choice.logprobs.text_offset == [0, 1]
        assert choice.logprobs.top_logprobs == [{}, {}]

        # Verify usage
        assert response.usage["completion_tokens"] > 0
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["total_tokens"] > 0


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
