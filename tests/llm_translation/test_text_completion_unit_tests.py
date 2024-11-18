import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock
import pytest
import httpx
from respx import MockRouter

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


@pytest.mark.asyncio
@pytest.mark.respx
async def test_huggingface_text_completion_logprobs(respx_mock: MockRouter):
    """Test text completion with Hugging Face, focusing on logprobs structure"""
    litellm.set_verbose = True

    # Mock the raw response from Hugging Face
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

    # Mock the API request
    mock_request = respx_mock.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
    ).mock(return_value=httpx.Response(200, json=mock_response))

    response = await litellm.atext_completion(
        model="huggingface/mistralai/Mistral-7B-v0.1",
        prompt="good morning",
    )

    # Verify the request
    assert mock_request.called
    request_body = json.loads(mock_request.calls[0].request.content)
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
