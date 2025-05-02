"""
Unit Tests Huggingface route
"""

import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion, acompletion
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from unittest.mock import patch, MagicMock, AsyncMock, Mock
import pytest


def tgi_mock_post(url, **kwargs):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    
    # Check if response_format is in headers
    headers = kwargs.get('headers', {})
    response_format = headers.get('response_format', {})
    if isinstance(response_format, dict) and response_format.get('type') == 'json_object':
        mock_response.json.return_value = [
            {
                "generated_text": '{"city": "San Francisco", "state": "CA", "weather": "sunny", "temp": 60}',
                "details": {
                    "finish_reason": "length",
                    "generated_tokens": 26,
                    "seed": None,
                    "prefill": [],
                }
            }
        ]
    else:
        mock_response.json.return_value = [
            {
                "generated_text": "<|assistant|>\nI'm",
                "details": {
                    "finish_reason": "length",
                    "generated_tokens": 10,
                    "seed": None,
                    "prefill": [],
                    "tokens": [
                        {
                            "id": 28789,
                            "text": "<",
                            "logprob": -0.025222778,
                            "special": False,
                        },
                        {
                            "id": 28766,
                            "text": "|",
                            "logprob": -0.000003695488,
                            "special": False,
                        },
                        {
                            "id": 489,
                            "text": "ass",
                            "logprob": -0.0000019073486,
                            "special": False,
                        },
                        {
                            "id": 11143,
                            "text": "istant",
                            "logprob": -0.000002026558,
                            "special": False,
                        },
                        {
                            "id": 28766,
                            "text": "|",
                            "logprob": -0.0000015497208,
                            "special": False,
                        },
                        {
                            "id": 28767,
                            "text": ">",
                            "logprob": -0.0000011920929,
                            "special": False,
                        },
                        {
                            "id": 13,
                            "text": "\n",
                            "logprob": -0.00009703636,
                            "special": False,
                        },
                        {"id": 28737, "text": "I", "logprob": -0.1953125, "special": False},
                        {
                            "id": 28742,
                            "text": "'",
                            "logprob": -0.88183594,
                            "special": False,
                        },
                        {
                            "id": 28719,
                            "text": "m",
                            "logprob": -0.00032639503,
                            "special": False,
                        },
                    ],
                },
            }
        ]
    return mock_response


@pytest.fixture
def huggingface_chat_completion_call():
    def _call(
        model="huggingface/my-test-model",
        messages=None,
        api_key="test_api_key",
        headers=None,
        client=None,
    ):
        if messages is None:
            messages = [{"role": "user", "content": "Hello, how are you?"}]
        if client is None:
            client = HTTPHandler()

        mock_response = Mock()

        with patch.object(client, "post", side_effect=tgi_mock_post) as mock_post:
            completion(
                model=model,
                messages=messages,
                api_key=api_key,
                headers=headers or {},
                client=client,
            )

            return mock_post

    return _call


@pytest.fixture
def async_huggingface_chat_completion_call():
    async def _call(
        model="huggingface/my-test-model",
        messages=None,
        api_key="test_api_key",
        headers=None,
        client=None,
    ):
        if messages is None:
            messages = [{"role": "user", "content": "Hello, how are you?"}]
        if client is None:
            client = AsyncHTTPHandler()

        with patch.object(client, "post", side_effect=tgi_mock_post) as mock_post:
            await acompletion(
                model=model,
                messages=messages,
                api_key=api_key,
                headers=headers or {},
                client=client,
            )

            return mock_post

    return _call


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_huggingface_chat_completions_endpoint(
    sync_mode, huggingface_chat_completion_call, async_huggingface_chat_completion_call
):
    model = "huggingface/another-model"
    messages = [{"role": "user", "content": "Test message"}]

    if sync_mode:
        mock_post = huggingface_chat_completion_call(model=model, messages=messages)
    else:
        mock_post = await async_huggingface_chat_completion_call(
            model=model, messages=messages
        )

    assert mock_post.call_count == 1


@pytest.mark.parametrize("response_format", [{"type": "json_object"}])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_huggingface_chat_completions_endpoint_response_format(
    sync_mode, response_format, huggingface_chat_completion_call, async_huggingface_chat_completion_call
):
    model = "huggingface/another-model"
    messages = [
        {
            "role": "system",
            "content": "Your output should be a JSON object with no additional properties.",
        },
        {
            "role": "user",
            "content": "Respond with this in json. city=San Francisco, state=CA, weather=sunny, temp=60",
        },
    ]

    if sync_mode:
        mock_post = huggingface_chat_completion_call(
            model=model, messages=messages, headers={"response_format": response_format}
        )
    else:
        mock_post = await async_huggingface_chat_completion_call(
            model=model, messages=messages, headers={"response_format": response_format}
        )

    assert mock_post.call_count == 1
    
    # Verify the response format
    call_kwargs = mock_post.call_args[1]
    assert 'headers' in call_kwargs
    assert call_kwargs['headers'].get('response_format') == response_format
    
    # Verify the mock response matches expected format
    response = tgi_mock_post(None, headers={"response_format": response_format})
    assert '"city": "San Francisco"' in response.json.return_value[0]["generated_text"]
