import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock, patch

import litellm


@pytest.fixture
def openai_api_response():
    mock_response_data = {
        "id": "chatcmpl-B0W3vmiM78Xkgx7kI7dr7PC949DMS",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "",
                    "refusal": None,
                    "role": "assistant",
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ],
        "created": 1739462947,
        "model": "gpt-4o-mini-2024-07-18",
        "object": "chat.completion",
        "service_tier": "default",
        "system_fingerprint": "fp_bd83329f63",
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 121,
            "total_tokens": 122,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
    }

    return mock_response_data


def test_completion_missing_role(openai_api_response):
    from openai import OpenAI

    from litellm.types.utils import ModelResponse

    client = OpenAI(api_key="test_api_key")

    mock_raw_response = MagicMock()
    mock_raw_response.headers = {
        "x-request-id": "123",
        "openai-organization": "org-123",
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
    }
    mock_raw_response.parse.return_value = ModelResponse(**openai_api_response)

    print(f"openai_api_response: {openai_api_response}")

    with patch.object(
        client.chat.completions.with_raw_response, "create", mock_raw_response
    ) as mock_create:
        litellm.completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hey"},
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_m0vFJjQmTH1McvaHBPR2YFwY",
                            "function": {
                                "arguments": '{"input": "dksjsdkjdhskdjshdskhjkhlk"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 0,
                        },
                        {
                            "id": "call_Vw6RaqV2n5aaANXEdp5pYxo2",
                            "function": {
                                "arguments": '{"input": "jkljlkjlkjlkjlk"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 1,
                        },
                        {
                            "id": "call_hBIKwldUEGlNh6NlSXil62K4",
                            "function": {
                                "arguments": '{"input": "jkjlkjlkjlkj;lj"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 2,
                        },
                    ],
                },
            ],
            client=client,
        )

        mock_create.assert_called_once()


@pytest.mark.parametrize(
    "model",
    [
        "gemini/gemini-1.5-flash",
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic/claude-3-5-sonnet",
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_url_with_format_param(model, sync_mode):
    from litellm import acompletion, completion
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    if sync_mode:
        client = HTTPHandler()
    else:
        client = AsyncHTTPHandler()

    args = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                            "format": "image/png",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ],
    }
    with patch.object(client, "post", new=MagicMock()) as mock_client:
        try:
            if sync_mode:
                response = completion(**args, client=client)
            else:
                response = await acompletion(**args, client=client)
            print(response)
        except Exception as e:
            print(e)

        mock_client.assert_called()

        print(mock_client.call_args.kwargs)

        if "data" in mock_client.call_args.kwargs:
            json_str = mock_client.call_args.kwargs["data"]
        else:
            json_str = json.dumps(mock_client.call_args.kwargs["json"])
        assert "png" in json_str
        assert "jpeg" not in json_str


@pytest.mark.parametrize("model", ["gpt-4o-mini"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_url_with_format_param_openai(model, sync_mode):
    from openai import AsyncOpenAI, OpenAI

    from litellm import acompletion, completion

    if sync_mode:
        client = OpenAI()
    else:
        client = AsyncOpenAI()

    args = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                            "format": "image/png",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ],
    }
    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            if sync_mode:
                response = completion(**args, client=client)
            else:
                response = await acompletion(**args, client=client)
            print(response)
        except Exception as e:
            print(e)

        mock_client.assert_called()

        print(mock_client.call_args.kwargs)

        json_str = json.dumps(mock_client.call_args.kwargs)

        assert "format" not in json_str
