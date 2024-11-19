# What is this?
## Unit tests for Azure AI integration

import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

import litellm.types
import litellm.types.utils
from litellm.llms.anthropic.chat import ModelResponseIterator
import httpx
import json
from respx import MockRouter

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import litellm


@pytest.mark.parametrize(
    "model_group_header, expected_model",
    [
        ("offer-cohere-embed-multili-paygo", "Cohere-embed-v3-multilingual"),
        ("offer-cohere-embed-english-paygo", "Cohere-embed-v3-english"),
    ],
)
def test_map_azure_model_group(model_group_header, expected_model):
    from litellm.llms.azure_ai.embed.cohere_transformation import AzureAICohereConfig

    config = AzureAICohereConfig()
    assert config._map_azure_model_group(model_group_header) == expected_model


@pytest.mark.asyncio
@pytest.mark.respx
async def test_azure_ai_with_image_url(respx_mock: MockRouter):
    """
    Important test:

    Test that Azure AI studio can handle image_url passed when content is a list containing both text and image_url
    """
    litellm.set_verbose = True

    # Mock response based on the actual API response
    mock_response = {
        "id": "cmpl-53860ea1efa24d2883555bfec13d2254",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "The image displays a graphic with the text 'LiteLLM' in black",
                    "role": "assistant",
                    "refusal": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ],
        "created": 1731801937,
        "model": "phi35-vision-instruct",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 69,
            "prompt_tokens": 617,
            "total_tokens": 686,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }

    # Mock the API request
    mock_request = respx_mock.post(
        "https://Phi-3-5-vision-instruct-dcvov.eastus2.models.ai.azure.com"
    ).mock(return_value=httpx.Response(200, json=mock_response))

    response = await litellm.acompletion(
        model="azure_ai/Phi-3-5-vision-instruct-dcvov",
        api_base="https://Phi-3-5-vision-instruct-dcvov.eastus2.models.ai.azure.com",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
                        },
                    },
                ],
            },
        ],
        api_key="fake-api-key",
    )

    # Verify the request was made
    assert mock_request.called

    # Check the request body
    request_body = json.loads(mock_request.calls[0].request.content)
    assert request_body == {
        "model": "Phi-3-5-vision-instruct-dcvov",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
                        },
                    },
                ],
            }
        ],
    }

    print(f"response: {response}")
