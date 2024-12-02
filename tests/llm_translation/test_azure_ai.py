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
async def test_azure_ai_with_image_url():
    """
    Important test:

    Test that Azure AI studio can handle image_url passed when content is a list containing both text and image_url
    """
    from openai import AsyncOpenAI

    litellm.set_verbose = True

    client = AsyncOpenAI(
        api_key="fake-api-key",
        base_url="https://Phi-3-5-vision-instruct-dcvov.eastus2.models.ai.azure.com",
    )

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
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
                client=client,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")

        # Verify the request was made
        mock_client.assert_called_once()

        # Check the request body
        request_body = mock_client.call_args.kwargs
        assert request_body["model"] == "Phi-3-5-vision-instruct-dcvov"
        assert request_body["messages"] == [
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
        ]
