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
from litellm.llms.custom_httpx.http_handler import HTTPHandler

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
from litellm import completion


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
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    litellm.set_verbose = True

    client = AsyncHTTPHandler()

    with patch.object(client, "post") as mock_client:
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

        print(f"mock_client.call_args.kwargs: {mock_client.call_args.kwargs}")
        # Check the request body
        request_body = json.loads(mock_client.call_args.kwargs["data"])
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


@pytest.mark.parametrize(
    "api_base, expected_url",
    [
        (
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview",
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview",
        ),
        (
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions",
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions",
        ),
        (
            "https://litellm8397336933.services.ai.azure.com/models",
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions",
        ),
        (
            "https://litellm8397336933.services.ai.azure.com",
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions",
        ),
    ],
)
def test_azure_ai_services_handler(api_base, expected_url):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    litellm.set_verbose = True

    client = HTTPHandler()

    with patch.object(client, "post") as mock_client:
        try:
            response = litellm.completion(
                model="azure_ai/Meta-Llama-3.1-70B-Instruct",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                api_key="my-fake-api-key",
                api_base=api_base,
                client=client,
            )

            print(response)

        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["headers"]["api-key"] == "my-fake-api-key"
        assert mock_client.call_args.kwargs["url"] == expected_url


def test_completion_azure_ai_command_r():
    try:
        import os

        litellm.set_verbose = True

        os.environ["AZURE_AI_API_BASE"] = os.getenv("AZURE_COHERE_API_BASE", "")
        os.environ["AZURE_AI_API_KEY"] = os.getenv("AZURE_COHERE_API_KEY", "")

        response = completion(
            model="azure_ai/command-r-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the meaning of life?"}
                    ],
                }
            ],
        )  # type: ignore

        assert "azure_ai" in response.model
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_azure_deepseek_reasoning_content():
    import json

    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        mock_response = MagicMock()
        
        mock_response.text = json.dumps(
            {
                "choices": [
                    {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "<think>I am thinking here</think>\n\nThe sky is a canvas of blue",
                        "role": "assistant",
                    }
                    }
                ],
            }
        ) 
            
        mock_response.status_code = 200
        # Add required response attributes
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response
        

        response = litellm.completion(
            model='azure_ai/deepseek-r1',
            messages=[{"role": "user", "content": "Hello, world!"}],
            api_base="https://litellm8397336933.services.ai.azure.com/models/chat/completions",
            api_key="my-fake-api-key",
            client=client
        )  

        print(response)
        assert(response.choices[0].message.reasoning_content == "I am thinking here")
        assert(response.choices[0].message.content == "\n\nThe sky is a canvas of blue")
