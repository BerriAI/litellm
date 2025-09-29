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

# from base_rerank_unit_tests import BaseLLMRerankTest

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


def test_azure_ai_services_with_api_version():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post") as mock_client:
        try:
            response = litellm.completion(
                model="azure_ai/Meta-Llama-3.1-70B-Instruct",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                api_key="my-fake-api-key",
                api_version="2024-05-01-preview",
                api_base="https://litellm8397336933.services.ai.azure.com/models",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["headers"]["api-key"] == "my-fake-api-key"
        assert (
            mock_client.call_args.kwargs["url"]
            == "https://litellm8397336933.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
        )


@pytest.mark.skip(reason="Skipping due to cohere ssl issues")
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
                        },
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
            model="azure_ai/deepseek-r1",
            messages=[{"role": "user", "content": "Hello, world!"}],
            api_base="https://litellm8397336933.services.ai.azure.com/models/chat/completions",
            api_key="my-fake-api-key",
            client=client,
        )

        print(response)
        assert response.choices[0].message.reasoning_content == "I am thinking here"
        assert response.choices[0].message.content == "\n\nThe sky is a canvas of blue"


# skipping due to cohere rbac issues
# class TestAzureAIRerank(BaseLLMRerankTest):
#     def get_custom_llm_provider(self) -> litellm.LlmProviders:
#         return litellm.LlmProviders.AZURE_AI

#     def get_base_rerank_call_args(self) -> dict:
#         return {
#             "model": "azure_ai/cohere-rerank-v3-english",
#             "api_base": os.getenv("AZURE_AI_COHERE_API_BASE"),
#             "api_key": os.getenv("AZURE_AI_COHERE_API_KEY"),
#         }


@pytest.mark.asyncio
async def test_azure_ai_request_format():
    """
    Test that Azure AI requests are formatted correctly with the proper endpoint and parameters
    for both synchronous and asynchronous calls
    """
    from openai import AsyncAzureOpenAI, AzureOpenAI

    litellm._turn_on_debug()

    # Set up the test parameters
    api_key = os.getenv("AZURE_API_KEY")
    api_base = os.getenv("AZURE_API_BASE")
    model = "azure_ai/gpt-4.1-nano"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {"role": "user", "content": "hi"},
    ]

    await litellm.acompletion(
        custom_llm_provider="azure_ai",
        api_key=api_key,
        api_base=api_base,
        model=model,
        messages=messages,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["azure/gpt5_series/gpt-5", "azure/gpt-5"])
async def test_azure_gpt5_reasoning(model):
    litellm._turn_on_debug()
    response = await litellm.acompletion(
        model="azure/gpt5_series/gpt-5",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        reasoning_effort="minimal",
        max_tokens=10,
        api_base=os.getenv("AZURE_GPT5_API_BASE"),
        api_key=os.getenv("AZURE_GPT5_API_KEY"),
    )
    print("response: ", response)
    assert response.choices[0].message.content is not None



def test_completion_azure():
    try:
        from litellm import completion_cost
        litellm.set_verbose = False
        ## Test azure call
        response = completion(
            model="azure/gpt-4.1-nano",
            messages=[
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                }
            ],
            api_key="os.environ/AZURE_API_KEY",
        )
        print(f"response: {response}")
        print(f"response hidden params: {response._hidden_params}")
        print(response)

        cost = completion_cost(completion_response=response)
        assert cost > 0.0
        print("Cost for azure completion request", cost)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
