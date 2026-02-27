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
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload


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
    model = "azure_ai/gpt-4.1-mini"
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
@pytest.mark.parametrize("model", ["azure/gpt5_series/gpt-5-mini", "azure/gpt-5-mini"])
async def test_azure_gpt5_reasoning(model):
    litellm._turn_on_debug()
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        reasoning_effort="minimal",
        max_tokens=10,
        api_base=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
    )
    print("response: ", response)
    assert response.choices[0].message.content is not None



def test_completion_azure():
    try:
        from litellm import completion_cost
        litellm.set_verbose = False
        ## Test azure call
        response = completion(
            model="azure/gpt-4.1-mini",
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


@pytest.mark.parametrize(
    "api_base",
    [
        "https://litellm-ci-cd-prod.cognitiveservices.azure.com/",
        "https://litellm-ci-cd-prod.cognitiveservices.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2023-03-15-preview",
    ],
)
def test_completion_azure_ai_gpt_4o_with_flexible_api_base(api_base):
    try:
        litellm.set_verbose = True

        response = completion(
            model="azure_ai/gpt-4.1-mini",
            api_base=api_base,
            api_key=os.getenv("AZURE_API_KEY"),
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
        )

        print(response)
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_azure_ai_model_router():
    """
    Test Azure AI model router non-streaming response cost tracking.
    Verifies that the flat cost of $0.14 per M input tokens is applied.
    
    Tests the pattern: azure_ai/model_router/<deployment-name>
    Where deployment-name is the Azure deployment (e.g., "azure-model-router").
    The model_router prefix is stripped before sending to Azure API.
    """
    from litellm.llms.azure_ai.cost_calculator import calculate_azure_model_router_flat_cost
    
    litellm._turn_on_debug()
    response = await litellm.acompletion(
        model="azure_ai/model_router/azure-model-router",
        messages=[{"role": "user", "content": "hi who is this"}],
        api_base="https://ishaa-mh6uutut-swedencentral.cognitiveservices.azure.com/openai/v1/",
        api_key=os.getenv("AZURE_MODEL_ROUTER_API_KEY"),
    )
    print("response: ", response)

    # Check response cost
    tracked_cost = response._hidden_params["response_cost"]
    assert tracked_cost > 0
    print("Tracked cost: ", tracked_cost)
    
    # Verify flat cost is included using the helper function
    usage = response.usage
    if usage and usage.prompt_tokens:
        expected_flat_cost = calculate_azure_model_router_flat_cost(
            model="model_router/azure-model-router",
            prompt_tokens=usage.prompt_tokens
        )
        print(f"Prompt tokens: {usage.prompt_tokens}")
        print(f"Expected flat cost: ${expected_flat_cost:.9f}")
        print(f"Total tracked cost: ${tracked_cost:.9f}")
        
        # Total cost should be at least the flat cost
        assert tracked_cost >= expected_flat_cost, (
            f"Cost ${tracked_cost:.9f} should be >= flat cost ${expected_flat_cost:.9f}"
        )
        
        # Verify the flat cost is non-zero
        assert expected_flat_cost > 0, "Flat cost should be greater than 0"


@pytest.mark.asyncio
async def test_azure_ai_model_router_streaming_model_in_chunk():
    """
    Test that Azure AI model router streaming returns the actual model in each chunk.
    The response should contain the actual model used (e.g., gpt-4.1-nano) not the request model (azure-model-router).
    """
    litellm._turn_on_debug()
    response = await litellm.acompletion(
        model="azure_ai/azure-model-router",
        messages=[{"role": "user", "content": "hi"}],
        api_base="https://ishaa-mh6uutut-swedencentral.cognitiveservices.azure.com/openai/v1/",
        api_key=os.getenv("AZURE_MODEL_ROUTER_API_KEY"),
        stream=True,
    )

    # Collect chunks and check model field
    chunks_with_model = []
    async for chunk in response:
        print(f"Chunk model: {chunk.model}")
        if chunk.model and chunk.model.strip():
            chunks_with_model.append(chunk.model)

    print(f"All chunk models: {chunks_with_model}")

    # At least some chunks should have a model
    assert len(chunks_with_model) > 0, "No chunks had a model field set"

    # The model should NOT be azure-model-router (the request model)
    # It should be the actual model from the response (e.g., gpt-4.1-nano, gpt-5-nano, etc.)
    for model in chunks_with_model:
        assert model != "azure-model-router", f"Chunk model should be actual model, not request model. Got: {model}"
        # The actual model should be a real model name like gpt-4.1-nano, gpt-5-nano, etc.
        print(f"Verified chunk has actual model: {model}")


class AzureModelRouterStreamingCallback(litellm.integrations.custom_logger.CustomLogger):
    """
    Custom callback to capture streaming cost tracking for Azure Model Router.
    """
    def __init__(self):
        self.standard_logging_payload = None
        self.response_cost = None
        self.async_success_called = False
        self.complete_streaming_response = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"async_log_success_event called")
        self.async_success_called = True
        self.standard_logging_payload = kwargs.get("standard_logging_object")
        self.complete_streaming_response = kwargs.get("complete_streaming_response")
        
        if self.standard_logging_payload:
            self.response_cost = self.standard_logging_payload.get("response_cost")
            print(f"standard_logging_payload model: {self.standard_logging_payload.get('model')}")
            print(f"standard_logging_payload response_cost: {self.response_cost}")
        
        if self.complete_streaming_response:
            print(f"complete_streaming_response model: {self.complete_streaming_response.model}")
            print(f"complete_streaming_response usage: {self.complete_streaming_response.usage}")




@pytest.mark.asyncio
async def test_azure_ai_model_router_streaming_cost_with_stream_options():
    """
    Test Azure AI model router streaming cost tracking with stream_options include_usage=True.
    This tests the specific case where cost tracking fails with stream_options.
    """
    litellm.logging_callback_manager._reset_all_callbacks()
    test_callback = AzureModelRouterStreamingCallback()
    litellm.callbacks = [test_callback]

    try:
        litellm._turn_on_debug()
        response = await litellm.acompletion(
            model="azure_ai/azure-model-router",
            messages=[{"role": "user", "content": "hi"}],
            api_base="https://ishaa-mh6uutut-swedencentral.cognitiveservices.azure.com/openai/v1/",
            api_key=os.getenv("AZURE_MODEL_ROUTER_API_KEY"),
            stream=True,
            stream_options={"include_usage": True},
        )

        # Consume the stream and check chunks
        full_response = ""
        chunks_with_model = []
        async for chunk in response:
            print(f"Chunk: model={chunk.model}, choices={len(chunk.choices) if chunk.choices else 0}")
            if chunk.model:
                chunks_with_model.append(chunk.model)
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content

        print(f"Full streamed response: {full_response}")
        print(f"Chunks with model: {chunks_with_model}")

        # Give async logging time to complete
        import asyncio
        await asyncio.sleep(1)

        # Verify callback was called
        assert test_callback.async_success_called is True, "async_log_success_event was not called"
        assert test_callback.standard_logging_payload is not None, "standard_logging_payload is None"

        # Check response cost
        print(f"Final response_cost: {test_callback.response_cost}")
        
        # The first chunk may have the request model (azure-model-router) because it's created
        # before the API response is received. Subsequent chunks should have the actual model.
        # At least some chunks should have the actual model (not azure-model-router)
        actual_model_chunks = [m for m in chunks_with_model if m != "azure-model-router"]
        assert len(actual_model_chunks) > 0, "No chunks had the actual model from the API response"
        print(f"Chunks with actual model: {actual_model_chunks}")

        # Verify response cost is tracked - this is the main goal of this test
        assert test_callback.response_cost is not None, "response_cost is None with stream_options"
        assert test_callback.response_cost > 0, f"response_cost should be > 0, got {test_callback.response_cost}"
        print(f"Streaming cost tracking with stream_options passed. Cost: {test_callback.response_cost}")

    finally:
        litellm.logging_callback_manager._reset_all_callbacks()
        litellm.callbacks = []