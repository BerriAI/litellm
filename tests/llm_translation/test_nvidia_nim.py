import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter
from unittest.mock import patch, MagicMock, AsyncMock

import litellm
from litellm import Choices, Message, ModelResponse, EmbeddingResponse, Usage
from litellm import completion
from base_rerank_unit_tests import BaseLLMRerankTest
import litellm


def test_completion_nvidia_nim():
    from openai import OpenAI

    litellm.set_verbose = True
    model_name = "nvidia_nim/databricks/dbrx-instruct"
    client = OpenAI(
        api_key="fake-api-key",
    )

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            completion(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in Boston today in Fahrenheit?",
                    }
                ],
                presence_penalty=0.5,
                frequency_penalty=0.1,
                client=client,
            )
        except Exception as e:
            print(e)
        # Add any assertions here to check the response

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["messages"] == [
            {
                "role": "user",
                "content": "What's the weather like in Boston today in Fahrenheit?",
            },
        ]
        assert request_body["model"] == "databricks/dbrx-instruct"
        assert request_body["frequency_penalty"] == 0.1
        assert request_body["presence_penalty"] == 0.5


def test_embedding_nvidia_nim():
    litellm.set_verbose = True
    from openai import OpenAI

    client = OpenAI(
        api_key="fake-api-key",
    )
    with patch.object(client.embeddings.with_raw_response, "create") as mock_client:
        try:
            litellm.embedding(
                model="nvidia_nim/nvidia/nv-embedqa-e5-v5",
                input="What is the meaning of life?",
                input_type="passage",
                dimensions=1024,
                client=client,
            )
        except Exception as e:
            print(e)
        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs
        print("request_body: ", request_body)
        assert request_body["input"] == "What is the meaning of life?"
        assert request_body["model"] == "nvidia/nv-embedqa-e5-v5"
        assert request_body["extra_body"]["input_type"] == "passage"
        assert request_body["dimensions"] == 1024


def test_chat_completion_nvidia_nim_with_tools():
    from openai import OpenAI

    litellm.set_verbose = True
    model_name = "nvidia_nim/meta/llama3-70b-instruct"
    client = OpenAI(
        api_key="fake-api-key",
    )

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature to use",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone, e.g. EST, PST",
                        },
                    },
                    "required": ["timezone"],
                },
            },
        },
    ]

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            completion(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in Boston today and what time is it in EST?",
                    }
                ],
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=True,
                temperature=0.7,
                client=client,
            )
        except Exception as e:
            print(e)
        
        # Add assertions to check the request
        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["messages"] == [
            {
                "role": "user",
                "content": "What's the weather like in Boston today and what time is it in EST?",
            },
        ]
        assert request_body["model"] == "meta/llama3-70b-instruct"
        assert request_body["temperature"] == 0.7
        assert request_body["tools"] == tools
        assert request_body["tool_choice"] == "auto"
        assert request_body["parallel_tool_calls"] == True

@pytest.mark.asyncio()
async def test_nvidia_nim_rerank_ranking_endpoint():
    """
    Test that using "nvidia_nim/ranking/<model>" forces the /v1/ranking endpoint.
    
    This allows users to explicitly use the /v1/ranking endpoint for models like
    nvidia/llama-3.2-nv-rerankqa-1b-v2.
    
    Reference: https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2/deploy
    """
    mock_response = AsyncMock()

    def return_val():
        return {
            "rankings": [
                {"index": 0, "logit": 0.95},
                {"index": 1, "logit": 0.75},
            ],
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        # Use "ranking/" prefix to force /v1/ranking endpoint
        response = await litellm.arerank(
            model="nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2",
            query="What is the GPU memory bandwidth?",
            documents=["H100 delivers 3TB/s memory bandwidth", "A100 has 2TB/s memory bandwidth"],
            top_n=2,
            api_key="fake-api-key",
        )

        mock_post.assert_called_once()
        
        args_to_api = mock_post.call_args.kwargs["data"]
        _url = mock_post.call_args.kwargs["url"]
        print("url = ", _url)

        # Verify URL is /v1/ranking
        assert _url == "https://ai.api.nvidia.com/v1/ranking"

        # Verify request body structure
        request_data = json.loads(args_to_api)
        print("request_data=", request_data)

        # Query should be an object with 'text' field
        assert request_data["query"] == {"text": "What is the GPU memory bandwidth?"}

        # Documents should be 'passages'
        assert request_data["passages"] == [
            {"text": "H100 delivers 3TB/s memory bandwidth"},
            {"text": "A100 has 2TB/s memory bandwidth"},
        ]

        # Model name in body should NOT have "ranking/" prefix
        assert request_data["model"] == "nvidia/llama-3.2-nv-rerankqa-1b-v2"


class TestNvidiaNim(BaseLLMRerankTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.NVIDIA_NIM

    def get_base_rerank_call_args(self) -> dict:
        return {
            "model": "nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
        }
    
    def get_expected_cost(self) -> float:
        """Nvidia NIM rerank models are free (cost = 0.0)"""
        return 0.0