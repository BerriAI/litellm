import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse
from base_llm_unit_tests import BaseLLMChatTest, BaseOSeriesModelsTest


class TestAzureOpenAIO3Mini(BaseOSeriesModelsTest, BaseLLMChatTest):
    def get_base_completion_call_args(self):
        # Clear the LLM client cache to prevent test pollution from cached clients
        litellm.in_memory_llm_clients_cache.flush_cache()
        return {
            "model": "azure/o3-mini",
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_base": os.getenv("AZURE_API_BASE"),
            "api_version": "2024-12-01-preview"
        }

    def get_client(self):
        from openai import AzureOpenAI

        return AzureOpenAI(
            api_key="my-fake-o1-key",
            base_url="https://openai-prod-test.openai.azure.com",
            api_version="2024-02-15-preview",
        )

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_basic_tool_calling(self):
        pass

    def test_prompt_caching(self):
        """Temporary override. o1 prompt caching is not working."""
        pass

    def test_override_fake_stream(self):
        """Test that native streaming is not supported for o1."""
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "azure/o1-preview",
                    "litellm_params": {
                        "model": "azure/o1-preview",
                        "api_key": "my-fake-o1-key",
                        "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com",
                    },
                    "model_info": {
                        "supports_native_streaming": True,
                    },
                }
            ]
        )

        ## check model info

        model_info = litellm.get_model_info(
            model="azure/o1-preview", custom_llm_provider="azure"
        )
        assert model_info["supports_native_streaming"] is True

        fake_stream = litellm.AzureOpenAIO1Config().should_fake_stream(
            model="azure/o1-preview", stream=True
        )
        assert fake_stream is False


class TestAzureOpenAIO3(BaseOSeriesModelsTest):
    def get_base_completion_call_args(self):
        return {
            "model": "azure/o3-mini",
            "api_key": "my-fake-o1-key",
            "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com",
        }

    def get_client(self):
        from openai import AzureOpenAI

        return AzureOpenAI(
            api_key="my-fake-o1-key",
            base_url="https://openai-gpt-4-test-v-1.openai.azure.com",
            api_version="2024-02-15-preview",
        )


def test_azure_o3_streaming():
    """
    Test that o3 models handles fake streaming correctly.
    """
    from openai import AzureOpenAI
    from litellm import completion

    client = AzureOpenAI(
        api_key="my-fake-o1-key",
        base_url="https://openai-gpt-4-test-v-1.openai.azure.com",
        api_version="2024-02-15-preview",
    )

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_create:
        try:
            completion(
                model="azure/o3-mini",
                messages=[{"role": "user", "content": "Hello, world!"}],
                stream=True,
                client=client,
            )
        except (
            Exception
        ) as e:  # expect output translation error as mock response doesn't return a json
            print(e)
        assert mock_create.call_count == 1
        assert "stream" in mock_create.call_args.kwargs


def test_azure_o_series_routing():
    """
    Allows user to pass model="azure/o_series/<any-deployment-name>" for explicit o_series model routing.
    """
    from openai import AzureOpenAI
    from litellm import completion

    client = AzureOpenAI(
        api_key="my-fake-o1-key",
        base_url="https://openai-gpt-4-test-v-1.openai.azure.com",
        api_version="2024-02-15-preview",
    )

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_create:
        try:
            completion(
                model="azure/o_series/my-random-deployment-name",
                messages=[{"role": "user", "content": "Hello, world!"}],
                stream=True,
                client=client,
            )
        except (
            Exception
        ) as e:  # expect output translation error as mock response doesn't return a json
            print(e)
        assert mock_create.call_count == 1
        assert "stream" not in mock_create.call_args.kwargs


@patch("litellm.main.azure_o1_chat_completions._get_openai_client")
def test_openai_o_series_max_retries_0(mock_get_openai_client):
    import litellm

    litellm.set_verbose = True
    response = litellm.completion(
        model="azure/o1-preview",
        messages=[{"role": "user", "content": "hi"}],
        max_retries=0,
    )

    mock_get_openai_client.assert_called_once()
    assert mock_get_openai_client.call_args.kwargs["max_retries"] == 0


@pytest.mark.asyncio
async def test_azure_o1_series_response_format_extra_params():
    """
    Tool calling should work for all azure o_series models.
    """
    litellm._turn_on_debug()

    from openai import AsyncAzureOpenAI

    litellm.set_verbose = True

    client = AsyncAzureOpenAI(
        api_key="fake-api-key", 
        base_url="https://openai-prod-test.openai.azure.com/openai/deployments/o1/chat/completions?api-version=2025-01-01-preview", 
        api_version="2025-01-01-preview"
    )

    tools = [{'type': 'function', 'function': {'name': 'get_current_time', 'description': 'Get the current time in a given location.', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city name, e.g. San Francisco'}}, 'required': ['location']}}}]
    response_format = {'type': 'json_object'}
    tool_choice = "auto"
    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                client=client,
                model="azure/o_series/<my-deployment-name>",
                api_key="xxxxx",
                api_base="https://openai-prod-test.openai.azure.com/openai/deployments/o1/chat/completions?api-version=2025-01-01-preview",
                api_version="2024-12-01-preview",
                messages=[{"role": "user", "content": "Hello! return a json object"}],
                tools=tools,
                response_format=response_format,
                tool_choice=tool_choice
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", json.dumps(request_body, indent=4))
        assert request_body["tools"] == tools
        assert request_body["response_format"] == response_format
        assert request_body["tool_choice"] == tool_choice

    


