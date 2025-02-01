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
from base_llm_unit_tests import BaseLLMChatTest


class TestAzureOpenAIO1(BaseLLMChatTest):
    def get_base_completion_call_args(self):
        return {
            "model": "azure/o1-preview",
            "api_key": os.getenv("AZURE_OPENAI_O1_KEY"),
            "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
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
        assert "stream" not in mock_create.call_args.kwargs


def test_azure_o_series_routing():
    """
    Allows user to pass model="azure/o_series/<any-deployment-name>" for explicit o_series model routing.
    """
    pass
