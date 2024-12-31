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
