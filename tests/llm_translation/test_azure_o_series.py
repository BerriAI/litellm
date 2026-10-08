import os


import pytest

import litellm
from base_llm_unit_tests import BaseLLMChatTest, BaseOSeriesModelsTest


class TestAzureOpenAIO3Mini(BaseOSeriesModelsTest, BaseLLMChatTest):
    test_content_list_handling = None
    test_empty_tools = None
    test_function_calling_with_tool_response = None

    def get_base_completion_call_args(self):
        # Clear the LLM client cache to prevent test pollution from cached clients
        litellm.in_memory_llm_clients_cache.flush_cache()
        return {
            "model": "azure/o3-mini",
            "api_key": os.getenv("AZURE_AI_API_KEY"),
            "api_base": os.getenv("AZURE_AI_API_BASE"),
            "api_version": "2024-12-01-preview",
        }

    def get_client(self):
        from openai import AzureOpenAI

        return AzureOpenAI(
            api_key="my-fake-o1-key",
            base_url="https://openai-prod-test.openai.azure.com",
            api_version="2024-02-15-preview",
        )

    def test_basic_tool_calling(self):
        pass



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
