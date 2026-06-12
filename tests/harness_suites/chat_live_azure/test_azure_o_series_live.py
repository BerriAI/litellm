import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path



import litellm
from base_llm_unit_tests import BaseLLMChatTest

class TestAzureOpenAIO3Mini(BaseLLMChatTest):
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

    def test_prompt_caching(self):
        """Temporary override. o1 prompt caching is not working."""
        pass

