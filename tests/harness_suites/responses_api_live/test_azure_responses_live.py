import os
import sys
import pytest
from typing import Optional

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from base_responses_api import BaseResponsesAPITest

class TestAzureResponsesAPITest(BaseResponsesAPITest):
    def get_base_completion_call_args(self):
        return {
            "model": "azure/gpt-4.1-mini",
            "truncation": "auto",
            "api_base": os.getenv("AZURE_AI_API_BASE"),
            "api_key": os.getenv("AZURE_AI_API_KEY"),
            "api_version": "2025-03-01-preview",
        }

    def get_advanced_model_for_shell_tool(self) -> Optional[str]:
        """If specified, overrides the model used by test_responses_api_shell_tool_streaming_sees_shell_output (e.g. openai/gpt-5.2 for shell support)."""
        return "azure/gpt-5-mini"


@pytest.mark.asyncio
async def test_azure_responses_api_preview_api_version():
    """
    Ensure new azure preview api version is working
    """
    litellm._turn_on_debug()
    response = await litellm.aresponses(
        model="azure/gpt-5-mini",
        truncation="auto",
        api_version="preview",
        api_base=os.getenv("AZURE_AI_API_BASE"),
        api_key=os.getenv("AZURE_AI_API_KEY"),
        input="Hello, can you tell me a short joke?",
    )
