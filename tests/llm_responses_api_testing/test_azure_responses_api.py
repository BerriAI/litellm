import os
import pytest

import litellm
from base_responses_api import BaseResponsesAPITest


class TestAzureResponsesAPITest(BaseResponsesAPITest):
    test_multiturn_responses_api = None
    test_responses_api_with_tool_calls = None

    def get_base_completion_call_args(self):
        return {
            "model": "azure/gpt-4.1-mini",
            "truncation": "auto",
            "api_base": os.getenv("AZURE_AI_API_BASE"),
            "api_key": os.getenv("AZURE_AI_API_KEY"),
            "api_version": "2025-03-01-preview",
        }


@pytest.mark.asyncio
async def test_azure_responses_api_preview_api_version():
    """
    Ensure new azure preview api version is working
    """
    litellm.turn_on_debug()
    response = await litellm.aresponses(
        model="azure/gpt-5-mini",
        truncation="auto",
        api_version="preview",
        api_base=os.getenv("AZURE_AI_API_BASE"),
        api_key=os.getenv("AZURE_AI_API_KEY"),
        input="Hello, can you tell me a short joke?",
    )
