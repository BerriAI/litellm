import os
import sys
import pytest
import asyncio
from typing import Optional
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.integrations.custom_logger import CustomLogger
import json
from litellm.types.utils import StandardLoggingPayload
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponseAPIUsage,
    IncompleteDetails,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from base_responses_api import BaseResponsesAPITest


class TestAzureResponsesAPITest(BaseResponsesAPITest):
    def get_base_completion_call_args(self):
        return {
            "model": "azure/computer-use-preview",
            "truncation": "auto",
            "api_base": os.getenv("AZURE_RESPONSES_OPENAI_ENDPOINT"),
            "api_key": os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
            "api_version": os.getenv("AZURE_RESPONSES_OPENAI_API_VERSION"),
        }


@pytest.mark.asyncio
async def test_azure_responses_api_preview_api_version():
    """
    Ensure new azure preview api version is working
    """
    litellm._turn_on_debug()
    response = await litellm.aresponses(
        model="azure/computer-use-preview",
        truncation="auto",
        api_version="preview",
        api_base=os.getenv("AZURE_RESPONSES_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
        input="Hello, can you tell me a short joke?",
    )


@pytest.mark.asyncio
async def test_azure_responses_api_status_error():
    """
    Ensure new azure preview api version is working
    """
    litellm._turn_on_debug()

    request_data = {
        "model": "gpt-5-mini",
        "input": [
            {"content": "tell me an interesting fact", "role": "user"},
            {
                "id": "rs_0ab687487834d9df0068e462a1b2d88197aabbc832c9ba5316",
                "summary": [],
                "type": "reasoning",
                "content": None,
                "encrypted_content": None,
                "status": "completed",
            },
            {
                "id": "msg_0ab687487834d9df0068e462a1df188197b74b1eef05102c18",
                "content": [
                    {
                        "annotations": [],
                        "text": "Octopuses have three hearts: two pump blood to the gills, while the third pumps it to the rest of the body. Even more unusual, their blood is blue because it uses the copper-containing protein hemocyanin to carry oxygen, which is more efficient than hemoglobin in cold, low-oxygen environments.",
                        "type": "output_text",
                        "logprobs": [],
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {"role": "user", "content": "tell me another"},
        ],
        "include": [],
        "instructions": "You are a helpful assistant.",
        "reasoning": {"effort": "minimal"},
        "stream": False,
        "tools": [],
    }
    response = await litellm.aresponses(
        model="azure/gpt-5-mini-2",
        truncation="auto",
        api_version="preview",
        api_base=os.getenv("AZURE_GPT5_MINI_API_BASE"),
        api_key=os.getenv("AZURE_GPT5_MINI_API_KEY"),
        input=request_data["input"],
    )
