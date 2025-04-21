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
    ResponseTextConfig,
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
async def test_responses_api_routing_with_previous_response_id():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview",
                    "api_key": os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
                    "api_version": os.getenv("AZURE_RESPONSES_OPENAI_API_VERSION"),
                    "api_base": os.getenv("AZURE_RESPONSES_OPENAI_ENDPOINT"),
                },
            },
            # {
            #     "model_name": "azure-computer-use-preview",
            #     "litellm_params": {
            #         "model": "azure/computer-use-preview",
            #         "api_key": os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
            #         "api_version": os.getenv("AZURE_RESPONSES_OPENAI_API_VERSION"),
            #         "api_base": os.getenv("AZURE_RESPONSES_OPENAI_ENDPOINT_2"),
            #     },
            # },
        ],
    )
    MODEL = "azure-computer-use-preview"

    litellm._turn_on_debug()
    response = await router.aresponses(
        model=MODEL,
        input="Hello, how are you?",
        truncation="auto",
    )

    expected_model_id = response._hidden_params["model_id"]
    response_id = response.id

    print("Response ID=", response_id, "came from model_id=", expected_model_id)

    # make 3 other requests with previous_response_id
    # assert that this was sent to the same model_id
    for i in range(3):
        response = await router.aresponses(
            model=MODEL,
            input="Hello, how are you?",
            truncation="auto",
            previous_response_id=response_id,
        )

        assert response._hidden_params["model_id"] == expected_model_id

  

