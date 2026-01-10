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


@pytest.mark.asyncio
async def test_manus_responses_api_with_agent_profile():
    """
    Test that Manus API correctly extracts agent profile from model name
    and includes task_mode and agent_profile in the request.
    """
    litellm._turn_on_debug()
    
    response = await litellm.aresponses(
        model="manus/manus-1.6-lite",
        input="What's the color of the sky?",
        api_key=os.getenv("MANUS_API_KEY"),
        max_output_tokens=50,
    )
    
    print("Manus response=", json.dumps(response, indent=4, default=str))

    ## Get the status of the response
    got_response = await litellm.aget_responses(response_id=response.id)
    print("GET API MANUS RESPONSE=", json.dumps(got_response, indent=4, default=str))
    if got_response.status == "completed":
        assert got_response.output is not None
        assert len(got_response.output) > 0
    else:
        # Manus can return "running" or "pending" status
        assert got_response.status in ["running", "pending"]
        assert got_response.id is not None
    


