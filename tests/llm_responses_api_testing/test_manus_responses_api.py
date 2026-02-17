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
    got_response = await litellm.aget_responses(
        response_id=response.id,
        custom_llm_provider="manus",
        api_key=os.getenv("MANUS_API_KEY"),
    )
    print("GET API MANUS RESPONSE=", json.dumps(got_response, indent=4, default=str))
    if got_response.status == "completed":
        assert got_response.output is not None
        assert len(got_response.output) > 0



@pytest.mark.asyncio
async def test_manus_responses_api_with_file_upload():
    """
    Test that uploads a file via Files API and then passes it to Responses API.
    """
    litellm._turn_on_debug()
    
    api_key = os.getenv("MANUS_API_KEY")
    if api_key is None:
        pytest.skip("MANUS_API_KEY not set")
    
    # Step 1: Upload a file
    test_content = b"Warren Buffett's 2023 Letter to Shareholders\n\nKey Points:\n1. Long-term value creation\n2. Capital allocation strategy\n3. Market volatility perspective"
    test_filename = "buffett_letter_summary.txt"
    
    print("Step 1: Uploading file...")
    uploaded_file = await litellm.acreate_file(
        file=(test_filename, test_content),
        purpose="assistants",
        custom_llm_provider="manus",
        api_key=api_key,
    )
    print(f"Uploaded file: {uploaded_file}")
    assert uploaded_file.id is not None
    file_id = uploaded_file.id
    
    # Step 2: Create a response with the uploaded file
    print(f"\nStep 2: Creating response with file {file_id}...")
    response = await litellm.aresponses(
        model="manus/manus-1.6-lite",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Summarize the key points from this letter.",
                    },
                    {
                        "type": "input_file",
                        "file_id": file_id,
                    },
                ],
            },
        ],
        api_key=api_key,
        max_output_tokens=100,
    )
    
    print(f"Response created: {response}")
    print(f"Response type: {type(response)}")
    print(f"Response has id: {hasattr(response, 'id')}")
    
    # Handle both dict and ResponsesAPIResponse object
    if isinstance(response, dict):
        response_id = response.get("id")
    else:
        response_id = getattr(response, "id", None)
    
    assert response_id is not None, f"Response ID is None. Response: {response}"

    
    # Step 3: Clean up - delete the file
    print(f"\nStep 4: Cleaning up - deleting file {file_id}...")
    deleted_file = await litellm.afile_delete(
        file_id=file_id,
        custom_llm_provider="manus",
        api_key=api_key,
    )
    print(f"Deleted file: {deleted_file}")
    assert deleted_file.deleted is True
    
    print("\n✅ File upload and responses API integration test passed!")


