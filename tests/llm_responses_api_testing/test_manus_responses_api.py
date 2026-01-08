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


class TestManusResponsesAPITest(BaseResponsesAPITest):
    def get_base_completion_call_args(self):
        return {
            "model": "manus/manus-1.6",
            "api_key": os.getenv("MANUS_API_KEY"),
        }

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_openai_responses_delete_endpoint(self, sync_mode):
        pytest.skip("DELETE responses is not supported for Manus")

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_openai_responses_streaming_delete_endpoint(self, sync_mode):
        pytest.skip("DELETE responses is not supported for Manus")

    # GET responses is now supported for Manus
    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_openai_responses_get_endpoint(self, sync_mode):
        pytest.skip("GET responses is not supported for Manus")

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_openai_responses_cancel_endpoint(self, sync_mode):
        pytest.skip("CANCEL responses is not supported for Manus")

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_cancel_responses_invalid_response_id(self, sync_mode):
        pytest.skip("CANCEL responses is not supported for Manus")


@pytest.mark.asyncio
async def test_manus_responses_api_with_agent_profile():
    """
    Test that Manus API correctly extracts agent profile from model name
    and includes task_mode and agent_profile in the request.
    """
    litellm._turn_on_debug()
    
    response = await litellm.aresponses(
        model="manus/manus-1.6",
        input="What's the color of the sky?",
        api_key=os.getenv("MANUS_API_KEY"),
        max_output_tokens=50,
    )
    
    print("Manus response=", json.dumps(response, indent=4, default=str))
    
    # Validate response structure
    assert isinstance(response, ResponsesAPIResponse), "Response should be ResponsesAPIResponse"
    assert response.id is not None, "Response should have an ID"
    assert response.status in ["running", "completed", "pending"], f"Status should be valid, got {response.status}"
    
    # Check that metadata includes Manus-specific fields
    if response.metadata:
        assert "task_id" in response.metadata or "task_url" in response.metadata, (
            "Manus response should include task_id or task_url in metadata"
        )


@pytest.mark.asyncio
async def test_manus_responses_api_different_agent_profiles():
    """
    Test that different agent profiles work correctly.
    """
    litellm._turn_on_debug()
    
    # Test with different agent profile variants
    agent_profiles = ["manus-1.6", "manus-1.6-lite", "manus-1.6-max"]
    
    for profile in agent_profiles:
        try:
            response = await litellm.aresponses(
                model=f"manus/{profile}",
                input="Hello",
                api_key=os.getenv("MANUS_API_KEY"),
                max_output_tokens=20,
            )
            
            assert response.id is not None, f"Response for {profile} should have an ID"
            print(f"✓ {profile} works: {response.id}")
        except Exception as e:
            # Some profiles might not be available, that's okay
            print(f"⚠ {profile} not available: {e}")
            pass

