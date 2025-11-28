"""
Test cost injection for Vertex AI Anthropic (streamRawPredict) passthrough streaming.

This test verifies that cost is correctly injected into streaming chunks
for Vertex AI streamRawPredict endpoints when include_cost_in_streaming_usage is enabled.
"""

import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../.."))

import httpx
import pytest
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)


@pytest.mark.asyncio
async def test_vertex_ai_anthropic_streaming_cost_injection_enabled():
    """
    Test that cost is injected into Vertex AI streamRawPredict streaming chunks
    when include_cost_in_streaming_usage is enabled.
    """
    # Enable cost injection
    original_value = getattr(litellm, "include_cost_in_streaming_usage", False)
    litellm.include_cost_in_streaming_usage = True

    try:
        # Mock response with Anthropic SSE format chunks
        response = AsyncMock(spec=httpx.Response)
        
        # Create chunks with message_delta event containing usage
        chunks_with_usage = [
            b'data: {"type": "content_block_delta", "delta": {"text": "Hello"}}\n\n',
            b'data: {"type": "message_delta", "usage": {"input_tokens": 10, "output_tokens": 5}}\n\n',
            b'data: {"type": "content_block_delta", "delta": {"text": " world"}}\n\n',
        ]

        async def mock_aiter_bytes():
            for chunk in chunks_with_usage:
                yield chunk

        response.aiter_bytes = mock_aiter_bytes

        # Setup logging object with model info
        litellm_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        litellm_logging_obj.model_call_details = {"model": "claude-sonnet-4@20250514"}
        litellm_logging_obj.async_success_handler = AsyncMock()

        request_body = {"model": "claude-sonnet-4@20250514"}
        start_time = datetime.now()
        passthrough_success_handler_obj = MagicMock(spec=PassThroughEndpointLogging)
        
        url_route = "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict"

        # Mock completion_cost to return a test cost value
        with patch("litellm.completion_cost", return_value=0.00015):
            received_chunks = []
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=response,
                request_body=request_body,
                litellm_logging_obj=litellm_logging_obj,
                endpoint_type=EndpointType.VERTEX_AI,
                start_time=start_time,
                passthrough_success_handler_obj=passthrough_success_handler_obj,
                url_route=url_route,
            ):
                received_chunks.append(chunk)

        # Verify that cost was injected into the message_delta chunk
        cost_injected = False
        for chunk in received_chunks:
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8", errors="ignore")
                if "message_delta" in chunk_str and "cost" in chunk_str:
                    # Parse the chunk to verify cost was added
                    for line in chunk_str.split("\n"):
                        if line.startswith("data:") and "message_delta" in line:
                            json_part = line.split("data:", 1)[1].strip()
                            if json_part and json_part != "[DONE]":
                                try:
                                    obj = json.loads(json_part)
                                    if (
                                        obj.get("type") == "message_delta"
                                        and "usage" in obj
                                        and "cost" in obj["usage"]
                                    ):
                                        assert obj["usage"]["cost"] == 0.00015
                                        cost_injected = True
                                except json.JSONDecodeError:
                                    pass

        assert cost_injected, "Cost was not injected into message_delta chunk"

    finally:
        # Restore original value
        litellm.include_cost_in_streaming_usage = original_value


@pytest.mark.asyncio
async def test_vertex_ai_anthropic_streaming_cost_injection_disabled():
    """
    Test that cost is NOT injected when include_cost_in_streaming_usage is disabled.
    """
    # Disable cost injection
    original_value = getattr(litellm, "include_cost_in_streaming_usage", False)
    litellm.include_cost_in_streaming_usage = False

    try:
        # Mock response with Anthropic SSE format chunks
        response = AsyncMock(spec=httpx.Response)
        
        chunks_with_usage = [
            b'data: {"type": "message_delta", "usage": {"input_tokens": 10, "output_tokens": 5}}\n\n',
        ]

        async def mock_aiter_bytes():
            for chunk in chunks_with_usage:
                yield chunk

        response.aiter_bytes = mock_aiter_bytes

        litellm_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        litellm_logging_obj.model_call_details = {"model": "claude-sonnet-4@20250514"}
        litellm_logging_obj.async_success_handler = AsyncMock()

        request_body = {"model": "claude-sonnet-4@20250514"}
        start_time = datetime.now()
        passthrough_success_handler_obj = MagicMock(spec=PassThroughEndpointLogging)
        
        url_route = "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict"

        received_chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
            endpoint_type=EndpointType.VERTEX_AI,
            start_time=start_time,
            passthrough_success_handler_obj=passthrough_success_handler_obj,
            url_route=url_route,
        ):
            received_chunks.append(chunk)

        # Verify that cost was NOT injected
        cost_found = False
        for chunk in received_chunks:
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8", errors="ignore")
                if "cost" in chunk_str:
                    cost_found = True

        assert not cost_found, "Cost should not be injected when feature is disabled"

    finally:
        # Restore original value
        litellm.include_cost_in_streaming_usage = original_value


@pytest.mark.asyncio
async def test_vertex_ai_anthropic_streaming_cost_injection_no_usage_chunk():
    """
    Test that chunks without usage are not modified.
    """
    original_value = getattr(litellm, "include_cost_in_streaming_usage", False)
    litellm.include_cost_in_streaming_usage = True

    try:
        response = AsyncMock(spec=httpx.Response)
        
        # Chunks without usage (should not be modified)
        chunks_without_usage = [
            b'data: {"type": "content_block_delta", "delta": {"text": "Hello"}}\n\n',
            b'data: {"type": "content_block_start", "index": 0}\n\n',
        ]

        async def mock_aiter_bytes():
            for chunk in chunks_without_usage:
                yield chunk

        response.aiter_bytes = mock_aiter_bytes

        litellm_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        litellm_logging_obj.model_call_details = {"model": "claude-sonnet-4@20250514"}
        litellm_logging_obj.async_success_handler = AsyncMock()

        request_body = {"model": "claude-sonnet-4@20250514"}
        start_time = datetime.now()
        passthrough_success_handler_obj = MagicMock(spec=PassThroughEndpointLogging)
        
        url_route = "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict"

        received_chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
            endpoint_type=EndpointType.VERTEX_AI,
            start_time=start_time,
            passthrough_success_handler_obj=passthrough_success_handler_obj,
            url_route=url_route,
        ):
            received_chunks.append(chunk)

        # Verify chunks remain unchanged (no cost injection attempted)
        assert len(received_chunks) == len(chunks_without_usage)
        # Chunks should be exactly as input since they don't contain usage
        for i, chunk in enumerate(received_chunks):
            assert chunk == chunks_without_usage[i]

    finally:
        litellm.include_cost_in_streaming_usage = original_value


@pytest.mark.asyncio
async def test_vertex_ai_anthropic_streaming_model_extraction():
    """
    Test that model name is correctly extracted for cost calculation.
    """
    original_value = getattr(litellm, "include_cost_in_streaming_usage", False)
    litellm.include_cost_in_streaming_usage = True

    try:
        response = AsyncMock(spec=httpx.Response)
        
        chunks = [
            b'data: {"type": "message_delta", "usage": {"input_tokens": 10, "output_tokens": 5}}\n\n',
        ]

        async def mock_aiter_bytes():
            for chunk in chunks:
                yield chunk

        response.aiter_bytes = mock_aiter_bytes

        litellm_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        litellm_logging_obj.model_call_details = {}
        litellm_logging_obj.async_success_handler = AsyncMock()

        # Test model extraction from request body
        request_body = {"model": "claude-sonnet-4@20250514"}
        start_time = datetime.now()
        passthrough_success_handler_obj = MagicMock(spec=PassThroughEndpointLogging)
        
        url_route = "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict"

        with patch("litellm.completion_cost") as mock_cost:
            mock_cost.return_value = 0.0001
            received_chunks = []
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=response,
                request_body=request_body,
                litellm_logging_obj=litellm_logging_obj,
                endpoint_type=EndpointType.VERTEX_AI,
                start_time=start_time,
                passthrough_success_handler_obj=passthrough_success_handler_obj,
                url_route=url_route,
            ):
                received_chunks.append(chunk)

            # Verify completion_cost was called with the correct model
            assert mock_cost.called
            call_args = mock_cost.call_args
            assert call_args[1]["model"] == "claude-sonnet-4@20250514"

    finally:
        litellm.include_cost_in_streaming_usage = original_value

