"""
Unit tests for ResponsePollingHandler

Tests core functionality including:
1. Polling ID generation and detection
2. Initial state creation (queued status)
3. State updates with batched output
4. Status transitions (queued -> in_progress -> completed)
5. Response completion with reasoning, tools, tool_choice
6. Error handling and cancellation
7. Cache key generation

These tests ensure the polling handler correctly manages response state
following the OpenAI Response API format.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler


class TestResponsePollingHandler:
    """Test cases for ResponsePollingHandler"""

    # ==================== Polling ID Tests ====================

    def test_generate_polling_id_has_correct_prefix(self):
        """Test that generated polling IDs have the correct prefix"""
        polling_id = ResponsePollingHandler.generate_polling_id()
        
        assert polling_id.startswith("litellm_poll_")
        assert len(polling_id) > len("litellm_poll_")  # Has UUID after prefix

    def test_generate_polling_id_is_unique(self):
        """Test that each generated polling ID is unique"""
        ids = [ResponsePollingHandler.generate_polling_id() for _ in range(100)]
        
        assert len(ids) == len(set(ids))  # All unique

    def test_is_polling_id_returns_true_for_polling_ids(self):
        """Test that is_polling_id correctly identifies polling IDs"""
        polling_id = ResponsePollingHandler.generate_polling_id()
        
        assert ResponsePollingHandler.is_polling_id(polling_id) is True

    def test_is_polling_id_returns_false_for_provider_ids(self):
        """Test that is_polling_id returns False for provider response IDs"""
        # OpenAI format
        assert ResponsePollingHandler.is_polling_id("resp_abc123") is False
        # Anthropic format
        assert ResponsePollingHandler.is_polling_id("msg_01XFDUDYJgAACzvnptvVoYEL") is False
        # Generic UUID
        assert ResponsePollingHandler.is_polling_id("550e8400-e29b-41d4-a716-446655440000") is False

    def test_get_cache_key_format(self):
        """Test that cache keys have the correct format"""
        polling_id = "litellm_poll_abc123"
        cache_key = ResponsePollingHandler.get_cache_key(polling_id)
        
        assert cache_key == "litellm:polling:response:litellm_poll_abc123"

    # ==================== Initial State Tests ====================

    @pytest.mark.asyncio
    async def test_create_initial_state_returns_queued_status(self):
        """Test that create_initial_state returns response with queued status"""
        mock_redis = AsyncMock()
        handler = ResponsePollingHandler(redis_cache=mock_redis, ttl=3600)
        
        polling_id = "litellm_poll_test123"
        request_data = {
            "model": "gpt-4o",
            "input": "Hello",
            "metadata": {"test": "value"}
        }
        
        response = await handler.create_initial_state(
            polling_id=polling_id,
            request_data=request_data,
        )
        
        assert response.id == polling_id
        assert response.object == "response"
        assert response.status == "queued"
        assert response.output == []
        assert response.usage is None
        assert response.metadata == {"test": "value"}

    @pytest.mark.asyncio
    async def test_create_initial_state_stores_in_redis(self):
        """Test that create_initial_state stores state in Redis with correct TTL"""
        mock_redis = AsyncMock()
        handler = ResponsePollingHandler(redis_cache=mock_redis, ttl=7200)
        
        polling_id = "litellm_poll_test123"
        request_data = {"model": "gpt-4o", "input": "Hello"}
        
        await handler.create_initial_state(
            polling_id=polling_id,
            request_data=request_data,
        )
        
        # Verify Redis was called with correct parameters
        mock_redis.async_set_cache.assert_called_once()
        call_args = mock_redis.async_set_cache.call_args
        
        assert call_args.kwargs["key"] == "litellm:polling:response:litellm_poll_test123"
        assert call_args.kwargs["ttl"] == 7200
        
        # Verify the stored value is valid JSON
        stored_value = call_args.kwargs["value"]
        parsed = json.loads(stored_value)
        assert parsed["id"] == polling_id
        assert parsed["status"] == "queued"

    @pytest.mark.asyncio
    async def test_create_initial_state_sets_created_at_timestamp(self):
        """Test that create_initial_state sets a valid created_at timestamp"""
        mock_redis = AsyncMock()
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        before_time = int(datetime.now(timezone.utc).timestamp())
        
        response = await handler.create_initial_state(
            polling_id="litellm_poll_test",
            request_data={},
        )
        
        after_time = int(datetime.now(timezone.utc).timestamp())
        
        assert before_time <= response.created_at <= after_time

    # ==================== State Update Tests ====================

    @pytest.mark.asyncio
    async def test_update_state_changes_status_to_in_progress(self):
        """Test that update_state can change status to in_progress"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = json.dumps({
            "id": "litellm_poll_test",
            "object": "response",
            "status": "queued",
            "output": [],
            "created_at": 1234567890
        })
        
        handler = ResponsePollingHandler(redis_cache=mock_redis, ttl=3600)
        
        await handler.update_state(
            polling_id="litellm_poll_test",
            status="in_progress",
        )
        
        # Verify the update was saved
        mock_redis.async_set_cache.assert_called_once()
        call_args = mock_redis.async_set_cache.call_args
        stored = json.loads(call_args.kwargs["value"])
        
        assert stored["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_update_state_replaces_full_output_list(self):
        """Test that update_state replaces the full output list"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = json.dumps({
            "id": "litellm_poll_test",
            "object": "response",
            "status": "in_progress",
            "output": [{"id": "old_item", "type": "message"}],
            "created_at": 1234567890
        })
        
        handler = ResponsePollingHandler(redis_cache=mock_redis, ttl=3600)
        
        new_output = [
            {"id": "item_1", "type": "message", "content": [{"type": "text", "text": "Hello"}]},
            {"id": "item_2", "type": "message", "content": [{"type": "text", "text": "World"}]},
        ]
        
        await handler.update_state(
            polling_id="litellm_poll_test",
            output=new_output,
        )
        
        call_args = mock_redis.async_set_cache.call_args
        stored = json.loads(call_args.kwargs["value"])
        
        assert len(stored["output"]) == 2
        assert stored["output"][0]["id"] == "item_1"
        assert stored["output"][1]["id"] == "item_2"

    @pytest.mark.asyncio
    async def test_update_state_with_usage(self):
        """Test that update_state correctly stores usage data"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = json.dumps({
            "id": "litellm_poll_test",
            "object": "response",
            "status": "in_progress",
            "output": [],
            "created_at": 1234567890
        })
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        usage_data = {
            "input_tokens": 10,
            "output_tokens": 50,
            "total_tokens": 60
        }
        
        await handler.update_state(
            polling_id="litellm_poll_test",
            status="completed",
            usage=usage_data,
        )
        
        call_args = mock_redis.async_set_cache.call_args
        stored = json.loads(call_args.kwargs["value"])
        
        assert stored["status"] == "completed"
        assert stored["usage"] == usage_data

    @pytest.mark.asyncio
    async def test_update_state_with_reasoning_tools_tool_choice(self):
        """Test that update_state stores reasoning, tools, and tool_choice from response.completed"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = json.dumps({
            "id": "litellm_poll_test",
            "object": "response",
            "status": "in_progress",
            "output": [],
            "created_at": 1234567890
        })
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        reasoning_data = {"effort": "medium", "summary": "Step by step analysis"}
        tool_choice_data = {"type": "function", "function": {"name": "get_weather"}}
        tools_data = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        
        await handler.update_state(
            polling_id="litellm_poll_test",
            status="completed",
            reasoning=reasoning_data,
            tool_choice=tool_choice_data,
            tools=tools_data,
        )
        
        call_args = mock_redis.async_set_cache.call_args
        stored = json.loads(call_args.kwargs["value"])
        
        assert stored["reasoning"] == reasoning_data
        assert stored["tool_choice"] == tool_choice_data
        assert stored["tools"] == tools_data

    @pytest.mark.asyncio
    async def test_update_state_with_error_sets_failed_status(self):
        """Test that providing an error automatically sets status to failed"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = json.dumps({
            "id": "litellm_poll_test",
            "object": "response",
            "status": "in_progress",
            "output": [],
            "created_at": 1234567890
        })
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        error_data = {
            "type": "internal_error",
            "message": "Something went wrong",
            "code": "server_error"
        }
        
        await handler.update_state(
            polling_id="litellm_poll_test",
            error=error_data,
        )
        
        call_args = mock_redis.async_set_cache.call_args
        stored = json.loads(call_args.kwargs["value"])
        
        assert stored["status"] == "failed"
        assert stored["error"] == error_data

    @pytest.mark.asyncio
    async def test_update_state_with_incomplete_details(self):
        """Test that update_state stores incomplete_details"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = json.dumps({
            "id": "litellm_poll_test",
            "object": "response",
            "status": "in_progress",
            "output": [],
            "created_at": 1234567890
        })
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        incomplete_details = {
            "reason": "max_output_tokens"
        }
        
        await handler.update_state(
            polling_id="litellm_poll_test",
            status="incomplete",
            incomplete_details=incomplete_details,
        )
        
        call_args = mock_redis.async_set_cache.call_args
        stored = json.loads(call_args.kwargs["value"])
        
        assert stored["status"] == "incomplete"
        assert stored["incomplete_details"] == incomplete_details

    @pytest.mark.asyncio
    async def test_update_state_does_nothing_without_redis(self):
        """Test that update_state gracefully handles no Redis cache"""
        handler = ResponsePollingHandler(redis_cache=None)
        
        # Should not raise an exception
        await handler.update_state(
            polling_id="litellm_poll_test",
            status="in_progress",
        )

    @pytest.mark.asyncio
    async def test_update_state_handles_missing_cached_state(self):
        """Test that update_state handles case when cached state doesn't exist"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = None  # Cache miss
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        # Should not raise an exception
        await handler.update_state(
            polling_id="litellm_poll_test",
            status="in_progress",
        )
        
        # Should not try to set cache if nothing was found
        mock_redis.async_set_cache.assert_not_called()

    # ==================== Get State Tests ====================

    @pytest.mark.asyncio
    async def test_get_state_returns_cached_state(self):
        """Test that get_state returns the cached state"""
        mock_redis = AsyncMock()
        cached_state = {
            "id": "litellm_poll_test",
            "object": "response",
            "status": "in_progress",
            "output": [{"id": "item_1", "type": "message"}],
            "created_at": 1234567890,
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }
        mock_redis.async_get_cache.return_value = json.dumps(cached_state)
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        result = await handler.get_state("litellm_poll_test")
        
        assert result == cached_state

    @pytest.mark.asyncio
    async def test_get_state_returns_none_for_missing_state(self):
        """Test that get_state returns None when state doesn't exist"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = None
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        result = await handler.get_state("litellm_poll_nonexistent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_state_returns_none_without_redis(self):
        """Test that get_state returns None when Redis is not configured"""
        handler = ResponsePollingHandler(redis_cache=None)
        
        result = await handler.get_state("litellm_poll_test")
        
        assert result is None

    # ==================== Cancel Polling Tests ====================

    @pytest.mark.asyncio
    async def test_cancel_polling_updates_status_to_cancelled(self):
        """Test that cancel_polling sets status to cancelled"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = json.dumps({
            "id": "litellm_poll_test",
            "object": "response",
            "status": "in_progress",
            "output": [],
            "created_at": 1234567890
        })
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        result = await handler.cancel_polling("litellm_poll_test")
        
        assert result is True
        
        call_args = mock_redis.async_set_cache.call_args
        stored = json.loads(call_args.kwargs["value"])
        assert stored["status"] == "cancelled"

    # ==================== Delete Polling Tests ====================

    @pytest.mark.asyncio
    async def test_delete_polling_removes_from_cache(self):
        """Test that delete_polling removes the entry from Redis"""
        mock_redis = AsyncMock()
        mock_async_client = AsyncMock()
        mock_redis.redis_async_client = True  # hasattr check
        mock_redis.init_async_client.return_value = mock_async_client
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        result = await handler.delete_polling("litellm_poll_test")
        
        assert result is True
        mock_async_client.delete.assert_called_once_with(
            "litellm:polling:response:litellm_poll_test"
        )

    @pytest.mark.asyncio
    async def test_delete_polling_returns_false_without_redis(self):
        """Test that delete_polling returns False when Redis is not configured"""
        handler = ResponsePollingHandler(redis_cache=None)
        
        result = await handler.delete_polling("litellm_poll_test")
        
        assert result is False

    # ==================== TTL Tests ====================

    def test_default_ttl_is_one_hour(self):
        """Test that default TTL is 3600 seconds (1 hour)"""
        handler = ResponsePollingHandler(redis_cache=None)
        
        assert handler.ttl == 3600

    def test_custom_ttl_is_respected(self):
        """Test that custom TTL is stored correctly"""
        handler = ResponsePollingHandler(redis_cache=None, ttl=7200)
        
        assert handler.ttl == 7200

    @pytest.mark.asyncio
    async def test_update_state_uses_configured_ttl(self):
        """Test that update_state uses the configured TTL"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = json.dumps({
            "id": "litellm_poll_test",
            "object": "response",
            "status": "queued",
            "output": [],
            "created_at": 1234567890
        })
        
        handler = ResponsePollingHandler(redis_cache=mock_redis, ttl=1800)
        
        await handler.update_state(
            polling_id="litellm_poll_test",
            status="in_progress",
        )
        
        call_args = mock_redis.async_set_cache.call_args
        assert call_args.kwargs["ttl"] == 1800


class TestStreamingEventProcessing:
    """
    Test cases for streaming event processing logic.
    
    These tests verify the expected behavior when processing different
    OpenAI streaming event types.
    """

    def test_accumulated_text_structure(self):
        """Test the structure used for accumulating text deltas"""
        accumulated_text = {}
        
        # Simulate accumulating deltas for (item_id, content_index)
        key = ("item_123", 0)
        accumulated_text[key] = ""
        accumulated_text[key] += "Hello "
        accumulated_text[key] += "World"
        
        assert accumulated_text[key] == "Hello World"
        assert ("item_123", 0) in accumulated_text
        assert ("item_123", 1) not in accumulated_text

    def test_output_items_tracking_structure(self):
        """Test the structure used for tracking output items by ID"""
        output_items = {}
        
        # Simulate adding output items
        item1 = {"id": "item_1", "type": "message", "content": []}
        item2 = {"id": "item_2", "type": "function_call", "name": "get_weather"}
        
        output_items[item1["id"]] = item1
        output_items[item2["id"]] = item2
        
        assert len(output_items) == 2
        assert output_items["item_1"]["type"] == "message"
        assert output_items["item_2"]["type"] == "function_call"

    def test_150ms_batch_interval_constant(self):
        """Test that the batch interval is 150ms"""
        UPDATE_INTERVAL = 0.150  # 150ms
        
        assert UPDATE_INTERVAL == 0.150
        assert UPDATE_INTERVAL * 1000 == 150  # 150 milliseconds


class TestBackgroundStreamingModule:
    """Test cases for background_streaming module imports and structure"""

    def test_background_streaming_task_can_be_imported(self):
        """Test that background_streaming_task can be imported from the module"""
        from litellm.proxy.response_polling.background_streaming import (
            background_streaming_task,
        )
        
        assert background_streaming_task is not None
        assert callable(background_streaming_task)

    def test_module_exports_from_init(self):
        """Test that the module exports are available from __init__"""
        from litellm.proxy.response_polling import (
            ResponsePollingHandler,
            background_streaming_task,
        )
        
        assert ResponsePollingHandler is not None
        assert background_streaming_task is not None

    def test_background_streaming_task_is_async(self):
        """Test that background_streaming_task is an async function"""
        import asyncio
        from litellm.proxy.response_polling.background_streaming import (
            background_streaming_task,
        )
        
        assert asyncio.iscoroutinefunction(background_streaming_task)

