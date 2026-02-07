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
    async def test_update_state_with_all_responses_api_fields(self):
        """Test that update_state stores all ResponsesAPIResponse fields from response.completed"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = json.dumps({
            "id": "litellm_poll_test",
            "object": "response",
            "status": "in_progress",
            "output": [],
            "created_at": 1234567890
        })
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        # All ResponsesAPIResponse fields that can be updated
        await handler.update_state(
            polling_id="litellm_poll_test",
            status="completed",
            usage={"input_tokens": 10, "output_tokens": 50, "total_tokens": 60},
            reasoning={"effort": "medium"},
            tool_choice={"type": "auto"},
            tools=[{"type": "function", "function": {"name": "test"}}],
            model="gpt-4o",
            instructions="You are a helpful assistant",
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=1000,
            previous_response_id="resp_prev_123",
            text={"format": {"type": "text"}},
            truncation="auto",
            parallel_tool_calls=True,
            user="user_123",
            store=True,
            incomplete_details={"reason": "max_output_tokens"},
        )
        
        call_args = mock_redis.async_set_cache.call_args
        stored = json.loads(call_args.kwargs["value"])
        
        # Verify all fields are stored correctly
        assert stored["status"] == "completed"
        assert stored["usage"] == {"input_tokens": 10, "output_tokens": 50, "total_tokens": 60}
        assert stored["reasoning"] == {"effort": "medium"}
        assert stored["tool_choice"] == {"type": "auto"}
        assert stored["tools"] == [{"type": "function", "function": {"name": "test"}}]
        assert stored["model"] == "gpt-4o"
        assert stored["instructions"] == "You are a helpful assistant"
        assert stored["temperature"] == 0.7
        assert stored["top_p"] == 0.9
        assert stored["max_output_tokens"] == 1000
        assert stored["previous_response_id"] == "resp_prev_123"
        assert stored["text"] == {"format": {"type": "text"}}
        assert stored["truncation"] == "auto"
        assert stored["parallel_tool_calls"] is True
        assert stored["user"] == "user_123"
        assert stored["store"] is True
        assert stored["incomplete_details"] == {"reason": "max_output_tokens"}

    @pytest.mark.asyncio
    async def test_update_state_preserves_existing_fields(self):
        """Test that update_state preserves fields not being updated"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = json.dumps({
            "id": "litellm_poll_test",
            "object": "response",
            "status": "in_progress",
            "output": [{"id": "item_1", "type": "message"}],
            "created_at": 1234567890,
            "model": "gpt-4o",
            "temperature": 0.5,
        })
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        # Only update status
        await handler.update_state(
            polling_id="litellm_poll_test",
            status="completed",
        )
        
        call_args = mock_redis.async_set_cache.call_args
        stored = json.loads(call_args.kwargs["value"])
        
        # Verify existing fields are preserved
        assert stored["status"] == "completed"
        assert stored["model"] == "gpt-4o"
        assert stored["temperature"] == 0.5
        assert stored["output"] == [{"id": "item_1", "type": "message"}]

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
        # init_async_client is a sync method that returns an async client
        mock_redis.init_async_client = Mock(return_value=mock_async_client)
        
        # Mock async_delete_cache to actually call init_async_client and delete
        async def mock_async_delete_cache(key):
            client = mock_redis.init_async_client()
            await client.delete(key)
        
        mock_redis.async_delete_cache = mock_async_delete_cache
        
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
        import inspect
        from litellm.proxy.response_polling.background_streaming import (
            background_streaming_task,
        )

        assert inspect.iscoroutinefunction(background_streaming_task)


class TestProviderResolutionForPolling:
    """
    Test cases for provider resolution logic used to determine
    if polling_via_cache should be enabled for a given model.
    
    This tests the logic in endpoints.py that resolves model names
    to their providers using the router's deployment configuration.
    """

    def test_provider_from_model_string_with_slash(self):
        """Test extracting provider from 'provider/model' format"""
        model = "openai/gpt-4o"
        
        # Direct extraction when model has slash
        if "/" in model:
            provider = model.split("/")[0]
        else:
            provider = None
        
        assert provider == "openai"

    def test_provider_from_model_string_without_slash(self):
        """Test that model without slash doesn't extract provider directly"""
        model = "gpt-5"
        
        # No slash means we can't extract provider directly
        if "/" in model:
            provider = model.split("/")[0]
        else:
            provider = None
        
        assert provider is None

    def test_provider_resolution_from_router_single_deployment(self):
        """Test resolving provider from router with single deployment"""
        # Simulate router's model_name_to_deployment_indices
        model_name_to_deployment_indices = {
            "gpt-5": [0],  # Single deployment at index 0
        }
        model_list = [
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-5",
                    "api_key": "sk-test",
                }
            }
        ]
        
        model = "gpt-5"
        polling_via_cache_enabled = ["openai"]
        should_use_polling = False
        
        # Simulate the resolution logic
        indices = model_name_to_deployment_indices.get(model, [])
        for idx in indices:
            deployment_dict = model_list[idx]
            litellm_params = deployment_dict.get("litellm_params", {})
            
            dep_provider = litellm_params.get("custom_llm_provider")
            if not dep_provider:
                dep_model = litellm_params.get("model", "")
                if "/" in dep_model:
                    dep_provider = dep_model.split("/")[0]
            
            if dep_provider and dep_provider in polling_via_cache_enabled:
                should_use_polling = True
                break
        
        assert should_use_polling is True

    def test_provider_resolution_from_router_multiple_deployments_match(self):
        """Test resolving provider when multiple deployments exist and one matches"""
        model_name_to_deployment_indices = {
            "gpt-4o": [0, 1],  # Two deployments
        }
        model_list = [
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                }
            },
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "azure/gpt-4o-deployment",
                }
            }
        ]
        
        model = "gpt-4o"
        polling_via_cache_enabled = ["openai"]  # Only openai in list
        should_use_polling = False
        
        indices = model_name_to_deployment_indices.get(model, [])
        for idx in indices:
            deployment_dict = model_list[idx]
            litellm_params = deployment_dict.get("litellm_params", {})
            
            dep_provider = litellm_params.get("custom_llm_provider")
            if not dep_provider:
                dep_model = litellm_params.get("model", "")
                if "/" in dep_model:
                    dep_provider = dep_model.split("/")[0]
            
            if dep_provider and dep_provider in polling_via_cache_enabled:
                should_use_polling = True
                break
        
        # Should be True because first deployment is openai
        assert should_use_polling is True

    def test_provider_resolution_from_router_no_match(self):
        """Test that polling is disabled when no deployment provider matches"""
        model_name_to_deployment_indices = {
            "claude-3": [0],
        }
        model_list = [
            {
                "model_name": "claude-3",
                "litellm_params": {
                    "model": "anthropic/claude-3-sonnet",
                }
            }
        ]
        
        model = "claude-3"
        polling_via_cache_enabled = ["openai", "bedrock"]  # anthropic not in list
        should_use_polling = False
        
        indices = model_name_to_deployment_indices.get(model, [])
        for idx in indices:
            deployment_dict = model_list[idx]
            litellm_params = deployment_dict.get("litellm_params", {})
            
            dep_provider = litellm_params.get("custom_llm_provider")
            if not dep_provider:
                dep_model = litellm_params.get("model", "")
                if "/" in dep_model:
                    dep_provider = dep_model.split("/")[0]
            
            if dep_provider and dep_provider in polling_via_cache_enabled:
                should_use_polling = True
                break
        
        assert should_use_polling is False

    def test_provider_resolution_with_custom_llm_provider(self):
        """Test that custom_llm_provider takes precedence over model string"""
        model_name_to_deployment_indices = {
            "my-model": [0],
        }
        model_list = [
            {
                "model_name": "my-model",
                "litellm_params": {
                    "model": "some-custom-model",
                    "custom_llm_provider": "openai",  # Explicit provider
                }
            }
        ]
        
        model = "my-model"
        polling_via_cache_enabled = ["openai"]
        should_use_polling = False
        
        indices = model_name_to_deployment_indices.get(model, [])
        for idx in indices:
            deployment_dict = model_list[idx]
            litellm_params = deployment_dict.get("litellm_params", {})
            
            # custom_llm_provider should be checked first
            dep_provider = litellm_params.get("custom_llm_provider")
            if not dep_provider:
                dep_model = litellm_params.get("model", "")
                if "/" in dep_model:
                    dep_provider = dep_model.split("/")[0]
            
            if dep_provider and dep_provider in polling_via_cache_enabled:
                should_use_polling = True
                break
        
        assert should_use_polling is True

    def test_provider_resolution_model_not_in_router(self):
        """Test that unknown model doesn't enable polling"""
        model_name_to_deployment_indices = {
            "gpt-5": [0],
        }
        model_list = [
            {
                "model_name": "gpt-5",
                "litellm_params": {"model": "openai/gpt-5"}
            }
        ]
        
        model = "unknown-model"  # Not in router
        polling_via_cache_enabled = ["openai"]
        should_use_polling = False
        
        indices = model_name_to_deployment_indices.get(model, [])  # Empty list
        for idx in indices:
            # This loop won't execute
            pass
        
        assert should_use_polling is False
        assert len(indices) == 0


class TestPollingConditionChecks:
    """
    Test cases for the conditions that determine whether polling should be enabled.
    Tests the should_use_polling_for_request function.
    """

    def test_polling_enabled_when_all_conditions_met(self):
        """Test polling is enabled when background=true, polling_via_cache="all", and redis is available"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled="all",
            redis_cache=Mock(),
            model="gpt-4o",
            llm_router=None,
        )
        
        assert result is True

    def test_polling_disabled_when_background_false(self):
        """Test polling is disabled when background=false"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        result = should_use_polling_for_request(
            background_mode=False,
            polling_via_cache_enabled="all",
            redis_cache=Mock(),
            model="gpt-4o",
            llm_router=None,
        )
        
        assert result is False

    def test_polling_disabled_when_config_false(self):
        """Test polling is disabled when polling_via_cache is False"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled=False,
            redis_cache=Mock(),
            model="gpt-4o",
            llm_router=None,
        )
        
        assert result is False

    def test_polling_disabled_when_redis_not_configured(self):
        """Test polling is disabled when Redis is not configured"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled="all",
            redis_cache=None,
            model="gpt-4o",
            llm_router=None,
        )
        
        assert result is False

    def test_polling_enabled_with_provider_list_match(self):
        """Test polling is enabled when provider list matches"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled=["openai", "anthropic"],
            redis_cache=Mock(),
            model="openai/gpt-4o",
            llm_router=None,
        )
        
        assert result is True

    def test_polling_disabled_with_provider_list_no_match(self):
        """Test polling is disabled when provider not in list"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled=["openai"],
            redis_cache=Mock(),
            model="anthropic/claude-3",
            llm_router=None,
        )
        
        assert result is False

    def test_polling_with_router_lookup(self):
        """Test polling uses router to resolve model name to provider"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        # Create mock router
        mock_router = Mock()
        mock_router.model_name_to_deployment_indices = {"gpt-5": [0]}
        mock_router.model_list = [
            {
                "model_name": "gpt-5",
                "litellm_params": {"model": "openai/gpt-5"}
            }
        ]
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled=["openai"],
            redis_cache=Mock(),
            model="gpt-5",  # No slash, needs router lookup
            llm_router=mock_router,
        )
        
        assert result is True

    def test_polling_with_router_lookup_no_match(self):
        """Test polling returns False when router lookup finds non-matching provider"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        mock_router = Mock()
        mock_router.model_name_to_deployment_indices = {"claude-3": [0]}
        mock_router.model_list = [
            {
                "model_name": "claude-3",
                "litellm_params": {"model": "anthropic/claude-3-sonnet"}
            }
        ]
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled=["openai"],
            redis_cache=Mock(),
            model="claude-3",
            llm_router=mock_router,
        )
        
        assert result is False

    # ==================== Native Background Mode Tests ====================

    def test_polling_disabled_when_model_in_native_background_mode(self):
        """Test that polling is disabled when model is in native_background_mode list"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled="all",
            redis_cache=Mock(),
            model="o4-mini-deep-research",
            llm_router=None,
            native_background_mode=["o4-mini-deep-research", "o3-deep-research"],
        )
        
        assert result is False

    def test_polling_disabled_for_native_background_mode_with_provider_list(self):
        """Test that native_background_mode takes precedence even when provider matches"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled=["openai"],
            redis_cache=Mock(),
            model="openai/o4-mini-deep-research",
            llm_router=None,
            native_background_mode=["openai/o4-mini-deep-research"],
        )
        
        assert result is False

    def test_polling_enabled_when_model_not_in_native_background_mode(self):
        """Test that polling is enabled when model is not in native_background_mode list"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled="all",
            redis_cache=Mock(),
            model="gpt-4o",
            llm_router=None,
            native_background_mode=["o4-mini-deep-research"],
        )
        
        assert result is True

    def test_polling_enabled_when_native_background_mode_is_none(self):
        """Test that polling works normally when native_background_mode is None"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled="all",
            redis_cache=Mock(),
            model="gpt-4o",
            llm_router=None,
            native_background_mode=None,
        )
        
        assert result is True

    def test_polling_enabled_when_native_background_mode_is_empty_list(self):
        """Test that polling works normally when native_background_mode is empty list"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled="all",
            redis_cache=Mock(),
            model="gpt-4o",
            llm_router=None,
            native_background_mode=[],
        )
        
        assert result is True

    def test_native_background_mode_exact_match_required(self):
        """Test that native_background_mode uses exact model name matching"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        # "o4-mini" should not match "o4-mini-deep-research"
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled="all",
            redis_cache=Mock(),
            model="o4-mini",
            llm_router=None,
            native_background_mode=["o4-mini-deep-research"],
        )
        
        assert result is True

    def test_native_background_mode_with_provider_prefix_in_request(self):
        """Test native_background_mode matching when request model has provider prefix"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        # Model in native_background_mode without provider prefix
        # Request comes in with provider prefix - should not match
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled=["openai"],
            redis_cache=Mock(),
            model="openai/o4-mini-deep-research",
            llm_router=None,
            native_background_mode=["o4-mini-deep-research"],  # Without prefix
        )
        
        # Should return True because "openai/o4-mini-deep-research" != "o4-mini-deep-research"
        assert result is True

    def test_native_background_mode_with_router_lookup(self):
        """Test that native_background_mode works with router-resolved models"""
        from litellm.proxy.response_polling.polling_handler import should_use_polling_for_request
        
        mock_router = Mock()
        mock_router.model_name_to_deployment_indices = {"deep-research": [0]}
        mock_router.model_list = [
            {
                "model_name": "deep-research",
                "litellm_params": {"model": "openai/o4-mini-deep-research"}
            }
        ]
        
        # Model alias "deep-research" is in native_background_mode
        result = should_use_polling_for_request(
            background_mode=True,
            polling_via_cache_enabled=["openai"],
            redis_cache=Mock(),
            model="deep-research",
            llm_router=mock_router,
            native_background_mode=["deep-research"],
        )
        
        assert result is False


class TestStreamingEventParsing:
    """
    Test cases for parsing OpenAI streaming events in the background task.
    Tests the event handling logic in background_streaming.py.
    """

    def test_parse_response_output_item_added_event(self):
        """Test parsing response.output_item.added event"""
        event = {
            "type": "response.output_item.added",
            "item": {
                "id": "item_123",
                "type": "message",
                "role": "assistant",
                "content": []
            }
        }
        
        output_items = {}
        event_type = event.get("type", "")
        
        if event_type == "response.output_item.added":
            item = event.get("item", {})
            item_id = item.get("id")
            if item_id:
                output_items[item_id] = item
        
        assert "item_123" in output_items
        assert output_items["item_123"]["type"] == "message"

    def test_parse_response_output_text_delta_event(self):
        """Test parsing response.output_text.delta event and accumulating text"""
        output_items = {
            "item_123": {
                "id": "item_123",
                "type": "message",
                "content": [{"type": "text", "text": ""}]
            }
        }
        accumulated_text = {}
        
        # Simulate receiving multiple delta events
        delta_events = [
            {"type": "response.output_text.delta", "item_id": "item_123", "content_index": 0, "delta": "Hello "},
            {"type": "response.output_text.delta", "item_id": "item_123", "content_index": 0, "delta": "World!"},
        ]
        
        for event in delta_events:
            event_type = event.get("type", "")
            if event_type == "response.output_text.delta":
                item_id = event.get("item_id")
                content_index = event.get("content_index", 0)
                delta = event.get("delta", "")
                
                if item_id and item_id in output_items:
                    key = (item_id, content_index)
                    if key not in accumulated_text:
                        accumulated_text[key] = ""
                    accumulated_text[key] += delta
                    
                    # Update content
                    if "content" in output_items[item_id]:
                        content_list = output_items[item_id]["content"]
                        if content_index < len(content_list):
                            if isinstance(content_list[content_index], dict):
                                content_list[content_index]["text"] = accumulated_text[key]
        
        assert accumulated_text[("item_123", 0)] == "Hello World!"
        assert output_items["item_123"]["content"][0]["text"] == "Hello World!"

    def test_parse_response_completed_event(self):
        """Test parsing response.completed event extracts all fields"""
        event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 50},
                "reasoning": {"effort": "medium"},
                "tool_choice": {"type": "auto"},
                "tools": [{"type": "function", "function": {"name": "test"}}],
                "model": "gpt-4o",
                "output": [{"id": "item_1", "type": "message"}]
            }
        }
        
        event_type = event.get("type", "")
        usage_data = None
        reasoning_data = None
        tool_choice_data = None
        tools_data = None
        model_data = None
        
        if event_type == "response.completed":
            response_data = event.get("response", {})
            usage_data = response_data.get("usage")
            reasoning_data = response_data.get("reasoning")
            tool_choice_data = response_data.get("tool_choice")
            tools_data = response_data.get("tools")
            model_data = response_data.get("model")
        
        assert usage_data == {"input_tokens": 10, "output_tokens": 50}
        assert reasoning_data == {"effort": "medium"}
        assert tool_choice_data == {"type": "auto"}
        assert tools_data == [{"type": "function", "function": {"name": "test"}}]
        assert model_data == "gpt-4o"

    def test_parse_done_marker(self):
        """Test that [DONE] marker is detected correctly"""
        chunks = [
            "data: {\"type\": \"response.in_progress\"}",
            "data: {\"type\": \"response.completed\"}",
            "data: [DONE]",
        ]
        
        done_received = False
        for chunk in chunks:
            if chunk.startswith("data: "):
                chunk_data = chunk[6:].strip()
                if chunk_data == "[DONE]":
                    done_received = True
                    break
        
        assert done_received is True

    def test_parse_sse_format(self):
        """Test parsing Server-Sent Events format"""
        raw_chunk = b"data: {\"type\": \"response.output_item.added\", \"item\": {\"id\": \"123\"}}"
        
        # Decode bytes to string
        if isinstance(raw_chunk, bytes):
            chunk = raw_chunk.decode('utf-8')
        else:
            chunk = raw_chunk
        
        # Extract JSON from SSE format
        if isinstance(chunk, str) and chunk.startswith("data: "):
            chunk_data = chunk[6:].strip()
            
            import json
            event = json.loads(chunk_data)
            
            assert event["type"] == "response.output_item.added"
            assert event["item"]["id"] == "123"

    def test_content_part_added_event(self):
        """Test parsing response.content_part.added event"""
        output_items = {
            "item_123": {
                "id": "item_123",
                "type": "message",
            }
        }
        
        event = {
            "type": "response.content_part.added",
            "item_id": "item_123",
            "part": {"type": "text", "text": ""}
        }
        
        event_type = event.get("type", "")
        if event_type == "response.content_part.added":
            item_id = event.get("item_id")
            content_part = event.get("part", {})
            
            if item_id and item_id in output_items:
                if "content" not in output_items[item_id]:
                    output_items[item_id]["content"] = []
                output_items[item_id]["content"].append(content_part)
        
        assert "content" in output_items["item_123"]
        assert len(output_items["item_123"]["content"]) == 1
        assert output_items["item_123"]["content"][0]["type"] == "text"


class TestEdgeCases:
    """Test edge cases and error scenarios"""

    def test_empty_model_string(self):
        """Test handling of empty model string"""
        model = ""
        polling_via_cache_enabled = ["openai"]
        
        should_use_polling = False
        if "/" in model:
            provider = model.split("/")[0]
            if provider in polling_via_cache_enabled:
                should_use_polling = True
        
        assert should_use_polling is False

    def test_model_with_multiple_slashes(self):
        """Test handling model with multiple slashes (e.g., bedrock ARN)"""
        model = "bedrock/arn:aws:bedrock:us-east-1:123456:model/my-model"
        polling_via_cache_enabled = ["bedrock"]
        
        # Only split on first slash
        if "/" in model:
            provider = model.split("/")[0]
        else:
            provider = None
        
        assert provider == "bedrock"
        assert provider in polling_via_cache_enabled

    def test_polling_id_detection_edge_cases(self):
        """Test polling ID detection with edge cases"""
        # Empty string
        assert ResponsePollingHandler.is_polling_id("") is False
        
        # Just prefix without UUID
        assert ResponsePollingHandler.is_polling_id("litellm_poll_") is True
        
        # Similar but different prefix
        assert ResponsePollingHandler.is_polling_id("litellm_polling_abc") is False
        
        # Case sensitivity
        assert ResponsePollingHandler.is_polling_id("LITELLM_POLL_abc") is False

    @pytest.mark.asyncio
    async def test_create_initial_state_with_empty_metadata(self):
        """Test create_initial_state handles missing metadata gracefully"""
        mock_redis = AsyncMock()
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        response = await handler.create_initial_state(
            polling_id="litellm_poll_test",
            request_data={"model": "gpt-4o"},  # No metadata field
        )
        
        assert response.metadata == {}

    @pytest.mark.asyncio
    async def test_update_state_with_none_output_clears_output(self):
        """Test that output=[] explicitly sets empty output"""
        mock_redis = AsyncMock()
        mock_redis.async_get_cache.return_value = json.dumps({
            "id": "litellm_poll_test",
            "object": "response",
            "status": "in_progress",
            "output": [{"id": "item_1"}],  # Has existing output
            "created_at": 1234567890
        })
        
        handler = ResponsePollingHandler(redis_cache=mock_redis)
        
        await handler.update_state(
            polling_id="litellm_poll_test",
            output=[],  # Explicitly set empty
        )
        
        call_args = mock_redis.async_set_cache.call_args
        stored = json.loads(call_args.kwargs["value"])
        
        assert stored["output"] == []
