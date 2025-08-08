"""
Test suite for Redis Session Cache implementation for Responses API.

This tests the Redis caching layer that provides fast session retrieval
as an alternative to slow database queries.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Import the modules we're testing
from litellm.responses.redis_session_cache import (
    ResponsesAPIRedisCache,
    get_responses_redis_cache,
    reset_responses_redis_cache,
)


@pytest.fixture
def mock_redis_cache():
    """Create a mock Redis cache for testing."""
    mock_cache = AsyncMock()
    mock_cache.async_set_cache = AsyncMock(return_value=None)
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.async_delete_cache = AsyncMock(return_value=None)
    return mock_cache


@pytest.fixture
def redis_session_cache(mock_redis_cache):
    """Create a ResponsesAPIRedisCache instance with mocked Redis."""
    cache = ResponsesAPIRedisCache(redis_cache=mock_redis_cache)
    return cache


@pytest.fixture
def sample_spend_log():
    """Sample spend log data for testing."""
    return {
        "request_id": "resp_test_123",
        "session_id": "sess_test_456",
        "messages": json.dumps([{"role": "user", "content": "Hello"}]),
        "response": json.dumps({"choices": [{"message": {"content": "Hi there!"}}]}),
        "startTime": "2024-01-01T00:00:00Z",
        "endTime": "2024-01-01T00:00:01Z",
        "model": "test-model",
        "call_type": "aresponses",
    }


@pytest.fixture
def sample_response_content():
    """Sample response content for testing."""
    return {
        "id": "resp_test_123",
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant", 
                    "content": "Hi there!"
                }
            }
        ]
    }


class TestResponsesAPIRedisCache:
    """Test the main Redis cache functionality."""

    def test_init_with_redis_cache(self, mock_redis_cache):
        """Test initialization with provided Redis cache."""
        cache = ResponsesAPIRedisCache(redis_cache=mock_redis_cache)
        assert cache.redis_cache == mock_redis_cache
        assert cache.ttl_seconds == 60
        assert cache.key_prefix == "litellm:responses_api:"

    def test_init_without_redis_cache(self):
        """Test initialization without Redis cache (should try to auto-detect)."""
        cache = ResponsesAPIRedisCache()
        # Without global litellm cache, should be None
        assert cache.redis_cache is None

    def test_cache_key_generation(self, redis_session_cache):
        """Test Redis key generation for sessions and responses."""
        session_key = redis_session_cache._get_session_cache_key("test_session")
        response_key = redis_session_cache._get_response_cache_key("test_response")
        
        assert session_key == "litellm:responses_api:session:test_session"
        assert response_key == "litellm:responses_api:response:test_response"

    def test_get_cache_key_info(self, redis_session_cache):
        """Test the debug utility for getting cache keys."""
        session_key = redis_session_cache.get_cache_key_info("session", "test_session")
        response_key = redis_session_cache.get_cache_key_info("response", "test_response")
        
        assert session_key == "litellm:responses_api:session:test_session"
        assert response_key == "litellm:responses_api:response:test_response"
        
        with pytest.raises(ValueError):
            redis_session_cache.get_cache_key_info("invalid", "test")

    def test_is_available(self, mock_redis_cache):
        """Test availability checking."""
        # With Redis cache
        cache_with_redis = ResponsesAPIRedisCache(redis_cache=mock_redis_cache)
        assert cache_with_redis.is_available() is True
        
        # Without Redis cache
        cache_without_redis = ResponsesAPIRedisCache(redis_cache=None)
        assert cache_without_redis.is_available() is False

    @pytest.mark.asyncio
    async def test_store_response_data_success(
        self, redis_session_cache, sample_spend_log, sample_response_content
    ):
        """Test successful response data storage."""
        success = await redis_session_cache.store_response_data(
            response_id="resp_test_123",
            session_id="sess_test_456", 
            spend_log_data=sample_spend_log,
            response_content=sample_response_content,
        )
        
        assert success is True
        
        # Verify Redis cache calls
        redis_session_cache.redis_cache.async_set_cache.assert_called()
        
        # Check that both individual response and session were stored
        assert redis_session_cache.redis_cache.async_set_cache.call_count >= 2

    @pytest.mark.asyncio
    async def test_store_response_data_no_redis(self):
        """Test response data storage when Redis is unavailable."""
        cache = ResponsesAPIRedisCache(redis_cache=None)
        
        success = await cache.store_response_data(
            response_id="resp_test_123",
            session_id="sess_test_456",
            spend_log_data={"test": "data"},
        )
        
        assert success is False

    @pytest.mark.asyncio
    async def test_get_session_spend_logs_success(
        self, redis_session_cache, sample_spend_log
    ):
        """Test successful session spend logs retrieval."""
        # Mock Redis to return session data
        session_data = {
            "session_id": "sess_test_456",
            "responses": [
                {
                    "response_id": "resp_test_123",
                    "data": {
                        "spend_log": sample_spend_log
                    }
                }
            ]
        }
        
        redis_session_cache.redis_cache.async_get_cache.return_value = json.dumps(session_data)
        
        spend_logs = await redis_session_cache.get_session_spend_logs("sess_test_456")
        
        assert len(spend_logs) == 1
        assert spend_logs[0] == sample_spend_log

    @pytest.mark.asyncio
    async def test_get_session_spend_logs_cache_miss(self, redis_session_cache):
        """Test session spend logs retrieval when not in cache."""
        redis_session_cache.redis_cache.async_get_cache.return_value = None
        
        spend_logs = await redis_session_cache.get_session_spend_logs("sess_not_found")
        
        assert spend_logs == []

    @pytest.mark.asyncio
    async def test_get_response_by_id_success(
        self, redis_session_cache, sample_spend_log
    ):
        """Test successful response retrieval by ID."""
        response_data = {
            "response_id": "resp_test_123",
            "session_id": "sess_test_456",
            "spend_log": sample_spend_log
        }
        
        redis_session_cache.redis_cache.async_get_cache.return_value = json.dumps(response_data)
        
        result = await redis_session_cache.get_response_by_id("resp_test_123")
        
        assert result == response_data

    @pytest.mark.asyncio
    async def test_get_response_by_id_cache_miss(self, redis_session_cache):
        """Test response retrieval when not in cache."""
        redis_session_cache.redis_cache.async_get_cache.return_value = None
        
        result = await redis_session_cache.get_response_by_id("resp_not_found")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_session_success(self, redis_session_cache):
        """Test successful session invalidation."""
        # Mock session data with responses
        session_data = {
            "session_id": "sess_test_456",
            "responses": [
                {"response_id": "resp_1"},
                {"response_id": "resp_2"},
            ]
        }
        
        redis_session_cache.redis_cache.async_get_cache.return_value = json.dumps(session_data)
        
        success = await redis_session_cache.invalidate_session("sess_test_456")
        
        assert success is True
        
        # Should delete session and individual responses
        assert redis_session_cache.redis_cache.async_delete_cache.call_count >= 3

    @pytest.mark.asyncio
    async def test_get_cache_stats_success(self, redis_session_cache):
        """Test cache statistics retrieval when Redis is working."""
        stats = await redis_session_cache.get_cache_stats()
        
        assert stats["available"] is True
        assert stats["ttl_seconds"] == 60
        assert stats["key_prefix"] == "litellm:responses_api:"

    @pytest.mark.asyncio
    async def test_get_cache_stats_redis_failure(self, redis_session_cache):
        """Test cache statistics when Redis fails."""
        redis_session_cache.redis_cache.async_set_cache.side_effect = Exception("Redis down")
        
        stats = await redis_session_cache.get_cache_stats()
        
        assert stats["available"] is False
        assert "Redis connectivity test failed" in stats["error"]

    @pytest.mark.asyncio
    async def test_clear_all_cached_sessions(self, redis_session_cache):
        """Test clearing all cached sessions."""
        result = await redis_session_cache.clear_all_cached_sessions()
        
        assert result["success"] is True
        assert "litellm:responses_api:*" in result["pattern"]

    @pytest.mark.asyncio
    async def test_session_size_limiting(self, redis_session_cache, sample_spend_log):
        """Test that sessions are limited to 50 responses."""
        # Mock a session with 51 responses
        responses = []
        for i in range(51):
            responses.append({
                "response_id": f"resp_{i}",
                "data": {"spend_log": sample_spend_log},
                "added_at": i
            })
        
        session_data = {
            "session_id": "test_session",
            "responses": responses
        }
        
        redis_session_cache.redis_cache.async_get_cache.return_value = json.dumps(session_data)
        
        # Add one more response
        await redis_session_cache._add_response_to_session(
            "test_session", "resp_new", {"spend_log": sample_spend_log}
        )
        
        # Verify that the new session data was stored
        set_cache_calls = redis_session_cache.redis_cache.async_set_cache.call_args_list
        
        # Find the call that stores session data
        session_store_call = None
        for call in set_cache_calls:
            if "session:" in call[1]["key"]:
                session_store_call = call
                break
        
        assert session_store_call is not None
        
        # Parse the stored session data
        stored_data = json.loads(session_store_call[1]["value"])
        
        # Should be limited to 50 responses + 1 new = 50 total (oldest removed)
        assert len(stored_data["responses"]) == 50


class TestGlobalCacheInstance:
    """Test the global cache instance management."""

    def test_get_responses_redis_cache_singleton(self):
        """Test that get_responses_redis_cache returns a singleton."""
        # Reset first
        reset_responses_redis_cache()
        
        cache1 = get_responses_redis_cache()
        cache2 = get_responses_redis_cache()
        
        assert cache1 is cache2

    def test_reset_responses_redis_cache(self):
        """Test cache instance reset."""
        cache1 = get_responses_redis_cache()
        reset_responses_redis_cache()
        cache2 = get_responses_redis_cache()
        
        assert cache1 is not cache2


class TestIntegrationWithSessionHandler:
    """Test integration with the session handler."""

    @pytest.mark.asyncio 
    async def test_session_handler_redis_integration(self):
        """Test that session handler can use Redis cache."""
        with patch('litellm.responses.litellm_completion_transformation.session_handler.get_responses_redis_cache') as mock_get_cache:
            mock_cache_instance = AsyncMock()
            mock_cache_instance.is_available.return_value = True
            mock_cache_instance.get_response_by_id.return_value = {
                "session_id": "test_session",
                "spend_log": {"request_id": "test_resp"}
            }
            mock_cache_instance.get_session_spend_logs.return_value = [
                {"request_id": "test_resp", "session_id": "test_session"}
            ]
            mock_get_cache.return_value = mock_cache_instance
            
            # Import here to ensure patch is active
            from litellm.responses.litellm_completion_transformation.session_handler import ResponsesSessionHandler
            
            # Test the session handler using Redis cache
            result = await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id("test_resp")
            
            # Should get data from Redis cache
            assert len(result) == 1
            assert result[0]["request_id"] == "test_resp"


class TestDBSpendUpdateWriterIntegration:
    """Test integration with database spend update writer."""

    @pytest.mark.asyncio
    async def test_cache_response_in_redis_responses_api(self):
        """Test that responses API calls are cached in Redis."""
        with patch('litellm.proxy.db.db_spend_update_writer.get_responses_redis_cache') as mock_get_cache:
            mock_cache_instance = AsyncMock()
            mock_cache_instance.is_available.return_value = True
            mock_cache_instance.store_response_data.return_value = True
            mock_get_cache.return_value = mock_cache_instance
            
            # Import here to ensure patch is active
            from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
            
            writer = DBSpendUpdateWriter()
            
            # Test payload for responses API
            payload = {
                "request_id": "resp_test_123",
                "session_id": "sess_test_456",
                "call_type": "aresponses",
            }
            
            kwargs = {"call_type": "aresponses"}
            
            # Call the Redis caching method
            await writer._cache_response_in_redis(
                payload=payload,
                kwargs=kwargs, 
                completion_response=None
            )
            
            # Verify Redis cache was called
            mock_cache_instance.store_response_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_response_skips_non_responses_api(self):
        """Test that non-responses API calls are not cached."""
        with patch('litellm.proxy.db.db_spend_update_writer.get_responses_redis_cache') as mock_get_cache:
            mock_cache_instance = AsyncMock()
            mock_cache_instance.is_available.return_value = True
            mock_cache_instance.store_response_data.return_value = True
            mock_get_cache.return_value = mock_cache_instance
            
            # Import here to ensure patch is active
            from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
            
            writer = DBSpendUpdateWriter()
            
            # Test payload for non-responses API
            payload = {
                "request_id": "req_test_123",
                "call_type": "completion",  # Not responses API
            }
            
            kwargs = {"call_type": "completion"}
            
            # Call the Redis caching method
            await writer._cache_response_in_redis(
                payload=payload,
                kwargs=kwargs,
                completion_response=None
            )
            
            # Verify Redis cache was NOT called
            mock_cache_instance.store_response_data.assert_not_called()


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])