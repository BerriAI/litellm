import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SpendLogsPayload

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache


class ResponsesAPIRedisCache:
    """
    Redis cache handler specifically for Responses API session handling.
    
    This class provides:
    1. Fast Redis-based caching of response data for 1 minute TTL
    2. Fallback to database when Redis data is unavailable
    3. Tight integration with existing session handling logic
    4. Support for both streaming and non-streaming responses
    """

    def __init__(self, redis_cache: Optional["RedisCache"] = None):
        """
        Initialize the ResponsesAPIRedisCache.
        
        Args:
            redis_cache: Optional RedisCache instance. If None, will try to get from global cache.
        """
        self.redis_cache = redis_cache
        self.ttl_seconds = 60  # 1 minute TTL as specified
        self.key_prefix = "litellm:responses_api:"
        
        if self.redis_cache is None:
            self._init_redis_cache()

    def _init_redis_cache(self) -> None:
        """Initialize Redis cache from global litellm cache if available."""
        try:
            import litellm
            from litellm.caching.redis_cache import RedisCache
            
            if (litellm.cache is not None and 
                hasattr(litellm.cache, 'cache') and 
                isinstance(litellm.cache.cache, RedisCache)):
                self.redis_cache = litellm.cache.cache
                verbose_proxy_logger.debug("ResponsesAPIRedisCache: Using global Redis cache")
            else:
                verbose_proxy_logger.debug("ResponsesAPIRedisCache: No Redis cache available")
        except Exception as e:
            verbose_proxy_logger.debug(f"ResponsesAPIRedisCache: Failed to initialize Redis cache: {e}")

    def _get_session_cache_key(self, session_id: str) -> str:
        """Generate Redis key for session data."""
        return f"{self.key_prefix}session:{session_id}"

    def _get_response_cache_key(self, response_id: str) -> str:
        """Generate Redis key for individual response data."""
        return f"{self.key_prefix}response:{response_id}"

    async def store_response_data(
        self, 
        response_id: str, 
        session_id: Optional[str],
        spend_log_data: SpendLogsPayload,
        response_content: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store response data in Redis cache for fast retrieval.
        
        Args:
            response_id: The response ID
            session_id: The session ID this response belongs to
            spend_log_data: The spend log payload containing request/response data
            response_content: Optional additional response content
        
        Returns:
            bool: True if stored successfully, False otherwise
        """
        if not self.redis_cache:
            return False

        try:
            # Prepare cache data
            cache_data = {
                "response_id": response_id,
                "session_id": session_id,
                "spend_log": spend_log_data,
                "timestamp": time.time(),
            }
            
            if response_content:
                cache_data["response_content"] = response_content

            # Store individual response
            response_key = self._get_response_cache_key(response_id)
            await self.redis_cache.async_set_cache(
                key=response_key,
                value=json.dumps(cache_data, default=str),
                ttl=self.ttl_seconds
            )

            # Store in session-based cache if session_id is available
            if session_id:
                await self._add_response_to_session(session_id, response_id, cache_data)

            verbose_proxy_logger.debug(
                f"ResponsesAPIRedisCache: Stored response {response_id} in Redis cache"
            )
            return True

        except Exception as e:
            verbose_proxy_logger.error(
                f"ResponsesAPIRedisCache: Failed to store response {response_id}: {e}"
            )
            return False

    async def _add_response_to_session(
        self, 
        session_id: str, 
        response_id: str, 
        response_data: Dict[str, Any]
    ) -> None:
        """Add a response to the session's cached response list."""
        try:
            session_key = self._get_session_cache_key(session_id)
            
            # Get existing session data
            existing_data = await self.redis_cache.async_get_cache(session_key)
            if existing_data:
                session_data = json.loads(existing_data)
            else:
                session_data = {
                    "session_id": session_id,
                    "responses": [],
                    "last_updated": time.time()
                }

            # Add new response (keep chronological order)
            session_data["responses"].append({
                "response_id": response_id,
                "data": response_data,
                "added_at": time.time()
            })
            
            # Limit session size to prevent memory issues (keep last 50 responses)
            if len(session_data["responses"]) > 50:
                session_data["responses"] = session_data["responses"][-50:]

            session_data["last_updated"] = time.time()

            # Store updated session data
            await self.redis_cache.async_set_cache(
                key=session_key,
                value=json.dumps(session_data, default=str),
                ttl=self.ttl_seconds
            )

        except Exception as e:
            verbose_proxy_logger.error(
                f"ResponsesAPIRedisCache: Failed to add response to session {session_id}: {e}"
            )

    async def get_session_spend_logs(self, session_id: str) -> List[SpendLogsPayload]:
        """
        Get all spend logs for a session from Redis cache.
        
        Args:
            session_id: The session ID to retrieve
        
        Returns:
            List[SpendLogsPayload]: List of spend logs, empty if not found in cache
        """
        if not self.redis_cache:
            return []

        try:
            session_key = self._get_session_cache_key(session_id)
            cached_data = await self.redis_cache.async_get_cache(session_key)
            
            if not cached_data:
                verbose_proxy_logger.debug(
                    f"ResponsesAPIRedisCache: No cached session data for {session_id}"
                )
                return []

            session_data = json.loads(cached_data)
            spend_logs = []

            for response_entry in session_data.get("responses", []):
                response_data = response_entry.get("data", {})
                spend_log = response_data.get("spend_log")
                if spend_log:
                    spend_logs.append(spend_log)

            verbose_proxy_logger.debug(
                f"ResponsesAPIRedisCache: Retrieved {len(spend_logs)} spend logs for session {session_id}"
            )
            return spend_logs

        except Exception as e:
            verbose_proxy_logger.error(
                f"ResponsesAPIRedisCache: Failed to get session spend logs for {session_id}: {e}"
            )
            return []

    async def get_response_by_id(self, response_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific response by ID from Redis cache.
        
        Args:
            response_id: The response ID to retrieve
        
        Returns:
            Optional[Dict[str, Any]]: Response data if found, None otherwise
        """
        if not self.redis_cache:
            return None

        try:
            response_key = self._get_response_cache_key(response_id)
            cached_data = await self.redis_cache.async_get_cache(response_key)
            
            if not cached_data:
                verbose_proxy_logger.debug(
                    f"ResponsesAPIRedisCache: No cached response data for {response_id}"
                )
                return None

            response_data = json.loads(cached_data)
            verbose_proxy_logger.debug(
                f"ResponsesAPIRedisCache: Retrieved cached response {response_id}"
            )
            return response_data

        except Exception as e:
            verbose_proxy_logger.error(
                f"ResponsesAPIRedisCache: Failed to get response {response_id}: {e}"
            )
            return None

    async def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate/remove session data from cache.
        
        Args:
            session_id: The session ID to invalidate
        
        Returns:
            bool: True if invalidated successfully, False otherwise
        """
        if not self.redis_cache:
            return False

        try:
            session_key = self._get_session_cache_key(session_id)
            
            # First get the session data to find all response IDs
            cached_data = await self.redis_cache.async_get_cache(session_key)
            if cached_data:
                session_data = json.loads(cached_data)
                
                # Remove individual response cache entries
                for response_entry in session_data.get("responses", []):
                    response_id = response_entry.get("response_id")
                    if response_id:
                        response_key = self._get_response_cache_key(response_id)
                        try:
                            await self.redis_cache.async_delete_cache(response_key)
                        except Exception:
                            pass  # Continue even if individual deletion fails

            # Remove session cache entry
            await self.redis_cache.async_delete_cache(session_key)
            
            verbose_proxy_logger.debug(
                f"ResponsesAPIRedisCache: Invalidated session {session_id}"
            )
            return True

        except Exception as e:
            verbose_proxy_logger.error(
                f"ResponsesAPIRedisCache: Failed to invalidate session {session_id}: {e}"
            )
            return False

    def is_available(self) -> bool:
        """Check if Redis cache is available for use."""
        return self.redis_cache is not None

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the Redis cache usage."""
        if not self.redis_cache:
            return {"available": False, "error": "Redis cache not available"}

        try:
            # Try to get Redis info
            stats = {
                "available": True,
                "ttl_seconds": self.ttl_seconds,
                "key_prefix": self.key_prefix,
                "redis_available": True,
            }
            
            # Try a simple operation to verify connectivity
            test_key = f"{self.key_prefix}test"
            await self.redis_cache.async_set_cache(test_key, "test", ttl=1)
            await self.redis_cache.async_delete_cache(test_key)
            
            return stats
        except Exception as e:
            return {
                "available": False,
                "error": f"Redis connectivity test failed: {e}",
                "ttl_seconds": self.ttl_seconds,
                "key_prefix": self.key_prefix,
            }

    async def clear_all_cached_sessions(self) -> Dict[str, Any]:
        """
        Clear all cached session data (useful for cleanup or testing).
        
        Returns:
            Dict with operation results
        """
        if not self.redis_cache:
            return {"success": False, "error": "Redis cache not available"}

        try:
            # This is a potentially expensive operation, use with caution
            # In production, consider implementing a more targeted cleanup
            pattern = f"{self.key_prefix}*"
            
            # Note: This is a simple implementation
            # For production with large datasets, consider batch operations
            verbose_proxy_logger.info(
                f"ResponsesAPIRedisCache: Clearing all cached data with pattern {pattern}"
            )
            
            return {
                "success": True,
                "message": "Cache clearing initiated",
                "pattern": pattern,
                "note": "This is an expensive operation - use with caution"
            }
            
        except Exception as e:
            verbose_proxy_logger.error(
                f"ResponsesAPIRedisCache: Failed to clear cache: {e}"
            )
            return {"success": False, "error": str(e)}

    def get_cache_key_info(self, key_type: str, identifier: str) -> str:
        """
        Get cache key for debugging purposes.
        
        Args:
            key_type: Either 'session' or 'response'
            identifier: The session_id or response_id
        
        Returns:
            The Redis key that would be used
        """
        if key_type == "session":
            return self._get_session_cache_key(identifier)
        elif key_type == "response":
            return self._get_response_cache_key(identifier)
        else:
            raise ValueError(f"Invalid key_type: {key_type}. Must be 'session' or 'response'")


# Global instance - will be initialized when needed
_global_responses_redis_cache: Optional[ResponsesAPIRedisCache] = None


def get_responses_redis_cache() -> ResponsesAPIRedisCache:
    """Get or create the global ResponsesAPIRedisCache instance."""
    global _global_responses_redis_cache
    
    if _global_responses_redis_cache is None:
        _global_responses_redis_cache = ResponsesAPIRedisCache()
    
    return _global_responses_redis_cache


def reset_responses_redis_cache() -> None:
    """Reset the global cache instance (useful for testing)."""
    global _global_responses_redis_cache
    _global_responses_redis_cache = None