import pytest
import asyncio
import os
from litellm import Router


# Mark as async test
@pytest.mark.asyncio
async def test_router_uses_correct_redis_db():
    """
    Verifies that when redis_db is passed to Router,
    items are actually stored in that specific Redis DB index.
    """
    # 1. Setup - Use a non-standard DB index (e.g., 5) to prove it's not using default 0
    test_db_index = 5

    # Ensure we have a Redis URL available (fallback to localhost if env var not set)
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = os.getenv("REDIS_PORT", "6379")

    # Initialize Router with specific redis_db
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        redis_host=redis_host,
        redis_port=int(redis_port),
        redis_db=test_db_index,
        cache_responses=True,  # Important: Enable caching to trigger Redis usage
    )

    # 2. Verify Internal State
    # Check if the underlying cache client is configured with the correct DB
    # Accessing internal attributes for verification purposes
    try:
        if router.cache.redis_cache:
            # Check connection kwargs or internal client db
            cache_client = router.cache.redis_cache.redis_client
            # Redis client stores connection args in connection_pool.connection_kwargs
            conn_kwargs = cache_client.connection_pool.connection_kwargs

            assert str(conn_kwargs.get("db")) == str(
                test_db_index
            ), f"Router Internal Check Failed: Expected DB {test_db_index}, got {conn_kwargs.get('db')}"
        else:
            pytest.fail("Redis cache was not initialized in Router")

    except Exception as e:
        pytest.fail(f"Failed to inspect Router internals: {e}")


if __name__ == "__main__":
    asyncio.run(test_router_uses_correct_redis_db())
