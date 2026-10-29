"""
Tests to prevent regression for issue #48589:
Sync Router.completion opens a new Redis connection on every call and never closes it
when a deployment has an rpm/tpm limit.
"""

from unittest.mock import MagicMock, patch

import litellm
from litellm import Router


def test_sync_router_completion_does_not_leak_async_redis_clients() -> None:
    """
    Synchronous router.completion() with rpm/tpm limits must use the synchronous
    Redis client/pipeline and must NOT instantiate or leak new async Redis clients
    into in_memory_llm_clients_cache on each call.
    """
    mock_sync_client = MagicMock()
    mock_pipeline = MagicMock()
    mock_sync_client.pipeline.return_value.__enter__.return_value = mock_pipeline
    mock_pipeline.execute.return_value = [1.0, 1.0]

    with (
        patch("litellm._redis.get_redis_client", return_value=mock_sync_client),
        patch("litellm._redis.get_redis_async_client", return_value=MagicMock()),
        patch("litellm._redis.get_redis_connection_pool", return_value=MagicMock()),
    ):
        router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-test",
                        "mock_response": "ok",
                        "rpm": 1000,
                    },
                }
            ],
            redis_host="127.0.0.1",
            redis_port=6379,
        )

        cache = litellm.in_memory_llm_clients_cache.cache_dict
        before_keys = set(cache.keys())

        for _ in range(5):
            response = router.completion(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
            )
            assert response is not None

        new_async_keys = [k for k in (set(cache.keys()) - before_keys) if k.startswith("async-redis-client")]
        assert len(new_async_keys) == 0, f"Leaked async Redis clients in cache: {new_async_keys}"
        assert mock_pipeline.incrbyfloat.called, "Sync pipeline incrbyfloat should be called"
        assert mock_pipeline.execute.called, "Sync pipeline execute should be called"


def test_sync_redis_cache_increment_pipeline_with_real_redis_return_types() -> None:
    """
    Real Redis pipeline execution returns [incrbyfloat_val (float), expire_val (bool/int)].
    increment_pipeline must properly extract only the increment results and ignore expire return codes.
    """
    from litellm.caching.redis_cache import RedisCache
    from litellm.types.caching import RedisPipelineIncrementOperation

    mock_client = MagicMock()
    mock_pipeline = MagicMock()
    mock_client.pipeline.return_value.__enter__.return_value = mock_pipeline
    # Real Redis execute returns [15.5 (float), True (bool), 1.0 (float), True (bool)]
    mock_pipeline.execute.return_value = [15.5, True, 1.0, True]

    with patch("litellm._redis.get_redis_client", return_value=mock_client):
        cache = RedisCache(host="127.0.0.1", port=6379)
        cache.redis_client = mock_client

        ops: list[RedisPipelineIncrementOperation] = [
            {"key": "tpm_key", "increment_value": 15.5, "ttl": 60},
            {"key": "rpm_key", "increment_value": 1.0, "ttl": 60},
        ]
        results = cache.increment_pipeline(increment_list=ops)

        assert results == [15.5, 1.0], f"Expected [15.5, 1.0], got {results}"
        assert mock_pipeline.incrbyfloat.call_count == 2
        assert mock_pipeline.expire.call_count == 2
