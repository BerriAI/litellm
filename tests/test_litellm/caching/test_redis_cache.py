import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock

from litellm.caching.redis_cache import RedisCache


@pytest.fixture
def redis_no_ping():
    """Patch RedisCache initialization to prevent async ping tasks from being created"""
    with patch("asyncio.get_running_loop") as mock_get_loop:
        # Either raise an exception or return a mock that will handle the task creation
        mock_get_loop.side_effect = RuntimeError("No running event loop")
        yield


@pytest.mark.parametrize("namespace", [None, "litellm"])
@pytest.mark.asyncio
async def test_async_delete_cache_applies_namespace(
    namespace, monkeypatch, redis_no_ping
):
    """async_delete_cache must prefix keys with the namespace, matching every
    other cache operation. Without this, Redis NOPERM errors occur when an
    ACL restricts DEL to the litellm:* pattern."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_delete_cache(key="3997c4abcdef")

    expected_key = "litellm:3997c4abcdef" if namespace else "3997c4abcdef"
    mock_redis_instance.delete.assert_awaited_once_with(expected_key)


@pytest.mark.parametrize("namespace", [None, "litellm"])
def test_delete_cache_applies_namespace(namespace, monkeypatch, redis_no_ping):
    """delete_cache must prefix keys with the namespace, matching every other
    cache operation."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_client = MagicMock()
    redis_cache.redis_client = mock_redis_client

    redis_cache.delete_cache(key="3997c4abcdef")

    expected_key = "litellm:3997c4abcdef" if namespace else "3997c4abcdef"
    mock_redis_client.delete.assert_called_once_with(expected_key)


@pytest.mark.asyncio
async def test_redis_client_init_with_socket_timeout(monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "my-fake-host")
    redis_cache = RedisCache(socket_timeout=1.0)
    assert redis_cache.redis_kwargs["socket_timeout"] == 1.0
    client = redis_cache.init_async_client()
    assert client is not None
    assert client.connection_pool.connection_kwargs["socket_timeout"] == 1.0


@pytest.mark.asyncio
async def test_handle_lpop_count_for_older_redis_versions(monkeypatch):
    """Test the helper method that handles LPOP with count for Redis versions < 7.0"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    # Create RedisCache instance
    redis_cache = RedisCache()

    # Create a mock pipeline
    mock_pipeline = AsyncMock()
    # Set up execute to return different values each time
    mock_pipeline.execute.side_effect = [
        [b"value1"],  # First execute returns first value
        [b"value2"],  # Second execute returns second value
    ]

    # Test the helper method
    result = await redis_cache.handle_lpop_count_for_older_redis_versions(
        pipe=mock_pipeline, key="test_key", count=2
    )

    # Verify results
    assert result == [b"value1", b"value2"]
    assert mock_pipeline.lpop.call_count == 2
    assert mock_pipeline.execute.call_count == 2


@pytest.mark.asyncio
async def test_async_rpush_pipeline_empty_list_returns_empty(
    monkeypatch, redis_no_ping
):
    """Empty rpush_list should return empty list without touching Redis"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    mock_redis_instance = AsyncMock()

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        result = await redis_cache.async_rpush_pipeline(rpush_list=[])

    assert result == []
    mock_redis_instance.pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_async_lpop_pipeline_empty_list(monkeypatch, redis_no_ping):
    """Empty lpop_list should return empty list without touching Redis"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    mock_redis_instance = AsyncMock()

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        result = await redis_cache.async_lpop_pipeline(lpop_list=[])

    assert result == []
    mock_redis_instance.pipeline.assert_not_called()


# LIT-3374: the namespace must be applied uniformly across every key-taking
# Redis operation, not just get/set/increment. Before the fix these paths wrote
# or read raw keys, so with a namespace configured the prefixed keys other
# operations created were silently missed.


@pytest.mark.parametrize(
    "namespace, raw_keys, expected_keys",
    [
        (None, ["{k:v}:tokens", "{k:v}:requests"], ["{k:v}:tokens", "{k:v}:requests"]),
        (
            "litellm_sandbox",
            ["{k:v}:tokens", "{k:v}:requests"],
            ["litellm_sandbox:{k:v}:tokens", "litellm_sandbox:{k:v}:requests"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_async_register_script_namespaces_keys(
    namespace, raw_keys, expected_keys, monkeypatch, redis_no_ping
):
    """The callable returned by async_register_script (used by the rate limiter
    Lua scripts, pod-lock release, and budget limiters) must namespace every key
    it is invoked with. The hash tag is preserved so cluster slotting is intact."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)

    registered_script = AsyncMock(return_value="ok")
    mock_redis_instance = MagicMock()
    mock_redis_instance.register_script = MagicMock(return_value=registered_script)

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        script = redis_cache.async_register_script("return 1")
        result = await script(keys=raw_keys, args=[60])

    assert result == "ok"
    registered_script.assert_awaited_once_with(
        keys=tuple(expected_keys), args=[60], client=None
    )


# LIT-3298: rate limits tripped at ~40M instead of 80M. async_register_script
# registered the Lua script once at startup and stored the object on the
# limiter, so a request running on a different event loop awaited a script bound
# to the startup loop's connection -> "got Future attached to a different loop".
# The limiter then fell back to a pipeline that reset the window TTL, so two
# minutes of tokens piled into one window. The script must instead be registered
# lazily against the calling loop's client and cached per loop.


@pytest.mark.parametrize("namespace", [None, "litellm_sandbox"])
def test_async_register_script_binds_per_event_loop(namespace, monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)

    clients_built = []

    def make_client():
        client = MagicMock()
        client.register_script = MagicMock(return_value=AsyncMock(return_value="ok"))
        clients_built.append(client)
        return client

    unique_script = "return 'lit3298'"

    with patch.object(redis_cache, "init_async_client", side_effect=make_client):
        script = redis_cache.async_register_script(unique_script)

        # Registration is deferred: no client is touched until the script runs.
        assert clients_built == []

        # Two loops kept alive at once so their ids can't be recycled into one
        # cache key. The buggy version reuses the first loop's bound object.
        loop_a = asyncio.new_event_loop()
        loop_b = asyncio.new_event_loop()
        try:
            result_a = loop_a.run_until_complete(
                script(keys=["{k:v}:tokens"], args=[60])
            )
            result_b = loop_b.run_until_complete(
                script(keys=["{k:v}:tokens"], args=[60])
            )
        finally:
            loop_a.close()
            loop_b.close()

    assert result_a == "ok"
    assert result_b == "ok"
    assert len(clients_built) == 2
    for client in clients_built:
        client.register_script.assert_called_once_with(unique_script)


@pytest.mark.asyncio
async def test_async_register_script_not_shared_across_namespaces(
    monkeypatch, redis_no_ping
):
    """Two caches with different namespaces registering the SAME script must
    each run against their own client and key prefix. A content-only executor
    cache would let the second cache reuse the first's executor and namespace."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    cache_a = RedisCache(namespace="ns_a")
    cache_b = RedisCache(namespace="ns_b")

    reg_a = AsyncMock(return_value="a")
    client_a = MagicMock()
    client_a.register_script = MagicMock(return_value=reg_a)
    reg_b = AsyncMock(return_value="b")
    client_b = MagicMock()
    client_b.register_script = MagicMock(return_value=reg_b)

    same_script = "return redis.call('GET', KEYS[1])"
    with patch.object(
        cache_a, "init_async_client", return_value=client_a
    ), patch.object(cache_b, "init_async_client", return_value=client_b):
        script_a = cache_a.async_register_script(same_script)
        script_b = cache_b.async_register_script(same_script)
        result_a = await script_a(keys=["k"], args=[])
        result_b = await script_b(keys=["k"], args=[])

    assert (result_a, result_b) == ("a", "b")
    reg_a.assert_awaited_once_with(keys=("ns_a:k",), args=[], client=None)
    reg_b.assert_awaited_once_with(keys=("ns_b:k",), args=[], client=None)


@pytest.mark.asyncio
async def test_async_register_script_cluster_path_uses_evalsha(
    monkeypatch, redis_no_ping
):
    """Redis Cluster exposes script_load/evalsha rather than register_script.
    The script is loaded once and invoked via evalsha with namespaced keys."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace="ns")

    cluster_client = MagicMock(spec=["script_load", "evalsha"])
    cluster_client.script_load = MagicMock(return_value="sha123")
    cluster_client.evalsha = AsyncMock(return_value="cluster-ok")

    with patch.object(
        redis_cache, "init_async_client", return_value=cluster_client
    ):
        script = redis_cache.async_register_script("return 'cluster'")
        result = await script(keys=["{k:v}:tokens"], args=[5, 60])

    assert result == "cluster-ok"
    cluster_client.script_load.assert_called_once_with("return 'cluster'")
    cluster_client.evalsha.assert_awaited_once_with(
        "sha123", 1, "ns:{k:v}:tokens", 5, 60
    )


@pytest.mark.asyncio
async def test_async_register_script_raises_for_unsupported_client(
    monkeypatch, redis_no_ping
):
    """A client exposing neither register_script nor script_load fails loudly
    rather than silently returning a no-op callable."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    bad_client = MagicMock(spec=[])

    with patch.object(redis_cache, "init_async_client", return_value=bad_client):
        script = redis_cache.async_register_script("return 'x'")
        with pytest.raises(ValueError, match="does not support Lua script"):
            await script(keys=["k"], args=[1])


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
@pytest.mark.asyncio
async def test_async_delete_cache_namespaces_key(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_delete_cache("k")
    mock_redis_instance.delete.assert_awaited_once_with(expected)


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
@pytest.mark.asyncio
async def test_delete_cache_keys_namespaces_keys(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.delete_cache_keys(["k"])
    mock_redis_instance.delete.assert_awaited_once_with(expected)


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
@pytest.mark.asyncio
async def test_async_get_ttl_namespaces_key(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()
    mock_redis_instance.ttl = AsyncMock(return_value=42)
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        ttl = await redis_cache.async_get_ttl("k")
    assert ttl == 42
    mock_redis_instance.ttl.assert_awaited_once_with(expected)


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
@pytest.mark.asyncio
async def test_async_lpop_namespaces_key(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()
    mock_redis_instance.lpop = AsyncMock(return_value=b"value")
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_lpop(key="k")
    mock_redis_instance.lpop.assert_awaited_once_with(expected, None)


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
@pytest.mark.asyncio
async def test_async_rpush_namespaces_key(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_redis_instance = AsyncMock()
    mock_redis_instance.rpush = AsyncMock(return_value=1)
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_rpush("k", ["v"])
    mock_redis_instance.rpush.assert_awaited_once_with(expected, "v")


@pytest.mark.parametrize("namespace, expected_match", [(None, "k*"), ("ns", "ns:k*")])
@pytest.mark.asyncio
async def test_async_scan_iter_namespaces_pattern(
    namespace, expected_match, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)

    captured = {}

    def scan_iter(match, count):
        captured["match"] = match

        async def gen():
            for _ in ():
                yield _

        return gen()

    mock_redis_instance = MagicMock()
    mock_redis_instance.scan_iter = scan_iter
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_scan_iter(pattern="k")
    assert captured["match"] == expected_match


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
def test_increment_cache_namespaces_key(
    namespace, expected, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_client = MagicMock()
    mock_client.incr.return_value = 5
    mock_client.ttl.return_value = 100
    redis_cache.redis_client = mock_client
    redis_cache.increment_cache(key="k", value=1)
    mock_client.incr.assert_called_once_with(name=expected, amount=1)


@pytest.mark.parametrize("namespace, expected", [(None, "k"), ("ns", "ns:k")])
def test_delete_cache_namespaces_key(namespace, expected, monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    mock_client = MagicMock()
    redis_cache.redis_client = mock_client
    redis_cache.delete_cache(key="k")
    mock_client.delete.assert_called_once_with(expected)
