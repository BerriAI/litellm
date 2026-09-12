import asyncio
import time
from collections.abc import Iterator
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm._service_logger import ServiceLogging
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreakerOpenError


@pytest.fixture
def redis_no_ping():
    """Patch RedisCache initialization to prevent async ping tasks from being created"""
    with patch("asyncio.get_running_loop") as mock_get_loop:
        # Either raise an exception or return a mock that will handle the task creation
        mock_get_loop.side_effect = RuntimeError("No running event loop")
        yield


@pytest.fixture
def sync_batch_redis_cache(redis_no_ping):
    with patch(  # test-quality-ok: RedisCache.__init__ builds its client eagerly, with no injection point
        "litellm._redis.get_redis_client", return_value=MagicMock()
    ) as get_client:
        cache = RedisCache(host="127.0.0.1", port=6379)
        cache.redis_client.mget.side_effect = OSError("redis unavailable")
        get_client.assert_called_once()
        yield cache


@pytest.mark.parametrize(
    ("namespace", "key", "expected"),
    [
        ("litellm", "litellm_spend_update_buffer", "litellm:litellm_spend_update_buffer"),
        ("litellm", "litellm_config:param:general_settings", "litellm:litellm_config:param:general_settings"),
        ("litellm", "litellm:3997c4abcdef", "litellm:3997c4abcdef"),
        ("litellm", "spend:key:3997c4abcdef", "litellm:spend:key:3997c4abcdef"),
        (None, "litellm_spend_update_buffer", "litellm_spend_update_buffer"),
        ("", "litellm_spend_update_buffer", "litellm_spend_update_buffer"),
    ],
)
def test_check_and_fix_namespace_prefixes_keys_sharing_the_namespace_prefix(
    namespace, key, expected, monkeypatch, redis_no_ping
):
    """A key whose name merely begins with the namespace string (e.g.
    litellm_spend_update_buffer under namespace "litellm") is not namespaced
    yet and must still get the "namespace:" prefix; only a key already carrying
    the delimited prefix is left alone. Without this, spend update buffers and
    litellm_config:param:* keys reach Redis unprefixed and NOPERM under an ACL
    scoped to the namespace pattern."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    assert redis_cache.check_and_fix_namespace(key=key) == expected


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
@pytest.mark.parametrize(
    "redis_config",
    [
        pytest.param({"host": "my-fake-host"}, id="host_port"),
        pytest.param({"url": "redis://my-fake-host:6379"}, id="url"),
    ],
)
async def test_redis_client_init_with_socket_timeout(monkeypatch, redis_no_ping, redis_config):
    """socket_timeout has to reach the connection however Redis was configured.

    A url config used to drop every connection kwarg, so redis-py was left with
    socket_timeout (and socket_connect_timeout, which falls back to it) unset. A
    Redis host that drops packets instead of refusing them then blocks each caller
    indefinitely, and the circuit breaker never trips because no call ever returns.
    """
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    redis_cache = RedisCache(socket_timeout=1.0, **redis_config)
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


def _closed_port() -> int:
    """A port with nothing listening, so Redis calls fail fast and deterministically."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_method",
    [
        pytest.param(lambda c: c.async_get_cache("lit4930"), id="async_get_cache"),
        pytest.param(lambda c: c.async_batch_get_cache(["lit4930"]), id="async_batch_get_cache"),
        pytest.param(lambda c: c.async_set_cache("lit4930", "v"), id="async_set_cache"),
        pytest.param(lambda c: c.async_get_ttl("lit4930"), id="async_get_ttl"),
    ],
)
async def test_circuit_breaker_opens_when_method_swallows_redis_failure(call_method):
    """A guarded method that swallows its own Redis error must still count as a failure.

    These methods catch connection errors and return a default so callers degrade instead
    of failing, which is correct. But that returns cleanly through the circuit breaker
    guard, and counting it as a success reset the failure streak on every call, so the
    breaker could never open. An unreachable Redis then stayed in the pool and every
    request kept paying the full socket timeout on it.
    """
    from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD

    cache = await asyncio.to_thread(RedisCache, host="127.0.0.1", port=_closed_port(), socket_timeout=0.5)

    for _ in range(REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        await call_method(cache)

    with pytest.raises(Exception, match="circuit breaker is open"):
        await call_method(cache)


def test_circuit_breaker_open_makes_sync_batch_get_cache_fast_fail(sync_batch_redis_cache, caplog):
    """Once the breaker is open the sync batch read refuses with the typed error instead of a miss.

    Swallowing the refusal into `{}` made every sync batch read on an open breaker emit an ERROR
    log and a service failure event per call, and the DualCache caller could not tell the
    refusal from a dead Redis, so it dropped its in-memory hits too.
    """
    from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD

    for _ in range(REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        assert sync_batch_redis_cache.batch_get_cache(key_list=["lit6729"]) == {}

    caplog.clear()
    with caplog.at_level("INFO"):
        with pytest.raises(RedisCircuitBreakerOpenError):
            sync_batch_redis_cache.batch_get_cache(key_list=["lit6729"])
    sync_batch_redis_cache.redis_client.mget.assert_called()
    assert caplog.records == []


def test_sync_get_cache_failure_feeds_the_breaker_and_logs_a_well_formed_record(sync_batch_redis_cache, caplog):
    """The sync get path swallowed its Redis error without recording it, and its log call was malformed.

    `verbose_logger.error("...: ", e)` passes the exception as a format argument to a message
    with no placeholder, so the record carried no error text. Nothing fed the breaker either,
    so a dead Redis read through this path never opened it.
    """
    from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD

    sync_batch_redis_cache.redis_client.get.side_effect = OSError("redis unavailable")

    with caplog.at_level("ERROR"):
        for _ in range(REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD):
            assert sync_batch_redis_cache.get_cache("lit7468") is None

    assert all("redis unavailable" in record.getMessage() for record in caplog.records)
    assert len(caplog.records) == REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD
    assert sync_batch_redis_cache._circuit_breaker.is_open() is True
    with pytest.raises(RedisCircuitBreakerOpenError):
        sync_batch_redis_cache.get_cache("lit7468")


def test_batch_get_counts_raises_where_batch_get_cache_reports_a_miss(sync_batch_redis_cache):
    """A caller that must fall back when Redis is unreachable needs the failure, not zeros.

    The batch read answers a dead Redis with an empty dict, which a counting caller cannot tell
    apart from "every counter is unset". Least-busy routing read that as an idle deployment and
    kept sending traffic to it instead of falling back to this worker's own in-flight counts.
    """
    assert sync_batch_redis_cache.batch_get_cache(key_list=["lit7039"]) == {}

    with pytest.raises(OSError, match="redis unavailable"):
        sync_batch_redis_cache.batch_get_counts(["lit7039"])


@pytest.mark.asyncio
async def test_async_batch_get_counts_raises_where_async_batch_get_cache_reports_a_miss(redis_no_ping: None):
    """Async twin: the async batch read hides the same failure behind an empty dict."""
    failing_client = AsyncMock()
    failing_client.mget.side_effect = OSError("redis unavailable")
    with patch(  # test-quality-ok: RedisCache.__init__ builds its client eagerly, with no injection point
        "litellm._redis.get_redis_client", return_value=MagicMock()
    ):
        cache = RedisCache(host="127.0.0.1", port=6379)

    with patch.object(cache, "init_async_client", return_value=failing_client):
        assert await cache.async_batch_get_cache(key_list=["lit7039"]) == {}

        with pytest.raises(OSError, match="redis unavailable"):
            await cache.async_batch_get_counts(["lit7039"])


@pytest.mark.parametrize("stored", [b"3", "3"])
def test_batch_get_counts_reads_counters_in_order_and_keeps_unset_keys_apart(stored, redis_no_ping: None):
    """Counters come back positionally, so an unset key has to stay a hole rather than shift the
    rest of the row onto the wrong deployments, and a count has to survive whether the client
    hands it back as bytes or as text."""
    with patch(  # test-quality-ok: RedisCache.__init__ builds its client eagerly, with no injection point
        "litellm._redis.get_redis_client", return_value=MagicMock()
    ):
        cache = RedisCache(host="127.0.0.1", port=6379)
    cache.redis_client.mget.return_value = [stored, None, b"0"]

    assert cache.batch_get_counts(["dep-a", "dep-b", "dep-c"]) == (3, None, 0)


@pytest.fixture
def sync_batch_cache_with_service_logger(redis_no_ping: None) -> Iterator[tuple[RedisCache, ServiceLogging]]:
    service_logger = ServiceLogging(mock_testing=True)
    failing_client = MagicMock()
    failing_client.mget.side_effect = OSError("redis unavailable")
    with patch(  # test-quality-ok: RedisCache.__init__ builds its client eagerly, with no injection point
        "litellm._redis.get_redis_client", return_value=failing_client
    ):
        cache = RedisCache(host="127.0.0.1", port=6379, service_logger_obj=service_logger)
    yield cache, service_logger


@pytest.mark.asyncio
async def test_sync_batch_get_cache_reports_a_failed_read_from_a_running_loop(
    sync_batch_cache_with_service_logger: tuple[RedisCache, ServiceLogging],
):
    """A swallowed Redis failure must still be reported as a service failure event.

    The routing strategies call this blocking read from inside the request's event loop,
    and the read hides the Redis error by returning an empty dict. Without an emitted
    failure event, litellm_redis_failed_requests_total stops moving during a Redis
    outage while the success path keeps reporting, so the dashboards read healthy.
    """
    cache, service_logger = sync_batch_cache_with_service_logger

    assert cache.batch_get_cache(key_list=["lit6729"]) == {}
    await asyncio.sleep(0.05)

    assert service_logger.mock_testing_sync_failure_hook == 1
    assert service_logger.mock_testing_async_failure_hook == 1


def test_sync_batch_get_cache_reports_a_failed_read_from_a_worker_thread(
    sync_batch_cache_with_service_logger: tuple[RedisCache, ServiceLogging],
):
    """The same report must reach the async hook when the caller has no event loop at all."""
    from concurrent.futures import ThreadPoolExecutor

    cache, service_logger = sync_batch_cache_with_service_logger

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(cache.batch_get_cache, key_list=["lit6729"]).result() == {}

    assert service_logger.mock_testing_async_failure_hook == 1


def test_sync_batch_get_cache_reports_a_failed_read_on_an_idle_event_loop(
    sync_batch_cache_with_service_logger: tuple[RedisCache, ServiceLogging],
):
    """The report must also go out when the caller holds an open loop that is not running."""
    cache, service_logger = sync_batch_cache_with_service_logger
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        assert cache.batch_get_cache(key_list=["lit6729"]) == {}
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    assert service_logger.mock_testing_async_failure_hook == 1


def test_sync_batch_get_cache_survives_a_service_callback_that_raises(
    sync_batch_cache_with_service_logger: tuple[RedisCache, ServiceLogging],
    monkeypatch: pytest.MonkeyPatch,
):
    """A failing service callback must not replace the swallowed Redis failure.

    A misconfigured callback raises while emitting (a datadog callback with no
    DD_API_KEY raises at construction), and the failure event is emitted from inside
    the except block that swallows the Redis error. If that exception escapes, a Redis
    outage surfaces to routing as a callback error and the circuit breaker never
    records the failed read.
    """
    from concurrent.futures import ThreadPoolExecutor

    import litellm
    from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD

    cache, service_logger = sync_batch_cache_with_service_logger
    monkeypatch.setattr(litellm, "service_callback", ["prometheus_system"])
    monkeypatch.setattr(
        service_logger,
        "init_prometheus_services_logger_if_none",
        AsyncMock(side_effect=Exception("callback is misconfigured")),
    )

    for _ in range(REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(cache.batch_get_cache, key_list=["lit6729"]).result() == {}

    with pytest.raises(RedisCircuitBreakerOpenError):
        cache.batch_get_cache(key_list=["lit6729"])


def test_call_stack_info_skips_breaker_guard_frames():
    """Guarded methods must still report their real callers in service-log call_type.

    The breaker guards put their own frames between a method body and its caller, so
    without skipping them every guarded method logged the guard machinery instead of
    who actually issued the Redis call.
    """
    from litellm.caching.redis_cache import (
        RedisCircuitBreaker,
        _get_call_stack_info,
        _redis_circuit_breaker_guard_sync,
    )

    class Guarded:
        _circuit_breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)

        @_redis_circuit_breaker_guard_sync
        def probe(self):
            return _get_call_stack_info()

    def caller_one():
        return Guarded().probe()

    def caller_two():
        return caller_one()

    assert caller_two() == "caller_one <- caller_two"


def test_call_stack_info_skips_guard_frames_when_deployed_without_sources(monkeypatch):
    """Guard-frame skipping must survive a bytecode-only deployment.

    Shipping `.pyc` files without their `.py` sources leaves the module's `__file__` pointing
    at the compiled file while every frame still carries the compile-time source path, so a
    check comparing those two paths stops skipping and the service log then names the guard
    machinery instead of the real caller.
    """
    from litellm.caching import redis_cache as redis_cache_module
    from litellm.caching.redis_cache import (
        RedisCircuitBreaker,
        _get_call_stack_info,
        _redis_circuit_breaker_guard_sync,
    )

    monkeypatch.setattr(redis_cache_module, "__file__", redis_cache_module.__file__ + "c")

    class Guarded:
        _circuit_breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)

        @_redis_circuit_breaker_guard_sync
        def probe(self):
            return _get_call_stack_info()

    def caller_one():
        return Guarded().probe()

    def caller_two():
        return caller_one()

    assert caller_two() == "caller_one <- caller_two"


@pytest.mark.asyncio
async def test_circuit_breaker_success_still_resets_the_failure_streak():
    """A reachable Redis must keep the breaker closed, however many earlier calls failed.

    The guard now records success only when nothing failed while the method ran, so this
    pins the other half of that contract: a call that genuinely reaches Redis has to clear
    the streak, or a healthy Redis would eventually be evicted from the pool.
    """
    from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD

    cache = await asyncio.to_thread(RedisCache, host="127.0.0.1", port=_closed_port(), socket_timeout=0.5)

    for _ in range(REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD - 1):
        await cache.async_get_cache("lit4930")
    assert cache._circuit_breaker.is_open() is False

    reachable_redis = AsyncMock()
    reachable_redis.get.return_value = None
    with patch.object(cache, "init_async_client", return_value=reachable_redis):
        await cache.async_get_cache("lit4930")

    for _ in range(REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD - 1):
        await cache.async_get_cache("lit4930")

    assert cache._circuit_breaker.is_open() is False, "one success must clear the streak"


@pytest.mark.asyncio
async def test_circuit_breaker_covers_lua_script_execution():
    """Lua script execution must feed the breaker like every other Redis call.

    The v3 rate limiter issues all of its Redis traffic through async_register_script, so
    leaving that path unguarded meant the coordination calls during an outage never
    counted toward taking Redis out of the pool and kept paying a full socket timeout
    each, which is the traffic the outage hurts most.
    """
    from redis.exceptions import ConnectionError as RedisConnectionError

    from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD

    cache = await asyncio.to_thread(RedisCache, host="127.0.0.1", port=_closed_port(), socket_timeout=0.5)
    run_script = cache.async_register_script("return 1")

    for _ in range(REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        with pytest.raises(RedisConnectionError):
            await run_script(keys=["lit4930"], args=[1])

    with pytest.raises(Exception, match="circuit breaker is open"):
        await run_script(keys=["lit4930"], args=[1])


@pytest.mark.asyncio
async def test_concurrent_success_is_not_cancelled_by_another_calls_failure():
    """One caller's failure must not discard a different caller's success.

    A breaker is shared by every concurrent caller, so tracking "did this call fail" on the
    breaker itself cannot tell my failure from someone else's. A Redis that is still
    answering would then be evicted from the pool by unrelated in-flight failures, which is
    the opposite of the outage this guard exists to handle.
    """
    from redis.exceptions import ConnectionError as RedisConnectionError

    from litellm.caching.redis_cache import (
        RedisCircuitBreaker,
        _record_swallowed_redis_failure,
        _run_under_circuit_breaker,
    )

    breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)

    # The failure has to land after both calls are already in flight, which is the only
    # ordering where a shared counter confuses the two. Failing before the healthy call
    # starts would leave its snapshot correct and prove nothing.
    async def swallows_a_failure():
        await asyncio.sleep(0.02)
        _record_swallowed_redis_failure(breaker, RedisConnectionError("redis unreachable"))

    async def succeeds_while_the_other_fails():
        await asyncio.sleep(0.05)
        return "ok"

    rounds = breaker.failure_threshold + 1
    for _ in range(rounds):
        await asyncio.gather(
            _run_under_circuit_breaker(breaker, "failing", swallows_a_failure),
            _run_under_circuit_breaker(breaker, "healthy", succeeds_while_the_other_fails),
        )

    assert breaker._failure_count < breaker.failure_threshold, "the healthy call must clear the streak"
    assert breaker.is_open() is False, "a Redis answering every round must stay in the pool"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, opens_breaker",
    [
        pytest.param("ConnectionError", True, id="connection_refused_is_unhealthy"),
        pytest.param("TimeoutError", False, id="timeout_burst_is_ambiguous"),
        pytest.param("BusyLoadingError", True, id="loading_is_unhealthy"),
        pytest.param("ResponseError", False, id="wrong_type_command_is_not"),
        pytest.param("DataError", False, id="bad_data_is_not"),
    ],
)
async def test_only_connectivity_failures_open_the_breaker(error, opens_breaker):
    """Command and data errors must not count against Redis health.

    They say nothing about connectivity, and a caller able to provoke them (an INCR against
    a non-numeric value, say) could otherwise trip the shared breaker on demand and drop
    rate limiting to per-process counters, which spreading traffic across replicas outruns.

    A rapid burst of timeouts is ambiguous too: the async timeout includes event-loop
    scheduling delay, so a loop stall times out every queued call at once against a
    healthy Redis. It must not open the breaker until the streak spans a minimum duration.
    """
    import redis.exceptions

    from litellm.caching.redis_cache import RedisCircuitBreaker, _run_under_circuit_breaker

    breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)
    raised = getattr(redis.exceptions, error)("boom")

    async def failing_call():
        raise raised

    for _ in range(breaker.failure_threshold):
        with pytest.raises(type(raised)):
            await _run_under_circuit_breaker(breaker, "op", failing_call)

    with pytest.raises(Exception, match="circuit breaker is open" if opens_breaker else "boom"):
        await _run_under_circuit_breaker(breaker, "op", failing_call)

    assert breaker.is_open() is opens_breaker


@pytest.mark.asyncio
async def test_event_loop_stall_timeout_burst_keeps_breaker_closed():
    """One blocking stall of the worker event loop must not trip the breaker.

    Every operation already waiting on the loop times out together when the loop resumes,
    so a purely consecutive threshold is satisfied instantly even though the Redis on the
    other end (here an in-process fake that answers immediately) is healthy.

    The fake checks its own client deadline against the clock, the way a client library
    does, rather than wrapping the call in asyncio.wait_for: before 3.12 wait_for returns
    the inner result when the inner future also completed during the stall, so the burst
    never materialises and the test cannot exercise the duration gate.
    """
    import time as time_mod

    from redis.exceptions import TimeoutError as RedisTimeoutError

    from litellm.caching.redis_cache import RedisCircuitBreaker, _run_under_circuit_breaker

    breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60, timeout_min_duration=5.0)

    async def healthy_redis_call_with_client_timeout():
        deadline = time_mod.monotonic() + 0.05
        await asyncio.sleep(0.001)
        if time_mod.monotonic() > deadline:
            raise RedisTimeoutError("read timed out")
        return "ok"

    async def stall_the_loop():
        await asyncio.sleep(0)
        time_mod.sleep(0.2)

    results = await asyncio.gather(
        *(_run_under_circuit_breaker(breaker, "op", healthy_redis_call_with_client_timeout) for _ in range(8)),
        stall_the_loop(),
        return_exceptions=True,
    )
    timeouts = [r for r in results if isinstance(r, RedisTimeoutError)]
    assert len(timeouts) >= breaker.failure_threshold, "the stall must time out a full burst"

    assert breaker.is_open() is False, "a healthy Redis behind one loop stall must stay in the pool"
    assert await _run_under_circuit_breaker(breaker, "op", healthy_redis_call_with_client_timeout) == "ok"


@pytest.mark.asyncio
async def test_persistent_timeouts_still_open_the_breaker():
    """A real outage that surfaces only as timeouts must still open the breaker.

    Once the timeout-only streak spans the minimum duration with no success in between,
    Redis is genuinely unusable from this worker and protection has to kick in.
    """
    from redis.exceptions import TimeoutError as RedisTimeoutError

    from litellm.caching.redis_cache import RedisCircuitBreaker, _run_under_circuit_breaker

    breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60, timeout_min_duration=0.1)

    async def timing_out_call():
        raise RedisTimeoutError("read timed out")

    for _ in range(breaker.failure_threshold):
        with pytest.raises(RedisTimeoutError):
            await _run_under_circuit_breaker(breaker, "op", timing_out_call)
    assert breaker.is_open() is False, "the burst has not spanned the minimum duration yet"

    await asyncio.sleep(0.12)
    with pytest.raises(RedisTimeoutError):
        await _run_under_circuit_breaker(breaker, "op", timing_out_call)

    assert breaker.is_open() is True


@pytest.mark.asyncio
async def test_stale_timeout_does_not_let_sub_threshold_hard_failures_open_the_breaker():
    """Hard connectivity failures below the threshold must not open the breaker just
    because an old timeout already started the streak and the duration has elapsed.

    Each class has to earn the open on its own terms: hard failures by reaching the
    threshold, timeouts by reaching the threshold and spanning the minimum duration.
    """
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError

    from litellm.caching.redis_cache import RedisCircuitBreaker, _is_redis_timeout_failure

    breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60, timeout_min_duration=0.05)

    breaker.record_failure(is_timeout=_is_redis_timeout_failure(RedisTimeoutError("read timed out")))
    await asyncio.sleep(0.06)
    for _ in range(breaker.failure_threshold - 1):
        breaker.record_failure(is_timeout=_is_redis_timeout_failure(RedisConnectionError("refused")))
    assert breaker.is_open() is False, "2 hard failures and 1 stale timeout are below both thresholds"

    breaker.record_failure(is_timeout=_is_redis_timeout_failure(RedisConnectionError("refused")))
    assert breaker.is_open() is True, "the threshold-th hard failure must still open it"


@pytest.mark.asyncio
async def test_hard_failure_resets_timeout_streak_so_a_later_burst_must_earn_its_own_duration():
    """A stale timeout followed by hard failures must not pre-age the duration gate:
    a later short timeout burst has to span timeout_min_duration on its own.
    """
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError

    from litellm.caching.redis_cache import RedisCircuitBreaker, _is_redis_timeout_failure

    breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60, timeout_min_duration=0.05)

    breaker.record_failure(is_timeout=_is_redis_timeout_failure(RedisTimeoutError("read timed out")))
    breaker.record_failure(is_timeout=_is_redis_timeout_failure(RedisConnectionError("refused")))
    await asyncio.sleep(0.06)
    for _ in range(breaker.failure_threshold):
        breaker.record_failure(is_timeout=_is_redis_timeout_failure(RedisTimeoutError("read timed out")))
    assert breaker.is_open() is False, "the burst is instantaneous, so the duration gate must hold it closed"

    await asyncio.sleep(0.06)
    breaker.record_failure(is_timeout=_is_redis_timeout_failure(RedisTimeoutError("read timed out")))
    assert breaker.is_open() is True, "the same run of timeouts persisting past the duration must open it"


@pytest.mark.asyncio
async def test_breaker_metrics_track_state_and_failure_class():
    """Breaker accounting must be observable: failure class, transitions, and state."""
    from prometheus_client import REGISTRY
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError

    from litellm.caching.redis_cache import RedisCircuitBreaker, _is_redis_timeout_failure

    def sample(name, labels=None):
        return REGISTRY.get_sample_value(name, labels) or 0.0

    timeout_before = sample("litellm_redis_circuit_breaker_failures_total", {"failure_class": "timeout"})
    hard_before = sample("litellm_redis_circuit_breaker_failures_total", {"failure_class": "connectivity"})
    opened_before = sample("litellm_redis_circuit_breaker_transitions_total", {"state": "open"})
    open_gauge_before = sample("litellm_redis_circuit_breaker_state", {"state": "open"})
    closed_gauge_before = sample("litellm_redis_circuit_breaker_state", {"state": "closed"})

    breaker = RedisCircuitBreaker(failure_threshold=2, recovery_timeout=60, timeout_min_duration=5.0)
    breaker.record_failure(is_timeout=_is_redis_timeout_failure(RedisTimeoutError("t")))
    breaker.record_failure(is_timeout=_is_redis_timeout_failure(RedisConnectionError("refused")))
    breaker.record_failure(is_timeout=_is_redis_timeout_failure(RedisConnectionError("refused")))

    assert sample("litellm_redis_circuit_breaker_failures_total", {"failure_class": "timeout"}) == timeout_before + 1
    assert sample("litellm_redis_circuit_breaker_failures_total", {"failure_class": "connectivity"}) == hard_before + 2
    assert sample("litellm_redis_circuit_breaker_transitions_total", {"state": "open"}) == opened_before + 1
    assert sample("litellm_redis_circuit_breaker_state", {"state": "open"}) == open_gauge_before + 1
    assert sample("litellm_redis_circuit_breaker_state", {"state": "closed"}) == closed_gauge_before

    breaker._opened_at = time.time() - 9999
    assert breaker.is_open() is False
    breaker.record_success()
    assert sample("litellm_redis_circuit_breaker_state", {"state": "open"}) == open_gauge_before
    assert sample("litellm_redis_circuit_breaker_state", {"state": "closed"}) == closed_gauge_before + 1


def test_sync_guard_counts_a_timeout_as_a_timeout():
    from redis.exceptions import TimeoutError as RedisTimeoutError

    from litellm.caching.redis_cache import RedisCircuitBreaker, _run_under_circuit_breaker_sync

    breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60, timeout_min_duration=5.0)

    def timing_out_call() -> str:
        raise RedisTimeoutError("read timed out")

    for _ in range(6):
        with pytest.raises(RedisTimeoutError):
            _run_under_circuit_breaker_sync(breaker, "op", timing_out_call)

    assert breaker.is_open() is False


def test_success_admitted_before_the_breaker_opened_cannot_close_it():
    """A stale in-flight success must not close a breaker that opened while it ran.

    Calls admitted while the breaker was still closed finish after later failures opened it.
    Recording their success unconditionally closed the breaker again, skipping the recovery
    timeout and the single half-open probe, so the breaker flapped between open and closed
    on every straggler while Redis was still down.
    """
    from litellm.caching.redis_cache import RedisCircuitBreaker

    breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)
    for _ in range(3):
        breaker.record_failure()
    assert breaker._state == breaker.OPEN

    breaker.record_success()

    assert breaker._state == breaker.OPEN
    assert breaker.is_open() is True


def test_recovery_probe_still_closes_the_breaker():
    from litellm.caching.redis_cache import RedisCircuitBreaker

    breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)
    for _ in range(3):
        breaker.record_failure()
    breaker._opened_at = time.time() - 9999
    assert breaker.is_open() is False
    assert breaker._state == breaker.HALF_OPEN

    breaker.record_success()

    assert breaker._state == breaker.CLOSED
    assert breaker.is_open() is False


@pytest.mark.asyncio
async def test_stale_success_during_the_recovery_probe_leaves_the_breaker_to_the_probe():
    """A call admitted before the trip that finishes while HALF_OPEN must not close the breaker.

    Only the one call designated as the recovery probe has actually reached Redis after the
    outage, so closing on the straggler's success resumed full Redis traffic before the probe
    had proven anything.
    """
    from litellm.caching.redis_cache import RedisCircuitBreaker, _run_under_circuit_breaker

    breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)
    stale_admitted = asyncio.Event()
    stale_release = asyncio.Event()
    probe_admitted = asyncio.Event()
    probe_release = asyncio.Event()

    async def stale_call() -> str:
        stale_admitted.set()
        await stale_release.wait()
        return "stale"

    async def probe_call() -> str:
        probe_admitted.set()
        await probe_release.wait()
        return "probe"

    stale = asyncio.ensure_future(_run_under_circuit_breaker(breaker, "op", stale_call))
    await stale_admitted.wait()
    for _ in range(3):
        breaker.record_failure()
    assert breaker._state == breaker.OPEN
    breaker._opened_at = time.time() - 9999
    probe = asyncio.ensure_future(_run_under_circuit_breaker(breaker, "op", probe_call))
    await probe_admitted.wait()
    assert breaker._state == breaker.HALF_OPEN

    stale_release.set()
    assert await stale == "stale"

    assert breaker._state == breaker.HALF_OPEN, "the straggler must not close the breaker for the probe"
    assert breaker.is_open() is True

    probe_release.set()
    assert await probe == "probe"

    assert breaker._state == breaker.CLOSED
    assert breaker.is_open() is False


@pytest.mark.asyncio
async def test_a_probe_overtaken_by_a_later_outage_leaves_the_breaker_to_the_new_probe():
    """A probe still in flight when a late failure reopens the breaker must not close it for the next probe.

    Once the breaker has reopened, only the probe admitted after that outage has reached
    Redis, so the older probe's success no longer says anything about whether Redis recovered.
    """
    from litellm.caching.redis_cache import RedisCircuitBreaker, _run_under_circuit_breaker

    breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)
    old_probe_admitted = asyncio.Event()
    old_probe_release = asyncio.Event()
    new_probe_admitted = asyncio.Event()
    new_probe_release = asyncio.Event()

    async def old_probe_call() -> str:
        old_probe_admitted.set()
        await old_probe_release.wait()
        return "old probe"

    async def new_probe_call() -> str:
        new_probe_admitted.set()
        await new_probe_release.wait()
        return "new probe"

    for _ in range(3):
        breaker.record_failure()
    breaker._opened_at = time.time() - 9999
    old_probe = asyncio.ensure_future(_run_under_circuit_breaker(breaker, "op", old_probe_call))
    await old_probe_admitted.wait()
    assert breaker._state == breaker.HALF_OPEN

    breaker.record_failure()
    assert breaker._state == breaker.OPEN
    breaker._opened_at = time.time() - 9999
    new_probe = asyncio.ensure_future(_run_under_circuit_breaker(breaker, "op", new_probe_call))
    await new_probe_admitted.wait()
    assert breaker._state == breaker.HALF_OPEN

    old_probe_release.set()
    assert await old_probe == "old probe"

    assert breaker._state == breaker.HALF_OPEN, "the overtaken probe must not close the breaker for the new probe"
    assert breaker.is_open() is True

    new_probe_release.set()
    assert await new_probe == "new probe"
    assert breaker._state == breaker.CLOSED


class _SpyRedisCommands:
    def __init__(self, eval_result: object) -> None:
        self.commands: list[str] = []
        self.eval_calls: list[tuple[object, ...]] = []
        self._eval_result = eval_result

    async def eval(self, script: str, numkeys: int, *keys_and_args: object) -> object:
        self.commands.append("eval")
        self.eval_calls.append((script, numkeys, *keys_and_args))
        return self._eval_result

    async def incrbyfloat(self, name: str, amount: float) -> float:
        self.commands.append("incrbyfloat")
        return amount

    async def ttl(self, name: str) -> int:
        self.commands.append("ttl")
        return -1

    async def expire(self, name: str, time: int) -> bool:
        self.commands.append("expire")
        return True


class _SpyRedisCache(RedisCache):
    def __init__(self, spy: _SpyRedisCommands, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.spy = spy

    def init_async_client(self, *args: object, **kwargs: object) -> object:
        return self.spy


@pytest.mark.parametrize(("namespace", "expected_key"), [(None, "spend:key:abc"), ("ns", "ns:spend:key:abc")])
@pytest.mark.asyncio
async def test_redis_cache_async_increment_arms_ttl_in_the_same_command(
    namespace, expected_key, monkeypatch, redis_no_ping
):
    """The increment and its TTL reach Redis as one server-side step."""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    spy = _SpyRedisCommands(eval_result=b"1.25")
    redis_cache = _SpyRedisCache(spy, namespace=namespace)

    result = await redis_cache.async_increment(key="spend:key:abc", value=0.25, ttl=30)

    assert result == 1.25
    assert spy.commands == ["eval"]
    script, numkeys, key, amount, ttl, refresh = spy.eval_calls[0]
    assert "INCRBYFLOAT" in script and "EXPIRE" in script and "TTL" in script
    assert (numkeys, key, amount, ttl, refresh) == (1, expected_key, 0.25, "30", "0")


@pytest.mark.asyncio
async def test_redis_cache_async_increment_refresh_ttl_sends_refresh_flag(monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    spy = _SpyRedisCommands(eval_result="2.5")
    redis_cache = _SpyRedisCache(spy)

    result = await redis_cache.async_increment(key="spend:team:t1", value=0.5, refresh_ttl=True)

    assert result == 2.5
    assert spy.commands == ["eval"]
    assert spy.eval_calls[0][3:] == (0.5, "60", "1")


@pytest.mark.parametrize(
    ("ttl", "default_ttl", "expected_ttl_arg"), [(None, None, ""), (0, None, "0"), (None, 15, "15")]
)
@pytest.mark.asyncio
async def test_redis_cache_async_increment_forwards_ttl_exactly(
    ttl, default_ttl, expected_ttl_arg, monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    spy = _SpyRedisCommands(eval_result=b"0.75")
    redis_cache = _SpyRedisCache(spy)
    redis_cache.default_ttl = default_ttl

    result = await redis_cache.async_increment(key="spend:key:abc", value=0.75, ttl=ttl)

    assert result == 0.75
    assert spy.eval_calls[0][4] == expected_ttl_arg

class _RoundTripCountingRedis:
    """Fake redis.asyncio client: one round trip per awaited command or pipeline execute."""

    def __init__(self, ttl: int) -> None:
        self.values: dict[str, float] = {}
        self.ttls: dict[str, int] = {}
        self.round_trips = 0
        self._initial_ttl = ttl

    async def incrbyfloat(self, name: str, amount: float) -> float:
        self.round_trips += 1
        return self._incr(name, amount)

    async def expire(self, name: str, time: int) -> bool:
        self.round_trips += 1
        self.ttls[name] = time
        return True

    async def eval(self, script: str, numkeys: int, key: str, amount: object, ttl_arg: str, refresh: str) -> bytes:
        self.round_trips += 1
        value = self._incr(key, float(amount))  # pyright: ignore[reportArgumentType]  # fake receives the raw float
        if ttl_arg != "" and (refresh == "1" or self.ttls.get(key) == -1):
            self.ttls[key] = int(ttl_arg)
        return str(value).encode()

    def _incr(self, name: str, amount: float) -> float:
        self.values[name] = self.values.get(name, 0.0) + amount
        self.ttls.setdefault(name, self._initial_ttl)
        return self.values[name]

    def pipeline(self, transaction: bool) -> "_RoundTripCountingRedis._Pipeline":
        return _RoundTripCountingRedis._Pipeline(self)

    class _Pipeline:
        def __init__(self, client: "_RoundTripCountingRedis") -> None:
            self._client = client
            self._commands: list[tuple[str, tuple[object, ...]]] = []

        async def __aenter__(self) -> "_RoundTripCountingRedis._Pipeline":
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        def incrbyfloat(self, name: str, amount: float) -> None:
            self._commands.append(("incrbyfloat", (name, amount)))

        def expire(self, name: str, time: int) -> None:
            self._commands.append(("expire", (name, time)))

        def ttl(self, name: str) -> None:
            self._commands.append(("ttl", (name,)))

        async def execute(self) -> list[object]:
            self._client.round_trips += 1
            results: list[object] = []
            for command, args in self._commands:
                if command == "incrbyfloat":
                    results.append(self._client._incr(str(args[0]), float(args[1])))  # pyright: ignore[reportArgumentType]  # fake stores str/float
                elif command == "expire":
                    self._client.ttls[str(args[0])] = int(args[1])  # pyright: ignore[reportArgumentType]  # fake stores int
                    results.append(True)
                else:
                    results.append(self._client.ttls.get(str(args[0]), -2))
            return results


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("refresh_ttl", "existing_ttl", "expected_round_trips", "expected_ttl"),
    [
        pytest.param(True, 100, 2, 60, id="refresh_ttl: one EVAL per increment, TTL re-armed"),
        pytest.param(False, 100, 2, 100, id="keep ttl: one EVAL per increment, existing TTL kept"),
        pytest.param(False, -1, 2, 60, id="unexpiring key: one EVAL per increment, TTL armed in the same call"),
    ],
)
async def test_async_increment_sets_the_ttl_in_one_round_trip(
    monkeypatch, redis_no_ping, refresh_ttl, existing_ttl, expected_round_trips, expected_ttl
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace="ns")
    client = _RoundTripCountingRedis(ttl=existing_ttl)

    with patch.object(redis_cache, "init_async_client", return_value=client):
        first = await redis_cache.async_increment(key="spend:key:k", value=1.5, ttl=60, refresh_ttl=refresh_ttl)
        second = await redis_cache.async_increment(key="spend:key:k", value=2.0, ttl=60, refresh_ttl=refresh_ttl)

    assert (first, second) == (1.5, 3.5)
    assert client.values == {"ns:spend:key:k": 3.5}
    assert client.ttls == {"ns:spend:key:k": expected_ttl}
    assert client.round_trips == expected_round_trips


class _SetRecordingPipeline:
    def __init__(self) -> None:
        self.sets: list[tuple[str, str, timedelta | None]] = []
        self.executes = 0

    async def __aenter__(self) -> "_SetRecordingPipeline":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    def set(self, name: str, value: str, ex: timedelta | None) -> None:
        self.sets.append((name, value, ex))

    async def execute(self) -> list[bool]:
        self.executes += 1
        return [True] * len(self.sets)


@pytest.mark.asyncio
async def test_async_set_cache_pipeline_with_ttls_keeps_each_entry_ttl(monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    monkeypatch.setattr(litellm, "default_redis_ttl", 300)
    redis_cache = RedisCache(namespace="ns")
    pipe = _SetRecordingPipeline()
    client = MagicMock()
    client.pipeline = MagicMock(return_value=pipe)

    with patch.object(redis_cache, "init_async_client", return_value=client):
        await redis_cache.async_set_cache_pipeline_with_ttls(
            (("team_id:t1", {"team_id": "t1"}, 60), ("u1", {"user_id": "u1"}, 7), ("org_id:o1", {"a": 1}, None))
        )

    client.pipeline.assert_called_once_with(transaction=False)
    assert pipe.executes == 1
    assert pipe.sets == [
        ("ns:team_id:t1", '{"team_id": "t1"}', timedelta(seconds=60)),
        ("ns:u1", '{"user_id": "u1"}', timedelta(seconds=7)),
        ("ns:org_id:o1", '{"a": 1}', timedelta(seconds=300)),
    ]
