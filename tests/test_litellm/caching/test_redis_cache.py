import asyncio
import logging
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
async def test_circuit_breaker_opens_when_method_swallows_redis_failure(redis_no_ping, call_method):
    """A guarded method that swallows its own Redis error must still count as a failure.

    These methods catch connection errors and return a default so callers degrade instead
    of failing, which is correct. But that returns cleanly through the circuit breaker
    guard, and counting it as a success reset the failure streak on every call, so the
    breaker could never open. An unreachable Redis then stayed in the pool and every
    request kept paying the full socket timeout on it.
    """
    from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD

    cache = RedisCache(host="127.0.0.1", port=_closed_port(), socket_timeout=0.5)

    for _ in range(REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        await call_method(cache)

    with pytest.raises(Exception, match="circuit breaker is open"):
        await call_method(cache)


@pytest.mark.asyncio
async def test_circuit_breaker_success_still_resets_the_failure_streak(redis_no_ping):
    """A reachable Redis must keep the breaker closed, however many earlier calls failed.

    The guard now records success only when nothing failed while the method ran, so this
    pins the other half of that contract: a call that genuinely reaches Redis has to clear
    the streak, or a healthy Redis would eventually be evicted from the pool.
    """
    from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD

    cache = RedisCache(host="127.0.0.1", port=_closed_port(), socket_timeout=0.5)

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
async def test_circuit_breaker_covers_lua_script_execution(redis_no_ping):
    """Lua script execution must feed the breaker like every other Redis call.

    The v3 rate limiter issues all of its Redis traffic through async_register_script, so
    leaving that path unguarded meant the coordination calls during an outage never
    counted toward taking Redis out of the pool and kept paying a full socket timeout
    each, which is the traffic the outage hurts most.
    """
    from redis.exceptions import ConnectionError as RedisConnectionError

    from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD

    cache = RedisCache(host="127.0.0.1", port=_closed_port(), socket_timeout=0.5)
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
        return None

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
        pytest.param("TimeoutError", True, id="timeout_is_unhealthy"),
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


# Every Redis write error logged the value being written (#11157), so an ordinary
# retryable failure — a pooled socket the server had already reaped, a read timeout —
# put the full upstream response object (choices[].message.content included) into the
# proxy's ERROR log, where it is shipped to wherever the proxy's stdout goes. The
# failure still has to be reported; the payload does not belong in the report.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fail_at",
    [
        pytest.param("client", id="client_init_failure"),
        pytest.param("write", id="write_failure"),
    ],
)
async def test_async_set_cache_error_log_omits_the_cached_value(
    fail_at, monkeypatch, redis_no_ping, caplog
):
    """A failed async_set_cache must report the error and the key, never the value."""
    from redis.exceptions import ConnectionError as RedisConnectionError

    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    response_marker = "response-body-marker"
    value = {"choices": [{"message": {"content": f"{response_marker} " * 20}}]}
    redis_error = RedisConnectionError("Connection closed by server.")

    mock_redis_instance = AsyncMock()
    if fail_at == "write":
        mock_redis_instance.set.side_effect = redis_error
        client_patch = patch.object(
            redis_cache, "init_async_client", return_value=mock_redis_instance
        )
    else:
        client_patch = patch.object(
            redis_cache, "init_async_client", side_effect=redis_error
        )

    with caplog.at_level(logging.ERROR, logger="LiteLLM"), client_patch:
        if fail_at == "client":
            with pytest.raises(RedisConnectionError):
                await redis_cache.async_set_cache("lit4930", value)
        else:
            await redis_cache.async_set_cache("lit4930", value)

    logged = [
        record.getMessage()
        for record in caplog.records
        if "async set()" in record.getMessage()
    ]
    assert logged, "a failed cache write still has to be reported"
    assert all(
        response_marker not in message for message in logged
    ), "the cached value must not reach the log record"
    assert any("Connection closed by server." in message for message in logged)
    assert any("lit4930" in message for message in logged)
    assert any(
        "value_bytes=" in message for message in logged
    ), "the size of the dropped write is the accounting the value used to carry"


@pytest.mark.asyncio
async def test_async_set_cache_pipeline_error_log_omits_the_cached_values(
    monkeypatch, redis_no_ping, caplog
):
    """Same contract for the bulk path, which logged a `cache_value` that was
    unconditionally None — it leaked nothing here only because it reported nothing."""
    from redis.exceptions import TimeoutError as RedisTimeoutError

    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    response_marker = "pipeline-body-marker"
    cache_list = [
        ("lit4930", {"choices": [{"message": {"content": f"{response_marker} " * 20}}]}),
        ("lit4931", {"choices": [{"message": {"content": f"{response_marker} " * 20}}]}),
    ]
    redis_error = RedisTimeoutError("Timeout reading from my-test-host:6379")

    mock_redis_instance = AsyncMock()
    mock_redis_instance.pipeline = MagicMock(side_effect=redis_error)

    with caplog.at_level(logging.ERROR, logger="LiteLLM"), patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_set_cache_pipeline(cache_list=cache_list)

    logged = [
        record.getMessage()
        for record in caplog.records
        if "async set_cache_pipeline()" in record.getMessage()
    ]
    assert logged, "a failed pipeline write still has to be reported"
    assert all(
        response_marker not in message for message in logged
    ), "the cached values must not reach the log record"
    assert any("Timeout reading from my-test-host:6379" in message for message in logged)
    assert any(
        "num_keys=2" in message for message in logged
    ), "how much was dropped is what the operator needs from this line"


@pytest.mark.asyncio
async def test_async_set_cache_sadd_error_log_omits_the_cached_value(
    monkeypatch, redis_no_ping, caplog
):
    """The set-member path takes caller-supplied values too, and swallows its error."""
    from redis.exceptions import ConnectionError as RedisConnectionError

    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    member_marker = "sadd-member-marker"
    redis_error = RedisConnectionError("Connection closed by server.")

    mock_redis_instance = AsyncMock()
    mock_redis_instance.sadd.side_effect = redis_error

    with caplog.at_level(logging.ERROR, logger="LiteLLM"), patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_set_cache_sadd(
            key="lit4930", value=[f"{member_marker}-1", f"{member_marker}-2"], ttl=60
        )

    logged = [
        record.getMessage()
        for record in caplog.records
        if "async set_cache_sadd()" in record.getMessage()
    ]
    assert logged, "a failed sadd still has to be reported"
    assert all(member_marker not in message for message in logged)
    assert any("Connection closed by server." in message for message in logged)
    assert any("lit4930" in message for message in logged)


# Dropping the value argument is only half the path. redis-py rewraps every pipelined
# failure as "Command # N (<the whole command>) of pipeline caused error: <reason>", so
# the cached value reaches the same ERROR log through the exception text while the value
# argument reads None. The failure still has to be reported; the payload does not.


def _payload_bearing_error(reason_first: bool, marker: str) -> str:
    """A redis error text carrying a response body, in each of the two observed shapes."""
    body = f"{'x' * 200}{marker}{'x' * 200}"
    if reason_first:
        return f"Connection closed by server. {body}"
    return f"Command # 1 (SET litellm:lit4930 {body}) of pipeline caused error: READONLY You can't write against a read only replica."


@pytest.mark.asyncio
async def test_async_set_cache_error_log_bounds_the_exception_text(
    monkeypatch, redis_no_ping, caplog
):
    """The exception text is bounded, so a client that embeds the command cannot use it
    as a second route for the value into the log."""
    from redis.exceptions import ConnectionError as RedisConnectionError

    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    response_marker = "exception-slot-marker"
    redis_error = RedisConnectionError(
        _payload_bearing_error(reason_first=True, marker=response_marker)
    )

    mock_redis_instance = AsyncMock()
    mock_redis_instance.set.side_effect = redis_error

    with caplog.at_level(logging.ERROR, logger="LiteLLM"), patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_set_cache("lit4930", {"choices": []})

    logged = [
        record.getMessage()
        for record in caplog.records
        if "async set()" in record.getMessage()
    ]
    assert logged, "a failed cache write still has to be reported"
    assert all(
        response_marker not in message for message in logged
    ), "the payload embedded in the exception must not reach the log record"
    assert any(
        "Connection closed by server." in message for message in logged
    ), "the reason leads this shape, so bounding must keep it"
    assert any("ConnectionError:" in message for message in logged)
    assert any("truncated" in message for message in logged)
    assert all(len(message) < 500 for message in logged)


@pytest.mark.asyncio
async def test_async_set_cache_pipeline_error_log_keeps_the_reason_at_the_tail(
    monkeypatch, redis_no_ping, caplog
):
    """redis-py's pipeline wrapper puts the command first and the reason last, so a
    head-only bound would print part of the payload and drop the only diagnostic."""
    from redis.exceptions import ResponseError

    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    response_marker = "pipeline-exception-slot-marker"
    redis_error = ResponseError(
        _payload_bearing_error(reason_first=False, marker=response_marker)
    )

    mock_redis_instance = AsyncMock()
    mock_redis_instance.pipeline = MagicMock(side_effect=redis_error)

    with caplog.at_level(logging.ERROR, logger="LiteLLM"), patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_set_cache_pipeline(
            cache_list=[("lit4930", {"choices": []})]
        )

    logged = [
        record.getMessage()
        for record in caplog.records
        if "async set_cache_pipeline()" in record.getMessage()
    ]
    assert logged, "a failed pipeline write still has to be reported"
    assert all(
        response_marker not in message for message in logged
    ), "the payload embedded in the exception must not reach the log record"
    assert any(
        "READONLY" in message for message in logged
    ), "the reason trails this shape, so a head-only bound would lose it"
    assert any("truncated" in message for message in logged)
    assert all(len(message) < 500 for message in logged)


@pytest.mark.parametrize(
    "raised",
    [
        pytest.param(ValueError("no"), id="exception"),
        pytest.param(KeyboardInterrupt("no"), id="base_exception"),
    ],
)
def test_redis_error_text_never_raises(raised):
    """__str__ is caller-controlled code and this runs inside except blocks that exist
    to keep the request alive, so an unrenderable exception has to render to something."""
    from litellm.caching.redis_cache import _redis_error_text

    class Unrenderable(Exception):
        def __str__(self):
            raise raised

    try:
        rendered = _redis_error_text(Unrenderable())
    except BaseException as escaped:  # noqa: BLE001  # the regression is precisely that anything escapes here
        pytest.fail(f"rendering must not re-raise: {escaped!r}")

    assert "UnrenderableException" in rendered


@pytest.mark.asyncio
async def test_error_log_survives_an_exception_that_cannot_be_rendered(
    monkeypatch, redis_no_ping, caplog
):
    """End to end: a write whose error cannot be rendered is still a swallowed write,
    not a failed request."""
    from redis.exceptions import ConnectionError as RedisConnectionError

    class UnrenderableRedisError(RedisConnectionError):
        def __str__(self):
            raise ValueError("__str__ is caller-controlled code")

    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    mock_redis_instance = AsyncMock()
    mock_redis_instance.set.side_effect = UnrenderableRedisError()

    with caplog.at_level(logging.ERROR, logger="LiteLLM"), patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        try:
            await redis_cache.async_set_cache("lit4930", {"choices": []})
        except BaseException as escaped:  # noqa: BLE001  # the regression is precisely that anything escapes here
            pytest.fail(f"rendering the exception must not escalate the failure: {escaped!r}")

    logged = [
        record.getMessage()
        for record in caplog.records
        if "async set()" in record.getMessage()
    ]
    assert logged, "the write failure still has to be reported"
    assert any("UnrenderableException" in message for message in logged)
    assert any("lit4930" in message for message in logged)
