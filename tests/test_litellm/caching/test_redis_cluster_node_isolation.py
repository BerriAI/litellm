"""Regression: a single cluster node's ConnectionError/TimeoutError must reset only that
node's connections, not tear down the whole cluster client for every other concurrent
caller. Live confirmation against a real 3-master local cluster (pausing one node with
CLIENT PAUSE) showed 100% of concurrent commands to the other two, untouched nodes
stalling for the full pause duration before this fix, and zero after -- these tests pin
the same behavior at the unit level so it can run without a live Redis Cluster."""

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import pytest
from redis.exceptions import (
    AskError,
    BusyLoadingError,
    ClusterDownError,
    ClusterError,
    MaxConnectionsError,
    MovedError,
    TryAgainError,
)
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
)
from redis.exceptions import (
    TimeoutError as RedisTimeoutError,
)

from litellm.caching.redis_cluster_node_isolation import (
    get_litellm_async_redis_cluster_class,
)

if TYPE_CHECKING:
    from redis.asyncio.cluster import RedisCluster as _AsyncRedisClusterType


class _NodeClassWithPerConnectionRecovery:
    def update_active_connections_for_reconnect(self) -> None: ...


class _NodeClassWithoutPerConnectionRecovery:
    pass


class _FakeClusterNode:
    def __init__(self, name: str, raises: Exception | None = None, response: object = None) -> None:
        self.name = name
        self.execute_command = AsyncMock(side_effect=raises, return_value=response)
        self.disconnect = AsyncMock()


class _Fake8xRedisCluster:
    def __init__(self) -> None:
        self._initialize = False

    async def _execute_command(
        self, target_node: _FakeClusterNode, *args: object, **kwargs: object
    ) -> object:
        try:
            return await target_node.execute_command(*args, **kwargs)
        except (RedisConnectionError, RedisTimeoutError):
            self._initialize = True
            await asyncio.sleep(0)
            raise

    async def aclose(self) -> None:
        self._initialize = True


class _FakeNodesManager:
    def __init__(self, node_to_return: _FakeClusterNode) -> None:
        self._moved_exception: object = None
        self._node_to_return = node_to_return

    def get_node_from_slot(
        self, slot: int, read_from_replicas: bool, load_balancing_strategy: object
    ) -> _FakeClusterNode:
        return self._node_to_return


def _build_cluster_instance() -> "_AsyncRedisClusterType":
    cluster_cls = get_litellm_async_redis_cluster_class(
        cluster_node_class=_NodeClassWithoutPerConnectionRecovery
    )
    instance = cluster_cls.__new__(cluster_cls)
    instance.RedisClusterRequestTTL = 1
    instance.reinitialize_counter = 0
    instance.reinitialize_steps = 5
    instance.read_from_replicas = False
    instance.load_balancing_strategy = None
    instance.aclose = AsyncMock()
    return instance


def _build_8x_cluster_instance() -> _Fake8xRedisCluster:
    cluster_cls = get_litellm_async_redis_cluster_class(
        cluster_node_class=_NodeClassWithPerConnectionRecovery,
        base_cluster_class=_Fake8xRedisCluster,
    )
    return cluster_cls()


def test_unverified_redis_version_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    import redis

    with patch.object(redis, "__version__", "8.0.1"):
        get_litellm_async_redis_cluster_class(cluster_node_class=_NodeClassWithoutPerConnectionRecovery)

    assert "not in the set this cluster-teardown-storm fix was verified against" in caplog.text


@pytest.mark.asyncio
async def test_single_timeout_does_not_request_topology_reinit() -> None:
    error = RedisTimeoutError("timeout")
    target_node = _FakeClusterNode("node-a")
    target_node.execute_command.side_effect = error
    instance = _build_8x_cluster_instance()

    with pytest.raises(RedisTimeoutError) as exc_info:
        await instance._execute_command(target_node, "GET", "k")

    assert exc_info.value is error
    assert instance._initialize is False


@pytest.mark.asyncio
async def test_connection_error_preserves_upstream_topology_reinit() -> None:
    error = RedisConnectionError("connection error")
    target_node = _FakeClusterNode("node-a")
    target_node.execute_command.side_effect = error
    instance = _build_8x_cluster_instance()

    with pytest.raises(RedisConnectionError) as exc_info:
        await instance._execute_command(target_node, "GET", "k")

    assert exc_info.value is error
    assert instance._initialize is True


@pytest.mark.asyncio
async def test_three_consecutive_timeouts_request_topology_reinit_and_reset_counter() -> None:
    errors = [
        RedisTimeoutError("timeout-1"),
        RedisTimeoutError("timeout-2"),
        RedisTimeoutError("timeout-3"),
    ]
    fourth_error = RedisTimeoutError("timeout-4")
    target_node = _FakeClusterNode("node-a")
    target_node.execute_command.side_effect = [*errors, fourth_error]
    instance = _build_8x_cluster_instance()

    for error in errors:
        with pytest.raises(RedisTimeoutError) as exc_info:
            await instance._execute_command(target_node, "GET", "k")
        assert exc_info.value is error

    assert instance._initialize is True
    instance._initialize = False

    with pytest.raises(RedisTimeoutError) as exc_info:
        await instance._execute_command(target_node, "GET", "k")

    assert exc_info.value is fourth_error
    assert instance._initialize is False


@pytest.mark.asyncio
async def test_success_resets_consecutive_timeout_counter() -> None:
    errors = [RedisTimeoutError("timeout-1"), RedisTimeoutError("timeout-2")]
    final_error = RedisTimeoutError("timeout-3")
    target_node = _FakeClusterNode("node-a")
    target_node.execute_command.side_effect = [*errors, b"value", final_error]
    instance = _build_8x_cluster_instance()

    for error in errors:
        with pytest.raises(RedisTimeoutError) as exc_info:
            await instance._execute_command(target_node, "GET", "k")
        assert exc_info.value is error
        assert instance._initialize is False

    result = await instance._execute_command(target_node, "GET", "k")
    assert result == b"value"
    assert instance._initialize is False

    with pytest.raises(RedisTimeoutError) as exc_info:
        await instance._execute_command(target_node, "GET", "k")

    assert exc_info.value is final_error
    assert instance._initialize is False


@pytest.mark.asyncio
async def test_timeout_counters_are_per_node() -> None:
    node_a_errors = [RedisTimeoutError("node-a-1"), RedisTimeoutError("node-a-2")]
    node_b_error = RedisTimeoutError("node-b-1")
    node_a = _FakeClusterNode("node-a")
    node_b = _FakeClusterNode("node-b")
    node_a.execute_command.side_effect = node_a_errors
    node_b.execute_command.side_effect = node_b_error
    instance = _build_8x_cluster_instance()

    for target_node, error in (
        (node_a, node_a_errors[0]),
        (node_b, node_b_error),
        (node_a, node_a_errors[1]),
    ):
        with pytest.raises(RedisTimeoutError) as exc_info:
            await instance._execute_command(target_node, "GET", "k")
        assert exc_info.value is error

    assert instance._initialize is False


@pytest.mark.asyncio
async def test_timeout_does_not_clear_concurrent_topology_reinit_request() -> None:
    error = RedisTimeoutError("timeout")
    instance = _build_8x_cluster_instance()

    async def request_reinit(*args: object, **kwargs: object) -> object:
        await instance.aclose()
        raise error

    target_node = _FakeClusterNode("node-a")
    target_node.execute_command.side_effect = request_reinit

    with pytest.raises(RedisTimeoutError) as exc_info:
        await instance._execute_command(target_node, "GET", "k")

    assert exc_info.value is error
    assert instance._initialize is True


@pytest.mark.asyncio
async def test_tolerated_timeout_does_not_erase_concurrent_connection_error_reinit() -> None:
    instance = _build_8x_cluster_instance()
    failing_node = _FakeClusterNode("node-a", raises=RedisConnectionError("gone"))
    slow_node = _FakeClusterNode("node-b", raises=RedisTimeoutError("slow"))

    results = await asyncio.gather(
        instance._execute_command(failing_node, "GET", "a"),
        instance._execute_command(slow_node, "GET", "b"),
        return_exceptions=True,
    )

    assert isinstance(results[0], RedisConnectionError)
    assert isinstance(results[1], RedisTimeoutError)
    assert instance._initialize is True


@pytest.mark.asyncio
async def test_tolerated_timeout_does_not_clear_pending_reinit() -> None:
    instance = _build_8x_cluster_instance()
    instance._initialize = True
    target_node = _FakeClusterNode("node-a", raises=RedisTimeoutError("slow"))

    with pytest.raises(RedisTimeoutError):
        await instance._execute_command(target_node, "GET", "k")

    assert instance._initialize is True


@pytest.mark.asyncio
async def test_success_returns_value_without_topology_reinit() -> None:
    target_node = _FakeClusterNode("node-a", response=b"value")
    instance = _build_8x_cluster_instance()

    result = await instance._execute_command(target_node, "GET", "k")

    assert result == b"value"
    assert instance._initialize is False


@pytest.mark.asyncio
@pytest.mark.parametrize("error_cls", [RedisConnectionError, RedisTimeoutError])
async def test_node_level_error_resets_only_that_node_not_the_whole_client(error_cls: type[Exception]) -> None:
    """The fix: a ConnectionError/TimeoutError must disconnect only the failing node
    and must NOT call the client-wide aclose() that tears down every node."""
    target_node = _FakeClusterNode("node-a", raises=error_cls("boom"))
    instance = _build_cluster_instance()

    with pytest.raises(error_cls):
        await instance._execute_command(target_node, "GET", "k")

    target_node.disconnect.assert_awaited_once()
    instance.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_moved_error_retries_without_full_reinit_before_threshold() -> None:
    moved_error = MovedError("1 127.0.0.1:7001")
    target_node = _FakeClusterNode("node-a")
    target_node.execute_command = AsyncMock(side_effect=[moved_error, b"value"])
    instance = _build_cluster_instance()
    instance.RedisClusterRequestTTL = 2
    instance.nodes_manager = _FakeNodesManager(node_to_return=target_node)
    instance._determine_slot = AsyncMock(return_value=0)

    result = await instance._execute_command(target_node, "GET", "k")

    assert result == b"value"
    assert instance.nodes_manager._moved_exception is moved_error
    instance.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_ask_error_sends_asking_and_retries_on_redirected_node() -> None:
    ask_error = AskError("0 127.0.0.1:7001")
    target_node = _FakeClusterNode("node-a")
    target_node.execute_command = AsyncMock(side_effect=[ask_error, None, b"value"])
    instance = _build_cluster_instance()
    instance.RedisClusterRequestTTL = 2
    instance.get_node = Mock(return_value=target_node)

    result = await instance._execute_command(target_node, "GET", "k")

    assert result == b"value"
    instance.get_node.assert_called_once_with(node_name="127.0.0.1:7001")


@pytest.mark.asyncio
async def test_try_again_error_exhausts_ttl() -> None:
    target_node = _FakeClusterNode("node-a", raises=TryAgainError("try again"))
    instance = _build_cluster_instance()
    instance.RedisClusterRequestTTL = 2

    with pytest.raises(ClusterError):
        await instance._execute_command(target_node, "GET", "k")


@pytest.mark.asyncio
async def test_successful_command_touches_neither_disconnect_nor_aclose() -> None:
    target_node = _FakeClusterNode("node-a", response=b"v")
    instance = _build_cluster_instance()

    result = await instance._execute_command(target_node, "GET", "k")

    assert result == b"v"
    target_node.disconnect.assert_not_awaited()
    instance.aclose.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_cls", [BusyLoadingError, MaxConnectionsError])
async def test_busy_loading_and_max_connections_reraise_without_any_reset(error_cls: type[Exception]) -> None:
    """Unchanged from upstream: these say nothing about node health, so neither the
    node nor the client should be reset."""
    target_node = _FakeClusterNode("node-a", raises=error_cls("boom"))
    instance = _build_cluster_instance()

    with pytest.raises(error_cls):
        await instance._execute_command(target_node, "GET", "k")

    target_node.disconnect.assert_not_awaited()
    instance.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_cluster_down_error_still_triggers_a_full_reinit() -> None:
    """Unchanged from upstream: ClusterDownError is real evidence the topology
    changed, so a full-client reinit (unlike a plain timeout) is still correct here."""
    target_node = _FakeClusterNode("node-a", raises=ClusterDownError("boom"))
    instance = _build_cluster_instance()

    with pytest.raises(ClusterDownError):
        await instance._execute_command(target_node, "GET", "k")

    instance.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_moved_error_still_triggers_reinit_after_reinitialize_steps() -> None:
    """Unchanged from upstream: repeated MOVED responses are real evidence of a
    slot migration, so they should still force a full reinit every `reinitialize_steps`."""
    target_node = _FakeClusterNode("node-a", raises=MovedError("1 127.0.0.1:7001"))
    instance = _build_cluster_instance()
    instance.reinitialize_steps = 1
    instance.RedisClusterRequestTTL = 2
    instance.nodes_manager = _FakeNodesManager(node_to_return=target_node)
    instance._determine_slot = AsyncMock(return_value=0)

    target_node.execute_command = AsyncMock(side_effect=[MovedError("1 127.0.0.1:7001"), b"v"])

    result = await instance._execute_command(target_node, "GET", "k")

    assert result == b"v"
    instance.aclose.assert_awaited_once()
    assert instance.reinitialize_counter == 0
