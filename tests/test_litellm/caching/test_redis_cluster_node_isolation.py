"""Regression: a single cluster node's ConnectionError/TimeoutError must reset only that
node's connections, not tear down the whole cluster client for every other concurrent
caller. Live confirmation against a real 3-master local cluster (pausing one node with
CLIENT PAUSE) showed 100% of concurrent commands to the other two, untouched nodes
stalling for the full pause duration before this fix, and zero after -- these tests pin
the same behavior at the unit level so it can run without a live Redis Cluster."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import (
    BusyLoadingError,
    ClusterDownError,
    MaxConnectionsError,
    MovedError,
)
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
)
from redis.exceptions import TimeoutError as RedisTimeoutError

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


def test_per_connection_recovery_redis_py_gets_the_unmodified_upstream_class() -> None:
    """Regression (redis-py 8.x): when upstream ClusterNode already recovers a node-level
    connection error per-connection, the factory must NOT install the copied override,
    whose node.disconnect() also kills connections other coroutines are mid-operation on."""
    from redis.asyncio.cluster import RedisCluster

    cluster_cls = get_litellm_async_redis_cluster_class(
        cluster_node_class=_NodeClassWithPerConnectionRecovery
    )

    assert cluster_cls is RedisCluster


def test_pre_recovery_redis_py_still_gets_the_node_isolation_override() -> None:
    """Old redis-py (5.x) responds to a node-level error with a full-cluster aclose(),
    so those versions must keep litellm's per-node isolation override."""
    from redis.asyncio.cluster import RedisCluster

    cluster_cls = get_litellm_async_redis_cluster_class(
        cluster_node_class=_NodeClassWithoutPerConnectionRecovery
    )

    assert cluster_cls is not RedisCluster
    assert issubclass(cluster_cls, RedisCluster)
    assert "_execute_command" in cluster_cls.__dict__


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
