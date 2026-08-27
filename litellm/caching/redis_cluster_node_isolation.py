"""Bounds the blast radius of a single node's transient connection error on the async
Redis Cluster client.

redis-py's ``RedisCluster._execute_command`` responds to a ``ConnectionError`` or
``TimeoutError`` on ANY one node by tearing down every node's connections and flipping
the client into "needs reinitialization", which forces every other concurrent caller
sharing this client through one reinit lock until the whole cluster topology is
re-walked. Under real proxy load, a client-side socket timeout on a single node is a
routine event (the event loop was too busy to read the response before ``socket_timeout``
elapsed) and does not mean the cluster's topology moved, so treating it as a full-cluster
event turns one slow node into a proxy-wide latency spike while Redis itself stays
healthy -- confirmed live: pausing one of three local cluster nodes made every concurrent
command against the other two, untouched nodes stall for the full pause duration too.

``get_litellm_async_redis_cluster_class`` returns a ``RedisCluster`` subclass that resets
only the node that actually failed (mirroring what a plain, non-cluster Redis client
already does when one of its pooled connections errors), leaving every other node's
connections untouched. Every other branch (MOVED, ASK, CLUSTERDOWN, slot-not-covered,
retry-exhaustion) is unchanged from upstream, since those already carry real evidence the
topology changed.

redis-py 8.x fixed this upstream with gentler machinery than this override's
``node.disconnect()`` (which also kills connections other coroutines are mid-operation
on, so one timeout cascades into a reconnect storm and, with TLS, a fresh handshake per
killed connection): it marks in-use connections for reconnect only after their current
operation completes, disconnects only the idle pooled ones, and defers reinitialization
to the outer retry loop. When the installed ``ClusterNode`` has that per-connection
recovery API, the factory returns the base ``RedisCluster`` unmodified.
"""

import asyncio
from typing import TYPE_CHECKING, Final, Protocol

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from redis.asyncio.cluster import RedisCluster as _AsyncRedisClusterType


class _ClusterNodeAttrs(Protocol):
    """The subset of ``redis.asyncio.cluster.ClusterNode`` this override reads. redis-py
    ships no resolvable stub for these members under the repo's current types-redis pin,
    so a plain attribute access resolves every downstream use to ``Unknown`` under strict
    mode; typing ``target_node`` as this Protocol at the one boundary keeps the override's
    own logic fully typed without a banned ``typing.cast``."""

    async def execute_command(
        self,
        *args: object,
        **kwargs: object,  # kwargs-ok: mirrors redis-py's own ClusterNode.execute_command signature, a raw command dispatch with no fixed keyword contract
    ) -> object: ...
    async def disconnect(self) -> None: ...


class _NodesManagerAttrs(Protocol):
    _moved_exception: object

    def get_node_from_slot(
        self, slot: int, read_from_replicas: bool, load_balancing_strategy: object
    ) -> _ClusterNodeAttrs: ...


class _ClusterAttrs(Protocol):
    RedisClusterRequestTTL: int
    reinitialize_counter: int
    reinitialize_steps: int
    read_from_replicas: bool
    load_balancing_strategy: object
    nodes_manager: _NodesManagerAttrs

    def get_node(self, node_name: str) -> _ClusterNodeAttrs: ...
    async def _determine_slot(self, *args: object) -> int: ...
    async def aclose(self) -> None: ...


#: redis-py versions this override's copied ``_execute_command`` body has been verified
#: against. A version outside this set may have changed the method's structure in a way
#: this override can't see (Python won't error -- it'll just run our now-stale copy), so
#: construction logs a loud warning rather than silently trusting an unverified copy.
_VERIFIED_REDIS_VERSIONS: Final = frozenset({"5.3.1"})


def get_litellm_async_redis_cluster_class(
    cluster_node_class: type | None = None,
) -> type["_AsyncRedisClusterType"]:
    """Returns the base ``RedisCluster`` when the installed redis-py already recovers a
    node-level connection error per-connection (8.x+), else builds the ``RedisCluster``
    subclass with the per-node isolation fix for older versions whose upstream branch
    tears down the whole cluster client.

    ``cluster_node_class`` exists for dependency injection in tests; production callers
    leave it unset and the installed ``ClusterNode`` is used.

    Imported lazily because this module is reachable from a base ``import litellm`` while
    redis is not a base dependency. Cheap to call repeatedly: the underlying redis
    submodules are cached in ``sys.modules`` after the first import.
    """
    import redis
    from redis.asyncio.cluster import (
        ClusterNode as _AsyncClusterNode,  # pyright: ignore[reportUnknownVariableType]  # redis-py ships no resolvable stub for this class under the repo's current (stale) types-redis pin
    )
    from redis.asyncio.cluster import (
        RedisCluster as _BaseAsyncRedisCluster,  # pyright: ignore[reportUnknownVariableType]  # same stale-stub gap as the import above
    )
    from redis.cluster import get_node_name
    from redis.commands import READ_COMMANDS
    from redis.exceptions import (
        AskError,
        BusyLoadingError,
        ClusterDownError,
        ClusterError,
        MaxConnectionsError,
        MovedError,
        SlotNotCoveredError,
        TryAgainError,
    )
    from redis.exceptions import ConnectionError as _RedisConnectionError
    from redis.exceptions import TimeoutError as _RedisTimeoutError

    node_class: Final = cluster_node_class if cluster_node_class is not None else _AsyncClusterNode
    if hasattr(node_class, "update_active_connections_for_reconnect"):
        verbose_logger.debug(
            "redis-py %s recovers a node-level connection error per-connection upstream; "
            "using the base RedisCluster without litellm's node-isolation override.",
            redis.__version__,
        )
        return _BaseAsyncRedisCluster

    if redis.__version__ not in _VERIFIED_REDIS_VERSIONS:
        verbose_logger.warning(
            "redis-py %s is not in the set this cluster-teardown-storm fix was verified "
            "against (%s). The per-node-isolation override may not match the installed library's "
            "real _execute_command behavior.",
            redis.__version__,
            sorted(_VERIFIED_REDIS_VERSIONS),
        )

    class LiteLLMAsyncRedisCluster(
        _BaseAsyncRedisCluster  # pyright: ignore[reportUntypedBaseClass]  # same stale-stub gap as the import above; the base class itself is unresolvable, not this subclass's own code
    ):
        async def _execute_command(
            self,
            target_node: _ClusterNodeAttrs,
            *args: object,
            **kwargs: object,  # kwargs-ok: overrides redis-py's own **kwargs signature; the keyword contract is defined by the Redis command being dispatched, not by this method
        ) -> object:
            cluster: _ClusterAttrs = self
            node = target_node

            asking = moved = False
            redirect_addr: str | None = None
            ttl = cluster.RedisClusterRequestTTL

            while ttl > 0:
                ttl -= 1
                try:
                    if asking:
                        assert redirect_addr is not None
                        node = cluster.get_node(node_name=redirect_addr)
                        await node.execute_command("ASKING")
                        asking = False
                    elif moved:
                        slot = await cluster._determine_slot(*args)  # pyright: ignore[reportPrivateUsage]  # mirrors upstream's own un-overridden branch, which makes this identical private call from the same subclass
                        node = cluster.nodes_manager.get_node_from_slot(
                            slot,
                            cluster.read_from_replicas and args[0] in READ_COMMANDS,
                            (cluster.load_balancing_strategy if args[0] in READ_COMMANDS else None),
                        )
                        moved = False

                    return await node.execute_command(*args, **kwargs)
                except (BusyLoadingError, MaxConnectionsError):
                    raise
                except (_RedisConnectionError, _RedisTimeoutError):
                    # Reset only the node that actually failed instead of the upstream
                    # default (`await self.aclose()`, a full-cluster teardown that forces
                    # every other concurrent caller through the shared reinit lock).
                    await node.disconnect()
                    raise
                except (ClusterDownError, SlotNotCoveredError):
                    await cluster.aclose()
                    await asyncio.sleep(0.25)
                    raise
                except MovedError as e:
                    cluster.reinitialize_counter += 1
                    if cluster.reinitialize_steps and cluster.reinitialize_counter % cluster.reinitialize_steps == 0:
                        await cluster.aclose()
                        cluster.reinitialize_counter = 0
                    else:
                        cluster.nodes_manager._moved_exception = e  # pyright: ignore[reportPrivateUsage]  # mirrors upstream's own un-overridden branch; redis-py exposes no public setter for this
                    moved = True
                except AskError as e:
                    redirect_addr = get_node_name(host=e.host, port=e.port)
                    asking = True
                except TryAgainError:
                    if ttl < cluster.RedisClusterRequestTTL / 2:
                        await asyncio.sleep(0.05)

            raise ClusterError("TTL exhausted.")

    return LiteLLMAsyncRedisCluster
