import socket
from collections.abc import Awaitable, Callable, Iterator
from typing import Final

import pytest

from litellm.caching.affinity_cache import claim_affinity_pin, read_affinity_pin
from litellm.caching.redis_batch import Batch
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.connection import RedisStores, connect
from litellm.redis.stores.router_session_pins import DeploymentSessionPin, RouterSessionPinStore
from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck
from litellm.rust_bridge._native import NativeRedisStoreError
from tests.unit.redis.stores.router_session_pins_memory import memory_pins

_KEY: Final = DeploymentSessionPin(model_group="group", caller=None, session_id="session")


@pytest.fixture
def unreachable_redis() -> Iterator[RedisStores]:
    """A real native client to a port nothing listens on, so every Redis call is refused at once."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: Final = probe.getsockname()[1]
    yield connect(RedisCache(host="127.0.0.1", port=port))


@pytest.fixture
def down_pins(unreachable_redis: RedisStores) -> RouterSessionPinStore:
    return RouterSessionPinStore(unreachable_redis.router_session_pins())


async def test_attaching_redis_keeps_the_memory_pins(unreachable_redis: RedisStores) -> None:
    pins: Final = memory_pins()
    pins.set_local(_KEY, {"model_id": "dep-a"}, 60)

    pins.attach_redis(unreachable_redis)

    assert pins.has_redis
    assert await pins.get(_KEY) == {"model_id": "dep-a"}


async def test_lookups_and_writes_serve_memory_while_redis_is_down(down_pins: RouterSessionPinStore) -> None:
    assert await down_pins.get(_KEY) is None

    await down_pins.set(_KEY, {"model_id": "dep-a"}, 60)

    assert await down_pins.get(_KEY) == {"model_id": "dep-a"}
    down_pins.delete_local(_KEY)
    assert down_pins.get_local(_KEY) is None


@pytest.mark.parametrize(
    "op",
    [
        lambda pins: pins.get_shared(_KEY),
        lambda pins: pins.read(_KEY, 60, batch=Batch.NONE),
        lambda pins: pins.claim(_KEY, {"model_id": "dep-a"}, 60, eligible_values=None, preserve_legacy_model_ids=False),
    ],
    ids=["get_shared", "read", "claim"],
)
async def test_redis_only_ops_raise_while_redis_is_down(
    down_pins: RouterSessionPinStore, op: Callable[[RouterSessionPinStore], Awaitable[object]]
) -> None:
    with pytest.raises(NativeRedisStoreError, match="Redis did not answer"):
        await op(down_pins)


async def test_a_read_without_redis_answers_from_memory() -> None:
    pins: Final = memory_pins()
    pins.set_local(_KEY, {"model_id": "dep-a"}, 60)

    assert await pins.read(_KEY, 60, batch=Batch.PRE_CALL) == {"model_id": "dep-a"}


async def test_a_failed_pin_read_keeps_the_pod_local_pin(down_pins: RouterSessionPinStore) -> None:
    down_pins.set_local(_KEY, {"model": "m", "tier": "SIMPLE"}, 60)

    assert await read_affinity_pin(down_pins, _KEY, 60) == {"model": "m", "tier": "SIMPLE"}
    assert await read_affinity_pin(down_pins, DeploymentSessionPin("group", None, "other"), 60) is None


async def test_a_failed_claim_falls_back_to_a_pod_local_first_writer(down_pins: RouterSessionPinStore) -> None:
    assert await claim_affinity_pin(down_pins, _KEY, {"model_id": "dep-a"}, 60) == {"model_id": "dep-a"}
    assert await claim_affinity_pin(down_pins, _KEY, {"model_id": "dep-b"}, 60) == {"model_id": "dep-a"}


@pytest.mark.parametrize("local", [None, "dep-a"])
async def test_session_routing_keeps_the_local_pin_when_the_shared_read_fails(
    down_pins: RouterSessionPinStore, local: str | None
) -> None:
    callback: Final = DeploymentAffinityCheck(
        pins=down_pins,
        ttl_seconds=60,
        enable_user_key_affinity=False,
        enable_responses_api_affinity=False,
        enable_session_id_affinity=True,
        session_affinity_repin_on_failover=True,
    )
    if local is not None:
        down_pins.set_local(_KEY, {"model_id": local}, 60)
    deployments: Final = [
        {"model_name": "group", "litellm_params": {"model": "openai/m"}, "model_info": {"id": deployment}}
        for deployment in ("dep-a", "dep-b")
    ]

    selected: Final = await callback.async_filter_deployments(
        model="group",
        healthy_deployments=deployments,
        messages=None,
        request_kwargs={"metadata": {"session_id": "session"}},
    )

    assert [deployment["model_info"]["id"] for deployment in selected] == (
        ["dep-a"] if local is not None else ["dep-a", "dep-b"]
    )
