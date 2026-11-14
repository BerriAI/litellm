import asyncio
import json
import os
import uuid
from collections.abc import Iterator, Mapping
from typing import Final

import pytest

from litellm.caching.affinity_cache import claim_affinity_pin, read_affinity_pin
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.connection import connect
from litellm.redis.stores.router_session_pins import (
    DeploymentPin,
    DeploymentSessionPin,
    PinKey,
    RouterSessionPinStore,
    TierModelPin,
    TierSessionPin,
)
from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck

_DEPLOYMENTS: Final = ("original", "replacement", "other")


@pytest.fixture
def redis() -> Iterator[RedisCache]:
    cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    yield cache
    cache.redis_client.close()


@pytest.fixture
def session_id() -> str:
    return uuid.uuid4().hex


def _pins(redis: RedisCache) -> RouterSessionPinStore:
    return RouterSessionPinStore(connect(redis).router_session_pins())


def _session_key(session_id: str) -> DeploymentSessionPin:
    return DeploymentSessionPin(model_group="repin-group", caller=None, session_id=session_id)


def _spelled(session_id: str) -> str:
    return f"deployment_affinity:v1:session:repin-group:unscoped:{session_id}"


def _repin_check(pins: RouterSessionPinStore, enabled: bool = True) -> DeploymentAffinityCheck:
    return DeploymentAffinityCheck(
        pins=pins,
        ttl_seconds=3600,
        enable_user_key_affinity=False,
        enable_responses_api_affinity=False,
        enable_session_id_affinity=True,
        session_affinity_repin_on_failover=enabled,
    )


async def _selected(callback: DeploymentAffinityCheck, session_id: str) -> tuple[str, ...]:
    selected: Final = await callback.async_filter_deployments(
        model="repin-group",
        healthy_deployments=[
            {"model_name": "repin-group", "litellm_params": {"model": "openai/m"}, "model_info": {"id": deployment}}
            for deployment in _DEPLOYMENTS
        ],
        messages=None,
        request_kwargs={"metadata": {"session_id": session_id}},
    )
    return tuple(deployment["model_info"]["id"] for deployment in selected)


def _break(redis: RedisCache, key: str) -> None:
    redis.redis_client.delete(key)
    redis.redis_client.rpush(key, "wrong-type")


# Keys main's Python spelled before they moved into the native store
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "spelled"),
    [
        (
            DeploymentPin(model_group="gpt-4o", caller="sk-caller"),
            "deployment_affinity:v1:gpt-4o:8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891",
        ),
        (
            DeploymentSessionPin(model_group="gpt-4o", caller=None, session_id="sess-1"),
            "deployment_affinity:v1:session:gpt-4o:unscoped:sess-1",
        ),
        (
            TierSessionPin(router="smart", caller="key-hash", session_id="sess-1"),
            "complexity_router_session_affinity:v1:smart:key-hash:sess-1",
        ),
        (
            TierModelPin(
                router="smart",
                tiers="cfg",
                caller=None,
                session_id="séss",
                tier="COMPLEX",
                replay_fallback="m-a",
            ),
            "complexity_router_tier_model_affinity:v1:c83437032b0b0d7d8229f7aaf444096a83510ee630e5a510e4770b0f6f03a8d3",
        ),
    ],
)
async def test_a_pin_key_reads_the_key_main_wrote(redis: RedisCache, key: PinKey, spelled: str) -> None:
    try:
        redis.redis_client.set(spelled, json.dumps({"model_id": "main-pod"}), ex=60)
        assert await _pins(redis).get_shared(key) == {"model_id": "main-pod"}
    finally:
        redis.redis_client.delete(spelled)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_session_selection_observes_another_workers_repin_only_when_enabled(
    redis: RedisCache, session_id: str, enabled: bool
) -> None:
    pins: Final = _pins(redis)
    try:
        pins.set_local(_session_key(session_id), {"model_id": "original"}, 3600)
        await _pins(redis).set(_session_key(session_id), {"model_id": "replacement"}, 3600)

        assert await _selected(_repin_check(pins, enabled), session_id) == (
            "replacement" if enabled else "original",
        )
    finally:
        redis.redis_client.delete(_spelled(session_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("has_pin", [False, True])
async def test_redis_read_failures_preserve_session_routing(
    redis: RedisCache, session_id: str, has_pin: bool
) -> None:
    pins: Final = _pins(redis)
    callback: Final = _repin_check(pins)
    try:
        if has_pin:
            redis.redis_client.set(_spelled(session_id), json.dumps({"model_id": "replacement"}), ex=60)
            pins.set_local(_session_key(session_id), {"model_id": "original"}, 3600)
            assert await _selected(callback, session_id) == ("replacement",)
        _break(redis, _spelled(session_id))

        assert set(await _selected(callback, session_id)) == ({"replacement"} if has_pin else set(_DEPLOYMENTS))
    finally:
        redis.redis_client.delete(_spelled(session_id))


@pytest.mark.asyncio
async def test_confirmed_redis_miss_does_not_revive_a_local_pin_during_outage(
    redis: RedisCache, session_id: str
) -> None:
    pins: Final = _pins(redis)
    callback: Final = _repin_check(pins)
    try:
        pins.set_local(_session_key(session_id), {"model_id": "original"}, 3600)
        assert set(await _selected(callback, session_id)) == set(_DEPLOYMENTS)
        _break(redis, _spelled(session_id))
        assert set(await _selected(callback, session_id)) == set(_DEPLOYMENTS)
    finally:
        redis.redis_client.delete(_spelled(session_id))


@pytest.mark.asyncio
async def test_shared_session_pin_resumes_after_a_transient_redis_failure(redis: RedisCache, session_id: str) -> None:
    pins: Final = _pins(redis)
    callback: Final = _repin_check(pins)
    try:
        pins.set_local(_session_key(session_id), {"model_id": "original"}, 3600)
        _break(redis, _spelled(session_id))
        assert await _selected(callback, session_id) == ("original",)
        redis.redis_client.delete(_spelled(session_id))
        redis.redis_client.set(_spelled(session_id), json.dumps({"model_id": "replacement"}), ex=60)
        assert await _selected(callback, session_id) == ("replacement",)
    finally:
        redis.redis_client.delete(_spelled(session_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_value", [None, {"model_id": "original"}])
async def test_a_shared_pin_read_in_flight_keeps_a_newer_local_write(
    redis: RedisCache, session_id: str, shared_value: Mapping[str, str] | None
) -> None:
    pins: Final = _pins(redis)
    callback: Final = _repin_check(pins)
    try:
        if shared_value is not None:
            redis.redis_client.set(_spelled(session_id), json.dumps(shared_value), ex=60)
        pending: Final = asyncio.create_task(_selected(callback, session_id))
        await asyncio.sleep(0)
        assert not pending.done()
        pins.set_local(_session_key(session_id), {"model_id": "replacement"}, 3600)
        await pending
        _break(redis, _spelled(session_id))

        assert await _selected(callback, session_id) == ("replacement",)
    finally:
        redis.redis_client.delete(_spelled(session_id))


@pytest.mark.asyncio
async def test_a_claim_uses_redis_attached_after_construction(redis: RedisCache, session_id: str) -> None:
    """The proxy attaches Redis after the Router and this callback are built, so a claim resolves Redis per call."""
    pins: Final = RouterSessionPinStore(connect(None).router_session_pins())
    callback: Final = _repin_check(pins)
    try:
        redis.redis_client.set(_spelled(session_id), json.dumps({"model_id": "other-pod-winner"}), ex=60)
        pins.attach_redis(connect(redis))

        claimed: Final = await callback._claim_pin(_session_key(session_id), {"model_id": "ours"}, 777)

        assert claimed == "other-pod-winner"
        assert pins.get_local(_session_key(session_id)) == {"model_id": "other-pod-winner"}
    finally:
        redis.redis_client.delete(_spelled(session_id))


@pytest.mark.asyncio
async def test_a_claim_falls_back_to_pod_local_when_redis_fails(redis: RedisCache, session_id: str) -> None:
    """A Redis outage costs cross-pod agreement, never same-pod stickiness."""
    pins: Final = _pins(redis)
    callback: Final = _repin_check(pins)
    try:
        _break(redis, _spelled(session_id))

        assert await callback._claim_pin(_session_key(session_id), {"model_id": "ours"}, 777) == "ours"
        assert pins.get_local(_session_key(session_id)) == {"model_id": "ours"}
        assert await callback._claim_pin(_session_key(session_id), {"model_id": "another"}, 777) == "ours"
    finally:
        redis.redis_client.delete(_spelled(session_id))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("held", "expected", "stored"),
    [
        ('{"model_id": "winner"}', "winner", {"model_id": "winner"}),
        ('"winner"', "winner", "winner"),
        ("winner", "winner", "winner"),
        ('{"model": "winner"}', None, {"model": "winner"}),
        (None, "candidate", {"model_id": "candidate"}),
    ],
)
async def test_a_deployment_claim_reads_the_winner_in_any_shape_main_wrote(
    redis: RedisCache, session_id: str, held: str | None, expected: str | None, stored: object
) -> None:
    pins: Final = _pins(redis)
    try:
        if held is not None:
            redis.redis_client.set(_spelled(session_id), held, ex=60)

        winner: Final = await _repin_check(pins)._claim_pin(_session_key(session_id), {"model_id": "candidate"}, 30)

        assert winner == expected
        assert pins.get_local(_session_key(session_id)) == stored
    finally:
        redis.redis_client.delete(_spelled(session_id))


@pytest.mark.asyncio
async def test_concurrent_session_repin_claims_keep_the_shared_winner_after_recovery(
    redis: RedisCache, session_id: str
) -> None:
    workers: Final = (_pins(redis), _pins(redis))
    key: Final = _session_key(session_id)
    choices: Final = ({"model_id": "replacement"}, {"model_id": "other"})
    try:
        redis.redis_client.set(_spelled(session_id), json.dumps({"model_id": "original"}), ex=60)
        for worker in workers:
            worker.set_local(key, {"model_id": "original"}, 60)
        winners: Final = await asyncio.gather(
            *(
                claim_affinity_pin(worker, key, choice, 60, eligible_values=choices, preserve_legacy_model_ids=True)
                for worker, choice in zip(workers, choices, strict=True)
            )
        )
        assert winners[0] == winners[1] and winners[0] in choices
        late_original: Final = await claim_affinity_pin(
            workers[0],
            key,
            {"model_id": "original"},
            60,
            eligible_values=({"model_id": "original"}, *choices),
            preserve_legacy_model_ids=True,
        )
        assert late_original == winners[0]
        assert json.loads(redis.redis_client.get(_spelled(session_id))) == winners[0]
    finally:
        redis.redis_client.delete(_spelled(session_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", ["original", '"original"', '{"model_id":"original"}'])
async def test_session_repin_lua_preserves_healthy_legacy_pins_and_replaces_unavailable_ones(
    redis: RedisCache, session_id: str, stored: str
) -> None:
    pins: Final = _pins(redis)
    key: Final = _session_key(session_id)
    replacement: Final = {"model_id": "replacement"}
    try:
        redis.redis_client.set(_spelled(session_id), stored, ex=60)
        preserved: Final = await claim_affinity_pin(
            pins,
            key,
            replacement,
            60,
            eligible_values=({"model_id": "original"}, replacement),
            preserve_legacy_model_ids=True,
        )
        assert preserved in ("original", {"model_id": "original"})
        moved: Final = await claim_affinity_pin(
            pins, key, replacement, 60, eligible_values=(replacement,), preserve_legacy_model_ids=True
        )
        assert moved == replacement
        assert json.loads(redis.redis_client.get(_spelled(session_id))) == replacement
    finally:
        redis.redis_client.delete(_spelled(session_id))


def _tier_key(session_id: str) -> TierSessionPin:
    return TierSessionPin(router="smart", caller=None, session_id=session_id)


def _tier_spelled(session_id: str) -> str:
    return f"complexity_router_session_affinity:v1:smart:unscoped:{session_id}"


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", ['{"model":"current","tier":"REASONING"}', '"current"', "current"])
async def test_tier_pin_reads_shared_state_refreshes_ttl_and_forgets_deleted_pins(
    redis: RedisCache, session_id: str, stored: str
) -> None:
    workers: Final = (_pins(redis), _pins(redis))
    key: Final = _tier_key(session_id)
    spelled: Final = _tier_spelled(session_id)
    try:
        for worker in workers:
            worker.set_local(key, {"model": "old", "tier": "SIMPLE"}, 60)
        redis.redis_client.set(spelled, stored, ex=10)
        first: Final = await read_affinity_pin(workers[0], key, 60)
        second: Final = await read_affinity_pin(workers[1], key, 60)
        assert first == second and first in ("current", {"model": "current", "tier": "REASONING"})
        assert all(worker.get_local(key) == first for worker in workers)
        assert 10 < redis.redis_client.ttl(spelled) <= 60
        replacement: Final = {"model": "newer", "tier": "SIMPLE"}
        redis.redis_client.set(spelled, json.dumps(replacement), ex=10)
        assert await read_affinity_pin(workers[0], key, 60) == replacement
        assert await read_affinity_pin(workers[1], key, 60) == replacement
        redis.redis_client.delete(spelled)
        for worker in workers:
            assert await read_affinity_pin(worker, key, 60) is None
            assert worker.get_local(key) is None
        assert redis.redis_client.exists(spelled) == 0
    finally:
        redis.redis_client.delete(spelled)


@pytest.mark.asyncio
@pytest.mark.parametrize("script_error", [False, True])
async def test_tier_pin_read_inside_a_request_follows_redis_and_keeps_memory_when_the_script_fails(
    redis: RedisCache, session_id: str, script_error: bool
) -> None:
    pins: Final = _pins(redis)
    key: Final = _tier_key(session_id)
    spelled: Final = _tier_spelled(session_id)
    stale: Final = {"model": "old", "tier": "SIMPLE"}
    current: Final = {"model": "current", "tier": "REASONING"}
    newer: Final = {"model": "newer", "tier": "COMPLEX"}
    try:
        redis.redis_client.set(spelled, json.dumps(current), ex=10)
        assert await read_affinity_pin(pins, key, 60) == current
        pins.set_local(key, stale, 60)
        if script_error:
            _break(redis, spelled)
        with request_redis_batch_scope():
            assert await read_affinity_pin(pins, key, 60) == (stale if script_error else current)
            if script_error:
                assert redis.redis_client.type(spelled) == b"list"
                redis.redis_client.delete(spelled)
            else:
                assert 10 < redis.redis_client.ttl(spelled) <= 60
            redis.redis_client.set(spelled, json.dumps(newer), ex=10)
            assert await read_affinity_pin(pins, key, 60) == newer
            assert pins.get_local(key) == newer
            assert 10 < redis.redis_client.ttl(spelled) <= 60
            redis.redis_client.delete(spelled)
            assert await read_affinity_pin(pins, key, 60) is None
            assert pins.get_local(key) is None
    finally:
        redis.redis_client.delete(spelled)
