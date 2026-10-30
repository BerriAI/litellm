import asyncio
import hashlib
import json
from collections.abc import Iterator
from typing import Final
from unittest.mock import patch

import fakeredis
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm import Router
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.router_strategy.complexity_router.complexity_router import (
    ComplexityRouter,
    _SessionAffinityPin,  # pyright: ignore[reportPrivateUsage]  # builds the retained pin the strategy carries between tiers
)
from tests._support.fake_redis import (
    fake_redis_cache,
    keys_matching,
    local_get,
    local_set,
    seed_async_client,
)


async def _pin(
    strategy: ComplexityRouter,
    tier: str,
    model: str,
    candidates: tuple[str, ...],
    request_kwargs: dict[str, object],
    retained_pin: _SessionAffinityPin | None = None,
) -> object:
    return await strategy._pin_model_for_tier(tier, model, candidates, request_kwargs, retained_pin)  # pyright: ignore[reportPrivateUsage]  # exercises the pin path without a full router request


@pytest.fixture(autouse=True)
def fixed_clock() -> Iterator[None]:
    with patch("time.time", return_value=1700000000.0):
        yield


def _strategy(server: fakeredis.FakeServer | None, *, name: str = "router", simple: str = "tiny") -> ComplexityRouter:
    parent: Final = Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/test"}, "model_info": {"id": name}}
            for name in ("tiny", "middle", "large", "top")
        ]
    )
    if server is not None:
        parent.cache.redis_cache = fake_redis_cache(server, namespace="tier")
    return ComplexityRouter(
        model_name=name,
        litellm_router_instance=parent,
        derive_savings_baseline=False,
        complexity_router_config={
            "tiers": {"SIMPLE": [simple, "middle"], "MEDIUM": "middle", "COMPLEX": "large", "REASONING": "top"},
            "session_affinity": True,
            "session_affinity_ttl_seconds": 60,
        },
    )


def _key(strategy: ComplexityRouter, caller: str | None = "caller", *, fallback: str | None = None) -> str:
    config: Final = strategy.config.model_dump_json(include={"tiers": True, "tier_model_configs": True})
    identity: Final = (strategy.model_name, hashlib.sha256(config.encode()).hexdigest(), caller, "same", "SIMPLE")
    encoded: Final = json.dumps((*identity, ("replay_fallback", fallback)) if fallback is not None else identity)
    return "tier:complexity_router_tier_model_affinity:v1:" + hashlib.sha256(encoded.encode()).hexdigest()


def _kwargs(caller: str | None = "caller") -> dict[str, object]:
    return {"metadata": {"session_id": "same", "user_api_key_hash": caller}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,caller,simple,fallback",
    [
        ("router", "caller", "tiny", None),
        ("路由", "caller", "tiny", None),
        ("router", "另一个", "tiny", None),
        ("router", None, "tiny", None),
        ("router", "caller", "模型", None),
        ("router", "caller", "tiny", "removed"),
    ],
)
async def test_identity_and_unicode_wire_match_legacy_hash(
    name: str, caller: str | None, simple: str, fallback: str | None
) -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server, name=name, simple=simple)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    retained: Final = _SessionAffinityPin(fallback, "SIMPLE") if fallback is not None else None
    assert await _pin(strategy, "SIMPLE", "模型", ("模型", "tiny"), _kwargs(caller), retained) == "模型"
    assert await raw.get(_key(strategy, caller, fallback=fallback)) == b'{"model": "\\u6a21\\u578b"}'
    assert await raw.pttl(_key(strategy, caller, fallback=fallback)) == 60000
    assert await keys_matching(raw, "tier:tier:*") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire,expected",
    [
        (b'{"model": "tiny"}', "tiny"),
        (b'{"model": "removed"}', "middle"),
        (b'{"model": "tiny", "extra": true}', "middle"),
        (b'"tiny"', "middle"),
        (b"tiny", "middle"),
        (b"{broken", "middle"),
    ],
)
async def test_claim_refreshes_only_complete_eligible_winners(wire: bytes, expected: str) -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(_key(strategy), wire, ex=20)
    assert await _pin(strategy, "SIMPLE", "middle", ("tiny", "middle"), _kwargs()) == expected
    assert await raw.get(_key(strategy)) == json.dumps({"model": expected}).encode()
    assert await raw.pttl(_key(strategy)) == 60000


@pytest.mark.asyncio
async def test_concurrent_pods_converge_then_replace_expired_pin() -> None:
    server: Final = fakeredis.FakeServer()
    left: Final = _strategy(server)
    right: Final = _strategy(server)
    winners: Final = await asyncio.gather(
        _pin(left, "SIMPLE", "tiny", ("tiny", "middle"), _kwargs()),
        _pin(right, "SIMPLE", "middle", ("tiny", "middle"), _kwargs()),
    )
    assert winners[0] == winners[1]
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    assert await raw.get(_key(left)) == json.dumps({"model": winners[0]}).encode()
    with patch("time.time", return_value=1700000061.0):
        assert await _pin(left, "SIMPLE", "middle", ("tiny", "middle"), _kwargs()) == "middle"
        assert await raw.get(_key(left)) == b'{"model": "middle"}'
        assert await raw.pttl(_key(left)) == 60000


@pytest.mark.asyncio
async def test_no_session_and_eligible_retained_pin_do_not_claim() -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    assert await _pin(strategy, "SIMPLE", "tiny", ("tiny", "middle"), {}) == "tiny"
    assert (
        await _pin(strategy, "SIMPLE", "middle", ("tiny", "middle"), _kwargs(), _SessionAffinityPin("tiny", "SIMPLE"))
        == "tiny"
    )
    assert await fakeredis.FakeAsyncRedis(server=server).keys("tier:*") == []


@pytest.mark.asyncio
async def test_late_redis_attachment_replaces_local_winner() -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(None)
    assert await _pin(strategy, "SIMPLE", "tiny", ("tiny", "middle"), _kwargs()) == "tiny"
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(_key(strategy), b'{"model": "middle"}', ex=20)
    strategy.litellm_router_instance.cache.redis_cache = fake_redis_cache(server, namespace="tier")
    assert await _pin(strategy, "SIMPLE", "tiny", ("tiny", "middle"), _kwargs()) == "middle"
    assert local_get(strategy.litellm_router_instance.cache, _key(strategy).removeprefix("tier:")) == {
        "model": "middle"
    }


class LostClaimReply(fakeredis.FakeAsyncRedis):
    async def eval(self, script: str, numkeys: int, *keys_and_args: str | bytes | int) -> object:
        await super().eval(script, numkeys, *keys_and_args)  # pyright: ignore[reportUnknownMemberType]  # fakeredis is untyped
        raise RedisTimeoutError("reply lost after claim")


class ConsumedClaimReply(fakeredis.FakeAsyncRedis):
    async def eval(self, script: str, numkeys: int, *keys_and_args: str | bytes | int) -> object:
        await super().eval(script, numkeys, *keys_and_args)  # pyright: ignore[reportUnknownMemberType]  # fakeredis is untyped
        return 1


@pytest.mark.asyncio
async def test_consumed_claim_returns_candidate_without_installing_local_winner() -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    cache: Final = strategy.litellm_router_instance.cache
    redis_cache: Final = cache.redis_cache
    assert redis_cache is not None
    seed_async_client(redis_cache, ConsumedClaimReply(server=server))
    local_set(cache, _key(strategy).removeprefix("tier:"), {"model": "tiny"}, ttl=20)
    assert await _pin(strategy, "SIMPLE", "middle", ("tiny", "middle"), _kwargs()) == "middle"
    assert await fakeredis.FakeAsyncRedis(server=server).get(_key(strategy)) == b'{"model": "middle"}'
    assert local_get(cache, _key(strategy).removeprefix("tier:")) == {"model": "tiny"}


@pytest.mark.asyncio
async def test_claim_leaves_pending_request_and_post_call_work_unflushed() -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    redis_cache: Final = strategy.litellm_router_instance.cache.redis_cache
    assert redis_cache is not None
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    with request_redis_batch_scope(post_call_deadline=3600) as batches:
        request: Final = batches.batch(redis_cache)
        post_call: Final = batches.post_call(redis_cache)
        request.set("pending-request", "ready", 60)
        post_call.set("pending-post-call", "later", 60)
        assert await _pin(strategy, "SIMPLE", "tiny", ("tiny", "middle"), _kwargs()) == "tiny"
        assert await raw.get(_key(strategy)) == b'{"model": "tiny"}'
        assert await raw.get("tier:pending-request") is None
        assert await raw.get("tier:pending-post-call") is None
        await batches.flush_post_call()
        await request.flush()
    assert await raw.get("tier:pending-request") == b'"ready"'
    assert await raw.get("tier:pending-post-call") == b'"later"'


@pytest.mark.asyncio
@pytest.mark.parametrize("applied", [False, True])
async def test_outage_and_unknown_claim_keep_local_winner(applied: bool) -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    cache: Final = strategy.litellm_router_instance.cache
    redis_cache: Final = cache.redis_cache
    assert redis_cache is not None
    local_set(cache, _key(strategy).removeprefix("tier:"), {"model": "tiny"}, ttl=20)
    if applied:
        seed_async_client(redis_cache, LostClaimReply(server=server))
    else:
        server.connected = False
    assert await _pin(strategy, "SIMPLE", "middle", ("tiny", "middle"), _kwargs()) == "tiny"
    server.connected = True
    assert await fakeredis.FakeAsyncRedis(server=server).get(_key(strategy)) == (
        b'{"model": "middle"}' if applied else None
    )
    with patch("time.time", return_value=1700000021.0):
        assert local_get(cache, _key(strategy).removeprefix("tier:")) == {"model": "tiny"}
