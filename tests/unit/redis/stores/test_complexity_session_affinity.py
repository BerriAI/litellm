import json
from collections.abc import Iterator, Mapping
from typing import Final
from unittest.mock import patch

import fakeredis
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm import Router
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.types.router import PreRoutingHookResponse
from tests._support.fake_redis import (
    fake_redis_cache,
    keys_matching,
    legacy_get,
    local_get,
    local_set,
    seed_async_client,
)


async def _store_pin(
    strategy: ComplexityRouter, session_id: str, request_kwargs: Mapping[str, object], model: str, tier: str | None
) -> None:
    await strategy._store_session_affinity_pin(session_id, request_kwargs, model, tier)  # pyright: ignore[reportPrivateUsage]  # exercises the pin path without a full router request


async def _read_pin(strategy: ComplexityRouter, session_id: str, request_kwargs: Mapping[str, object]) -> object:
    return await strategy._read_session_affinity_pin(session_id, request_kwargs)  # pyright: ignore[reportPrivateUsage]  # exercises the pin path without a full router request


async def _pre_routing(
    strategy: ComplexityRouter, *, model: str, request_kwargs: dict[str, object], messages: list[dict[str, object]]
) -> PreRoutingHookResponse | None:
    response: PreRoutingHookResponse | None = await strategy.async_pre_routing_hook(  # pyright: ignore[reportUnknownMemberType]  # legacy hook API is untyped
        model=model, request_kwargs=request_kwargs, messages=messages
    )
    return response


@pytest.fixture(autouse=True)
def fixed_clock() -> Iterator[None]:
    with patch("time.time", return_value=1700000000.0):
        yield


def _strategy(server: fakeredis.FakeServer | None = None) -> ComplexityRouter:
    parent: Final = Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/test"}, "model_info": {"id": name}}
            for name in ("tiny", "middle", "large", "top")
        ]
    )
    if server is not None:
        parent.cache.redis_cache = fake_redis_cache(server, namespace="complexity")
    return ComplexityRouter(
        model_name="router",
        litellm_router_instance=parent,
        derive_savings_baseline=False,
        complexity_router_config={
            "tiers": {"SIMPLE": "tiny", "MEDIUM": "middle", "COMPLEX": "large", "REASONING": "top"},
            "session_affinity": True,
            "session_affinity_ttl_seconds": 60,
        },
    )


def _key(caller: str = "caller") -> str:
    return f"complexity:complexity_router_session_affinity:v1:router:{caller}:same"


def _kwargs(caller: str = "caller") -> dict[str, object]:
    return {"metadata": {"session_id": "same", "user_api_key_hash": caller}}


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", [None, "自定义"])
async def test_overwrite_preserves_wire_bytes_scope_and_ttl(tier: str | None) -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await _store_pin(strategy, "same", _kwargs(), "模型", tier)
    expected: Final = json.dumps({"model": "模型", "tier": tier}).encode()
    assert await raw.get(_key()) == expected
    assert await raw.pttl(_key()) == 60000
    redis_cache: Final = strategy.litellm_router_instance.cache.redis_cache
    assert redis_cache is not None
    assert await legacy_get(redis_cache, _key().removeprefix("complexity:")) == {"model": "模型", "tier": tier}
    await _store_pin(strategy, "same", _kwargs(), "replacement", None)
    await _store_pin(strategy, "same", _kwargs("second"), "other", None)
    assert await raw.get(_key()) == b'{"model": "replacement", "tier": null}'
    assert await raw.get(_key("second")) == b'{"model": "other", "tier": null}'
    assert await keys_matching(raw, "complexity:complexity:*") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [0, -1, 0.5, 1.9])
async def test_overwrite_preserves_legacy_integer_seconds_and_invalid_expiry(ttl: float) -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    strategy.config = strategy.config.model_copy(update={"session_affinity_ttl_seconds": ttl})
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(_key(), b'"old"', ex=20)
    await _store_pin(strategy, "same", _kwargs(), "tiny", None)
    assert await raw.get(_key()) == (b'{"model": "tiny", "tier": null}' if int(ttl) > 0 else b'"old"')
    assert await raw.pttl(_key()) == (int(ttl) * 1000 if int(ttl) > 0 else 20000)


@pytest.mark.asyncio
async def test_nonpositive_read_refresh_returns_old_pin_but_deletes_shared_key() -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    strategy.config = strategy.config.model_copy(update={"session_affinity_ttl_seconds": 0})
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(_key(), b'"tiny"', ex=20)
    assert await _read_pin(strategy, "same", _kwargs()) == "tiny"
    with patch("time.time", return_value=1700000000.001):
        assert await raw.get(_key()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire,expected",
    [
        (b'"tiny"', "tiny"),
        (b"tiny", "tiny"),
        (b'{"model": "tiny", "tier": null}', {"model": "tiny", "tier": None}),
        (b"malformed{", "malformed{"),
    ],
)
async def test_read_refreshes_without_reencoding_legacy_values(wire: bytes, expected: object) -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(_key(), wire, ex=20)
    assert await _read_pin(strategy, "same", _kwargs()) == expected
    assert await raw.get(_key()) == wire
    assert await raw.pttl(_key()) == 60000


@pytest.mark.asyncio
async def test_successful_miss_invalidates_but_outage_refreshes_local_fallback() -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    cache: Final = strategy.litellm_router_instance.cache
    logical: Final = _key().removeprefix("complexity:")
    local_set(cache, logical, "stale", ttl=20)
    assert await _read_pin(strategy, "same", _kwargs()) is None
    assert local_get(cache, logical) is None
    local_set(cache, logical, "local", ttl=20)
    server.connected = False
    assert await _read_pin(strategy, "same", _kwargs()) == "local"
    with patch("time.time", return_value=1700000021.0):
        assert local_get(cache, logical) == "local"


@pytest.mark.asyncio
async def test_attached_redis_supersedes_local_only_session() -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy()
    await _store_pin(strategy, "same", _kwargs(), "local", None)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(_key(), b'"shared"', ex=20)
    strategy.litellm_router_instance.cache.redis_cache = fake_redis_cache(server, namespace="complexity")
    assert await _read_pin(strategy, "same", _kwargs()) == "shared"


class LostWriteReply(fakeredis.FakeAsyncRedis):
    async def set(self, *args: object, **kwargs: object) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # passes the write through fakeredis
        await super().set(*args, **kwargs)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]  # fakeredis is untyped
        raise RedisTimeoutError("reply lost after overwrite")


@pytest.mark.asyncio
async def test_unknown_overwrite_keeps_the_value_already_written_locally() -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    redis_cache: Final = strategy.litellm_router_instance.cache.redis_cache
    assert redis_cache is not None
    seed_async_client(redis_cache, LostWriteReply(server=server))
    await _store_pin(strategy, "same", _kwargs(), "tiny", None)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    assert await raw.get(_key()) == b'{"model": "tiny", "tier": null}'
    assert local_get(strategy.litellm_router_instance.cache, _key().removeprefix("complexity:")) == {
        "model": "tiny",
        "tier": None,
    }


@pytest.mark.asyncio
async def test_classification_and_pin_escalation_both_overwrite_shared_session() -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    first: Final = await _pre_routing(
        strategy, model="router", request_kwargs=_kwargs(), messages=[{"role": "user", "content": "Hi"}]
    )
    assert first is not None and first.model == "tiny"
    assert await raw.get(_key()) == b'{"model": "tiny", "tier": "SIMPLE"}'
    second: Final = await _pre_routing(
        strategy, model="router", request_kwargs=_kwargs(), messages=[{"role": "user", "content": "LITELLM ESCALATE"}]
    )
    assert second is not None and second.model == "middle"
    assert await raw.get(_key()) == b'{"model": "middle", "tier": "MEDIUM"}'


@pytest.mark.asyncio
async def test_reads_join_only_pending_request_work_not_post_call_writes() -> None:
    server: Final = fakeredis.FakeServer()
    strategy: Final = _strategy(server)
    redis_cache: Final = strategy.litellm_router_instance.cache.redis_cache
    assert redis_cache is not None
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(_key(), b'"tiny"', ex=20)
    with request_redis_batch_scope(post_call_deadline=3600) as batches:
        request: Final = batches.batch(redis_cache)
        post_call: Final = batches.post_call(redis_cache)
        post_call.set("unrelated", "later", 60)
        assert await _read_pin(strategy, "same", _kwargs()) == "tiny"
        assert request.flushes == 0
        request.set("before-read", "ready", 60)
        assert await _read_pin(strategy, "same", _kwargs()) == "tiny"
        assert request.flushes == 1
        assert await raw.get("complexity:before-read") == b'"ready"'
        assert await raw.get("complexity:unrelated") is None
        assert post_call.pending == 1
        await batches.flush_post_call()
    assert await raw.pttl(_key()) == 60000
