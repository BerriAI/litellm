import json
from collections.abc import Iterator, Mapping
from datetime import timedelta
from typing import Final, Protocol, cast
from unittest.mock import patch

import fakeredis
import pytest
from pydantic import TypeAdapter
from redis.exceptions import ConnectionError as RedisConnectionError

from litellm import Router
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.claude_code_router_binding import SessionRouterScope, TransportClaudeCodeRouterBindingStore
from litellm.redis.transport.contract import Arg, RedisFailure
from litellm.types.router import PreRoutingHookResponse
from tests._support.fake_redis import fake_redis_cache

KEY: Final = "claude_code_session_router:v1:key:scope:session-1234"
MODEL: Final = 'smart-雪 "router"'
SCOPE: Final = SessionRouterScope("key:scope", "session-1234")
_METADATA: Final = TypeAdapter(dict[str, object])


class BindingRouter(Protocol):
    def update_redis_cache(self, cache: RedisCache) -> None: ...

    async def async_pre_routing_hook(
        self, model: str, request_kwargs: Mapping[str, object]
    ) -> PreRoutingHookResponse | None: ...


class LegacyBindingCache(Protocol):
    async def async_get_cache(self, key: str) -> object: ...

    async def async_set_cache(self, key: str, value: str, *, ttl: int) -> None: ...


@pytest.fixture(autouse=True)
def clock() -> Iterator[None]:
    with patch("time.time", return_value=1800000000.0):
        yield


def _router(cache: RedisCache | None) -> BindingRouter:
    router: Final = Router(
        model_list=[
            {"model_name": "plain", "litellm_params": {"model": "openai/gpt-4.1-mini", "api_key": "fake"}},
            *(
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": "auto_router/complexity_router",
                        "complexity_router_config": {
                            "tiers": dict.fromkeys(("SIMPLE", "MEDIUM", "COMPLEX", "REASONING"), "plain")
                        },
                    },
                }
                for model in (MODEL, "premium-router")
            ),
        ]
    )
    typed: Final = cast(BindingRouter, router)  # cast-ok: typed view of the actual binding caller, without replacing it
    if cache is not None:
        typed.update_redis_cache(cache)
    return typed


def _kwargs(
    *, agent: bool = False, caller: str = "key:scope", session: str = "session-1234", depth: int = 0
) -> Mapping[str, object]:
    return {
        "metadata": {"user_api_key_hash": caller},
        "proxy_server_request": {
            "headers": {
                "X-Claude-Code-Session-Id": session,
                **({"x-claude-code-agent-id": "agent-1234"} if agent else {}),
            }
        },
        "fallback_depth": depth,
    }


async def _route(router: BindingRouter, model: str, kwargs: Mapping[str, object]) -> str:
    await router.async_pre_routing_hook(model, dict(kwargs))
    bound: Final = _METADATA.validate_python(kwargs["metadata"]).get("model_group")
    return bound if isinstance(bound, str) else model


@pytest.mark.asyncio
async def test_binding_old_new_wire_namespace_and_read_refresh(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server, namespace="tenant")
    legacy: Final = cast(LegacyBindingCache, cache)  # cast-ok: rolling-deploy writer and reader
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await legacy.async_set_cache(KEY, MODEL, ttl=3600)
    router: Final = _router(cache)
    assert await _route(router, "plain", _kwargs(agent=True)) == MODEL
    assert await raw.get("tenant:" + KEY) == json.dumps(MODEL).encode()
    assert await raw.exists(KEY) == 0
    with patch("time.time", return_value=1800000100.0):
        assert await raw.pttl("tenant:" + KEY) == 3500000
        assert await _route(router, "plain", _kwargs(agent=True)) == MODEL
        assert await raw.pttl("tenant:" + KEY) == 3600000
        assert await _route(router, "premium-router", _kwargs()) == "premium-router"
    assert await legacy.async_get_cache(KEY) == "premium-router"
    assert await raw.get("tenant:" + KEY) == b'"premium-router"'


@pytest.mark.asyncio
async def test_shared_miss_and_remote_update_override_a_fresh_local_binding(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    main: Final = _router(cache)
    agent: Final = _router(cache)
    await _route(main, MODEL, _kwargs())
    assert await _route(agent, "plain", _kwargs(agent=True)) == MODEL
    await _route(main, "premium-router", _kwargs())
    assert await _route(agent, "plain", _kwargs(agent=True)) == "premium-router"
    await raw.delete(KEY)
    assert await _route(agent, "plain", _kwargs(agent=True)) == "plain"


@pytest.mark.asyncio
async def test_local_bindings_survive_without_redis_but_are_not_backfilled_on_attach(
    server: fakeredis.FakeServer,
) -> None:
    router: Final = _router(None)
    await _route(router, MODEL, _kwargs())
    assert await _route(router, "plain", _kwargs(agent=True)) == MODEL
    router.update_redis_cache(fake_redis_cache(server))
    assert await _route(router, "plain", _kwargs(agent=True)) == "plain"
    assert await fakeredis.FakeAsyncRedis(server=server).exists(KEY) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_value", [b"null", b"4", b"[]", b'{"router": "other"}', b"not-json", b"\xff"])
async def test_non_string_and_malformed_bindings_are_ignored_not_rewritten(
    server: fakeredis.FakeServer, raw_value: bytes
) -> None:
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(KEY, raw_value, ex=600)
    assert await _route(_router(fake_redis_cache(server)), "plain", _kwargs(agent=True)) == "plain"
    assert await raw.get(KEY) == raw_value
    assert await raw.pttl(KEY) == 600000


@pytest.mark.asyncio
async def test_legacy_python_literal_binding_refreshes_as_json(server: fakeredis.FakeServer) -> None:
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(KEY, b"'premium-router'", ex=600)
    assert await _route(_router(fake_redis_cache(server)), "plain", _kwargs(agent=True)) == "premium-router"
    assert await raw.get(KEY) == b'"premium-router"'
    assert await raw.pttl(KEY) == 3600000


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [b'"removed-router"', b'""'])
async def test_unknown_strategy_deletes_the_binding_instead_of_tombstoning(
    server: fakeredis.FakeServer, value: bytes
) -> None:
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(KEY, value, ex=3600)
    assert await _route(_router(fake_redis_cache(server)), "plain", _kwargs(agent=True)) == "plain"
    assert await raw.exists(KEY) == 0


@pytest.mark.asyncio
async def test_invalid_scope_other_caller_and_fallback_cannot_reuse_the_binding(server: fakeredis.FakeServer) -> None:
    router: Final = _router(fake_redis_cache(server))
    await _route(router, MODEL, _kwargs())
    assert await _route(router, "plain", _kwargs(agent=True, caller="other")) == "plain"
    assert await _route(router, "plain", _kwargs(agent=True, caller="")) == "plain"
    assert await _route(router, "plain", _kwargs(agent=True, session="short")) == "plain"
    assert await _route(router, "plain", _kwargs(agent=True, depth=1)) == "plain"
    assert await fakeredis.FakeAsyncRedis(server=server).get(KEY) == json.dumps(MODEL).encode()


@pytest.mark.asyncio
async def test_binding_outage_never_uses_stale_local_and_shares_the_breaker() -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server)
    router: Final = _router(cache)
    await _route(router, MODEL, _kwargs())
    server.connected = False
    assert await _route(router, "plain", _kwargs(agent=True)) == "plain"
    assert await _route(router, "premium-router", _kwargs()) == "premium-router"
    store: Final = TransportClaudeCodeRouterBindingStore(cache.transport())
    for _ in range(5):
        await store.read_binding(SCOPE)
    result: Final = await store.bind_router(SCOPE, MODEL, ttl=timedelta(seconds=3600))
    assert isinstance(result, RedisFailure) and result.kind == "breaker_open"


class RedisWriter(Protocol):
    async def set(
        self, name: str, value: Arg, *, ex: timedelta | None = None, px: int | None = None, nx: bool = False
    ) -> object: ...


@pytest.mark.asyncio
async def test_unknown_applied_binding_write_remains_routable_without_replay(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    router: Final = _router(cache)
    original_set: Final = cast(type[RedisWriter], fakeredis.FakeAsyncRedis).set  # cast-ok: typed redis-py boundary

    async def apply_then_disconnect(
        client: RedisWriter,
        name: str,
        value: Arg,
        *,
        ex: timedelta | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> object:
        await original_set(client, name, value, ex=ex, px=px, nx=nx)
        raise RedisConnectionError("response lost after SET applied")

    with patch.object(fakeredis.FakeAsyncRedis, "set", new=apply_then_disconnect):
        assert await _route(router, MODEL, _kwargs()) == MODEL
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    assert await raw.get(KEY) == json.dumps(MODEL).encode()
    assert await raw.pttl(KEY) == 3600000
    assert await _route(_router(cache), "plain", _kwargs(agent=True)) == MODEL
