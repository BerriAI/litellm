import json
from typing import Final, NoReturn
from unittest.mock import patch

import fakeredis
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_endpoints.ui_sso import delete_login_code, get_login_code, put_login_code
from litellm.redis.stores.login_codes import TransportLoginCodeStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache


@pytest.mark.asyncio
async def test_login_code_request_write_and_tombstone_share_existing_batch(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="sso")
    local: Final = UserApiKeyCache(redis_cache=cache)
    with request_redis_batch_scope():
        await put_login_code("new", {"token": "t"}, None, local)
        assert await get_login_code("new", None, local) == {"token": "t"}
        assert await raw.get("sso:login_code:new") is None
        batch: Final = cache.transport().request_batch()
        assert batch is not None
        await batch.flush()
        assert await raw.get("sso:login_code:new") == b'{"token": "t"}'
        pending: Final = await local.async_delete_cache_pre_call("login_code:new")
        assert pending is not None
        assert await get_login_code("new", None, local) is None
        assert await raw.get("sso:login_code:new") == b'{"token": "t"}'
        await pending
        assert await raw.get("sso:login_code:new") is None


@pytest.mark.asyncio
async def test_login_code_rolling_pods_preserve_wire_and_non_atomic_consume(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="sso")
    local: Final = UserApiKeyCache()
    payload: Final = {"token": "token-é", "redirect_url": "https://example.test/登录?a=1"}
    with patch("time.time", return_value=1700000000):
        await cache.async_set_cache("login_code:old", payload, ttl=60)
        assert await get_login_code("old", cache, local) == payload
        assert await get_login_code("old", cache, local) == payload
        await delete_login_code("old", cache, local)
        assert await get_login_code("old", cache, local) is None
        await put_login_code("new", payload, cache, local)
        assert await cache.async_get_cache("login_code:new") == payload
        assert await raw.get("sso:login_code:new") == json.dumps(payload).encode()
        assert await raw.pttl("sso:login_code:new") == 60000
        assert await raw.keys("sso:sso:*") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("attached", [False, True])
async def test_login_code_fallback_keeps_local_first_and_deletes_both_tiers(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, attached: bool
) -> None:
    cache: Final = fake_redis_cache(server, namespace="fallback")
    local: Final = UserApiKeyCache(redis_cache=cache if attached else None)
    payload: Final = {"token": "local", "redirect_url": "/"}
    await put_login_code("code", payload, None, local)
    assert await get_login_code("code", None, local) == payload
    await delete_login_code("code", None, local)
    assert await get_login_code("code", None, local) is None
    assert await raw.get("fallback:login_code:code") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [b"broken", b"[]", b"null", b'"string"'])
async def test_login_code_malformed_rows_remain_invalid_without_deletion(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, wire: bytes
) -> None:
    cache: Final = fake_redis_cache(server)
    await raw.set("login_code:bad", wire)
    assert await get_login_code("bad", cache, UserApiKeyCache()) is None
    assert await raw.get("login_code:bad") == wire


@pytest.mark.asyncio
async def test_login_code_outage_keeps_direct_read_miss_and_local_fallback(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    local: Final = UserApiKeyCache(redis_cache=cache)
    server.connected = False
    await put_login_code("fallback", {"token": "t"}, None, local)
    assert await get_login_code("fallback", None, local) == {"token": "t"}
    assert await get_login_code("missing", cache, local) is None
    with pytest.raises(RuntimeError):
        await delete_login_code("missing", cache, local)
    for _ in range(10):
        await TransportLoginCodeStore(cache.transport()).get("missing")
    with pytest.raises(RedisCircuitBreakerOpenError):
        await get_login_code("missing", cache, local)


def _lost_reply(reply: object, **options: object) -> NoReturn:
    raise RedisTimeoutError("reply lost after apply")


@pytest.mark.asyncio
async def test_login_code_ambiguous_write_is_not_retried_or_copied_to_local(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server)
    local: Final = UserApiKeyCache()
    client: Final = cache.init_async_client()
    with patch.dict(client.response_callbacks, {"SET": _lost_reply}):
        await put_login_code("ambiguous", {"token": "t"}, cache, local)
    assert await raw.get("login_code:ambiguous") == b'{"token": "t"}'
    assert await get_login_code("ambiguous", None, local) is None
    server.connected = False
    failure: Final = await TransportLoginCodeStore(cache.transport()).get("ambiguous")
    assert isinstance(failure, RedisFailure)
    assert failure.kind == "transport"
