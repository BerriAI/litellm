import json
from typing import Final, NoReturn
from unittest.mock import patch

import fakeredis
import pytest
from fastapi import Request
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.proxy import proxy_server
from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler
from tests._support.fake_redis import fake_redis_cache


def _request(state: str) -> Request:
    return Request({"type": "http", "query_string": f"state={state}".encode(), "headers": []})


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [True, False])
async def test_pkce_old_and_new_pods_read_exact_wire_and_keep_verifier_until_cleanup(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch, direct: bool
) -> None:
    cache: Final = fake_redis_cache(server, namespace="sso")
    monkeypatch.setattr(proxy_server, "redis_usage_cache", cache if direct else None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache(redis_cache=cache))
    monkeypatch.setenv("GENERIC_CLIENT_USE_PKCE", "true")
    with patch("time.time", return_value=1700000000):
        await cache.async_set_cache("pkce_verifier:old", {"code_verifier": "é-verifier"}, ttl=600)
        params: Final = await SSOAuthenticationHandler.prepare_token_exchange_parameters(_request("old"), True)
        assert params == {
            "include_client_id": True,
            "code_verifier": "é-verifier",
            "_pkce_cache_key": "pkce_verifier:old",
        }
        assert await SSOAuthenticationHandler.prepare_token_exchange_parameters(_request("old"), True) == params
        await SSOAuthenticationHandler.delete_pkce_verifier(params["_pkce_cache_key"])
        assert await raw.get("sso:pkce_verifier:old") is None
        await SSOAuthenticationHandler.put_pkce_verifier("new", "é-verifier")
        assert await cache.async_get_cache("pkce_verifier:new") == {"code_verifier": "é-verifier"}
        assert await raw.get("sso:pkce_verifier:new") == json.dumps({"code_verifier": "é-verifier"}).encode()
        assert await raw.pttl("sso:pkce_verifier:new") == 600000


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [b'"legacy-verifier"', b"'legacy-verifier'"])
async def test_pkce_legacy_string_is_still_a_verifier(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch, wire: bytes
) -> None:
    monkeypatch.setattr(proxy_server, "redis_usage_cache", fake_redis_cache(server))
    monkeypatch.setenv("GENERIC_CLIENT_USE_PKCE", "true")
    await raw.set("pkce_verifier:legacy", wire)
    assert await SSOAuthenticationHandler.prepare_token_exchange_parameters(_request("legacy"), False) == {
        "include_client_id": False,
        "code_verifier": "legacy-verifier",
        "_pkce_cache_key": "pkce_verifier:legacy",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("value", [{"code_verifier": ""}, {"code_verifier": None}, {"unexpected": "shape"}, []])
async def test_pkce_invalid_decoded_values_keep_strict_policy_and_cleanup(
    server: fakeredis.FakeServer,
    raw: fakeredis.FakeAsyncRedis,
    monkeypatch: pytest.MonkeyPatch,
    strict: bool,
    value: object,
) -> None:
    cache: Final = fake_redis_cache(server)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", cache)
    monkeypatch.setenv("GENERIC_CLIENT_USE_PKCE", "true")
    monkeypatch.setenv("PKCE_STRICT_CACHE_MISS", str(strict).lower())
    await cache.async_set_cache("pkce_verifier:bad", value, ttl=600)
    if strict:
        with pytest.raises(ProxyException) as exc:
            await SSOAuthenticationHandler.prepare_token_exchange_parameters(_request("bad"), True)
        assert str(exc.value.code) == "401"
        assert exc.value.param == "PKCE_CACHE_MISS"
    else:
        assert await SSOAuthenticationHandler.prepare_token_exchange_parameters(_request("bad"), True) == {
            "include_client_id": True
        }
    assert await raw.get("pkce_verifier:bad") is None


@pytest.mark.asyncio
async def test_pkce_request_batch_tombstone_and_outage_fallback(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache: Final = fake_redis_cache(server)
    local: Final = UserApiKeyCache(redis_cache=cache)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", local)
    with request_redis_batch_scope():
        await SSOAuthenticationHandler.put_pkce_verifier("state", "v")
        assert await SSOAuthenticationHandler.get_pkce_verifier("state") == {"code_verifier": "v"}
        assert await raw.get("pkce_verifier:state") is None
        batch: Final = cache.transport().request_batch()
        assert batch is not None
        await batch.flush()
        pending: Final = await local.async_delete_cache_pre_call("pkce_verifier:state")
        assert pending is not None
        assert await SSOAuthenticationHandler.get_pkce_verifier("state") is None
        await pending
    server.connected = False
    await SSOAuthenticationHandler.put_pkce_verifier("offline", "local")
    assert await SSOAuthenticationHandler.get_pkce_verifier("offline") == {"code_verifier": "local"}
    await SSOAuthenticationHandler.delete_pkce_verifier("pkce_verifier:offline")
    assert await SSOAuthenticationHandler.get_pkce_verifier("offline") is None


def _lost_reply(reply: object, **options: object) -> NoReturn:
    raise RedisTimeoutError("reply lost after apply")


@pytest.mark.asyncio
async def test_pkce_unknown_applied_write_and_malformed_read_do_not_change_cleanup_policy(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache: Final = fake_redis_cache(server)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", cache)
    monkeypatch.setenv("GENERIC_CLIENT_USE_PKCE", "true")
    monkeypatch.setenv("PKCE_STRICT_CACHE_MISS", "false")
    with patch.dict(cache.init_async_client().response_callbacks, {"SET": _lost_reply}):
        await SSOAuthenticationHandler.put_pkce_verifier("ambiguous", "v")
    assert await raw.get("pkce_verifier:ambiguous") == b'{"code_verifier": "v"}'
    await raw.set("pkce_verifier:bad", b"not-json")
    assert await SSOAuthenticationHandler.prepare_token_exchange_parameters(_request("bad"), True) == {
        "include_client_id": True
    }
    assert await raw.get("pkce_verifier:bad") == b"not-json"
