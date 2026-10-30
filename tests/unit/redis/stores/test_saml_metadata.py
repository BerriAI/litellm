import hashlib
import json
from datetime import timedelta
from typing import Final, NoReturn
from unittest.mock import patch

import fakeredis
import pytest
from fastapi import HTTPException
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.proxy.management_endpoints.sso.saml_sso import SAMLAuthHandler
from litellm.redis.stores.saml_metadata import TransportSAMLMetadataStore, saml_metadata_key
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache

_XML: Final = (
    '<EntityDescriptor xmlns="urn:oasis:names:tc:SAML:2.0:metadata" entityID="https://idp.example/é">'
    '<IDPSSODescriptor protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">'
    '<SingleSignOnService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect" '
    'Location="https://idp.example/login"/></IDPSSODescriptor></EntityDescriptor>'
)
_KEY: Final = f"saml_idp_settings:{hashlib.sha256(_XML.encode()).hexdigest()}"


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [0, -1, 0.9, 1.9])
async def test_saml_metadata_seconds_expiry_preserves_legacy_rejection_and_truncation(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, seconds: float
) -> None:
    redis: Final = fake_redis_cache(server)
    store: Final = TransportSAMLMetadataStore(redis.transport())
    with patch("time.time", return_value=1700000000):
        await redis.async_set_cache(saml_metadata_key("old"), {"idp": "wire"}, ttl=seconds)
        result: Final = await store.put("new", {"idp": "wire"}, ttl=timedelta(seconds=seconds))
        assert await raw.get(saml_metadata_key("new")) == await raw.get(saml_metadata_key("old"))
        assert await raw.pttl(saml_metadata_key("new")) == await raw.pttl(saml_metadata_key("old"))
        if seconds < 1:
            assert isinstance(result, RedisFailure)
            assert result.kind == "server"
        else:
            assert result is None


@pytest.mark.asyncio
async def test_saml_metadata_new_fill_is_readable_by_old_pods_with_exact_wire_and_ttl(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis: Final = fake_redis_cache(server, namespace="saml")
    cache: Final = DualCache(redis_cache=redis, default_redis_ttl=600)
    monkeypatch.setenv("SAML_IDP_METADATA_XML", _XML)
    with patch("time.time", return_value=1700000000):
        settings: Final = await SAMLAuthHandler._load_idp_settings(cache)
        assert settings["idp"] == {
            "entityId": "https://idp.example/é",
            "singleSignOnService": {
                "url": "https://idp.example/login",
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
        }
        assert await redis.async_get_cache(_KEY) == settings
        assert await raw.get(f"saml:{_KEY}") == json.dumps(settings).encode()
        assert await raw.pttl(f"saml:{_KEY}") == 3600000
        assert await SAMLAuthHandler._load_idp_settings(DualCache(redis_cache=redis)) == settings


@pytest.mark.asyncio
async def test_saml_metadata_old_fill_and_url_precedence_survive_rolling_pods(
    server: fakeredis.FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis: Final = fake_redis_cache(server, namespace="saml")
    source: Final = "https://idp.example/metadata"
    key: Final = f"saml_idp_settings:{hashlib.sha256(source.encode()).hexdigest()}"
    payload: Final = {"idp": {"entityId": "old-ée", "x509cert": "old-cert"}, "legacy_number": 2}
    monkeypatch.setenv("SAML_IDP_METADATA_URL", source)
    monkeypatch.setenv("SAML_IDP_METADATA_XML", "invalid XML must not be parsed")
    await redis.async_set_cache(key, payload, ttl=3600)
    assert await SAMLAuthHandler._load_idp_settings(DualCache(redis_cache=redis)) == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [b"broken", b"[]", b"null", b'"text"'])
async def test_saml_metadata_malformed_cache_falls_back_to_parser_and_replaces_row(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch, wire: bytes
) -> None:
    redis: Final = fake_redis_cache(server)
    monkeypatch.setenv("SAML_IDP_METADATA_XML", _XML)
    await raw.set(_KEY, wire)
    result: Final = await SAMLAuthHandler._load_idp_settings(DualCache(redis_cache=redis))
    assert await raw.get(_KEY) == json.dumps(result).encode()


@pytest.mark.asyncio
async def test_saml_metadata_outage_keeps_local_fill_even_when_breaker_opens(
    server: fakeredis.FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis: Final = fake_redis_cache(server)
    cache: Final = DualCache(redis_cache=redis)
    monkeypatch.setenv("SAML_IDP_METADATA_XML", _XML)
    server.connected = False
    result: Final = await SAMLAuthHandler._load_idp_settings(cache)
    assert await SAMLAuthHandler._load_idp_settings(cache) == result
    for _ in range(6):
        assert await SAMLAuthHandler._load_idp_settings(DualCache(redis_cache=redis)) == result


@pytest.mark.asyncio
async def test_saml_metadata_request_batch_and_tombstone_do_not_resurrect_old_metadata(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis: Final = fake_redis_cache(server)
    cache: Final = DualCache(redis_cache=redis)
    monkeypatch.setenv("SAML_IDP_METADATA_XML", _XML)
    await redis.async_set_cache(_KEY, {"idp": "stale"}, ttl=3600)
    with request_redis_batch_scope():
        pending: Final = await cache.async_delete_cache_pre_call(_KEY)
        assert pending is not None
        result: Final = await SAMLAuthHandler._load_idp_settings(cache)
        assert result != {"idp": "stale"}
        assert await redis.async_get_cache(_KEY) == result
        batch: Final = redis.transport().request_batch()
        assert batch is not None
        await batch.flush()
        await pending
    assert await raw.get(_KEY) is None
    assert await cache.async_get_cache(_KEY, local_only=True) == result


def _lost_reply(reply: object, **options: object) -> NoReturn:
    raise RedisTimeoutError("reply lost after apply")


@pytest.mark.asyncio
async def test_saml_metadata_unknown_applied_write_keeps_local_result(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis: Final = fake_redis_cache(server)
    cache: Final = DualCache(redis_cache=redis)
    monkeypatch.setenv("SAML_IDP_METADATA_XML", _XML)
    with patch.dict(redis.init_async_client().response_callbacks, {"SET": _lost_reply}):
        result: Final = await SAMLAuthHandler._load_idp_settings(cache)
    assert await raw.get(_KEY) == json.dumps(result).encode()
    assert await cache.async_get_cache(_KEY, local_only=True) == result


@pytest.mark.asyncio
async def test_saml_metadata_unconfigured_still_returns_501(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAML_IDP_METADATA_XML", raising=False)
    monkeypatch.delenv("SAML_IDP_METADATA_URL", raising=False)
    with pytest.raises(HTTPException) as exc:
        await SAMLAuthHandler._load_idp_settings(DualCache())
    assert exc.value.status_code == 501
