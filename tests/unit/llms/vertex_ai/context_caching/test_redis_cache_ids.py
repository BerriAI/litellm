from datetime import datetime, timezone
from unittest.mock import Mock

import httpx
import pytest
from redis import RedisError

from litellm.llms.vertex_ai.context_caching.redis_cache_ids import Registry

NAME = "projects/project/locations/global/cachedContents/cache"
NOW = 1_900_000_000.0


def registry(store, **kwargs):
    return Registry(store, ("project", "global", "auth"), {"model": "model"}, clock=lambda: NOW, **kwargs)


def created(seconds=3600):
    return {"name": NAME, "expireTime": datetime.fromtimestamp(NOW + seconds, timezone.utc).isoformat()}


def test_only_owner_publishes_until_provider_expiry_minus_margin():
    store = Mock()
    store.eval.return_value = 1
    resolved = registry(store)
    assert resolved.publish("owner", created()) == NAME
    assert store.eval.call_args.args[-1] == 3_540_000
    assert store.eval.call_args.args[2:5] == (resolved.lock, resolved.key, "owner")
    store.eval.return_value = 0
    assert resolved.publish("expired-owner", created()) is None


@pytest.mark.parametrize("seconds", [-1, 0, 30, 60])
def test_expired_or_near_expiry_names_are_not_published(seconds):
    store = Mock()
    assert registry(store).publish("owner", created(seconds)) is None
    store.eval.assert_not_called()


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"name": NAME},
        {"name": NAME, "expireTime": "not-a-date"},
        {"name": "invalid", "expireTime": "2030-01-01T00:00:00Z"},
        {"name": NAME, "expireTime": "2030-01-01T00:00:00"},
    ],
)
def test_unusable_provider_metadata_never_publishes(result):
    store = Mock()
    assert registry(store).publish("owner", result) is None
    store.eval.assert_not_called()


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "null",
        "[]",
        '{"name":"bad","expires":1900003600}',
        '{"name":"' + NAME + '","expires":1900000030}',
        '{"name":"' + NAME + '","expires":"NaN"}',
    ],
)
def test_untrusted_or_expiring_redis_records_are_misses(raw):
    store = Mock()
    store.get.return_value = raw
    assert registry(store).read() is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_redis_failure_bypasses_cache_creation(asynchronous):
    store = Mock()
    store.get.side_effect = RedisError("unavailable")
    create = Mock()
    resolved = registry(store)
    result = await resolved.async_resolve(create) if asynchronous else resolved.resolve(create)
    assert result is None
    create.assert_not_called()


def test_transient_creation_failure_sets_shared_cooldown_and_releases_lease():
    store = Mock()
    store.get.return_value = None
    store.exists.return_value = 0
    store.set.return_value = True
    response = httpx.Response(429, request=httpx.Request("POST", "https://provider.invalid"))
    create = Mock(side_effect=httpx.HTTPStatusError("quota", request=response.request, response=response))
    resolved = registry(store)
    assert resolved.resolve(create) is None
    create.assert_called_once()
    store.set.assert_any_call(resolved.cooldown, "1", ex=30)
    assert store.eval.call_args.args[2] == resolved.lock


def test_keys_separate_tenant_and_payload_without_storing_either():
    store = Mock()
    first = Registry(store, ("tenant-a",), {"text": "private prefix"})
    tenant = Registry(store, ("tenant-b",), {"text": "private prefix"})
    prefix = Registry(store, ("tenant-a",), {"text": "different prefix"})
    assert len({first.key, tenant.key, prefix.key}) == 3
    assert "private prefix" not in first.key
    assert "tenant-a" not in first.key
    assert first.key.split("{")[1].split("}")[0] == first.lock.split("{")[1].split("}")[0]


def test_valid_hit_does_not_create_or_acquire_a_lease():
    import json

    store = Mock()
    store.get.return_value = json.dumps({"name": NAME, "expires": NOW + 3600})
    create = Mock()
    assert registry(store).resolve(create) == NAME
    create.assert_not_called()
    store.set.assert_not_called()


def test_contention_exits_without_duplicate_creation_when_wait_budget_expires():
    store = Mock()
    store.get.return_value = None
    store.exists.return_value = 0
    store.set.return_value = False
    ticks = iter((0.0, 3.0))
    create = Mock()
    resolved = registry(store, monotonic=lambda: next(ticks), sleep=Mock())
    assert resolved.resolve(create) is None
    create.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_expired_managed_cache_retries_once_with_original_content(asynchronous):
    from litellm.llms.vertex_ai.context_caching.redis_cache_ids import (
        CacheReference,
        async_post_with_recovery,
        post_with_recovery,
    )

    store = Mock()
    reference = CacheReference(NAME, registry(store), [], {})
    full_body = {"contents": [{"role": "user", "parts": [{"text": "complete original request"}]}]}
    reference.rebuild = lambda: full_body
    body = {"cachedContent": reference, "contents": []}
    received = []

    def post(request_body):
        received.append(dict(request_body))
        request = httpx.Request("POST", "https://provider.invalid/generateContent")
        if len(received) == 1:
            return httpx.Response(404, request=request, json={"error": {"message": "CachedContent not found"}})
        return httpx.Response(200, request=request, json={})

    async def async_post(request_body):
        return post(request_body)

    response = await async_post_with_recovery(async_post, body) if asynchronous else post_with_recovery(post, body)
    assert response.status_code == 200
    assert len(received) == 2
    assert received[0]["cachedContent"] == NAME
    assert received[1] == full_body
    store.eval.assert_called_once()
    assert store.eval.call_args.args[-1] == NAME


@pytest.mark.parametrize(
    "status,message",
    [(403, "permission denied"), (429, "quota exceeded"), (404, "model not found"), (400, "invalid request")],
)
def test_unrelated_provider_errors_are_not_retried(status, message):
    from litellm.llms.vertex_ai.context_caching.redis_cache_ids import CacheReference, post_with_recovery

    store = Mock()
    reference = CacheReference(NAME, registry(store), [], {})
    reference.rebuild = lambda: {"contents": []}
    body = {"cachedContent": reference, "contents": []}
    request = httpx.Request("POST", "https://provider.invalid/generateContent")
    post = Mock(return_value=httpx.Response(status, request=request, json={"error": {"message": message}}))
    with pytest.raises(httpx.HTTPStatusError):
        post_with_recovery(post, body)
    post.assert_called_once()
    store.eval.assert_not_called()
