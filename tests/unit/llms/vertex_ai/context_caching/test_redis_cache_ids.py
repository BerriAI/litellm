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
        {"name": NAME, "expireTime": None},
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


@pytest.fixture
def redis_store(monkeypatch):
    from unittest.mock import patch

    from litellm.llms.vertex_ai.context_caching.redis_cache_ids import connection

    store = Mock()
    store.get.return_value = None
    store.exists.return_value = 0
    store.set.return_value = True
    store.eval.return_value = 1
    monkeypatch.setenv("LITELLM_VERTEX_CACHE_ID_REDIS_URL", "redis://cache.invalid/0")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    connection.cache_clear()
    with patch("redis.Redis.from_url", return_value=store):
        yield store
    connection.cache_clear()


def cache_arguments():
    from litellm.litellm_core_utils.litellm_logging import Logging

    return dict(
        messages=[
            {
                "role": "system",
                "content": "Synthetic reusable instructions. " * 10000,
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": "Uncached question"},
        ],
        optional_params={
            "tools": [{"functionDeclarations": [{"name": "lookup", "parameters": {"type": "OBJECT"}}]}],
            "tool_choice": {"functionCallingConfig": {"mode": "AUTO"}},
            "temperature": 0.3,
        },
        api_key="synthetic-api-key",
        api_base=None,
        model="gemini-2.5-flash",
        timeout=10,
        logging_obj=Mock(spec=Logging),
        custom_llm_provider="vertex_ai",
        vertex_project="project",
        vertex_location="global",
        vertex_auth_header="synthetic-token",
        cache_id_auth_scope="synthetic-credential-fingerprint",
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("supplied_client", [False, True])
@pytest.mark.asyncio
async def test_public_entry_creates_then_reuses_id_preserving_tools_and_accounting(
    redis_store, respx_mock, asynchronous, supplied_client
):
    import copy
    import json

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    arguments = cache_arguments()
    before = copy.deepcopy((arguments["messages"], arguments["optional_params"]))
    response = {**created(), "expireTime": "9999-01-01T00:00:00Z", "usageMetadata": {"totalTokenCount": 1234}}
    route = respx_mock.post(url__regex=r"https://.*?/cachedContents$").respond(200, json=response)
    client = (AsyncHTTPHandler() if asynchronous else HTTPHandler()) if supplied_client else None
    endpoint = ContextCachingEndpoints()
    try:
        first = (
            await endpoint.async_check_and_create_cache(client=client, **arguments)
            if asynchronous
            else endpoint.check_and_create_cache(client=client, **arguments)
        )
        assert first[2] == NAME
        assert first[0] == before[0][1:]
        assert first[1] == {"temperature": 0.3}
        assert arguments["logging_obj"].vertex_ai_cache_creation_input_tokens == 1234
        outbound = json.loads(route.calls[0].request.content)
        assert outbound["tools"] == before[1]["tools"]
        assert outbound["toolConfig"] == before[1]["tool_choice"]
        assert route.calls[0].request.headers["authorization"] == "Bearer synthetic-token"
        redis_store.get.return_value = json.dumps({"name": NAME, "expires": 253370764800})
        second = (
            await endpoint.async_check_and_create_cache(client=client, **arguments)
            if asynchronous
            else endpoint.check_and_create_cache(client=client, **arguments)
        )
        assert second == first
        assert route.call_count == 1
        assert (arguments["messages"], arguments["optional_params"]) == before
        if client is not None:
            assert not client.client.is_closed
    finally:
        if isinstance(client, AsyncHTTPHandler):
            await client.close()
        elif client is not None:
            client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reason", ["unmarked", "too_short", "short_ttl", "assistant_turn", "redis_down", "cooldown"])
@pytest.mark.asyncio
async def test_ineligible_or_unavailable_cache_keeps_full_request(redis_store, respx_mock, asynchronous, reason):
    import copy

    from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    arguments = cache_arguments()
    if reason == "unmarked":
        del arguments["messages"][0]["cache_control"]
    elif reason == "too_short":
        arguments["messages"][0]["content"] = "short"
    elif reason == "short_ttl":
        arguments["messages"][0]["content"] = [
            {
                "type": "text",
                "text": arguments["messages"][0]["content"],
                "cache_control": {"type": "ephemeral", "ttl": "30s"},
            }
        ]
    elif reason == "assistant_turn":
        arguments["messages"][0]["role"] = "assistant"
    elif reason == "redis_down":
        redis_store.get.side_effect = RedisError("unavailable")
    else:
        redis_store.exists.return_value = 1
    before = copy.deepcopy((arguments["messages"], arguments["optional_params"]))
    endpoint = ContextCachingEndpoints()
    result = (
        await endpoint.async_check_and_create_cache(client=None, **arguments)
        if asynchronous
        else endpoint.check_and_create_cache(client=None, **arguments)
    )
    assert result == (*before, None)
    assert result[0] is arguments["messages"]
    assert result[1] is arguments["optional_params"]
    assert len(respx_mock.calls) == 0


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [403, 429])
@pytest.mark.asyncio
async def test_public_creation_error_preserves_auth_failure_or_fails_open_on_quota(
    redis_store, respx_mock, asynchronous, status
):
    from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    arguments = cache_arguments()
    route = respx_mock.post(url__regex=r"https://.*?/cachedContents$").respond(status, json={"error": "test"})
    endpoint = ContextCachingEndpoints()

    async def call():
        if asynchronous:
            return await endpoint.async_check_and_create_cache(client=None, **arguments)
        return endpoint.check_and_create_cache(client=None, **arguments)

    if status == 403:
        with pytest.raises(httpx.HTTPStatusError) as error:
            await call()
        assert error.value.response.status_code == status
    else:
        assert await call() == (arguments["messages"], arguments["optional_params"], None)
        assert any(call.kwargs.get("ex") == 30 for call in redis_store.set.call_args_list)
    assert route.call_count == 1
    assert redis_store.eval.call_args.args[2].endswith(":lock")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("published", [True, False])
@pytest.mark.asyncio
async def test_waiter_rechecks_publication_or_cooldown_after_winning_lease(asynchronous, published):
    import json
    from unittest.mock import AsyncMock

    store = Mock()
    store.get.side_effect = [None, json.dumps({"name": NAME, "expires": NOW + 3600}) if published else None]
    store.exists.side_effect = [0, 1]
    store.set.return_value = True
    create = AsyncMock() if asynchronous else Mock()
    resolved = registry(store)
    result = await resolved.async_resolve(create) if asynchronous else resolved.resolve(create)
    assert result == (NAME if published else None)
    create.assert_not_called()
    assert store.eval.call_args.args[2] == resolved.lock


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("published", [True, False])
@pytest.mark.asyncio
async def test_contender_observes_winner_or_exits_at_deadline(asynchronous, published):
    import json
    from unittest.mock import AsyncMock

    store = Mock()
    store.get.side_effect = [None, json.dumps({"name": NAME, "expires": NOW + 3600}) if published else None]
    store.exists.return_value = 0
    store.set.return_value = False
    ticks = iter((0.0, 0.1, 3.0))
    create = AsyncMock() if asynchronous else Mock()
    resolved = registry(store, monotonic=lambda: next(ticks), sleep=Mock(), async_sleep=AsyncMock())
    result = await resolved.async_resolve(create) if asynchronous else resolved.resolve(create)
    assert result == (NAME if published else None)
    create.assert_not_called()
    store.eval.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_async_creator_releases_lease():
    import asyncio

    store = Mock()
    store.get.return_value = None
    store.exists.return_value = 0
    store.set.return_value = True
    started = asyncio.Event()
    blocked = asyncio.Event()

    async def create():
        started.set()
        await blocked.wait()
        return created()

    resolved = registry(store)
    task = asyncio.create_task(resolved.async_resolve(create))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.eval.call_args.args[2] == resolved.lock


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("error", [httpx.ConnectError("connection failed"), ValueError("bad provider response")])
@pytest.mark.asyncio
async def test_creation_transport_failure_fails_open_but_unexpected_failure_propagates(asynchronous, error):
    from unittest.mock import AsyncMock

    store = Mock()
    store.get.return_value = None
    store.exists.return_value = 0
    store.set.return_value = True
    create = AsyncMock(side_effect=error) if asynchronous else Mock(side_effect=error)
    resolved = registry(store)
    if isinstance(error, ValueError):
        with pytest.raises(ValueError, match="bad provider response"):
            if asynchronous:
                await resolved.async_resolve(create)
            else:
                resolved.resolve(create)
    else:
        assert (await resolved.async_resolve(create) if asynchronous else resolved.resolve(create)) is None
        store.set.assert_any_call(resolved.cooldown, "1", ex=30)
    assert store.eval.call_args.args[2] == resolved.lock


@pytest.mark.parametrize("operation", ["release", "invalidate", "cool_down"])
def test_redis_cleanup_failure_does_not_interrupt_generation(operation):
    store = Mock()
    store.eval.side_effect = RedisError("unavailable")
    store.set.side_effect = RedisError("unavailable")
    resolved = registry(store)
    if operation == "release":
        assert resolved.release("owner") is None
    elif operation == "invalidate":
        assert resolved.invalidate(NAME) is None
    else:
        assert resolved.cool_down() is None


def test_expiry_recreates_instead_of_reusing_stale_name():
    import json

    store = Mock()
    store.get.return_value = json.dumps({"name": NAME, "expires": NOW + 30})
    store.exists.return_value = 0
    store.set.return_value = True
    store.eval.return_value = 1
    replacement = NAME + "-replacement"
    create = Mock(return_value={**created(), "name": replacement})
    assert registry(store).resolve(create) == replacement
    create.assert_called_once()


@pytest.mark.parametrize("source", ["mapping", "json", "file"])
def test_credential_fingerprint_stays_stable_without_exposing_secret(monkeypatch, source):
    import json
    from unittest.mock import mock_open, patch

    from litellm.llms.vertex_ai.context_caching.redis_cache_ids import credential_scope

    monkeypatch.setenv("LITELLM_VERTEX_CACHE_ID_REDIS_URL", "redis://cache.invalid/0")
    credentials = {"client_email": "test@example.invalid", "private_key": "synthetic-secret"}
    expected = credential_scope(credentials)
    argument = credentials if source == "mapping" else json.dumps(credentials) if source == "json" else "key.json"
    with (
        patch("os.path.isfile", return_value=source == "file"),
        patch("builtins.open", mock_open(read_data=json.dumps(credentials))),
    ):
        assert credential_scope(argument) == expected
    assert len(expected) == 64
    assert "synthetic-secret" not in expected
    assert credential_scope({**credentials, "client_email": "other@example.invalid"}) != expected


@pytest.mark.parametrize("credentials", [None, {}, "not-json", "{}", "[]"])
def test_unknown_credentials_fall_back_to_token_scope(monkeypatch, credentials):
    from litellm.llms.vertex_ai.context_caching.redis_cache_ids import credential_scope

    monkeypatch.setenv("LITELLM_VERTEX_CACHE_ID_REDIS_URL", "redis://cache.invalid/0")
    assert credential_scope(credentials) is None
    monkeypatch.delenv("LITELLM_VERTEX_CACHE_ID_REDIS_URL")
    assert credential_scope({"private_key": "synthetic-secret"}) is None


@pytest.mark.parametrize("override", [False, True])
def test_rotating_token_reuses_stable_scope_unless_authorization_is_overridden(redis_store, override):
    from litellm.llms.vertex_ai.context_caching.redis_cache_ids import prepare
    from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    endpoint = ContextCachingEndpoints()
    first_args = cache_arguments()
    second_args = {**first_args, "vertex_auth_header": "new-token"}
    if override:
        first_args["extra_headers"] = {"Authorization": "Bearer first"}
        second_args["extra_headers"] = {"Authorization": "Bearer second"}
    first = prepare(endpoint, first_args)
    second = prepare(endpoint, second_args)
    assert first is not None and second is not None
    assert (first.registry.key == second.registry.key) is (not override)
    assert first.headers["Authorization"] == ("Bearer first" if override else "Bearer synthetic-token")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_unusable_redis_connection_returns_original_request(monkeypatch, asynchronous):
    from unittest.mock import patch

    from litellm.llms.vertex_ai.context_caching.redis_cache_ids import connection
    from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    monkeypatch.setenv("LITELLM_VERTEX_CACHE_ID_REDIS_URL", "redis://cache.invalid/0")
    connection.cache_clear()
    arguments = cache_arguments()
    endpoint = ContextCachingEndpoints()
    with patch("redis.Redis.from_url", side_effect=RedisError("unavailable")):
        result = (
            await endpoint.async_check_and_create_cache(client=None, **arguments)
            if asynchronous
            else endpoint.check_and_create_cache(client=None, **arguments)
        )
    assert result == (arguments["messages"], arguments["optional_params"], None)
    connection.cache_clear()


def test_redis_is_optional_for_base_install():
    import builtins
    from unittest.mock import patch

    from litellm.llms.vertex_ai.context_caching.redis_cache_ids import redis_error_type

    original_import = builtins.__import__

    def without_redis(name, *args, **kwargs):
        if name == "redis.exceptions":
            raise ImportError("redis is not installed")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=without_redis):
        assert redis_error_type() is ImportError


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.asyncio
async def test_successful_generation_never_invalidates_or_retries(asynchronous, managed):
    from unittest.mock import AsyncMock

    from litellm.llms.vertex_ai.context_caching.redis_cache_ids import (
        CacheReference,
        async_post_with_recovery,
        post_with_recovery,
    )

    store = Mock()
    body = {"cachedContent": CacheReference(NAME, registry(store), [], {}) if managed else NAME}
    response = httpx.Response(200, request=httpx.Request("POST", "https://provider.invalid"), json={"answer": "ok"})
    post = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    assert (await async_post_with_recovery(post, body) if asynchronous else post_with_recovery(post, body)) is response
    post.assert_called_once_with(body)
    store.eval.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"error":{"message":"model not found"}}'])
@pytest.mark.asyncio
async def test_unknown_error_body_propagates_without_retry(asynchronous, raw):
    from unittest.mock import AsyncMock

    from litellm.llms.vertex_ai.context_caching.redis_cache_ids import (
        CacheReference,
        async_post_with_recovery,
        post_with_recovery,
    )

    store = Mock()
    reference = CacheReference(NAME, registry(store), [], {})
    reference.rebuild = lambda: {"contents": []}
    response = httpx.Response(404, request=httpx.Request("POST", "https://provider.invalid"), content=raw)
    post = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    with pytest.raises(httpx.HTTPStatusError) as error:
        if asynchronous:
            await async_post_with_recovery(post, {"cachedContent": reference})
        else:
            post_with_recovery(post, {"cachedContent": reference})
    assert error.value.response is response
    post.assert_called_once()
    store.eval.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_recovery_rebuilds_full_gemini_body_and_propagates_second_failure(asynchronous):
    import copy
    from unittest.mock import AsyncMock

    from litellm.llms.vertex_ai.context_caching.redis_cache_ids import (
        CacheReference,
        async_post_with_recovery,
        post_with_recovery,
    )
    from litellm.llms.vertex_ai.gemini.transformation import transform_request_body

    arguments = cache_arguments()
    params = {"temperature": 0.3}
    original = copy.deepcopy((arguments["messages"], params))
    store = Mock()
    reference = copy.deepcopy(CacheReference(NAME, registry(store), arguments["messages"], params))
    body = transform_request_body(
        messages=arguments["messages"][1:],
        model=arguments["model"],
        optional_params=params,
        custom_llm_provider="vertex_ai",
        litellm_params={},
        cached_content=reference,
    )
    expected = transform_request_body(
        messages=copy.deepcopy(original[0]),
        model=arguments["model"],
        optional_params=copy.deepcopy(original[1]),
        custom_llm_provider="vertex_ai",
        litellm_params={},
        cached_content=None,
    )
    request = httpx.Request("POST", "https://provider.invalid")
    expired = httpx.Response(404, request=request, json={"error": {"message": "CachedContent expired"}})
    denied = httpx.Response(403, request=request, json={"error": {"message": "permission denied"}})
    post = AsyncMock(side_effect=[expired, denied]) if asynchronous else Mock(side_effect=[expired, denied])
    with pytest.raises(httpx.HTTPStatusError) as error:
        if asynchronous:
            await async_post_with_recovery(post, body)
        else:
            post_with_recovery(post, body)
    assert error.value.response is denied
    assert post.call_count == 2
    assert body == expected
    assert "cachedContent" not in body
    assert (arguments["messages"], params) == original
    assert expired.is_closed
