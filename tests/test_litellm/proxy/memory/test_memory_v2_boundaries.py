"""Failure and authorization boundaries, with only the database/model edges replaced."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from prisma.models import LiteLLM_MemoryTable

from litellm.proxy.memory.gateway import GatewayMemoryLoop
from litellm.proxy.memory.knowledge import MEMORY_TOOL_NAMES, execute_memory_tool
from litellm.litellm_core_utils.prompt_templates.server_tool_responses import executable_server_calls, object_items
from litellm.proxy.memory.continuation import MemoryContinuation, MemoryContinuations, prefix_hashes
from litellm.proxy.memory.policy import MemoryAccess, MemoryIdentity, resolve_memory_access
from litellm.proxy.memory.store import MemoryStore
from litellm.types.memory_v2 import MemoryCapture, MemoryPolicy, MemorySearch

_NOW: Final = datetime(2026, 9, 12, tzinfo=timezone.utc)
_IDENTITY: Final = MemoryIdentity("a" * 64, "owner", "team", "project", "org", False)
_POLICY: Final = MemoryPolicy(
    policy_id="policy",
    target_type="team",
    target_id="team",
    activation="automatic",
    scope="key",
    updated_at=_NOW,
    updated_by="admin",
)
_CAPTURE: Final = MemoryCapture(key="demo", title="Demo", content="Use port 8347", evidence="User selected this port")


@pytest.fixture
def prisma_edge() -> MagicMock:
    client = MagicMock()
    client.db.litellm_memorypolicy.find_many = AsyncMock(return_value=[_POLICY])
    client.db.litellm_memorypreference.find_unique = AsyncMock(return_value=None)
    table = client.db.litellm_memorytable
    table.find_unique = AsyncMock(return_value=None)
    table.find_first = AsyncMock(return_value=None)
    table.find_many = AsyncMock(return_value=[])
    table.create = AsyncMock()
    table.count = AsyncMock(return_value=0)
    client.db.tx.return_value.__aenter__.return_value = client.db
    client.db.execute_raw = AsyncMock()
    continuations = client.db.litellm_memorycontinuation
    continuations.find_many = AsyncMock(return_value=[])
    continuations.find_first = AsyncMock(return_value=None)
    continuations.count = AsyncMock(return_value=0)
    continuations.delete_many = AsyncMock(return_value=0)
    continuations.upsert = AsyncMock()
    table.update_many = AsyncMock(return_value=1)
    table.delete_many = AsyncMock(return_value=1)
    return client


def store(client: MagicMock, identity: MemoryIdentity = _IDENTITY) -> MemoryStore:
    return MemoryStore(client, MemoryAccess(identity, _POLICY, False))


def row(**changes: object) -> LiteLLM_MemoryTable:
    return LiteLLM_MemoryTable.model_validate(
        {
            "memory_id": "entry",
            "key": f"memory-v2:{_IDENTITY.namespace('key')}:demo",
            "namespace": _IDENTITY.namespace("key"),
            "value": _CAPTURE.content,
            "metadata": json.dumps({"title": _CAPTURE.title, "evidence": _CAPTURE.evidence}),
            "created_at": _NOW,
            "updated_at": _NOW,
            **changes,
        }
    )


@pytest.mark.asyncio
async def test_policy_precedence_and_opt_in_are_resolved_from_database(prisma_edge: MagicMock) -> None:
    team = _POLICY.model_copy(update={"activation": "opt_in"})
    key = _POLICY.model_copy(update={"target_type": "key", "target_id": "a" * 64, "activation": "disabled"})
    policies = prisma_edge.db.litellm_memorypolicy.find_many
    policies.return_value = [team, key]
    prisma_edge.db.litellm_memorypreference.find_unique.return_value = SimpleNamespace(enabled=True)
    access = await resolve_memory_access(prisma_edge, _IDENTITY)
    assert access.policy == key and not access.active and access.opted_in
    policies.return_value = [team]
    assert (await resolve_memory_access(prisma_edge, _IDENTITY)).active
    prisma_edge.db.litellm_memorypreference.find_unique.return_value = None
    assert not (await resolve_memory_access(prisma_edge, _IDENTITY)).active
    policies.return_value = []
    assert (await resolve_memory_access(prisma_edge, _IDENTITY)).namespace is None


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["disabled", "scope", "missing", "readonly"])
async def test_store_rechecks_policy_before_writing(prisma_edge: MagicMock, change: str) -> None:
    identity = _IDENTITY
    if change == "disabled":
        prisma_edge.db.litellm_memorypolicy.find_many.return_value = [
            _POLICY.model_copy(update={"activation": "disabled"})
        ]
    elif change == "scope":
        prisma_edge.db.litellm_memorypolicy.find_many.return_value = [_POLICY.model_copy(update={"scope": "team"})]
    elif change == "missing":
        prisma_edge.db.litellm_memorypolicy.find_many.return_value = []
    else:
        identity = MemoryIdentity("a" * 64, "owner", "team", "project", "org", True)
    with pytest.raises(HTTPException) as exc:
        await store(prisma_edge, identity).capture(_CAPTURE)
    assert exc.value.status_code == 403
    prisma_edge.db.litellm_memorytable.create.assert_not_awaited()
    prisma_edge.db.litellm_memorytable.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_applies_fuzzy_ranking_before_pagination_with_namespace_boundaries(prisma_edge: MagicMock) -> None:
    prisma_edge.db.litellm_memorytable.find_many.return_value = [
        row(memory_id="first", value="Use port 8347"),
        row(memory_id="second", value="Use port 8348"),
    ]
    entries = await store(prisma_edge).search(MemorySearch(query="prto demo", limit=1, offset=1))
    assert [entry.memory_id for entry in entries] == ["second"]
    query = prisma_edge.db.litellm_memorytable.find_many.call_args.kwargs
    assert query["where"]["namespace"] == _IDENTITY.namespace("key")
    assert query["take"] == 1000


@pytest.mark.asyncio
async def test_read_and_delete_cannot_address_another_namespace(prisma_edge: MagicMock) -> None:
    memory = store(prisma_edge)
    with pytest.raises(HTTPException) as exc:
        await memory.read("foreign-entry")
    assert exc.value.status_code == 404
    assert prisma_edge.db.litellm_memorytable.find_first.call_args.kwargs["where"] == {
        "namespace": _IDENTITY.namespace("key"),
        "memory_id": "foreign-entry",
    }
    prisma_edge.db.litellm_memorypolicy.find_many.return_value = [_POLICY.model_copy(update={"activation": "disabled"})]
    assert await memory.delete("entry")
    assert prisma_edge.db.litellm_memorytable.delete_many.call_args.kwargs["where"] == {
        "namespace": _IDENTITY.namespace("key"),
        "memory_id": "entry",
    }
    with pytest.raises(HTTPException) as inactive:
        await memory.read("entry")
    assert inactive.value.status_code == 403


@pytest.mark.asyncio
async def test_identical_capture_is_idempotent_and_new_capture_has_scoped_identity(prisma_edge: MagicMock) -> None:
    from litellm.proxy.db.prisma_client import PrismaWrapper

    table = prisma_edge.db.litellm_memorytable
    table.create.return_value = row()
    wrapped: Final = MemoryStore(
        SimpleNamespace(db=PrismaWrapper(prisma_edge.db)), MemoryAccess(_IDENTITY, _POLICY, False)
    )
    saved = await wrapped.capture(_CAPTURE)
    assert saved.content == _CAPTURE.content
    data = table.create.call_args.kwargs["data"]
    assert data["namespace"] == _IDENTITY.namespace("key") and data["user_id"] == "owner" and data["team_id"] == "team"
    assert data["key"].startswith("memory-v2:" + _IDENTITY.namespace("key") + ":")
    table.find_unique.return_value = row()
    assert await wrapped.capture(_CAPTURE) == saved
    table.create.assert_awaited_once()
    table.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["missing", "foreign", "stale", "concurrent"])
async def test_capture_rejects_stale_or_conflicting_replacements(prisma_edge: MagicMock, race: str) -> None:
    table = prisma_edge.db.litellm_memorytable
    table.find_unique.return_value = (
        None if race == "missing" else row(namespace="foreign" if race == "foreign" else _IDENTITY.namespace("key"))
    )
    table.update_many.return_value = 0 if race == "concurrent" else 1
    correction = _CAPTURE.model_copy(
        update={
            "content": "Use port 8348",
            "expected_revision": _NOW - timedelta(seconds=1) if race == "stale" else _NOW,
        }
    )
    with pytest.raises(HTTPException) as exc:
        await store(prisma_edge).capture(correction)
    assert exc.value.status_code == 409
    table.create.assert_not_awaited()
    if race == "concurrent":
        where = table.update_many.call_args.kwargs["where"]
        assert (
            where["namespace"] == _IDENTITY.namespace("key")
            and where["updated_at"] == _NOW
            and where["value"] == _CAPTURE.content
        )
        assert "metadata" in where
    else:
        table.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_replacement_reads_confirmed_updated_row(prisma_edge: MagicMock) -> None:
    table = prisma_edge.db.litellm_memorytable
    table.find_unique.side_effect = [row(), row(value="Use port 8348", updated_at=_NOW + timedelta(seconds=1))]
    result = await store(prisma_edge).capture(
        _CAPTURE.model_copy(update={"content": "Use port 8348", "expected_revision": _NOW})
    )
    assert result.content == "Use port 8348" and result.updated_at > _NOW
    table.update_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_argument_errors_and_revocation_return_receipts_without_writing(prisma_edge: MagicMock) -> None:
    memory = store(prisma_edge)
    invalid = await execute_memory_tool(
        memory, {"id": "a", "name": "litellm_memory_capture", "arguments": {"key": "missing-fields"}}, "checkpoint"
    )
    assert not invalid.reflected and "error" in invalid.output
    missing = await execute_memory_tool(
        memory, {"id": "a", "name": "litellm_memory_read", "arguments": {"id": "missing"}}, "checkpoint"
    )
    assert missing.output == {"error": "Memory not found", "status": 404}
    unknown = await execute_memory_tool(memory, {"id": "a", "name": "other_tool", "arguments": {}}, "checkpoint")
    assert unknown.output == {"error": "Unknown memory tool"}
    prisma_edge.db.litellm_memorypolicy.find_many.return_value = []
    revoked = await execute_memory_tool(
        memory, {"id": "a", "name": "litellm_memory_capture", "arguments": {"observations": []}}, "checkpoint"
    )
    assert revoked.output["status"] == 403 and not revoked.reflected
    prisma_edge.db.litellm_memorytable.create.assert_not_awaited()


def request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "scheme": "https",
            "server": ("gateway.example", 443),
            "client": ("203.0.113.7", 41231),
            "query_string": b"api-version=test-version",
            "headers": [(b"host", b"gateway.example")],
        }
    )


@pytest.mark.asyncio
async def test_read_only_injection_and_forced_no_tools_do_not_request_reflection(prisma_edge: MagicMock) -> None:
    read_only: Final = MemoryIdentity("a" * 64, "owner", "team", "project", "org", True)
    loop: Final = GatewayMemoryLoop(
        FastAPI(),
        request(),
        {"messages": [{"role": "user", "content": "Hello"}]},
        "anthropic_messages",
        store(prisma_edge, read_only),
    )
    await loop.prepare()
    assert "litellm_memory_capture" not in str(loop.data)
    assert {tool["name"] for tool in object_items(loop.data["tools"])} == MEMORY_TOOL_NAMES - {"litellm_memory_capture"}
    assert loop.reflected
    forced: Final = GatewayMemoryLoop(
        FastAPI(),
        request(),
        {"tool_choice": {"type": "none"}, "messages": []},
        "anthropic_messages",
        store(prisma_edge),
    )
    await forced.prepare()
    assert forced.data["tool_choice"] == {"type": "none"}
    assert forced.reflected


@pytest.mark.parametrize("arguments,status", (('{"query":"valid"}', "incomplete"), ('{"query":', "completed")))
def test_incomplete_or_malformed_memory_arguments_never_become_executable(arguments: str, status: str) -> None:
    with pytest.raises(ValueError, match=r"complete|Invalid JSON"):
        executable_server_calls(
            {
                "status": status,
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "memory_1",
                        "name": "litellm_memory_search",
                        "arguments": arguments,
                    }
                ],
            },
            "aresponses",
            MEMORY_TOOL_NAMES,
        )


@pytest.mark.asyncio
async def test_restore_preserves_hidden_memory_tool_results_and_client_cache_markers(prisma_edge: MagicMock) -> None:
    items: Final = (
        {"role": "user", "content": "Read the fixture"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "client_1", "name": "Read", "input": {"path": "README.md"}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "client_1",
                    "content": "Fixture content",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    )
    continuations: Final = MemoryContinuations(store(prisma_edge), "anthropic_messages")
    anchor: Final = prefix_hashes(items, "anthropic_messages")[1]
    replacement: Final = (
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "memory_1", "name": "litellm_memory_read", "input": {"id": "entry"}},
                *object_items(items[1]["content"]),
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "memory_1", "content": "Stored fact"}]},
    )
    prisma_edge.db.litellm_memorycontinuation.find_many.return_value = [
        SimpleNamespace(
            id=continuations.identifier(anchor),
            payload=MemoryContinuation(replaces=1, replacement=replacement).model_dump(),
        )
    ]
    restored: Final = await continuations.restore(items)
    assert restored[0] == items[0]
    assert restored[1] == replacement[0]
    assert object_items(restored[2]["content"]) == (
        *object_items(replacement[1]["content"]),
        *object_items(items[2]["content"]),
    )
    sibling: Final = MemoryIdentity("b" * 64, "owner", "team", "project", "org", False)
    assert MemoryContinuations(store(prisma_edge, sibling), "anthropic_messages").identifier(
        anchor
    ) != continuations.identifier(anchor)
    without_markers: Final = (
        *items[:2],
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "client_1", "content": "Fixture content"}]},
    )
    assert prefix_hashes(items, "anthropic_messages") == prefix_hashes(without_markers, "anthropic_messages")


@pytest.mark.asyncio
async def test_model_loop_is_bounded_and_search_results_reach_the_active_model(prisma_edge: MagicMock) -> None:
    prisma_edge.db.litellm_memorytable.find_many.return_value = [row()]
    provider = FastAPI()
    observed = []

    @provider.post("/v1/messages")
    async def model(incoming: Request):
        body = await incoming.json()
        observed.append(body)
        return {
            "id": "msg_search",
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "search", "name": "litellm_memory_search", "input": {"query": "demo"}},
            ],
        }

    loop = GatewayMemoryLoop(
        provider,
        request(),
        {"messages": [{"role": "user", "content": "My demo port?"}]},
        "anthropic_messages",
        store(prisma_edge),
    )
    with pytest.raises(HTTPException) as exc:
        async for _ in loop.run():
            pass
    assert exc.value.status_code == 429 and len(observed) == 8
    continuation = observed[1]["messages"]
    assert continuation[-1]["content"][0]["tool_use_id"] == "search"
    assert "8347" in continuation[-1]["content"][0]["content"]
    prisma_edge.db.litellm_memorycontinuation.upsert.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id,count", [(True, 1), (False, 17)])
async def test_invalid_model_calls_are_rejected_before_storage(
    prisma_edge: MagicMock, bad_id: bool, count: int
) -> None:
    loop = GatewayMemoryLoop(FastAPI(), request(), {"messages": []}, "anthropic_messages", store(prisma_edge))
    loop.last_response = {
        "id": "msg_invalid",
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "tool_use",
                "id": "" if bad_id else str(i),
                "name": "litellm_memory_search",
                "input": {"query": "demo"},
            }
            for i in range(count)
        ],
    }
    with pytest.raises(HTTPException) as exc:
        await loop.advance(0)
    assert exc.value.status_code == 502
    prisma_edge.db.litellm_memorytable.find_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("writer_unavailable", [False, True])
async def test_replica_lag_cannot_authorize_memory_after_primary_revocation(
    prisma_edge: MagicMock, writer_unavailable: bool
) -> None:
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer = MagicMock(spec=PrismaWrapper)
    reader = MagicMock(spec=PrismaWrapper)
    writer.litellm_memorypolicy = SimpleNamespace(
        find_many=AsyncMock(return_value=[_POLICY.model_copy(update={"activation": "disabled"})])
    )
    reader.litellm_memorypolicy = SimpleNamespace(find_many=AsyncMock(return_value=[_POLICY]))
    writer.litellm_memorypreference = SimpleNamespace(find_unique=AsyncMock(return_value=None))
    writer.litellm_memorytable = prisma_edge.db.litellm_memorytable
    writer.is_connected = MagicMock(return_value=False)
    reader.is_connected = MagicMock(return_value=False)
    routed = RoutingPrismaWrapper(writer, reader)
    if writer_unavailable:
        writer.connect = AsyncMock(side_effect=RuntimeError("primary unavailable"))
        reader.connect = AsyncMock()
        await routed.connect()
    client = SimpleNamespace(db=routed)
    access = await resolve_memory_access(client, _IDENTITY)
    assert not access.active
    reader.litellm_memorypolicy.find_many.assert_not_awaited()
    with pytest.raises(HTTPException) as exc:
        await MemoryStore(client, MemoryAccess(_IDENTITY, _POLICY, False)).capture(_CAPTURE)
    assert exc.value.status_code == 403
    writer.litellm_memorytable.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_unconfigured_gate_caches_presence_without_caching_authorization(prisma_edge: MagicMock) -> None:
    from litellm.caching.caching import DualCache
    from litellm.proxy.memory.policy import gateway_memory_is_configured

    cache = DualCache()
    policies = prisma_edge.db.litellm_memorypolicy.find_many
    policies.return_value = []
    assert not await gateway_memory_is_configured(prisma_edge, cache)
    assert not await gateway_memory_is_configured(prisma_edge, cache)
    policies.assert_awaited_once()
    assert policies.call_args.kwargs == {"take": 1}
    enabled_cache = DualCache()
    policies.return_value = [_POLICY]
    assert await gateway_memory_is_configured(prisma_edge, enabled_cache)
    policies.return_value = [_POLICY.model_copy(update={"activation": "disabled"})]
    assert await gateway_memory_is_configured(prisma_edge, enabled_cache)
    assert not (await resolve_memory_access(prisma_edge, _IDENTITY)).active


@pytest.mark.asyncio
async def test_gateway_rounds_keep_separate_limiter_contexts_and_original_client_tools(prisma_edge: MagicMock) -> None:
    import asyncio

    from litellm.proxy.hooks.parallel_request_limiter_v3 import claim_request_stash_for_data, get_request_stash

    provider = FastAPI()
    observed = []
    deferred = []
    release = asyncio.Event()
    prisma_edge.db.litellm_memorytable.find_many.return_value = [row()]

    @provider.post("/v1/messages")
    async def model(incoming: Request):
        assert incoming.client is not None and incoming.client.host == "203.0.113.7"
        assert incoming.url.scheme == "https" and incoming.headers["host"] == "gateway.example"
        assert incoming.query_params["api-version"] == "test-version"
        body = await incoming.json()
        call_id = body["litellm_call_id"]
        stash = claim_request_stash_for_data(body)
        observed.append((call_id, stash, body))

        async def logged_owner():
            await release.wait()
            return get_request_stash().owner_litellm_call_id

        deferred.append(asyncio.create_task(logged_owner()))
        if len(observed) == 1:
            return {
                "id": "msg_search",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": "search", "name": "litellm_memory_search", "input": {"query": "demo"}},
                ],
            }
        return {
            "id": "msg_client",
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "original_client_id", "name": "client_tool", "input": {"path": "README.md"}},
            ],
        }

    original = {
        "model": "demo",
        "stream": False,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "My port?"}],
        "tools": [{"name": "client_tool", "input_schema": {"type": "object"}}],
    }
    before = get_request_stash()
    loop = GatewayMemoryLoop(provider, request(), original, "anthropic_messages", store(prisma_edge))
    async for _ in loop.run():
        pass
    release.set()
    owners = await asyncio.gather(*deferred)
    assert len(observed) == 2 and len({id(stash) for _, stash, _ in observed}) == 2
    assert owners == [call_id for call_id, _, _ in observed] and len(set(owners)) == 2
    assert get_request_stash() is before
    assert all(body["stream"] is False and body["max_tokens"] == 100 for _, _, body in observed)
    assert all(body["tools"][0] == original["tools"][0] and len(body["tools"]) == 5 for _, _, body in observed)
    assert loop.stream.response()["content"] == [
        {
            "type": "tool_use",
            "id": "original_client_id",
            "name": "client_tool",
            "input": {"path": "README.md"},
        }
    ]
    assert "8347" in json.dumps(observed[1][2]["messages"]) and "8347" not in json.dumps(original)


@pytest.mark.asyncio
@pytest.mark.parametrize("share_auth_cache", [False, True])
async def test_backend_activation_invalidates_a_gateway_negative_hint_without_pubsub(
    prisma_edge: MagicMock, share_auth_cache: bool
) -> None:
    from unittest.mock import patch

    from litellm.caching.caching import DualCache
    from litellm.proxy.memory.policy import gateway_memory_is_configured, invalidate_memory_configuration

    shared = {}

    async def get(key, **kwargs):
        return shared.get(key)

    async def set_value(key, value, **kwargs):
        shared[key] = value

    async def delete(key, **kwargs):
        shared.pop(key, None)

    redis = MagicMock(
        async_get_cache=AsyncMock(side_effect=get),
        async_set_cache=AsyncMock(side_effect=set_value),
        async_delete_cache=AsyncMock(side_effect=delete),
    )
    gateway_cache = DualCache(redis_cache=redis if share_auth_cache else None)
    backend_cache = DualCache(redis_cache=redis if share_auth_cache else None)
    policies = prisma_edge.db.litellm_memorypolicy.find_many
    with patch.multiple(  # test-quality-ok: Inject external worker caches and Redis; run real invalidation.
        "litellm.proxy.proxy_server", user_api_key_cache=backend_cache, redis_usage_cache=redis
    ):
        policies.return_value = []
        assert not await gateway_memory_is_configured(prisma_edge, gateway_cache)
        assert not await gateway_memory_is_configured(prisma_edge, gateway_cache)
        policies.assert_awaited_once()
        policies.return_value = [_POLICY]
        await invalidate_memory_configuration()
        assert await gateway_memory_is_configured(prisma_edge, gateway_cache)
        assert policies.await_count == 2


@pytest.mark.asyncio
async def test_redis_circuit_breaker_falls_back_to_primary_configuration(prisma_edge: MagicMock) -> None:
    from unittest.mock import patch

    from litellm.caching.caching import DualCache
    from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
    from litellm.proxy.memory.policy import gateway_memory_is_configured, invalidate_memory_configuration

    redis = MagicMock(
        async_get_cache=AsyncMock(side_effect=RedisCircuitBreakerOpenError("open")),
        async_set_cache=AsyncMock(side_effect=RedisCircuitBreakerOpenError("open")),
        async_delete_cache=AsyncMock(side_effect=RedisCircuitBreakerOpenError("open")),
    )
    cache = DualCache()
    prisma_edge.db.litellm_memorypolicy.find_many.return_value = []
    with patch.multiple(  # test-quality-ok: Inject external Redis failure and local worker cache; exercise real fallback.
        "litellm.proxy.proxy_server", user_api_key_cache=cache, redis_usage_cache=redis
    ):
        assert not await gateway_memory_is_configured(prisma_edge, cache)
        prisma_edge.db.litellm_memorypolicy.find_many.return_value = [_POLICY]
        assert await gateway_memory_is_configured(prisma_edge, cache)
        await invalidate_memory_configuration()
    assert prisma_edge.db.litellm_memorypolicy.find_many.await_count == 2
    redis.async_get_cache.assert_awaited_once()


@pytest.mark.asyncio
async def test_full_scope_blocks_creation_but_permits_correction_and_reclaimed_capacity(prisma_edge: MagicMock) -> None:
    table = prisma_edge.db.litellm_memorytable
    table.count.return_value = 1000
    with pytest.raises(HTTPException) as full:
        await store(prisma_edge).capture(_CAPTURE)
    assert full.value.status_code == 429
    table.create.assert_not_awaited()
    assert table.count.call_args.kwargs["where"] == {"namespace": _IDENTITY.namespace("key")}
    prisma_edge.db.execute_raw.assert_awaited_once()
    assert "pg_advisory_xact_lock" in prisma_edge.db.execute_raw.call_args.args[0]
    table.find_unique.side_effect = [row(), row(value="Corrected", updated_at=_NOW + timedelta(seconds=1))]
    corrected = await store(prisma_edge).capture(
        _CAPTURE.model_copy(update={"content": "Corrected", "expected_revision": _NOW})
    )
    assert corrected.content == "Corrected"
    table.count.assert_awaited_once()
    assert await store(prisma_edge).delete("entry")
    table.find_unique.side_effect = None
    table.find_unique.return_value = None
    table.count.return_value = 999
    table.create.return_value = row()
    assert (await store(prisma_edge).capture(_CAPTURE)).memory_id == "entry"
    table.create.assert_awaited_once()
