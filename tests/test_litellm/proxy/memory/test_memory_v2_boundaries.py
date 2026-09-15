"""Failure and authorization boundaries, with only the database/model edges replaced."""

import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request
from prisma.models import LiteLLM_MemoryTable
from starlette.responses import JSONResponse, Response, StreamingResponse

from litellm.constants import DEFAULT_MEMORY_CAPTURE_INSTRUCTIONS
from litellm.litellm_core_utils.prompt_templates.server_tool_responses import (
    combined_usage,
    executable_server_calls,
    object_items,
)
from litellm.litellm_core_utils.prompt_templates.server_tools import ServerToolRoute
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.memory.continuation import MemoryContinuation
from litellm.proxy.memory.gateway import GatewayMemoryLoop
from litellm.proxy.memory.knowledge import MEMORY_TOOL_NAMES, execute_memory_tool
from litellm.proxy.memory.policy import MemoryAccess, MemoryIdentity, memory_digest, resolve_memory_access
from litellm.proxy.memory.responses import serve_memory_response
from litellm.proxy.memory.store import MemoryStore
from litellm.proxy.memory.transport import in_gateway_round
from litellm.types.memory_v2 import MemoryCapture, MemoryEnrollment, MemorySearch, MemorySettings

_NOW: Final = datetime(2026, 9, 12, tzinfo=timezone.utc)
_IDENTITY: Final = MemoryIdentity("a" * 64, "owner", "team", "org", False)
_SETTINGS: Final = MemorySettings(enabled=True, read=MemoryEnrollment(enabled=True))
_CAPTURE: Final = MemoryCapture(key="demo", title="Demo", content="Use port 8347", evidence="User selected this port")


@pytest.fixture
def prisma_edge() -> MagicMock:
    client = MagicMock()
    client.db.litellm_config.find_unique = AsyncMock(return_value=SimpleNamespace(param_value=_SETTINGS.model_dump()))
    client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    table = client.db.litellm_memorytable
    table.find_unique = AsyncMock(return_value=None)
    table.find_first = AsyncMock(return_value=None)
    table.find_many = AsyncMock(return_value=[])
    table.create = AsyncMock()
    table.count = AsyncMock(return_value=0)
    client.db.tx.return_value.__aenter__.return_value = client.db
    client.db.execute_raw = AsyncMock()
    client.db.query_raw = AsyncMock(return_value=[{"key_count": 0, "bytes": 0}])
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
    return MemoryStore(client, access_for(identity))


def access_for(identity: MemoryIdentity = _IDENTITY) -> MemoryAccess:
    return MemoryAccess(
        identity, _SETTINGS, permission_revision=memory_digest(identity.namespace, identity.role, "False")
    )


def row(**changes: object) -> LiteLLM_MemoryTable:
    return LiteLLM_MemoryTable.model_validate(
        {
            "memory_id": "entry",
            "key": f"memory-v2:{_IDENTITY.namespace}:demo",
            "namespace": _IDENTITY.namespace,
            "user_id": "owner",
            "team_id": "team",
            "organization_id": "org",
            "owner_key_id": "a" * 64,
            "value": _CAPTURE.content,
            "metadata": json.dumps({"title": _CAPTURE.title, "evidence": _CAPTURE.evidence}),
            "created_at": _NOW,
            "updated_at": _NOW,
            **changes,
        }
    )


def test_nested_conversation_and_cyclic_usage_fail_with_bounded_errors() -> None:
    usage: dict[str, object] = {}
    usage["details"] = usage
    with pytest.raises(ValueError, match="nesting exceeds"):
        combined_usage((usage,))


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "input_items", "missing"])
async def test_saved_response_reads_are_scoped_and_never_return_internal_input(
    prisma_edge: MagicMock, operation: str
) -> None:
    patch = MemoryContinuation(
        permission_revision=access_for().continuation_revision,
        response={"id": "resp_litellm_memory_test", "output": [{"type": "message", "content": []}]},
        upstream_ids=("native=one",),
    )
    prisma_edge.db.litellm_memorycontinuation.find_first.return_value = (
        None if operation == "missing" else SimpleNamespace(payload=patch.model_dump())
    )

    async def app(inner: Request, body: dict[str, object], auth: UserAPIKeyAuth) -> JSONResponse:
        pytest.fail("Public reads must not fetch the hidden upstream transcript")

    request = Request({"type": "http", "method": "GET", "path": "/v1/responses/resp_litellm_memory_test"})
    route = "alist_input_items" if operation == "input_items" else "aget_responses"
    if operation == "get":
        response = await serve_memory_response(
            "resp_litellm_memory_test", request, route, store(prisma_edge), app, UserAPIKeyAuth()
        )
        assert isinstance(response, JSONResponse) and json.loads(response.body) == patch.response
    else:
        with pytest.raises(HTTPException) as exc:
            await serve_memory_response(
                "resp_litellm_memory_test", request, route, store(prisma_edge), app, UserAPIKeyAuth()
            )
        assert exc.value.status_code == (404 if operation == "missing" else 501)
    where = prisma_edge.db.litellm_memorycontinuation.find_first.call_args.kwargs["where"]
    assert where["namespace"] == _IDENTITY.namespace and where["key_id"] == _IDENTITY.key_id
    assert where["expires_at"]["gt"] <= datetime.now(timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "already_missing", "missing_exception", "upstream_error", "readonly"])
async def test_response_deletion_preserves_auth_paths_and_retry_state(prisma_edge: MagicMock, outcome: str) -> None:
    patch = MemoryContinuation(
        permission_revision=access_for().continuation_revision,
        response={"id": "resp_litellm_memory_test"},
        upstream_ids=("native=one", "native=two"),
    )
    prisma_edge.db.litellm_memorycontinuation.find_first.return_value = SimpleNamespace(payload=patch.model_dump())
    paths: list[str] = []

    async def app(inner: Request, body: dict[str, object], auth: UserAPIKeyAuth) -> JSONResponse:
        scope = inner.scope
        paths.append(scope["path"])
        if outcome == "missing_exception":
            raise HTTPException(status_code=404, detail="Not found")
        assert scope["raw_path"] == scope["path"].replace("=", "%3D").encode()
        assert Request(scope).headers["authorization"] == "Bearer synthetic-test-credential"
        assert scope["method"] == "DELETE" and scope["query_string"] == b"api-version=test"
        status = (
            502 if outcome == "upstream_error" and len(paths) == 2 else 404 if outcome == "already_missing" else 200
        )
        return JSONResponse({"deleted": True}, status_code=status)

    request = Request(
        {
            "type": "http",
            "method": "DELETE",
            "path": "/v1/responses/resp_litellm_memory_test",
            "headers": [(b"authorization", b"Bearer synthetic-test-credential")],
            "query_string": b"api-version=test",
        }
    )
    identity = MemoryIdentity("a" * 64, "owner", "team", "org", outcome == "readonly")
    if outcome in ("upstream_error", "readonly"):
        with pytest.raises(HTTPException) as exc:
            await serve_memory_response(
                "resp_litellm_memory_test",
                request,
                "adelete_responses",
                store(prisma_edge, identity),
                app,
                UserAPIKeyAuth(),
            )
        assert exc.value.status_code == (403 if outcome == "readonly" else 502)
        prisma_edge.db.litellm_memorycontinuation.delete_many.assert_not_awaited()
    else:
        response = await serve_memory_response(
            "resp_litellm_memory_test", request, "adelete_responses", store(prisma_edge), app, UserAPIKeyAuth()
        )
        assert isinstance(response, JSONResponse)
        assert json.loads(response.body) == {
            "id": "resp_litellm_memory_test",
            "object": "response.deleted",
            "deleted": True,
        }
        prisma_edge.db.litellm_memorycontinuation.delete_many.assert_awaited_once()
    assert paths == ([] if outcome == "readonly" else ["/v1/responses/native=one", "/v1/responses/native=two"])
    prisma_edge.db.litellm_memorytable.delete_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_flat_enrollment_follows_the_user_across_keys(prisma_edge: MagicMock) -> None:
    config = prisma_edge.db.litellm_config.find_unique
    config.return_value = None
    assert not (await resolve_memory_access(prisma_edge, _IDENTITY)).active
    config.return_value = SimpleNamespace(
        param_value=MemorySettings(enabled=True, everyone=False, user_ids=("owner",)).model_dump()
    )
    assert (await resolve_memory_access(prisma_edge, _IDENTITY)).active
    sibling = MemoryIdentity("b" * 64, "owner", "team", "org", False)
    assert (await resolve_memory_access(prisma_edge, sibling)).active
    config.return_value = SimpleNamespace(
        param_value=MemorySettings(enabled=True, everyone=False, user_ids=("other",)).model_dump()
    )
    assert not (await resolve_memory_access(prisma_edge, _IDENTITY)).active


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["disabled", "unenrolled", "missing", "readonly"])
async def test_store_rechecks_access_before_writing(prisma_edge: MagicMock, change: str) -> None:
    identity = _IDENTITY
    if change == "missing":
        prisma_edge.db.litellm_config.find_unique.return_value = None
    elif change == "readonly":
        identity = MemoryIdentity("a" * 64, "owner", "team", "org", True)
    else:
        config = MemorySettings(enabled=change != "disabled", everyone=False, user_ids=("other",))
        prisma_edge.db.litellm_config.find_unique.return_value = SimpleNamespace(param_value=config.model_dump())
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
    assert query["where"]["AND"][0]["AND"][0]["namespace"] == {"startswith": "v2:"}
    assert query["take"] == 128


@pytest.mark.asyncio
async def test_read_and_delete_cannot_address_another_namespace(prisma_edge: MagicMock) -> None:
    memory = store(prisma_edge)
    with pytest.raises(HTTPException) as exc:
        await memory.read("foreign-entry")
    assert exc.value.status_code == 404
    assert prisma_edge.db.litellm_memorytable.find_first.call_args.kwargs["where"] == {
        "AND": [access_for().visible_rows(), {"memory_id": "foreign-entry"}],
    }
    prisma_edge.db.litellm_config.find_unique.return_value = SimpleNamespace(param_value=MemorySettings().model_dump())
    assert await memory.delete("entry")
    assert prisma_edge.db.litellm_memorytable.delete_many.call_args.kwargs["where"] == {
        "AND": [access_for().visible_rows(write=True), {"memory_id": "entry"}],
    }
    with pytest.raises(HTTPException) as inactive:
        await memory.read("entry")
    assert inactive.value.status_code == 403


@pytest.mark.asyncio
async def test_identical_capture_is_idempotent_and_new_capture_has_scoped_identity(prisma_edge: MagicMock) -> None:
    from litellm.proxy.db.prisma_client import PrismaWrapper

    table = prisma_edge.db.litellm_memorytable
    table.create.return_value = row()
    wrapped: Final = MemoryStore(SimpleNamespace(db=PrismaWrapper(prisma_edge.db)), access_for())
    saved = await wrapped.capture(_CAPTURE)
    assert saved.content == _CAPTURE.content
    data = table.create.call_args.kwargs["data"]
    assert data["namespace"] == _IDENTITY.namespace and data["user_id"] == "owner" and data["team_id"] == "team"
    assert data["key"].startswith("memory-v2:" + _IDENTITY.namespace + ":")
    table.find_unique.return_value = row()
    assert await wrapped.capture(_CAPTURE) == saved
    table.create.assert_awaited_once()
    table.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("revoked", [False, True])
async def test_capture_rechecks_policy_on_its_transaction_connection(prisma_edge: MagicMock, revoked: bool) -> None:
    prisma_edge.db.litellm_memorytable.create.return_value = row()
    prisma_edge.db.litellm_config.find_unique.side_effect = [
        SimpleNamespace(param_value=_SETTINGS.model_dump()),
        RuntimeError("The only pooled connection belongs to the active transaction"),
    ]
    transaction = SimpleNamespace(
        litellm_memorytable=prisma_edge.db.litellm_memorytable,
        litellm_config=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(
                    param_value=MemorySettings(enabled=not revoked, read=MemoryEnrollment(enabled=True)).model_dump()
                )
            )
        ),
        litellm_usertable=prisma_edge.db.litellm_usertable,
        litellm_teamtable=prisma_edge.db.litellm_teamtable,
        execute_raw=AsyncMock(),
    )
    prisma_edge.db.tx.return_value.__aenter__.return_value = transaction
    if revoked:
        with pytest.raises(HTTPException) as exc:
            await store(prisma_edge).capture(_CAPTURE)
        assert exc.value.status_code == 403
    else:
        assert (await store(prisma_edge).capture(_CAPTURE)).content == _CAPTURE.content


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["missing", "foreign", "stale", "concurrent"])
async def test_capture_rejects_stale_or_conflicting_replacements(prisma_edge: MagicMock, race: str) -> None:
    table = prisma_edge.db.litellm_memorytable
    table.find_unique.return_value = (
        None if race == "missing" else row(namespace="foreign" if race == "foreign" else _IDENTITY.namespace)
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
            where["namespace"] == _IDENTITY.namespace
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
        memory, {"id": "a", "name": "litellm_memory_capture", "arguments": {"key": "missing-fields"}}, ()
    )
    assert "error" in invalid
    missing = await execute_memory_tool(
        memory, {"id": "a", "name": "litellm_memory_read", "arguments": {"id": "missing"}}, ()
    )
    assert missing == {"error": "Memory not found", "status": 404}
    unknown = await execute_memory_tool(memory, {"id": "a", "name": "other_tool", "arguments": {}}, ())
    assert unknown == {"error": "Unknown memory tool"}
    prisma_edge.db.litellm_config.find_unique.return_value = None
    revoked = await execute_memory_tool(
        memory, {"id": "a", "name": "litellm_memory_capture", "arguments": {"observations": []}}, ()
    )
    assert "error" in revoked
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
    read_only: Final = MemoryIdentity("a" * 64, "owner", "team", "org", True)
    loop: Final = GatewayMemoryLoop(
        AsyncMock(),
        request(),
        {"messages": [{"role": "user", "content": "Hello"}]},
        "anthropic_messages",
        store(prisma_edge, read_only),
        UserAPIKeyAuth(),
    )
    await loop.prepare()
    assert "litellm_memory_capture" not in str(loop.data)
    assert {tool["name"] for tool in object_items(loop.data["tools"])} == MEMORY_TOOL_NAMES - {"litellm_memory_capture"}
    forced: Final = GatewayMemoryLoop(
        AsyncMock(),
        request(),
        {"tool_choice": {"type": "none"}, "messages": []},
        "anthropic_messages",
        store(prisma_edge),
        UserAPIKeyAuth(),
    )
    await forced.prepare()
    assert forced.data["tool_choice"] == {"type": "none"}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["acompletion", "aresponses", "anthropic_messages"])
@pytest.mark.parametrize("save,read", [(True, False), (True, True), (False, True)])
async def test_admin_capture_guidance_is_stable_and_only_injected_when_saving(
    prisma_edge: MagicMock, route: ServerToolRoute, save: bool, read: bool
) -> None:
    guidance: Final = "Remember only architecture decisions, including {service} ownership."
    configured: Final = MemorySettings(enabled=save, read=MemoryEnrollment(enabled=read), capture_instructions=guidance)
    prisma_edge.db.litellm_config.find_unique.return_value = SimpleNamespace(param_value=configured.model_dump())
    access: Final = await resolve_memory_access(prisma_edge, _IDENTITY)
    original: Final = {"input" if route == "aresponses" else "messages": [{"role": "user", "content": "Hello"}]}
    snapshot: Final = json.dumps(original)
    loops: Final = tuple(
        GatewayMemoryLoop(AsyncMock(), request(), original, route, MemoryStore(prisma_edge, access), UserAPIKeyAuth())
        for _ in range(2)
    )
    for loop in loops:
        await loop.prepare()
        payload: Final = json.dumps(loop.data)
        assert payload.count(guidance) == int(save)
        assert "Each observation must quote its evidence verbatim" in payload if save else guidance not in payload
        assert DEFAULT_MEMORY_CAPTURE_INSTRUCTIONS not in payload
        assert "litellm_memory_capture" in payload if save else "litellm_memory_capture" not in payload
    for field in ("input", "messages", "instructions", "system", "tools"):
        assert loops[0].data.get(field) == loops[1].data.get(field)
    assert json.dumps(original) == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["acompletion", "aresponses", "anthropic_messages"])
async def test_disabled_memory_tool_names_cannot_be_intercepted_as_application_tools(
    prisma_edge: MagicMock, route: ServerToolRoute
) -> None:
    configure = MemorySettings(enabled=True)
    prisma_edge.db.litellm_config.find_unique.return_value = SimpleNamespace(param_value=configure.model_dump())
    access = await resolve_memory_access(prisma_edge, _IDENTITY)
    client_tool = {"name": "litellm_memory_read"}
    loop = GatewayMemoryLoop(
        AsyncMock(),
        request(),
        {"tools": [{"type": "function", "function": client_tool}] if route == "acompletion" else [client_tool]},
        route,
        MemoryStore(prisma_edge, access),
        UserAPIKeyAuth(),
    )
    with pytest.raises(ValueError, match="conflicts"):
        await loop.prepare()


@pytest.mark.parametrize("arguments,status", (('{"query":', "completed"),))
def test_incomplete_or_malformed_memory_arguments_never_become_executable(arguments: str, status: str) -> None:
    with pytest.raises(ValueError, match="Invalid JSON"):
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
@pytest.mark.parametrize("writer_unavailable", [False, True])
async def test_replica_lag_cannot_authorize_memory_after_primary_revocation(
    prisma_edge: MagicMock, writer_unavailable: bool
) -> None:
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer = MagicMock(spec=PrismaWrapper)
    reader = MagicMock(spec=PrismaWrapper)
    writer.litellm_config = SimpleNamespace(find_unique=AsyncMock(return_value=None))
    reader.litellm_config = SimpleNamespace(
        find_unique=AsyncMock(return_value=SimpleNamespace(param_value=_SETTINGS.model_dump()))
    )
    writer.litellm_usertable = prisma_edge.db.litellm_usertable
    writer.litellm_teamtable = prisma_edge.db.litellm_teamtable
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
    reader.litellm_config.find_unique.assert_not_awaited()
    with pytest.raises(HTTPException) as exc:
        await MemoryStore(client, access_for()).capture(_CAPTURE)
    assert exc.value.status_code == 403
    writer.litellm_memorytable.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_unconfigured_gate_caches_presence_without_caching_authorization(prisma_edge: MagicMock) -> None:
    from litellm.caching.caching import DualCache
    from litellm.proxy.memory.policy import gateway_memory_is_configured

    cache = DualCache()
    config = prisma_edge.db.litellm_config.find_unique
    config.return_value = None
    assert not await gateway_memory_is_configured(prisma_edge, cache)
    assert not await gateway_memory_is_configured(prisma_edge, cache)
    config.assert_awaited_once()
    assert config.call_args.kwargs == {"where": {"param_name": "memory_v2"}}
    enabled_cache = DualCache()
    config.return_value = SimpleNamespace(param_value=_SETTINGS.model_dump())
    assert await gateway_memory_is_configured(prisma_edge, enabled_cache)
    config.return_value = None
    assert await gateway_memory_is_configured(prisma_edge, enabled_cache)
    assert not (await resolve_memory_access(prisma_edge, _IDENTITY)).active


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
    config = prisma_edge.db.litellm_config.find_unique
    with patch.multiple(  # test-quality-ok: Inject external worker caches and Redis; run real invalidation.
        "litellm.proxy.proxy_server", user_api_key_cache=backend_cache, redis_usage_cache=redis
    ):
        config.return_value = None
        assert not await gateway_memory_is_configured(prisma_edge, gateway_cache)
        assert not await gateway_memory_is_configured(prisma_edge, gateway_cache)
        config.assert_awaited_once()
        config.return_value = SimpleNamespace(param_value=_SETTINGS.model_dump())
        await invalidate_memory_configuration()
        assert await gateway_memory_is_configured(prisma_edge, gateway_cache)
        assert config.await_count == 2


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
    prisma_edge.db.litellm_config.find_unique.return_value = None
    with patch.multiple(  # test-quality-ok: Inject external Redis failure and local worker cache; exercise real fallback.
        "litellm.proxy.proxy_server", user_api_key_cache=cache, redis_usage_cache=redis
    ):
        assert not await gateway_memory_is_configured(prisma_edge, cache)
        prisma_edge.db.litellm_config.find_unique.return_value = SimpleNamespace(param_value=_SETTINGS.model_dump())
        assert await gateway_memory_is_configured(prisma_edge, cache)
        await invalidate_memory_configuration()
    assert prisma_edge.db.litellm_config.find_unique.await_count == 2
    redis.async_get_cache.assert_awaited_once()


@pytest.mark.asyncio
async def test_full_scope_blocks_creation_but_permits_correction_and_reclaimed_capacity(prisma_edge: MagicMock) -> None:
    table = prisma_edge.db.litellm_memorytable
    table.count.return_value = 1000
    with pytest.raises(HTTPException) as full:
        await store(prisma_edge).capture(_CAPTURE)
    assert full.value.status_code == 429
    table.create.assert_not_awaited()
    assert table.count.call_args.kwargs["where"] == {"namespace": _IDENTITY.namespace}
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


@pytest.mark.asyncio
async def test_memory_lookup_failure_leaves_inference_unchanged_but_never_leaks_owned_response_ids(
    prisma_edge: MagicMock,
) -> None:
    from unittest.mock import patch

    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.memory.gateway import process_gateway_memory

    prisma_edge.db.litellm_config.find_unique.side_effect = RuntimeError("database unavailable")
    caller = UserAPIKeyAuth(user_id="owner", token="a" * 64)
    with patch.multiple(  # test-quality-ok: Inject unavailable external DB and an empty worker cache.
        "litellm.proxy.proxy_server", prisma_client=prisma_edge, user_api_key_cache=DualCache()
    ):
        assert await process_gateway_memory({"messages": []}, request(), caller, "acompletion", AsyncMock()) is None
        with pytest.raises(HTTPException) as exc:
            await process_gateway_memory(
                {"previous_response_id": "resp_litellm_memory_private"}, request(), caller, "aresponses", AsyncMock()
            )
        assert exc.value.status_code == 404
    prisma_edge.db.litellm_memorytable.find_many.assert_not_awaited()
    prisma_edge.db.litellm_memorytable.create.assert_not_awaited()


_ROUTES: Final = ("acompletion", "aresponses", "anthropic_messages")


@pytest.mark.asyncio
@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("configuration", ("absent", "disabled", "no_database"))
async def test_disabled_memory_leaves_inference_untouched(
    prisma_edge: MagicMock, route: ServerToolRoute, streaming: bool, configuration: str
) -> None:
    from unittest.mock import patch

    from litellm.caching.caching import DualCache
    from litellm.proxy.memory.gateway import process_gateway_memory

    prisma_edge.db.litellm_config.find_unique.return_value = (
        None if configuration == "absent" else SimpleNamespace(param_value=MemorySettings().model_dump())
    )
    data: Final = {
        "messages": [{"role": "system", "content": "Original instructions"}, {"role": "user", "content": "hi"}],
        "input": "hi",
        "stream": streaming,
        "caching": True,
        "tools": [{"type": "function", "function": {"name": "litellm_memory_search"}}],
    }
    original: Final = json.dumps(data)
    execute: Final = AsyncMock()
    with patch.multiple(  # test-quality-ok: Inject the external configuration DB and worker cache.
        "litellm.proxy.proxy_server",
        prisma_client=None if configuration == "no_database" else prisma_edge,
        user_api_key_cache=DualCache(),
    ):
        assert await process_gateway_memory(data, request(), UserAPIKeyAuth(user_id="owner"), route, execute) is None
    assert json.dumps(data) == original
    execute.assert_not_awaited()
    prisma_edge.db.litellm_memorytable.find_many.assert_not_awaited()
    prisma_edge.db.litellm_memorytable.create.assert_not_awaited()
    prisma_edge.db.litellm_memorycontinuation.upsert.assert_not_awaited()
    prisma_edge.db.litellm_usertable.find_unique.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", _ROUTES)
async def test_malformed_nonstreaming_provider_body_returns_502(prisma_edge: MagicMock, route: ServerToolRoute) -> None:
    from unittest.mock import patch

    from litellm.caching.caching import DualCache
    from litellm.proxy.memory.gateway import process_gateway_memory

    execute = AsyncMock(return_value=Response(b"not JSON", media_type="text/html"))
    caller = UserAPIKeyAuth(user_id="owner", token="a" * 64)
    with patch.multiple(  # test-quality-ok: Use external database, cache, and provider response boundaries.
        "litellm.proxy.proxy_server", prisma_client=prisma_edge, user_api_key_cache=DualCache(), llm_router=None
    ):
        with pytest.raises(HTTPException) as exc:
            await process_gateway_memory({"messages": [], "input": "hi"}, request(), caller, route, execute)
        assert exc.value.status_code == 502
    execute.assert_awaited_once()


@pytest.mark.parametrize(
    "choice",
    [
        {"type": "tool", "name": "litellm_memory_search"},
        {"type": "function", "name": "litellm_memory_read"},
        {"type": "function", "function": {"name": "litellm_memory_capture"}},
    ],
)
def test_clients_cannot_force_private_memory_rounds(choice: Mapping[str, object]) -> None:
    from litellm.proxy.memory.gateway import validate_memory_request

    with pytest.raises(HTTPException) as exc:
        validate_memory_request({"tool_choice": choice}, request())
    assert exc.value.status_code == 400


def provider_response(
    route: ServerToolRoute, text: str, calls: tuple[Mapping[str, object], ...] = (), truncated: bool = False
) -> dict[str, object]:
    if route == "acompletion":
        return {
            "id": "chat-test",
            "object": "chat.completion",
            "model": "test",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length" if truncated else "tool_calls" if calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                            }
                            for call in calls
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }
    if route == "anthropic_messages":
        return {
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [
                {"type": "text", "text": text},
                *[
                    {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["arguments"]}
                    for call in calls
                ],
            ],
            "stop_reason": "max_tokens" if truncated else "tool_use" if calls else "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }
    return {
        "id": "resp-native",
        "object": "response",
        "model": "test",
        "status": "incomplete" if truncated else "completed",
        "output": [
            {
                "type": "message",
                "id": "answer",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
            *[
                {
                    "type": "function_call",
                    "id": call["id"],
                    "call_id": call["id"],
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"]),
                }
                for call in calls
            ],
        ],
        "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
    }


def wire_response(body: dict[str, object], route: ServerToolRoute, streaming: bool) -> Response:
    if not streaming:
        return JSONResponse(body)
    if route == "acompletion":
        choice: Final = body["choices"][0]
        events: Final = (
            {
                **body,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            **choice["message"],
                            "tool_calls": [
                                {**call, "index": index} for index, call in enumerate(choice["message"]["tool_calls"])
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                **body,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
            },
        )
    elif route == "anthropic_messages":
        events = (
            {"type": "message_start", "message": {**body, "content": []}},
            *(
                event
                for index, block in enumerate(body["content"])
                for event in (
                    {"type": "content_block_start", "index": index, "content_block": block},
                    {"type": "content_block_stop", "index": index},
                )
            ),
            {"type": "message_delta", "delta": {"stop_reason": body["stop_reason"]}, "usage": body["usage"]},
            {"type": "message_stop"},
        )
    else:
        events = (
            {"type": "response.created", "response": {**body, "output": [], "status": "in_progress"}},
            *(
                event
                for index, item in enumerate(body["output"])
                for event in (
                    {"type": "response.output_item.added", "output_index": index, "item": item},
                    {"type": "response.output_item.done", "output_index": index, "item": item},
                )
            ),
            {
                "type": "response.incomplete" if body["status"] == "incomplete" else "response.completed",
                "response": body,
            },
        )

    async def chunks():
        for event in events:
            yield ("event: " + str(event.get("type", "message")) + "\ndata: " + json.dumps(event) + "\n\n").encode()

    return StreamingResponse(chunks(), media_type="text/event-stream")


@pytest.mark.asyncio
@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("truncated", (False, True))
async def test_plain_answer_needs_one_round_and_preserves_truncation(
    prisma_edge: MagicMock, route: ServerToolRoute, streaming: bool, truncated: bool
) -> None:
    execute: Final = AsyncMock(
        return_value=wire_response(provider_response(route, "", truncated=truncated), route, streaming)
    )
    loop: Final = GatewayMemoryLoop(
        execute,
        request(),
        {"messages": [{"role": "user", "content": "hi"}], "input": "hi", "stream": streaming, "store": False},
        route,
        store(prisma_edge),
        UserAPIKeyAuth(),
    )
    chunks: Final = b"".join([chunk async for chunk in loop.run()])
    assert execute.await_count == 1
    result: Final = loop.stream.response()
    terminal: Final = (
        result["choices"][0]["finish_reason"]
        if route == "acompletion"
        else result["stop_reason"]
        if route == "anthropic_messages"
        else result["status"]
    )
    assert terminal == (
        {"acompletion": "length", "anthropic_messages": "max_tokens", "aresponses": "incomplete"}[route]
        if truncated
        else {"acompletion": "stop", "anthropic_messages": "end_turn", "aresponses": "completed"}[route]
    )
    if streaming:
        assert terminal.encode() in chunks
    prisma_edge.db.litellm_memorycontinuation.upsert.assert_not_awaited()
    prisma_edge.db.litellm_memorytable.create.assert_not_awaited()
    assert "checkpoint" not in json.dumps(execute.call_args.args[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("client_tool", (False, True))
async def test_search_is_private_and_client_tools_keep_their_ids(
    prisma_edge: MagicMock, route: ServerToolRoute, streaming: bool, client_tool: bool
) -> None:
    observed: Final = []
    search: Final = {"id": "memory-1", "name": "litellm_memory_read", "arguments": {"id": "entry"}}
    application: Final = {"id": "client-1", "name": "Read", "arguments": {"path": "README.md"}}
    prisma_edge.db.litellm_memorytable.find_first.return_value = row()

    async def execute(inner: Request, body: dict[str, object], auth: UserAPIKeyAuth) -> Response:
        observed.append(body)
        assert in_gateway_round()
        if len(observed) == 1:
            reply: Final = provider_response(
                route, "INTERNAL HOUSEKEEPING", (search, application) if client_tool else (search,)
            )
        else:
            assert "Use port 8347" in json.dumps(body)
            reply = provider_response(route, "8347")
        return wire_response(reply, route, streaming)

    loop: Final = GatewayMemoryLoop(
        execute,
        request(),
        {
            "messages": [{"role": "user", "content": "Which port?"}],
            "input": "Which port?",
            "stream": streaming,
            "store": False,
        },
        route,
        store(prisma_edge),
        UserAPIKeyAuth(),
    )
    chunks: Final = b"".join([chunk async for chunk in loop.run()])
    result: Final = json.dumps(loop.stream.response())
    assert len(observed) == (1 if client_tool else 2)
    assert "INTERNAL HOUSEKEEPING" not in result
    if streaming:
        assert b"INTERNAL HOUSEKEEPING" not in chunks
        assert b"litellm_memory_read" not in chunks
        assert (b"client-1" if client_tool else b"8347") in chunks
    assert ("client-1" if client_tool else "8347") in result
    assert loop.stream.response()["usage"]["prompt_tokens" if route == "acompletion" else "input_tokens"] == (
        10 if client_tool else 20
    )


@pytest.mark.asyncio
async def test_chat_never_uses_continuation_capacity(prisma_edge: MagicMock) -> None:
    execute: Final = AsyncMock(return_value=JSONResponse(provider_response("acompletion", "Hello")))
    for _ in range(270):
        loop: Final = GatewayMemoryLoop(
            execute,
            request(),
            {"messages": [{"role": "user", "content": "hi"}]},
            "acompletion",
            store(prisma_edge),
            UserAPIKeyAuth(),
        )
        async for _ in loop.run():
            pass
        assert loop.stream.response()["choices"][0]["message"]["content"] == "Hello"
    assert execute.await_count == 270
    prisma_edge.db.litellm_memorycontinuation.upsert.assert_not_awaited()
    prisma_edge.db.litellm_memorycontinuation.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_retention_failure_does_not_discard_paid_answer(prisma_edge: MagicMock) -> None:
    prisma_edge.db.litellm_memorycontinuation.upsert.side_effect = RuntimeError("storage unavailable")
    execute: Final = AsyncMock(return_value=JSONResponse(provider_response("aresponses", "Answered")))
    loop: Final = GatewayMemoryLoop(
        execute, request(), {"input": "hi"}, "aresponses", store(prisma_edge), UserAPIKeyAuth()
    )
    async for _ in loop.run():
        pass
    assert loop.stream.response()["output"][0]["content"][0]["text"] == "Answered"
    assert loop.stream.response()["store"] is False


@pytest.mark.asyncio
async def test_capture_rejects_fabricated_evidence_and_deduplicates_across_keys(prisma_edge: MagicMock) -> None:
    observation: Final = {
        "title": "Use port 8347",
        "content": "Use port 8347 for the deployment",
        "evidence": "Use port 8347",
        "kind": "context",
        "scope": "deployment",
        "certainty": "user_stated",
        "when_to_use": "Deploying the application",
    }
    call: Final = {"id": "memory-1", "name": "litellm_memory_capture", "arguments": {"observations": [observation]}}
    memory: Final = store(prisma_edge)
    rejected: Final = await execute_memory_tool(memory, call, ({"role": "user", "content": "What is two plus two?"},))
    assert "exact evidence quote" in rejected["error"]
    prisma_edge.db.litellm_memorytable.create.assert_not_awaited()
    prisma_edge.db.litellm_memorytable.create.return_value = row()
    saved: Final = await execute_memory_tool(memory, call, ({"role": "user", "content": "Use port 8347"},))
    assert saved["saved"] == 1
    prisma_edge.db.litellm_memorytable.find_unique.return_value = row(
        key=prisma_edge.db.litellm_memorytable.create.call_args.kwargs["data"]["key"],
        metadata=json.dumps(prisma_edge.db.litellm_memorytable.create.call_args.kwargs["data"]["metadata"].data),
        value=observation["content"],
    )
    repeated: Final = await execute_memory_tool(memory, call, ({"role": "user", "content": "Use port 8347"},))
    assert repeated["saved"] == 1
    assert prisma_edge.db.litellm_memorytable.create.await_count == 1


@pytest.mark.asyncio
async def test_rounds_share_trace_and_keep_live_auth_objects(prisma_edge: MagicMock) -> None:
    from opentelemetry.sdk.trace import TracerProvider

    span = TracerProvider().get_tracer(__name__).start_span("memory")
    auth = UserAPIKeyAuth(parent_otel_span=span)
    observed = []

    async def execute(inner: Request, body: dict[str, object], round_auth: UserAPIKeyAuth) -> Response:
        assert round_auth.parent_otel_span is span
        observed.append(body)
        calls = (
            ({"id": "read", "name": "litellm_memory_read", "arguments": {"id": "entry"}},) if len(observed) == 1 else ()
        )
        return JSONResponse(provider_response("acompletion", "answer", calls))

    loop = GatewayMemoryLoop(
        execute,
        request(),
        {"messages": [{"role": "user", "content": "Recall the port"}]},
        "acompletion",
        store(prisma_edge),
        auth,
    )
    async for _ in loop.run():
        pass
    assert len(observed) == 2
    assert observed[0]["litellm_trace_id"] == observed[1]["litellm_trace_id"]
    assert observed[0]["litellm_call_id"] != observed[1]["litellm_call_id"]
    span.end()


@pytest.mark.asyncio
async def test_previous_response_uses_owned_upstream_and_pending_tool_outputs(prisma_edge: MagicMock) -> None:
    pending = {"type": "function_call_output", "call_id": "memory-call", "output": "Memory saved"}
    patch = MemoryContinuation(
        permission_revision=access_for().continuation_revision,
        response={"id": "resp_litellm_memory_owned"},
        upstream_ids=("native-first", "native-last"),
        pending_results=(pending,),
    )
    prisma_edge.db.litellm_memorycontinuation.find_first.return_value = SimpleNamespace(payload=patch.model_dump())
    execute = AsyncMock(return_value=JSONResponse(provider_response("aresponses", "answer")))
    loop = GatewayMemoryLoop(
        execute,
        request(),
        {
            "previous_response_id": "resp_litellm_memory_owned",
            "input": [{"type": "function_call_output", "call_id": "client-call", "output": "File contents"}],
            "store": False,
        },
        "aresponses",
        store(prisma_edge),
        UserAPIKeyAuth(),
    )
    async for _ in loop.run():
        pass
    body = execute.call_args.args[1]
    assert body["previous_response_id"] == "native-last"
    assert pending in body["input"]
    assert any(item.get("call_id") == "client-call" for item in body["input"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    (
        b"",
        b"<html>private provider error</html>",
        b'data: {"error":"private provider error"}\n\n',
        b'{"error":"private provider error"}',
    ),
)
@pytest.mark.parametrize("status", (429, 502))
@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("streaming", (False, True))
async def test_error_response_retains_status_and_retry_after_without_parsing_provider_body(
    prisma_edge: MagicMock, body: bytes, status: int, route: ServerToolRoute, streaming: bool
) -> None:
    from unittest.mock import patch

    from litellm.caching.caching import DualCache
    from litellm.proxy.memory.gateway import process_gateway_memory

    execute = AsyncMock(return_value=Response(body, status_code=status, headers={"retry-after": "7"}))
    caller = UserAPIKeyAuth(user_id="owner", token="a" * 64)
    with patch.multiple(  # test-quality-ok: Inject external database, cache, and provider error responses.
        "litellm.proxy.proxy_server", prisma_client=prisma_edge, user_api_key_cache=DualCache(), llm_router=None
    ):
        with pytest.raises(HTTPException) as error:
            await process_gateway_memory(
                {"messages": [{"role": "user", "content": "hi"}], "input": "hi", "stream": streaming},
                request(),
                caller,
                route,
                execute,
            )
    assert error.value.status_code == status
    assert error.value.headers["retry-after"] == "7"
    assert "private provider" not in error.value.detail
