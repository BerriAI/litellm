"""Failure and authorization boundaries, with only the database/model edges replaced."""

import json
from datetime import datetime, timedelta, timezone
from functools import reduce
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from prisma.models import LiteLLM_MemoryTable
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from litellm.litellm_core_utils.prompt_templates.server_tool_responses import (
    combined_usage,
    executable_server_calls,
    object_items,
)
from litellm.proxy.memory.continuation import MemoryContinuation, MemoryContinuations, prefix_hashes
from litellm.proxy.memory.gateway import GatewayMemoryLoop
from litellm.proxy.memory.knowledge import MEMORY_TOOL_NAMES, execute_memory_tool
from litellm.proxy.memory.policy import MemoryAccess, MemoryIdentity, memory_digest, resolve_memory_access
from litellm.proxy.memory.responses import serve_memory_response
from litellm.proxy.memory.store import MemoryStore
from litellm.types.memory_v2 import MemoryCapture, MemorySearch, MemorySettings

_NOW: Final = datetime(2026, 9, 12, tzinfo=timezone.utc)
_IDENTITY: Final = MemoryIdentity("a" * 64, "owner", "team", "project", "org", False)
_SETTINGS: Final = MemorySettings(enabled=True)
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
    nested = reduce(lambda value, _: {"nested": value}, range(70), {})
    with pytest.raises(HTTPException, match="nesting exceeds") as exc:
        prefix_hashes(({"role": "user", "content": nested},), "acompletion")
    assert exc.value.status_code == 400
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
        permission_revision=access_for().permission_revision,
        replaces=1,
        response={"id": "resp_litellm_memory_test", "output": [{"type": "message", "content": []}]},
        upstream_ids=("native=one",),
    )
    prisma_edge.db.litellm_memorycontinuation.find_first.return_value = (
        None if operation == "missing" else SimpleNamespace(payload=patch.model_dump())
    )

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        pytest.fail("Public reads must not fetch the hidden upstream transcript")

    request = Request({"type": "http", "method": "GET", "path": "/v1/responses/resp_litellm_memory_test"})
    route = "alist_input_items" if operation == "input_items" else "aget_responses"
    if operation == "get":
        response = await serve_memory_response("resp_litellm_memory_test", request, route, store(prisma_edge), app)
        assert isinstance(response, JSONResponse) and json.loads(response.body) == patch.response
    else:
        with pytest.raises(HTTPException) as exc:
            await serve_memory_response("resp_litellm_memory_test", request, route, store(prisma_edge), app)
        assert exc.value.status_code == (404 if operation == "missing" else 501)
    where = prisma_edge.db.litellm_memorycontinuation.find_first.call_args.kwargs["where"]
    assert where["namespace"] == _IDENTITY.namespace and where["key_id"] == _IDENTITY.key_id
    assert where["expires_at"]["gt"] <= datetime.now(timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "already_missing", "upstream_error", "readonly"])
async def test_response_deletion_preserves_auth_paths_and_retry_state(prisma_edge: MagicMock, outcome: str) -> None:
    patch = MemoryContinuation(
        permission_revision=access_for().permission_revision,
        replaces=1,
        response={"id": "resp_litellm_memory_test"},
        upstream_ids=("native=one", "native=two"),
        transcript_anchor="transcript",
    )
    prisma_edge.db.litellm_memorycontinuation.find_first.return_value = SimpleNamespace(payload=patch.model_dump())
    paths: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        paths.append(scope["path"])
        assert scope["raw_path"] == scope["path"].replace("=", "%3D").encode()
        assert Request(scope).headers["authorization"] == "Bearer synthetic-test-credential"
        assert scope["method"] == "DELETE" and scope["query_string"] == b"api-version=test"
        status = (
            502 if outcome == "upstream_error" and len(paths) == 2 else 404 if outcome == "already_missing" else 200
        )
        await JSONResponse({"deleted": True}, status_code=status)(scope, receive, send)

    request = Request(
        {
            "type": "http",
            "method": "DELETE",
            "path": "/v1/responses/resp_litellm_memory_test",
            "headers": [(b"authorization", b"Bearer synthetic-test-credential")],
            "query_string": b"api-version=test",
        }
    )
    identity = MemoryIdentity("a" * 64, "owner", "team", "project", "org", outcome == "readonly")
    if outcome in ("upstream_error", "readonly"):
        with pytest.raises(HTTPException) as exc:
            await serve_memory_response(
                "resp_litellm_memory_test", request, "adelete_responses", store(prisma_edge, identity), app
            )
        assert exc.value.status_code == (403 if outcome == "readonly" else 502)
        prisma_edge.db.litellm_memorycontinuation.delete_many.assert_not_awaited()
    else:
        response = await serve_memory_response(
            "resp_litellm_memory_test", request, "adelete_responses", store(prisma_edge), app
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
    sibling = MemoryIdentity("b" * 64, "owner", "team", "project", "org", False)
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
        identity = MemoryIdentity("a" * 64, "owner", "team", "project", "org", True)
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
                return_value=SimpleNamespace(param_value=MemorySettings(enabled=not revoked).model_dump())
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
        memory, {"id": "a", "name": "litellm_memory_capture", "arguments": {"key": "missing-fields"}}, "checkpoint"
    )
    assert not invalid.reflected and "error" in invalid.output
    missing = await execute_memory_tool(
        memory, {"id": "a", "name": "litellm_memory_read", "arguments": {"id": "missing"}}, "checkpoint"
    )
    assert missing.output == {"error": "Memory not found", "status": 404}
    unknown = await execute_memory_tool(memory, {"id": "a", "name": "other_tool", "arguments": {}}, "checkpoint")
    assert unknown.output == {"error": "Unknown memory tool"}
    prisma_edge.db.litellm_config.find_unique.return_value = None
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
            payload=MemoryContinuation(
                replaces=1, replacement=replacement, permission_revision=access_for().permission_revision
            ).model_dump(),
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
async def test_trailing_system_messages_survive_client_tool_continuation(prisma_edge: MagicMock) -> None:
    provider = FastAPI()
    observed = []
    client_call = {"type": "tool_use", "id": "client_read", "name": "Read", "input": {"path": "README.md"}}

    @provider.post("/v1/messages")
    async def model(incoming: Request):
        body = await incoming.json()
        observed.append(body)
        messages = body["messages"]
        assert json.dumps(body).count('"cache_control"') == 4
        assert all(
            message["role"] != "system" or messages[index + 1]["role"] == "assistant"
            for index, message in enumerate(messages[:-1])
        )
        return {
            "id": "msg_" + str(len(observed)),
            "role": "assistant",
            "type": "message",
            "stop_reason": "tool_use" if len(observed) == 1 else "end_turn",
            "content": [client_call] if len(observed) == 1 else [{"type": "text", "text": "Read complete"}],
        }

    prefix = {
        "role": "user",
        "content": [{"type": "text", "text": "Read README.md", "cache_control": {"type": "ephemeral"}}],
    }
    directive = {
        "role": "system",
        "content": [{"type": "text", "text": "Use concise answers", "cache_control": {"type": "ephemeral"}}],
    }
    earlier_directive = {"role": "system", "content": [{"type": "text", "text": "Use concise answers"}]}
    original = {
        "system": [
            {"type": "text", "text": "Cached prefix " + str(i), "cache_control": {"type": "ephemeral"}}
            for i in range(2)
        ],
        "messages": [prefix, directive],
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "Read"},
    }
    first = GatewayMemoryLoop(provider, request(), original, "anthropic_messages", store(prisma_edge))
    async for _ in first.run():
        pass
    saved = prisma_edge.db.litellm_memorycontinuation.upsert.call_args.kwargs
    prisma_edge.db.litellm_memorycontinuation.find_many.return_value = [
        SimpleNamespace(id=saved["where"]["id"], payload=json.loads(saved["data"]["create"]["payload"]))
    ]
    following = {
        **original,
        "tool_choice": {"type": "none"},
        "messages": [
            prefix,
            earlier_directive,
            {"role": "assistant", "content": [client_call]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "client_read", "content": "File content"}],
            },
            directive,
        ],
    }
    second = GatewayMemoryLoop(provider, request(), following, "anthropic_messages", store(prisma_edge))
    async for _ in second.run():
        pass
    assert len(observed) == 2
    assert observed[0]["messages"][0] == observed[1]["messages"][0] == prefix
    assert observed[1]["messages"].count(directive) == 1
    assert observed[1]["messages"].count(earlier_directive) == 1
    assert observed[1]["messages"].count({"role": "assistant", "content": [client_call]}) == 1
    assert observed[1]["messages"][-1] == directive
    assert observed[1]["messages"][-3]["content"][0]["tool_use_id"] == "client_read"
    assert original["messages"] == [prefix, directive]


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_directive", [False, True])
async def test_claude_output_directives_reach_search_answer_and_reflection_rounds(
    prisma_edge: MagicMock, cached_directive: bool
) -> None:
    provider = FastAPI()
    observed = []
    directive = {
        "role": "system",
        "content": [{"type": "text", "text": "Reply briefly", "cache_control": {"type": "ephemeral"}}]
        if cached_directive
        else [],
        "output_config": {"effort": "low"},
    }

    @provider.post("/v1/messages")
    async def model(incoming: Request):
        body = await incoming.json()
        observed.append(body)
        assert json.dumps(body).count('"cache_control"') == 3 + int(cached_directive)
        last = body["messages"][-1]
        assert last["role"] == "system" and last["output_config"] == {"effort": "low"}
        assert last["content"] == (
            [{"type": "text", "text": "Reply briefly"}]
            if cached_directive and len(observed) > 1
            else directive["content"]
        )
        if len(observed) == 1:
            content = [
                {"type": "tool_use", "id": "search", "name": "litellm_memory_search", "input": {"query": "demo"}}
            ]
        elif len(observed) == 2:
            content = [{"type": "text", "text": "The port is 8347"}]
        else:
            content = [
                {"type": "tool_use", "id": "reflect", "name": "litellm_memory_capture", "input": {"observations": []}}
            ]
        return {
            "id": "msg_" + str(len(observed)),
            "stop_reason": "end_turn" if len(observed) == 2 else "tool_use",
            "content": content,
        }

    original = {
        "system": [
            {"type": "text", "text": "Cached prefix " + str(i), "cache_control": {"type": "ephemeral"}}
            for i in range(2)
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "My demo port?", "cache_control": {"type": "ephemeral"}}],
            },
            directive,
        ],
    }
    loop = GatewayMemoryLoop(provider, request(), original, "anthropic_messages", store(prisma_edge))
    async for _ in loop.run():
        pass
    assert len(observed) == 3
    assert observed[1]["messages"][-2]["content"][0]["tool_use_id"] == "search"
    assert "reflect once" in observed[2]["messages"][-2]["content"]
    assert observed[0]["messages"][0] == original["messages"][0]
    assert sum(message["role"] == "system" for message in object_items(loop.data.get("messages"))) == 1


@pytest.mark.asyncio
async def test_duplicate_directives_preserve_each_current_cache_breakpoint(prisma_edge: MagicMock) -> None:
    first = {"role": "system", "content": [{"type": "text", "text": "Same directive"}]}
    second = {
        "role": "system",
        "content": [{"type": "text", "text": "Same directive", "cache_control": {"type": "ephemeral"}}],
    }
    assistant = {"role": "assistant", "content": [{"type": "text", "text": "Reply"}]}
    items = (first, second, assistant)
    continuations = MemoryContinuations(store(prisma_edge), "anthropic_messages")
    patch = MemoryContinuation(
        permission_revision=access_for().permission_revision,
        replaces=3,
        replacement=({"role": "user", "content": "Memory reference"}, second, first, assistant),
    )
    prisma_edge.db.litellm_memorycontinuation.find_many.return_value = [
        SimpleNamespace(
            id=continuations.identifier(prefix_hashes(items, "anthropic_messages")[-1]), payload=patch.model_dump()
        )
    ]
    restored = await continuations.restore(items)
    assert restored[1:3] == (first, second)
    assert restored[-1] == assistant


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
@pytest.mark.parametrize("between_rounds", [False, True])
@pytest.mark.parametrize("disconnect", [False, True])
async def test_silent_memory_rounds_keep_the_client_alive_and_cancel_upstream(
    prisma_edge: MagicMock, between_rounds: bool, disconnect: bool
) -> None:
    import asyncio
    from unittest.mock import patch

    from starlette.responses import StreamingResponse

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.memory.gateway import process_gateway_memory

    waiting = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def provider(scope: Scope, receive: Receive, send: Send) -> None:
        calls.append(await receive())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        frames = (
            {
                "type": "message_start",
                "message": {
                    "id": "msg_slow",
                    "role": "assistant",
                    "model": "test",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "search",
                    "name": "litellm_memory_search",
                    "input": {"query": "demo"},
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}},
            {"type": "message_stop"},
        )
        if len(calls) == 1:
            for frame in frames if between_rounds else frames[:1]:
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"data: " + json.dumps(frame).encode() + b"\n\n",
                        "more_body": True,
                    }
                )
            if between_rounds:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
        waiting.set()
        try:
            await release.wait()
            raise RuntimeError("private upstream failure")
        finally:
            cancelled.set()

    with (
        patch(  # test-quality-ok: Set the real operator configuration.
            "litellm.sse_keepalive_ping_interval_seconds", 0.01
        ),
        patch.multiple(  # test-quality-ok: Replace the model HTTP boundary, preserving the real internal ASGI transport.
            "litellm.proxy.proxy_server", app=provider, llm_router=None
        ),
        patch(  # test-quality-ok: Inject authorized database edge; execute the real loop, SSE serialization and teardown.
            "litellm.proxy.memory.gateway.gateway_memory_store", new=AsyncMock(return_value=store(prisma_edge))
        ),
    ):
        response = await process_gateway_memory(
            {"model": "test", "stream": True, "messages": []}, request(), UserAPIKeyAuth(), "anthropic_messages"
        )
        assert isinstance(response, StreamingResponse)
        public = response.body_iterator
        assert b"message_start" in await anext(public)
        next_chunk = asyncio.create_task(anext(public))
        await asyncio.wait_for(waiting.wait(), timeout=1)
        assert await asyncio.wait_for(next_chunk, timeout=0.5) == b": ping\n\n"
        if disconnect:
            await public.aclose()
        else:
            release.set()
            remaining = b"".join([chunk async for chunk in public])
            assert remaining.count(b"event: error") == 1
            assert b"private upstream failure" not in remaining and b"message_stop" not in remaining
    assert cancelled.is_set() and len(calls) == (2 if between_rounds else 1)
    prisma_edge.db.litellm_memorycontinuation.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_preserves_upstream_retry_delay_without_exposing_provider_details(prisma_edge: MagicMock) -> None:
    async def provider(scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse({"error": "private provider detail"}, status_code=429, headers={"Retry-After": "17"})(
            scope, receive, send
        )

    loop = GatewayMemoryLoop(provider, request(), {"messages": []}, "anthropic_messages", store(prisma_edge))
    with pytest.raises(HTTPException) as exc:
        async for _ in loop.run():
            pass
    assert exc.value.status_code == 429 and exc.value.headers == {"retry-after": "17"}
    assert "private provider detail" not in str(exc.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"", b'data: {"error": {"message": "private provider detail"}}\n\n'])
async def test_invalid_model_stream_before_first_public_byte_returns_bad_gateway(
    prisma_edge: MagicMock, body: bytes
) -> None:
    from unittest.mock import patch

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.memory.gateway import process_gateway_memory

    async def provider(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    with (
        patch.multiple(  # test-quality-ok: Inject the upstream HTTP boundary, preserving the actual memory loop.
            "litellm.proxy.proxy_server", app=provider, llm_router=None
        ),
        patch(  # test-quality-ok: Inject the authorized persistence edge for a model transport failure.
            "litellm.proxy.memory.gateway.gateway_memory_store", new=AsyncMock(return_value=store(prisma_edge))
        ),
        pytest.raises(HTTPException) as exc,
    ):
        await process_gateway_memory({"stream": True, "messages": []}, request(), UserAPIKeyAuth(), "acompletion")
    assert exc.value.status_code == 502 and "private provider detail" not in str(exc.value.detail)
    prisma_edge.db.litellm_memorycontinuation.upsert.assert_not_awaited()


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
@pytest.mark.parametrize("key_count,used_bytes", [(256, 0), (0, 32 * 1024 * 1024 + 1)])
async def test_continuation_quota_rejects_excess_without_writing(
    prisma_edge: MagicMock, key_count: int, used_bytes: int
) -> None:
    prisma_edge.db.query_raw.return_value = [{"key_count": key_count, "bytes": used_bytes}]
    with pytest.raises(HTTPException) as exc:
        await MemoryContinuations(store(prisma_edge), "aresponses").save("response", MemoryContinuation(replaces=1))
    assert exc.value.status_code == 429
    prisma_edge.db.litellm_memorycontinuation.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_continuation_quota_shares_namespace_lock_across_keys_and_allows_replacements(
    prisma_edge: MagicMock,
) -> None:
    prisma_edge.db.query_raw.return_value = [{"key_count": 255, "bytes": 32 * 1024 * 1024}]
    other_key = MemoryIdentity("b" * 64, "owner", "team", "project", "org", False)
    for identity in (_IDENTITY, other_key):
        continuations = MemoryContinuations(MemoryStore(prisma_edge, access_for(identity)), "aresponses")
        await continuations.save("response", MemoryContinuation(replaces=1, response={"text": "é漢字"}))
        query = prisma_edge.db.query_raw.call_args.args
        assert query[1:4] == (identity.namespace, identity.key_id, [continuations.identifier("response")])
        assert json.loads(query[4])[0]["response"]["text"] == "é漢字"
    locks = prisma_edge.db.execute_raw.call_args_list
    assert locks[0] == locks[1]
    cleanup = prisma_edge.db.litellm_memorycontinuation.delete_many.call_args.kwargs["where"]
    assert cleanup["namespace"] == _IDENTITY.namespace and "key_id" not in cleanup
    assert prisma_edge.db.litellm_memorycontinuation.upsert.await_count == 2


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
        assert await process_gateway_memory({"messages": []}, request(), caller, "acompletion") is None
        with pytest.raises(HTTPException) as exc:
            await process_gateway_memory(
                {"previous_response_id": "resp_litellm_memory_private"}, request(), caller, "aresponses"
            )
        assert exc.value.status_code == 404
    prisma_edge.db.litellm_memorytable.find_many.assert_not_awaited()
    prisma_edge.db.litellm_memorytable.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("acompletion", "aresponses", "anthropic_messages"))
@pytest.mark.parametrize("stream", (False, True))
async def test_structured_output_hides_preparation_and_restores_final_constraints(
    prisma_edge: MagicMock, route: str, stream: bool
) -> None:
    from starlette.responses import StreamingResponse

    from litellm.litellm_core_utils.prompt_templates.server_tool_stream import sse_bytes

    provider = FastAPI()
    observed = []
    prisma_edge.db.litellm_memorytable.find_first.return_value = row()
    schema = {"type": "object", "properties": {"port": {"type": "integer"}}, "required": ["port"]}
    formatting = (
        {"response_format": {"type": "json_schema", "json_schema": {"name": "port", "schema": schema}}}
        if route == "acompletion"
        else {"text": {"format": {"type": "json_schema", "name": "port", "schema": schema}}}
        if route == "aresponses"
        else {"output_config": {"format": {"type": "json_schema", "schema": schema}, "effort": "low"}}
    )
    function = {"name": "client_tool", "description": "Client tool", "parameters": {"type": "object"}}
    client_tool = (
        {"type": "function", "function": function}
        if route == "acompletion"
        else {"type": "function", **function}
        if route == "aresponses"
        else {"name": "client_tool", "input_schema": {"type": "object"}}
    )
    original = {
        "model": "test",
        "stream": stream,
        "tools": [client_tool],
        **formatting,
        **({"input": "My port?"} if route == "aresponses" else {"messages": [{"role": "user", "content": "My port?"}]}),
    }
    endpoint = {
        "acompletion": "/v1/chat/completions",
        "aresponses": "/v1/responses",
        "anthropic_messages": "/v1/messages",
    }[route]

    @provider.post(endpoint)
    async def model(incoming: Request):
        body = await incoming.json()
        observed.append(body)
        index = len(observed)
        final = index == 4
        if final:
            assert body["tools"] == [client_tool] and "tool_choice" not in body
            assert all(body[key] == value for key, value in formatting.items())
            assert body["stream"] is stream
            assert "8347" in json.dumps(body)
        else:
            assert body["stream"] is False
            assert "client_tool" not in json.dumps(body["tools"])
            assert "json_schema" not in json.dumps({key: body.get(key) for key in formatting})
            if route == "anthropic_messages":
                assert body["output_config"] == {"effort": "low"}
        name = "litellm_memory_read" if index == 1 else "litellm_memory_capture"
        arguments = {"id": "entry"} if index == 1 else {"observations": [], "checkpoint": loop.checkpoint}
        text = '{"port":8347}' if final else "Hidden preparation draft"
        call = index <= 2
        usage = (
            {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
            if route == "acompletion"
            else {"input_tokens": 10, "output_tokens": 1}
        )
        if route == "acompletion":
            message = {
                "role": "assistant",
                **(
                    {
                        "tool_calls": [
                            {
                                "id": "call",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ]
                    }
                    if call
                    else {"content": text}
                ),
            }
            response = {
                "id": f"chat_{index}",
                "model": "test",
                "created": 1,
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if call else "stop"}],
                "usage": usage,
            }
            events = (
                {
                    **response,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
                    "usage": None,
                },
                {
                    **response,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            )
        elif route == "aresponses":
            item = (
                {
                    "type": "function_call",
                    "id": f"fc_{index}",
                    "call_id": f"call_{index}",
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
                if call
                else {
                    "id": f"item_{index}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            )
            response = {
                "id": f"resp_{index}",
                "object": "response",
                "status": "completed",
                "output": [item],
                "usage": usage,
            }
            events = (
                {"type": "response.created", "response": {**response, "output": []}},
                {"type": "response.output_item.added", "output_index": 0, "item": item},
                {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": text},
                {"type": "response.completed", "response": response},
            )
        else:
            response = {
                "id": f"msg_{index}",
                "type": "message",
                "role": "assistant",
                "model": "test",
                "stop_reason": "tool_use" if call else "end_turn",
                "content": [{"type": "tool_use", "id": f"call_{index}", "name": name, "input": arguments}]
                if call
                else [{"type": "text", "text": text}],
                "usage": usage,
            }
            events = (
                {
                    "type": "message_start",
                    "message": {**response, "content": [], "usage": {"input_tokens": 10, "output_tokens": 0}},
                },
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                {"type": "message_stop"},
            )
        if final and stream:
            return StreamingResponse(iter(sse_bytes(event) for event in events), media_type="text/event-stream")
        return response

    loop = GatewayMemoryLoop(
        provider, Request({**request().scope, "path": endpoint}), original, route, store(prisma_edge)
    )
    chunks = [chunk async for chunk in loop.run()]
    public = loop.stream.response()
    actual = (
        public["choices"][0]["message"]["content"]
        if route == "acompletion"
        else public["output"][0]["content"][0]["text"]
        if route == "aresponses"
        else public["content"][0]["text"]
    )
    assert json.loads(actual) == {"port": 8347}
    assert "Hidden preparation draft" not in json.dumps(public) and b"Hidden preparation draft" not in b"".join(chunks)
    assert len(observed) == len(loop.upstream_ids) == 4
    assert public["usage"]["prompt_tokens" if route == "acompletion" else "input_tokens"] == 40
    assert public["usage"]["completion_tokens" if route == "acompletion" else "output_tokens"] == 4
    assert original["tools"] == [client_tool]
    if stream:
        wire = b"".join(chunks)
        assert b"litellm_memory_read" not in wire and b"litellm_memory_capture" not in wire
        if route == "aresponses":
            assert wire.count(b'"type": "response.created"') == 1
            assert wire.count(b'"type": "response.completed"') == 1
        elif route == "anthropic_messages":
            assert wire.count(b'"type": "message_start"') == 1
            assert wire.count(b'"type": "message_stop"') == 1
