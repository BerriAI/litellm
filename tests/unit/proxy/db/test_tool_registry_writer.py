"""
Unit tests for tool_registry_writer.py — uses a mock prisma client
that exposes litellm_tooltable.upsert / find_many / find_unique.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.errors import PrismaError, RawQueryError

from litellm.constants import DB_SPEND_UPDATE_JOB_NAME, REDIS_TOOL_DISCOVERY_BUFFER_KEY, TOOL_DISCOVERY_COMMIT_JOB_NAME
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.db.db_transaction_queue.tool_discovery_queue import ToolDiscoveryQueue
from litellm.proxy.db.tool_registry_writer import (
    ToolPolicyRegistry,
    batch_upsert_tools,
    commit_tool_discoveries,
    get_tool,
    get_tool_policy_registry,
    get_tools_by_names,
    list_tools,
    update_tool_policy,
)
from litellm.types.caching import RedisPipelineLpopOperation, RedisPipelineRpushOperation


def _mock_row(**kwargs):
    """Build a row-like object with real attributes (no MagicMock) for _row_to_model."""

    class Row:
        pass

    default = {
        "tool_id": "uuid-1",
        "tool_name": "my_tool",
        "origin": "user_defined",
        "input_policy": "untrusted",
        "output_policy": "untrusted",
        "call_count": 1,
        "assignments": {},
        "key_hash": None,
        "team_id": None,
        "key_alias": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "created_by": None,
        "updated_by": None,
    }
    default.update(kwargs)
    row = Row()
    for k, v in default.items():
        setattr(row, k, v)
    return row


def _make_prisma(
    *,
    upsert_return=None,
    find_many_rows=None,
    find_unique_row=None,
    key_rows=(),
    user_rows=(),
):
    """Return a mock prisma_client with litellm_tooltable.upsert, find_many, find_unique."""
    prisma = MagicMock()
    prisma.db.litellm_tooltable = MagicMock()
    prisma.db.litellm_tooltable.upsert = AsyncMock(return_value=upsert_return)
    prisma.db.execute_raw = AsyncMock(return_value=1)
    prisma.db.litellm_tooltable.find_many = AsyncMock(
        return_value=find_many_rows if find_many_rows is not None else []
    )
    prisma.db.litellm_tooltable.find_unique = AsyncMock(return_value=find_unique_row)
    prisma.db.litellm_verificationtoken = MagicMock()
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=list(key_rows))
    prisma.db.litellm_usertable = MagicMock()
    prisma.db.litellm_usertable.find_many = AsyncMock(return_value=list(user_rows))
    return prisma


def _statement(prisma) -> tuple[str, tuple[object, ...]]:
    prisma.db.execute_raw.assert_awaited_once()
    sql, *params = prisma.db.execute_raw.call_args.args
    return sql, tuple(params)


@pytest.mark.asyncio
async def test_batch_upsert_tools_inserts_a_new_tool_with_the_batch_call_count_and_adds_it_on_conflict():
    prisma = _make_prisma()
    await batch_upsert_tools(
        prisma, [{"tool_name": "tool_a", "origin": "mcp_server", "created_by": None, "call_count": 4}]
    )
    sql, params = _statement(prisma)
    assert sql.startswith('INSERT INTO "LiteLLM_ToolTable"')
    assert 'ON CONFLICT ("tool_name") DO UPDATE SET' in sql
    assert '"call_count" = "LiteLLM_ToolTable"."call_count" + EXCLUDED."call_count"' in sql
    assert '"input_policy"' not in sql, "policies come from the table defaults and are never overwritten"
    assert params[1:] == ("tool_a", "mcp_server", 4, "system", "system", None, None, None, None)
    prisma.db.litellm_tooltable.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_upsert_tools_empty_list():
    prisma = _make_prisma()
    await batch_upsert_tools(prisma, [])
    prisma.db.execute_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_upsert_tools_skips_empty_names():
    prisma = _make_prisma()
    items = [{"tool_name": "", "origin": None}, {"tool_name": None}]  # type: ignore[list-item]
    await batch_upsert_tools(prisma, items)
    prisma.db.execute_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_upsert_many_tools_is_one_statement_with_one_row_per_distinct_tool():
    prisma = _make_prisma()
    items = [
        {"tool_name": "tool_a", "origin": "mcp_server", "created_by": None, "call_count": 2},
        {"tool_name": "tool_b", "origin": "user_defined", "created_by": "alice", "call_count": 1},
        {"tool_name": "tool_a", "origin": "user_defined", "created_by": "bob", "call_count": 5},
    ]
    await batch_upsert_tools(prisma, items)
    sql, params = _statement(prisma)
    assert sql.count("$1::text") == 1 and "$11::text" in sql and "$21::text" not in sql
    assert params[1:10] == ("tool_a", "mcp_server", 7, "system", "system", None, None, None, None)
    assert params[11:20] == ("tool_b", "user_defined", 1, "alice", "alice", None, None, None, None)


@pytest.mark.asyncio
async def test_list_tools_no_filter():
    row = _mock_row(
        tool_id="id1",
        tool_name="tool_a",
        origin="mcp",
        input_policy="untrusted",
        output_policy="untrusted",
        call_count=5,
    )
    prisma = _make_prisma(find_many_rows=[row])
    result = await list_tools(prisma)
    assert len(result) == 1
    assert result[0].tool_name == "tool_a"
    assert result[0].call_count == 5
    prisma.db.litellm_tooltable.find_many.assert_awaited_once()
    call_kw = prisma.db.litellm_tooltable.find_many.call_args.kwargs
    assert call_kw["where"] == {}
    assert call_kw["order"] == {"created_at": "desc"}


@pytest.mark.asyncio
async def test_list_tools_attaches_the_owner_of_the_discovering_key():
    owned = _mock_row(tool_id="id1", tool_name="owned_tool", key_hash="hash-owned")
    orphan = _mock_row(tool_id="id2", tool_name="orphan_tool", key_hash="hash-orphan")
    unknown_owner = _mock_row(tool_id="id3", tool_name="unknown_owner_tool", key_hash="hash-unknown-owner")
    keyless = _mock_row(tool_id="id4", tool_name="keyless_tool", key_hash=None)
    prisma = _make_prisma(
        find_many_rows=[owned, orphan, unknown_owner, keyless],
        key_rows=[
            {"token": "hash-owned", "user_id": "user-1"},
            {"token": "hash-orphan", "user_id": None},
            {"token": "hash-unknown-owner", "user_id": "user-gone"},
        ],
        user_rows=[{"user_id": "user-1", "user_email": "one@example.com", "user_alias": "One"}],
    )
    result = await list_tools(prisma)
    assert [tool.model_dump(include={"tool_name", "user"}) for tool in result] == [
        {
            "tool_name": "owned_tool",
            "user": {"user_id": "user-1", "user_email": "one@example.com", "user_alias": "One"},
        },
        {"tool_name": "orphan_tool", "user": None},
        {"tool_name": "unknown_owner_tool", "user": None},
        {"tool_name": "keyless_tool", "user": None},
    ]
    key_where = prisma.db.litellm_verificationtoken.find_many.call_args.kwargs["where"]
    assert key_where == {"token": {"in": ["hash-orphan", "hash-owned", "hash-unknown-owner"]}}
    user_where = prisma.db.litellm_usertable.find_many.call_args.kwargs["where"]
    assert user_where == {"user_id": {"in": ["user-1", "user-gone"]}}


@pytest.mark.asyncio
async def test_list_tools_keeps_tools_without_owners_when_the_owner_lookup_fails():
    prisma = _make_prisma(find_many_rows=[_mock_row(tool_name="my_tool", key_hash="hash-owned")])
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(side_effect=PrismaError("verification token table down"))
    result = await list_tools(prisma)
    assert [tool.model_dump(include={"tool_name", "user"}) for tool in result] == [
        {"tool_name": "my_tool", "user": None}
    ]


@pytest.mark.asyncio
async def test_list_tools_skips_owner_lookup_when_no_tool_has_a_key_hash():
    prisma = _make_prisma(find_many_rows=[_mock_row(key_hash=None)])
    result = await list_tools(prisma)
    assert [tool.user for tool in result] == [None]
    prisma.db.litellm_verificationtoken.find_many.assert_not_awaited()
    prisma.db.litellm_usertable.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_tools_with_input_policy_filter():
    row = _mock_row(
        tool_id="id1",
        tool_name="blocked_tool",
        origin=None,
        input_policy="blocked",
        output_policy="untrusted",
        call_count=2,
        assignments=None,
    )
    prisma = _make_prisma(find_many_rows=[row])
    result = await list_tools(prisma, input_policy="blocked")
    assert result[0].input_policy == "blocked"
    call_kw = prisma.db.litellm_tooltable.find_many.call_args.kwargs
    assert call_kw["where"] == {"input_policy": "blocked"}


@pytest.mark.asyncio
async def test_get_tool_found():
    row = _mock_row(tool_name="my_tool")
    prisma = _make_prisma(find_unique_row=row)
    result = await get_tool(prisma, "my_tool")
    assert result is not None
    assert result.tool_name == "my_tool"
    prisma.db.litellm_tooltable.find_unique.assert_awaited_once_with(
        where={"tool_name": "my_tool"}
    )


@pytest.mark.asyncio
async def test_get_tool_attaches_the_owner_of_the_discovering_key():
    row = _mock_row(tool_name="my_tool", key_hash="hash-owned")
    prisma = _make_prisma(
        find_unique_row=row,
        key_rows=[{"token": "hash-owned", "user_id": "user-1"}],
        user_rows=[{"user_id": "user-1", "user_email": "one@example.com", "user_alias": "One"}],
    )
    result = await get_tool(prisma, "my_tool")
    assert result is not None
    assert result.model_dump(include={"tool_name", "user"}) == {
        "tool_name": "my_tool",
        "user": {"user_id": "user-1", "user_email": "one@example.com", "user_alias": "One"},
    }
    key_where = prisma.db.litellm_verificationtoken.find_many.call_args.kwargs["where"]
    assert key_where == {"token": {"in": ["hash-owned"]}}


@pytest.mark.asyncio
async def test_get_tool_not_found():
    prisma = _make_prisma(find_unique_row=None)
    result = await get_tool(prisma, "nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_update_tool_policy_calls_upsert_then_get_tool():
    row = _mock_row(
        tool_name="my_tool",
        input_policy="blocked",
        output_policy="untrusted",
        updated_by="admin",
    )
    prisma = _make_prisma(find_unique_row=row)
    result = await update_tool_policy(
        prisma, "my_tool", updated_by="admin", input_policy="blocked"
    )
    assert result is not None
    assert result.input_policy == "blocked"
    prisma.db.litellm_tooltable.upsert.assert_awaited_once()
    call_kw = prisma.db.litellm_tooltable.upsert.call_args.kwargs
    assert call_kw["where"] == {"tool_name": "my_tool"}
    assert call_kw["data"]["update"]["input_policy"] == "blocked"
    assert call_kw["data"]["update"]["updated_by"] == "admin"
    prisma.db.litellm_tooltable.find_unique.assert_awaited_with(where={"tool_name": "my_tool"})


@pytest.mark.asyncio
async def test_get_tools_by_names_returns_policy_map():
    rows = [
        _mock_row(tool_name="tool_a", input_policy="trusted", output_policy="untrusted"),
        _mock_row(tool_name="tool_b", input_policy="blocked", output_policy="untrusted"),
    ]
    prisma = _make_prisma(find_many_rows=rows)
    result = await get_tools_by_names(prisma, ["tool_a", "tool_b"])
    assert result == {
        "tool_a": ("trusted", "untrusted"),
        "tool_b": ("blocked", "untrusted"),
    }
    prisma.db.litellm_tooltable.find_many.assert_awaited_once_with(where={"tool_name": {"in": ["tool_a", "tool_b"]}})


@pytest.mark.asyncio
async def test_get_tools_by_names_empty_list():
    prisma = _make_prisma()
    result = await get_tools_by_names(prisma, [])
    assert result == {}
    prisma.db.litellm_tooltable.find_many.assert_not_awaited()


# --- ToolPolicyRegistry ---


def _mock_tool_row(
    tool_name: str,
    input_policy: str = "untrusted",
    output_policy: str = "untrusted",
):
    row = MagicMock()
    row.tool_name = tool_name
    row.input_policy = input_policy
    row.output_policy = output_policy
    return row


def _mock_perm_row(object_permission_id: str, blocked_tools: list):
    row = MagicMock()
    row.object_permission_id = object_permission_id
    row.blocked_tools = blocked_tools
    return row


@pytest.mark.asyncio
async def test_tool_policy_registry_sync_and_get_effective_policies():
    """Registry syncs from DB; get_effective_policies returns merged blocked + global."""
    prisma = MagicMock()
    prisma.db.litellm_tooltable.find_many = AsyncMock(
        return_value=[
            _mock_tool_row("tool_a", input_policy="trusted"),
            _mock_tool_row("tool_b", input_policy="blocked"),
            _mock_tool_row("tool_c", input_policy="untrusted"),
        ]
    )
    prisma.db.litellm_objectpermissiontable.find_many = AsyncMock(
        return_value=[
            _mock_perm_row("op-key-1", ["tool_a"]),
            _mock_perm_row("op-team-1", ["tool_c"]),
        ]
    )
    registry = get_tool_policy_registry()
    await registry.sync_tool_policy_from_db(prisma)
    assert registry.is_initialized()
    # Key blocked: tool_a. Team blocked: tool_c. Global: tool_b blocked.
    result = registry.get_effective_policies(
        ["tool_a", "tool_b", "tool_c"],
        object_permission_id="op-key-1",
        team_object_permission_id="op-team-1",
    )
    assert result["tool_a"] == "blocked"
    assert result["tool_b"] == "blocked"
    assert result["tool_c"] == "blocked"
    # No op ids: only global
    result_global = registry.get_effective_policies(["tool_a", "tool_b", "tool_c"])
    assert result_global["tool_a"] == "trusted"
    assert result_global["tool_b"] == "blocked"
    assert result_global["tool_c"] == "untrusted"


@pytest.mark.asyncio
async def test_tool_policy_registry_not_initialized_returns_untrusted():
    """When not synced, get_effective_policies still returns untrusted for unknown tools."""
    registry = ToolPolicyRegistry()
    assert not registry.is_initialized()
    result = registry.get_effective_policies(["unknown_tool"])
    assert result == {"unknown_tool": "untrusted"}


@pytest.mark.asyncio
async def test_sync_tool_policy_from_db_retries_on_transport_error_first_read():
    """`ToolPolicyRegistry.sync_tool_policy_from_db` self-heals across one
    ClientNotConnectedError on the tools read — the perms read still fires
    after the recovery and the registry initializes cleanly."""
    import prisma as prisma_pkg

    registry = ToolPolicyRegistry()
    invocations: list = []

    async def _flaky_find_many():
        invocations.append(None)
        if len(invocations) == 1:
            raise prisma_pkg.errors.ClientNotConnectedError()
        return []

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_tooltable.find_many = AsyncMock(side_effect=_flaky_find_many)
    mock_prisma_client.db.litellm_objectpermissiontable.find_many = AsyncMock(return_value=[])
    mock_prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)
    mock_prisma_client._db_auth_reconnect_timeout_seconds = 2.0
    mock_prisma_client._db_auth_reconnect_lock_timeout_seconds = 0.1

    await registry.sync_tool_policy_from_db(mock_prisma_client)

    assert len(invocations) == 2
    mock_prisma_client.attempt_db_reconnect.assert_awaited_once()
    reconnect_kwargs = mock_prisma_client.attempt_db_reconnect.await_args.kwargs
    assert reconnect_kwargs["reason"] == "sync_tool_policy_from_db_tools_lookup_failure"
    assert registry.is_initialized()


@pytest.mark.asyncio
async def test_sync_tool_policy_from_db_retries_on_transport_error_second_read():
    """Same as above but the blip happens on the perms read — distinct reason
    tag in telemetry confirms the second wrap is wired separately."""
    import prisma as prisma_pkg

    registry = ToolPolicyRegistry()
    perms_invocations: list = []

    async def _flaky_perms_find_many():
        perms_invocations.append(None)
        if len(perms_invocations) == 1:
            raise prisma_pkg.errors.ClientNotConnectedError()
        return []

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_tooltable.find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_objectpermissiontable.find_many = AsyncMock(side_effect=_flaky_perms_find_many)
    mock_prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)
    mock_prisma_client._db_auth_reconnect_timeout_seconds = 2.0
    mock_prisma_client._db_auth_reconnect_lock_timeout_seconds = 0.1

    await registry.sync_tool_policy_from_db(mock_prisma_client)

    assert len(perms_invocations) == 2
    mock_prisma_client.attempt_db_reconnect.assert_awaited_once()
    reconnect_kwargs = mock_prisma_client.attempt_db_reconnect.await_args.kwargs
    assert reconnect_kwargs["reason"] == "sync_tool_policy_from_db_perms_lookup_failure"


class _InMemoryRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.lists: dict[str, list[object]] = {}

    async def async_set_cache(self, key: str, value: str, nx: bool = False, **kwargs: object) -> bool | None:
        if nx and key in self.store:
            return None
        self.store[key] = json.dumps(value)
        return True

    async def async_get_cache(self, key: str, **kwargs: object) -> str | None:
        raw = self.store.get(key)
        return json.loads(raw) if raw is not None else None

    async def async_delete_cache(self, key: str) -> int:
        return 1 if self.store.pop(key, None) is not None else 0

    async def async_rpush_pipeline(self, rpush_list: list[RedisPipelineRpushOperation]) -> list[int]:
        for op in rpush_list:
            self.lists.setdefault(op["key"], []).extend(op["values"])
        return [len(self.lists[op["key"]]) for op in rpush_list]

    async def async_lpop_pipeline(self, lpop_list: list[RedisPipelineLpopOperation]) -> list[list[object] | None]:
        return [self.lists.pop(op["key"], None) for op in lpop_list]


def _postgres_rejection(sqlstate: str) -> RawQueryError:
    return RawQueryError(
        data={"user_facing_error": {"error_code": "P2010", "meta": {"code": sqlstate, "message": "db error"}}}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lock_held_by_another_pod",
    [None, DB_SPEND_UPDATE_JOB_NAME, TOOL_DISCOVERY_COMMIT_JOB_NAME],
    ids=["free", "spend_lock_held", "tool_lock_held"],
)
async def test_commit_with_redis_writes_every_workers_tools_in_one_statement_from_the_tool_lock_holder_only(
    lock_held_by_another_pod: str | None,
):
    redis = _InMemoryRedis()
    if lock_held_by_another_pod is not None:
        assert await PodLockManager(redis_cache=redis).acquire_lock(cronjob_id=lock_held_by_another_pod)
    other_worker = ToolDiscoveryQueue()
    other_worker.add_update({"tool_name": "get_weather", "origin": "mcp_server"})
    other_worker.add_update({"tool_name": "search", "origin": "user_defined"})
    await RedisUpdateBuffer(redis_cache=redis).store_tool_discoveries_in_redis(other_worker)

    queue = ToolDiscoveryQueue()
    queue.add_update({"tool_name": "get_weather", "origin": "user_defined"})
    queue.add_update({"tool_name": "get_weather", "origin": "user_defined"})
    prisma_client = MagicMock()
    prisma_client.db.execute_raw = AsyncMock(return_value=2)
    prisma_client.db.litellm_tooltable.upsert = AsyncMock()

    await commit_tool_discoveries(
        prisma_client=prisma_client,
        tool_discovery_queue=queue,
        redis_update_buffer=RedisUpdateBuffer(redis_cache=redis),
        pod_lock_manager=PodLockManager(redis_cache=redis),
        use_redis_buffer=True,
    )

    assert queue.flush() == [], "every worker hands its tools to redis each tick"
    prisma_client.db.litellm_tooltable.upsert.assert_not_awaited()
    if lock_held_by_another_pod == TOOL_DISCOVERY_COMMIT_JOB_NAME:
        prisma_client.db.execute_raw.assert_not_awaited()
        assert len(redis.lists[REDIS_TOOL_DISCOVERY_BUFFER_KEY]) == 2, "non-holders leave their tools for the holder"
        return
    assert REDIS_TOOL_DISCOVERY_BUFFER_KEY not in redis.lists, "the spend lock must not gate the tool drain"
    prisma_client.db.execute_raw.assert_awaited_once()
    sql, *params = prisma_client.db.execute_raw.call_args.args
    assert sql.startswith('INSERT INTO "LiteLLM_ToolTable"') and 'ON CONFLICT ("tool_name")' in sql
    rows = {params[i + 1]: params[i + 3] for i in range(0, len(params), 10)}
    assert rows == {"get_weather": 3, "search": 1}, f"one row per tool summed across workers, got {params}"


@pytest.mark.asyncio
async def test_commit_without_redis_writes_only_this_workers_tools_in_one_statement_and_never_touches_redis():
    redis = _InMemoryRedis()
    queue = ToolDiscoveryQueue()
    queue.add_update({"tool_name": "get_weather", "origin": "user_defined"})
    queue.add_update({"tool_name": "get_weather", "origin": "user_defined"})
    queue.add_update({"tool_name": "search", "origin": "user_defined"})
    prisma_client = MagicMock()
    prisma_client.db.execute_raw = AsyncMock(return_value=2)

    await commit_tool_discoveries(
        prisma_client=prisma_client,
        tool_discovery_queue=queue,
        redis_update_buffer=RedisUpdateBuffer(redis_cache=redis),
        pod_lock_manager=PodLockManager(redis_cache=redis),
        use_redis_buffer=False,
    )

    assert redis.lists == {}, "without the buffer flag nothing goes through redis"
    assert queue.flush() == []
    prisma_client.db.execute_raw.assert_awaited_once()
    sql, *params = prisma_client.db.execute_raw.call_args.args
    assert 'ON CONFLICT ("tool_name")' in sql
    rows = {params[i + 1]: params[i + 3] for i in range(0, len(params), 10)}
    assert rows == {"get_weather": 2, "search": 1}, f"got {params}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "requeued"),
    [
        (Exception("connection reset"), True),
        (_postgres_rejection("23502"), False),
    ],
    ids=["connection_error_goes_back_to_redis", "rejected_row_is_dropped"],
)
async def test_failed_tool_table_write_requeues_the_rows_only_when_they_provably_did_not_apply(
    failure: Exception, requeued: bool
):
    redis = _InMemoryRedis()
    queue = ToolDiscoveryQueue()
    queue.add_update({"tool_name": "get_weather", "origin": "user_defined"})
    prisma_client = MagicMock()
    prisma_client.db.execute_raw = AsyncMock(side_effect=failure)

    await commit_tool_discoveries(
        prisma_client=prisma_client,
        tool_discovery_queue=queue,
        redis_update_buffer=RedisUpdateBuffer(redis_cache=redis),
        pod_lock_manager=PodLockManager(redis_cache=redis),
        use_redis_buffer=True,
    )

    prisma_client.db.execute_raw.assert_awaited_once()
    left_in_redis = [json.loads(entry) for entry in redis.lists.get(REDIS_TOOL_DISCOVERY_BUFFER_KEY, [])]
    expected = [[{"tool_name": "get_weather", "origin": "user_defined", "call_count": 1}]] if requeued else []
    assert left_in_redis == expected, f"requeued={requeued} but redis holds {left_in_redis}"
