import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm._auth.failure import NativeAuthFailure
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.repositories.native_auth_repository import NativeAuthRepository, native_auth_repository


@pytest.mark.asyncio
async def test_native_key_preserves_joined_budget_and_membership_projection():
    raw = {
        "token": "token-hash",
        "user_id": "user",
        "organization_id": "key-org",
        "org_id": "team-org",
        "team_models": None,
        "team_blocked": None,
        "max_budget": None,
        "litellm_budget_table_max_budget": 25.0,
        "model_max_budget": {},
        "litellm_budget_table_model_max_budget": {"model": {"max_budget": 1.0}},
        "team_model_aliases": {"alias": "model"},
        "team_members_with_roles": [{"user_id": "user", "role": "user"}, {"user_id": "user", "role": "admin"}],
        "expires": datetime(2027, 1, 1, tzinfo=timezone.utc),
    }
    handle = SimpleNamespace(auth_read=AsyncMock(return_value=[raw]))
    row = await NativeAuthRepository(handle).key("token-hash", writer=True)

    assert row is not None
    assert (row.org_id, row.team_models, row.team_blocked, row.max_budget) == ("key-org", [], False, 25.0)
    assert row.model_max_budget == {"model": {"max_budget": 1.0}}
    assert row.team_model_aliases == {"alias": "model"}
    assert row.team_member is not None and row.team_member.role == "admin"
    assert row.expires == "2027-01-01T00:00:00+00:00"
    assert row.last_refreshed_at is not None
    assert raw["max_budget"] is None and raw["team_models"] is None
    handle.auth_read.assert_awaited_once_with("key", ["token-hash"], writer=True)


@pytest.mark.asyncio
async def test_native_rows_keep_null_limits_and_make_request_local_models():
    raw = {"user_id": "u", "team_id": "t", "spend": None, "litellm_budget_table": {"max_budget": None}}
    handle = SimpleNamespace(auth_read=AsyncMock(return_value=[raw]))
    repo = NativeAuthRepository(handle)
    first = await repo.one("member", ["u", "t"], LiteLLM_TeamMembership)
    second = await repo.one("member", ["u", "t"], LiteLLM_TeamMembership)

    assert first is not None and second is not None
    assert first.spend is None and second.spend is None
    assert first.litellm_budget_table is not None and second.litellm_budget_table is not None
    first.litellm_budget_table.max_budget = 3.0
    assert second.litellm_budget_table.max_budget is None
    assert raw["litellm_budget_table"] == {"max_budget": None}


@pytest.mark.asyncio
async def test_native_large_id_batch_is_chunked_deduplicated_and_keeps_missing_rows_absent():
    async def read(kind, ids, writer=False):
        assert kind == "budget" and writer is True
        assert len(ids) <= 1000
        return [{"budget_id": entity_id} for entity_id in ids if entity_id != "missing"]

    ids = tuple(f"budget-{i}" for i in range(1001))
    handle = SimpleNamespace(auth_read=AsyncMock(side_effect=read))
    rows = await NativeAuthRepository(handle).many(
        "budget", (*ids, ids[0], "missing"), LiteLLM_BudgetTable, writer=True
    )
    assert tuple(row.budget_id for row in rows) == ids
    assert [len(call.args[1]) for call in handle.auth_read.call_args_list] == [1000, 2]


@pytest.mark.asyncio
async def test_native_large_batch_error_never_returns_partial_rows():
    from litellm.repositories.native_auth_repository import NativeAuthDatabaseError

    handle = SimpleNamespace(auth_read=AsyncMock(side_effect=[[{"budget_id": "first"}], OSError("offline")]))
    with pytest.raises(NativeAuthDatabaseError):
        await NativeAuthRepository(handle).many("budget", tuple(str(i) for i in range(1001)), LiteLLM_BudgetTable)
    assert handle.auth_read.await_count == 2


@pytest.mark.asyncio
async def test_native_empty_batch_preserves_native_empty_read_contract():
    handle = SimpleNamespace(auth_read=AsyncMock(return_value=[]))
    rows = await NativeAuthRepository(handle).many("budget", (), LiteLLM_BudgetTable)
    assert rows == ()
    handle.auth_read.assert_awaited_once_with("budget", [], writer=False)


@pytest.mark.asyncio
async def test_native_missing_row_is_none_and_zero_budget_is_not_missing():
    handle = SimpleNamespace(auth_read=AsyncMock(side_effect=[[], [{"budget_id": "b", "max_budget": 0.0}]]))
    repo = NativeAuthRepository(handle)
    assert await repo.one("budget", ["missing"], LiteLLM_BudgetTable) is None
    found = await repo.one("budget", ["b"], LiteLLM_BudgetTable)
    assert found is not None and found.max_budget == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [None, {}, ["bad"], [{"max_budget": "bad"}]])
async def test_native_invalid_rows_are_terminal(answer):
    handle = SimpleNamespace(auth_read=AsyncMock(return_value=answer))
    with pytest.raises(NativeAuthFailure):
        await NativeAuthRepository(handle).one("budget", ["b"], LiteLLM_BudgetTable)
    assert handle.auth_read.await_count == 1


@pytest.mark.asyncio
async def test_native_failure_and_cancellation_are_not_clean_misses():
    handle = SimpleNamespace(auth_read=AsyncMock(side_effect=RuntimeError("connection closed")))
    with pytest.raises(NativeAuthFailure, match="read failed"):
        await NativeAuthRepository(handle).key("hash")
    cancelled = SimpleNamespace(auth_read=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await NativeAuthRepository(cancelled).key("hash")


@pytest.mark.asyncio
async def test_native_timeout_keeps_deadline_identity_and_terminal_native_semantics():
    from litellm.proxy.db.db_lookup_gate import DBLookupDeadlineExceeded

    handle = SimpleNamespace(auth_read=AsyncMock(side_effect=asyncio.TimeoutError))
    with pytest.raises(NativeAuthFailure) as raised:
        await NativeAuthRepository(handle).key("hash")
    assert isinstance(raised.value, DBLookupDeadlineExceeded)
    assert raised.value.lookup == "key"


@pytest.mark.asyncio
async def test_native_spend_preserves_missing_and_authoritative_zero():
    handle = SimpleNamespace(auth_spend=AsyncMock(side_effect=[None, 0.0, RuntimeError("offline")]))
    repo = NativeAuthRepository(handle)
    assert await repo.spend("user", "u") is None
    assert await repo.spend("user", "u") == 0.0
    with pytest.raises(NativeAuthFailure):
        await repo.spend("user", "u")


def test_repository_selection_requires_an_explicit_handle():
    assert native_auth_repository(SimpleNamespace(native_auth_database=None)) is None
    assert native_auth_repository(MagicMock()) is None
    handle = SimpleNamespace(auth_read=AsyncMock())
    repo = native_auth_repository(SimpleNamespace(native_auth_database=handle))
    assert repo is not None and repo.database is handle
