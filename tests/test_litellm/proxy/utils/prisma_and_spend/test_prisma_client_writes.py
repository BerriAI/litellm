"""Pin ``PrismaClient`` write-side data operations.

Symbols pinned here:
  - ``PrismaClient.insert_data``
  - ``PrismaClient.update_data``
  - ``PrismaClient.delete_data``
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy.utils import PrismaClient


@pytest.mark.asyncio
async def test_insert_data_hashes_token_and_upserts(prisma_client: PrismaClient) -> None:
    token = "sk-secret-1"
    response = SimpleNamespace(token=hashlib.sha256(token.encode()).hexdigest(),
                               key_alias="alias", user_id="u1")
    prisma_client.db.litellm_verificationtoken.upsert = AsyncMock(return_value=response)
    data = {
        "token": token,
        "user_id": "u1",
        "team_id": "t1",
        "metadata": {"a": 1},
    }
    result = await prisma_client.insert_data(data=data, table_name="key")
    upsert_kwargs = prisma_client.db.litellm_verificationtoken.upsert.await_args.kwargs
    actual = {
        "returned": result,
        "where": upsert_kwargs["where"],
        "include": upsert_kwargs["include"],
        "create_token": upsert_kwargs["data"]["create"]["token"],
        "create_metadata_serialized": isinstance(
            upsert_kwargs["data"]["create"]["metadata"], str
        ),
        "update_empty": upsert_kwargs["data"]["update"],
    }
    expected_hash = hashlib.sha256(token.encode()).hexdigest()
    assert actual == {
        "returned": response,
        "where": {"token": expected_hash},
        "include": {"litellm_budget_table": True},
        "create_token": expected_hash,
        "create_metadata_serialized": True,
        "update_empty": {},
    }


@pytest.mark.asyncio
async def test_insert_data_strips_null_budget_limits(prisma_client: PrismaClient) -> None:
    prisma_client.db.litellm_verificationtoken.upsert = AsyncMock(return_value=None)
    await prisma_client.insert_data(
        data={"token": "sk-1", "budget_limits": None}, table_name="key"
    )
    create_payload = prisma_client.db.litellm_verificationtoken.upsert.await_args.kwargs[
        "data"
    ]["create"]
    assert "budget_limits" not in create_payload


@pytest.mark.asyncio
async def test_insert_data_team_serializes_members(prisma_client: PrismaClient) -> None:
    prisma_client.db.litellm_teamtable.upsert = AsyncMock(
        return_value=SimpleNamespace(team_id="t1", team_alias="x", spend=0)
    )
    data = {
        "team_id": "t1",
        "team_alias": "x",
        "members_with_roles": [{"role": "admin", "user_id": "u1"}],
    }
    result = await prisma_client.insert_data(data=data, table_name="team")
    create_payload = prisma_client.db.litellm_teamtable.upsert.await_args.kwargs["data"][
        "create"
    ]
    assert result.team_id == "t1"
    assert create_payload["members_with_roles"] == json.dumps(data["members_with_roles"])
    assert create_payload["team_id"] == "t1"


@pytest.mark.asyncio
async def test_insert_data_user_organization_fk_raises_400(
    prisma_client: PrismaClient,
) -> None:
    err = RuntimeError(
        "Foreign key constraint failed on the field: `LiteLLM_UserTable_organization_id_fkey (index)`"
    )
    prisma_client.db.litellm_usertable.upsert = AsyncMock(side_effect=err)
    with pytest.raises(HTTPException) as excinfo:
        await prisma_client.insert_data(
            data={"user_id": "u1", "organization_id": "org-bad"}, table_name="user"
        )
    raised = excinfo.value
    assert "Foreign Key Constraint failed" in raised.detail["error"]
    assert raised.status_code == 400


@pytest.mark.asyncio
async def test_insert_data_logs_and_raises_generic_error(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.litellm_verificationtoken.upsert = AsyncMock(
        side_effect=RuntimeError("write boom")
    )
    with pytest.raises(RuntimeError, match="write boom"):
        await prisma_client.insert_data(data={"token": "sk-1"}, table_name="key")


@pytest.mark.asyncio
async def test_update_data_token_hashes_and_updates(
    prisma_client: PrismaClient,
) -> None:
    token = "sk-update-1"
    response = SimpleNamespace(
        token=hashlib.sha256(token.encode()).hexdigest(),
        model_dump=lambda: {
            "token": hashlib.sha256(token.encode()).hexdigest(),
            "spend": 1.0,
            "user_id": "u1",
        },
    )
    prisma_client.db.litellm_verificationtoken.update = AsyncMock(return_value=response)
    result = await prisma_client.update_data(
        token=token,
        data={"spend": 1.0},
    )
    update_kwargs = prisma_client.db.litellm_verificationtoken.update.await_args.kwargs
    hashed = hashlib.sha256(token.encode()).hexdigest()
    actual = {
        "result": result,
        "where": update_kwargs["where"],
        "data_token": update_kwargs["data"]["token"],
        "data_spend": update_kwargs["data"]["spend"],
    }
    assert actual == {
        "result": {
            "token": hashed,
            "data": {"token": hashed, "spend": 1.0, "user_id": "u1"},
        },
        "where": {"token": hashed},
        "data_token": hashed,
        "data_spend": 1.0,
    }


@pytest.mark.asyncio
async def test_update_data_user_upsert_returns_user_envelope(
    prisma_client: PrismaClient,
) -> None:
    row = SimpleNamespace(user_id="u2", spend=2.0)
    prisma_client.db.litellm_usertable.upsert = AsyncMock(return_value=row)
    result = await prisma_client.update_data(
        data={"user_id": "u2", "spend": 2.0},
        table_name="user",
    )
    assert result == {"user_id": "u2", "data": row}


@pytest.mark.asyncio
async def test_update_data_team_serializes_members_when_list(
    prisma_client: PrismaClient,
) -> None:
    row = SimpleNamespace(team_id="t9", team_alias="x")
    prisma_client.db.litellm_teamtable.upsert = AsyncMock(return_value=row)
    members = [{"role": "admin", "user_id": "u1"}]
    result = await prisma_client.update_data(
        data={"team_id": "t9", "members_with_roles": members},
        update_key_values={"members_with_roles": members},
        table_name="team",
    )
    upsert_kwargs = prisma_client.db.litellm_teamtable.upsert.await_args.kwargs
    actual = {
        "result_team_id": result["team_id"],
        "result_data": result["data"],
        "create_members": upsert_kwargs["data"]["create"]["members_with_roles"],
        "update_members": upsert_kwargs["data"]["update"]["members_with_roles"],
    }
    assert actual == {
        "result_team_id": "t9",
        "result_data": row,
        "create_members": json.dumps(members),
        "update_members": json.dumps(members),
    }


@pytest.mark.asyncio
async def test_update_data_logs_and_raises_on_error(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.litellm_verificationtoken.update = AsyncMock(
        side_effect=RuntimeError("update fail")
    )
    with pytest.raises(RuntimeError, match="update fail"):
        await prisma_client.update_data(token="sk-x", data={"spend": 1.0})


@pytest.mark.asyncio
async def test_delete_data_hashes_sk_tokens_and_calls_delete_many(
    prisma_client: PrismaClient,
) -> None:
    deleted = SimpleNamespace(count=2)
    prisma_client.db.litellm_verificationtoken.delete_many = AsyncMock(
        return_value=deleted
    )
    tokens = ["sk-one", "sk-two", "raw-hashed-token"]
    result = await prisma_client.delete_data(tokens=tokens)
    where = prisma_client.db.litellm_verificationtoken.delete_many.await_args.kwargs[
        "where"
    ]
    expected_hashes = sorted(
        [
            hashlib.sha256(b"sk-one").hexdigest(),
            hashlib.sha256(b"sk-two").hexdigest(),
            "raw-hashed-token",
        ]
    )
    actual = {
        "deleted_keys_attr": result["deleted_keys"],
        "where_keys": list(where.keys()),
        "filter_in_sorted": sorted(where["token"]["in"]),
        "delete_call_count": prisma_client.db.litellm_verificationtoken.delete_many.await_count,
    }
    assert actual == {
        "deleted_keys_attr": deleted,
        "where_keys": ["token"],
        "filter_in_sorted": expected_hashes,
        "delete_call_count": 1,
    }


@pytest.mark.asyncio
async def test_delete_data_team_calls_team_delete_many(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.litellm_teamtable.delete_many = AsyncMock()
    result = await prisma_client.delete_data(
        team_id_list=["t1", "t2"], table_name="team"
    )
    where = prisma_client.db.litellm_teamtable.delete_many.await_args.kwargs["where"]
    assert result == {"deleted_teams": ["t1", "t2"]}
    assert where == {"team_id": {"in": ["t1", "t2"]}}


@pytest.mark.asyncio
async def test_delete_data_logs_and_raises_on_error(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.litellm_verificationtoken.delete_many = AsyncMock(
        side_effect=RuntimeError("delete fail")
    )
    with pytest.raises(RuntimeError, match="delete fail"):
        await prisma_client.delete_data(tokens=["sk-x"])
