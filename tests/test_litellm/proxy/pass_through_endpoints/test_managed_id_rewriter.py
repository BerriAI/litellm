import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.managed_id_codec import new_managed_id
from litellm.proxy.pass_through_endpoints.managed_id_rewriter import (
    list_passthrough_ids_from_db,
)


def _user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="user-1", team_id="team-1")


def _prisma_client(file_rows=None, batch_rows=None) -> MagicMock:
    pc = MagicMock()
    pc.db = MagicMock()
    pc.db.litellm_managedfiletable = MagicMock()
    pc.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)
    pc.db.litellm_managedfiletable.find_many = AsyncMock(
        side_effect=lambda *args, take=None, **kwargs: list(file_rows or [])[:take]
    )
    pc.db.litellm_managedobjecttable = MagicMock()
    pc.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=None)
    pc.db.litellm_managedobjecttable.find_many = AsyncMock(
        side_effect=lambda *args, take=None, **kwargs: list(batch_rows or [])[:take]
    )
    return pc


def _file_row(unified_id: str) -> MagicMock:
    row = MagicMock()
    row.unified_file_id = unified_id
    row.created_by = "user-1"
    row.team_id = "team-1"
    row.file_object = {"filename": "test.jsonl", "bytes": 42, "purpose": "batch"}
    row.created_at = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    return row


def _batch_row(unified_id: str) -> MagicMock:
    row = MagicMock()
    row.unified_object_id = unified_id
    row.created_by = "user-1"
    row.team_id = "team-1"
    row.file_object = {"status": "completed", "input_file_id": "file-managed-1"}
    row.file_purpose = "batch"
    row.created_at = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit,expected_message,expected_openai_code",
    [
        (
            "-1",
            "Invalid 'limit': integer below minimum value. Expected a value >= 0, but got -1 instead.",
            "integer_below_min_value",
        ),
        (
            "101",
            "Invalid 'limit': integer above maximum value. Expected a value <= 100, but got 101 instead.",
            "integer_above_max_value",
        ),
    ],
)
async def test_list_batches_out_of_range_limit_raises_400(
    limit, expected_message, expected_openai_code
):
    pc = _prisma_client(batch_rows=[_batch_row(new_managed_id("openai", "batch_abc"))])

    with pytest.raises(ProxyException) as exc:
        await list_passthrough_ids_from_db(
            provider="openai",
            route="/openai/v1/batches",
            user_api_key_dict=_user(),
            prisma_client=pc,
            query_params={"limit": limit},
        )

    assert exc.value.code == "400"
    assert exc.value.param == "limit"
    assert exc.value.type == "invalid_request_error"
    assert exc.value.openai_code == expected_openai_code
    assert exc.value.message == expected_message
    pc.db.litellm_managedobjecttable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_list_batches_limit_zero_returns_empty_page_without_db_query():
    pc = _prisma_client(batch_rows=[_batch_row(new_managed_id("openai", "batch_abc"))])

    result = await list_passthrough_ids_from_db(
        provider="openai",
        route="/openai/v1/batches",
        user_api_key_dict=_user(),
        prisma_client=pc,
        query_params={"limit": "0"},
    )

    assert result == {
        "object": "list",
        "data": [],
        "first_id": None,
        "last_id": None,
        "has_more": False,
    }
    pc.db.litellm_managedobjecttable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_list_files_limit_above_batch_cap_still_served():
    managed_id = new_managed_id("openai", "file-abc")
    pc = _prisma_client(file_rows=[_file_row(managed_id)])

    result = await list_passthrough_ids_from_db(
        provider="openai",
        route="/openai/v1/files",
        user_api_key_dict=_user(),
        prisma_client=pc,
        query_params={"limit": "101"},
    )

    assert result is not None
    assert [item["id"] for item in result["data"]] == [managed_id]


@pytest.mark.asyncio
async def test_list_files_drops_batch_guardrail_key_persisted_by_an_older_proxy():
    """Rows written before the response serializer dropped the key still carry an explicit null."""
    managed_id = new_managed_id("openai", "file-abc")
    row = _file_row(managed_id)
    row.file_object = {**row.file_object, "litellm_batch_guardrail": None}
    pc = _prisma_client(file_rows=[row])

    result = await list_passthrough_ids_from_db(
        provider="openai",
        route="/openai/v1/files",
        user_api_key_dict=_user(),
        prisma_client=pc,
        query_params={},
    )

    assert result is not None
    assert "litellm_batch_guardrail" not in result["data"][0]
    assert result["data"][0]["filename"] == "test.jsonl"
