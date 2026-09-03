from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.errors import PrismaError

from litellm.proxy.spend_tracking.key_metadata_recovery import (
    fill_missing_api_key_aliases,
    recover_double_hashed_key_metadata,
)
from litellm.proxy.utils import hash_token


@pytest.mark.asyncio
async def test_recover_double_hashed_key_metadata_via_reverse_hash():
    token = "a" * 64
    double_hashed = hash_token(token)
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[
            SimpleNamespace(
                token=token,
                key_alias="batch-worker",
                team_id="team-1",
                user_id="alice",
            )
        ]
    )
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[SimpleNamespace(user_id="alice", user_email="alice@example.com")]
    )
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed})

    assert result[double_hashed]["key_alias"] == "batch-worker"
    assert result[double_hashed]["team_id"] == "team-1"
    assert result[double_hashed]["user_email"] == "alice@example.com"
    mock_prisma.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_fill_missing_api_key_aliases_updates_null_alias_and_email_rows():
    token = "c" * 64
    double_hashed = hash_token(token)
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[
            SimpleNamespace(
                token=token,
                key_alias="recovered-alias",
                team_id="team-9",
                user_id="bob",
            )
        ]
    )
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[SimpleNamespace(user_id="bob", user_email="bob@example.com")]
    )
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    rows = (
        {
            "api_key": double_hashed,
            "api_key_alias": None,
            "team_id": None,
            "user_email": None,
            "spend": 12.5,
        },
        {
            "api_key": "already-joined-token",
            "api_key_alias": "named-key",
            "team_id": "team-ok",
            "user_email": "other@example.com",
            "spend": 1.0,
        },
    )

    filled = await fill_missing_api_key_aliases(mock_prisma, rows)

    assert filled[0]["api_key_alias"] == "recovered-alias"
    assert filled[0]["team_id"] == "team-9"
    assert filled[0]["user_email"] == "bob@example.com"
    assert filled[1]["api_key_alias"] == "named-key"


@pytest.mark.asyncio
async def test_recover_falls_back_to_spend_logs_when_token_scan_raises_prisma_error():
    token = "b" * 64
    double_hashed = hash_token(token)
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(side_effect=PrismaError("db down"))
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(side_effect=PrismaError("db down"))
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.query_raw = AsyncMock(
        return_value=[
            {
                "api_key": double_hashed,
                "key_alias": "from-spend-logs",
                "team_id": "team-sl",
                "user_id": "carol",
                "user_email": "carol@example.com",
            }
        ]
    )

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed})

    assert result[double_hashed]["key_alias"] == "from-spend-logs"
    assert result[double_hashed]["team_id"] == "team-sl"
    assert result[double_hashed]["user_email"] == "carol@example.com"


@pytest.mark.asyncio
async def test_recover_double_hashed_key_metadata_scans_past_first_page():
    token = "z" * 64
    double_hashed = hash_token(token)
    decoys = (
        SimpleNamespace(token="1" * 64, key_alias="decoy-1", team_id=None, user_id=None),
        SimpleNamespace(token="2" * 64, key_alias="decoy-2", team_id=None, user_id=None),
    )
    match = SimpleNamespace(token=token, key_alias="late-key", team_id="team-late", user_id="dana")

    async def find_many(*, take: int | None = None, skip: int | None = None, order: object = None):
        if skip == 0:
            return list(decoys)
        if skip == 2:
            return [match]
        return []

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(side_effect=find_many)
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[SimpleNamespace(user_id="dana", user_email="dana@example.com")]
    )
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed}, token_scan_page_size=2)

    assert result[double_hashed]["key_alias"] == "late-key"
    assert result[double_hashed]["team_id"] == "team-late"
    assert result[double_hashed]["user_email"] == "dana@example.com"
    assert [call.kwargs["skip"] for call in mock_prisma.db.litellm_verificationtoken.find_many.call_args_list] == [0, 2]
    mock_prisma.db.litellm_deletedverificationtoken.find_many.assert_not_called()
    mock_prisma.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_recover_double_hashed_key_metadata_pages_deleted_tokens():
    token = "y" * 64
    double_hashed = hash_token(token)
    decoys = (
        SimpleNamespace(token="3" * 64, key_alias="deleted-decoy-1", team_id=None, user_id=None),
        SimpleNamespace(token="4" * 64, key_alias="deleted-decoy-2", team_id=None, user_id=None),
    )
    match = SimpleNamespace(token=token, key_alias="deleted-late-key", team_id="team-del", user_id="erin")

    async def find_deleted(*, take: int | None = None, skip: int | None = None, order: object = None):
        if skip == 0:
            return list(decoys)
        if skip == 2:
            return [match]
        return []

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(side_effect=find_deleted)
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[SimpleNamespace(user_id="erin", user_email="erin@example.com")]
    )
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed}, token_scan_page_size=2)

    assert result[double_hashed]["key_alias"] == "deleted-late-key"
    assert result[double_hashed]["user_email"] == "erin@example.com"
    assert [
        call.kwargs["skip"] for call in mock_prisma.db.litellm_deletedverificationtoken.find_many.call_args_list
    ] == [
        0,
        2,
    ]
    mock_prisma.db.query_raw.assert_not_called()
