from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.errors import PrismaError

from litellm.proxy.spend_tracking.key_metadata_recovery import (
    fill_missing_api_key_aliases,
    recover_double_hashed_key_metadata,
)
from litellm.proxy.utils import hash_token


def _digest_row(digest: str, key_alias: str, team_id: str | None, user_id: str | None) -> dict[str, str | None]:
    return {"digest": digest, "key_alias": key_alias, "team_id": team_id, "user_id": user_id}


def _query_raw_by_table(
    active_rows: Sequence[dict[str, str | None]],
    deleted_rows: Sequence[dict[str, str | None]],
) -> AsyncMock:
    async def query_raw(sql: str, *params: object) -> list[dict[str, str | None]]:
        if '"LiteLLM_VerificationToken"' in sql:
            return list(active_rows)
        if '"LiteLLM_DeletedVerificationToken"' in sql:
            return list(deleted_rows)
        raise AssertionError(f"unexpected query: {sql}")

    return AsyncMock(side_effect=query_raw)


@pytest.mark.asyncio
async def test_recover_double_hashed_key_metadata_via_active_token_digest():
    double_hashed = hash_token("a" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(
        active_rows=[_digest_row(double_hashed, "batch-worker", "team-1", "alice")],
        deleted_rows=[],
    )

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed})

    assert result[double_hashed]["key_alias"] == "batch-worker"
    assert result[double_hashed]["team_id"] == "team-1"
    assert result[double_hashed]["user_id"] == "alice"
    ((_, digests),) = [call.args for call in mock_prisma.db.query_raw.call_args_list]
    assert digests == [double_hashed]


@pytest.mark.asyncio
async def test_recover_double_hashed_key_metadata_falls_back_to_deleted_tokens():
    double_hashed = hash_token("y" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(
        active_rows=[],
        deleted_rows=[_digest_row(double_hashed, "deleted-key", "team-del", "erin")],
    )

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed})

    assert result[double_hashed]["key_alias"] == "deleted-key"
    assert result[double_hashed]["team_id"] == "team-del"
    assert result[double_hashed]["user_id"] == "erin"
    assert [call.args[1] for call in mock_prisma.db.query_raw.call_args_list] == [[double_hashed], [double_hashed]]


@pytest.mark.asyncio
async def test_recover_only_asks_deleted_tokens_for_digests_active_keys_missed():
    found_active = hash_token("1" * 64)
    found_deleted = hash_token("2" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(
        active_rows=[_digest_row(found_active, "active-key", None, None)],
        deleted_rows=[_digest_row(found_deleted, "deleted-key", None, None)],
    )

    result = await recover_double_hashed_key_metadata(mock_prisma, {found_active, found_deleted})

    assert result[found_active]["key_alias"] == "active-key"
    assert result[found_deleted]["key_alias"] == "deleted-key"
    assert [call.args[1] for call in mock_prisma.db.query_raw.call_args_list] == [
        sorted((found_active, found_deleted)),
        [found_deleted],
    ]


@pytest.mark.asyncio
async def test_recover_permanent_miss_costs_two_digest_lookups_and_no_table_walk():
    double_hashed = hash_token("b" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(active_rows=[], deleted_rows=[])

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed})

    assert result == {}
    assert len(mock_prisma.db.query_raw.call_args_list) == 2
    mock_prisma.db.litellm_verificationtoken.find_many.assert_not_called()
    mock_prisma.db.litellm_deletedverificationtoken.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_recover_skips_keys_that_are_not_sha256_digests():
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    result = await recover_double_hashed_key_metadata(mock_prisma, {"sk-plain-key", "key-hash-short"})

    assert result == {}
    mock_prisma.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_recover_returns_empty_when_digest_lookup_raises_prisma_error():
    double_hashed = hash_token("c" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(side_effect=PrismaError("db down"))

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed})

    assert result == {}


@pytest.mark.asyncio
async def test_fill_missing_api_key_aliases_updates_null_alias_and_email_rows():
    double_hashed = hash_token("d" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(
        active_rows=[_digest_row(double_hashed, "recovered-alias", "team-9", "bob")],
        deleted_rows=[],
    )
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[SimpleNamespace(user_id="bob", user_email="bob@example.com")]
    )

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
    assert filled[0]["spend"] == 12.5
    assert filled[1]["api_key_alias"] == "named-key"
    assert mock_prisma.db.litellm_usertable.find_many.call_args.kwargs["where"] == {"user_id": {"in": ["bob"]}}


@pytest.mark.asyncio
async def test_fill_missing_api_key_aliases_leaves_rows_untouched_when_nothing_is_missing():
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])
    rows = ({"api_key": hash_token("e" * 64), "api_key_alias": "named", "user_email": "x@example.com"},)

    filled = await fill_missing_api_key_aliases(mock_prisma, rows)

    assert filled == rows
    mock_prisma.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_fill_missing_api_key_aliases_keeps_spend_user_email_when_alias_is_missing():
    double_hashed = hash_token("f" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(
        active_rows=[_digest_row(double_hashed, "team-key", "team-9", "key-owner")],
        deleted_rows=[],
    )
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[SimpleNamespace(user_id="key-owner", user_email="owner@example.com")]
    )

    rows = (
        {
            "api_key": double_hashed,
            "api_key_alias": None,
            "team_id": None,
            "user_email": "spender@example.com",
            "spend": 4.0,
        },
    )

    filled = await fill_missing_api_key_aliases(mock_prisma, rows)

    assert filled[0]["api_key_alias"] == "team-key"
    assert filled[0]["team_id"] == "team-9"
    assert filled[0]["user_email"] == "spender@example.com"


@pytest.mark.asyncio
async def test_fill_missing_api_key_aliases_skips_named_keys_that_have_no_email():
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])
    rows = (
        {
            "api_key": hash_token("g" * 64),
            "api_key_alias": "service-key",
            "team_id": "team-svc",
            "user_email": None,
        },
    )

    filled = await fill_missing_api_key_aliases(mock_prisma, rows)

    assert filled == rows
    mock_prisma.db.query_raw.assert_not_called()
