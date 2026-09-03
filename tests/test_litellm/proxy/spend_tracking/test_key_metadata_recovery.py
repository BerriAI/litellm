from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

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
        return_value=[SimpleNamespace(token=token, key_alias="batch-worker", team_id="team-1")]
    )
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[])
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed})

    assert result[double_hashed]["key_alias"] == "batch-worker"
    assert result[double_hashed]["team_id"] == "team-1"
    mock_prisma.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_fill_missing_api_key_aliases_updates_null_alias_rows():
    token = "c" * 64
    double_hashed = hash_token(token)
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[SimpleNamespace(token=token, key_alias="recovered-alias", team_id="team-9")]
    )
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[])
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    rows = (
        {
            "api_key": double_hashed,
            "api_key_alias": None,
            "team_id": None,
            "user_email": "owner@example.com",
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
    assert filled[0]["user_email"] == "owner@example.com"
    assert filled[1]["api_key_alias"] == "named-key"
