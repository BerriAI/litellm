"""
Tests for `PrismaClient.get_data(table_name="combined_view", query_type="find_unique")`.

The combined_view SQL projects `t.access_group_ids AS team_access_group_ids` so
the listing path (`/v1/models`) can read team-level DB access groups directly off
the verification-token row without an extra DB / cache round-trip. Postgres
returns SQL NULL for teams that haven't set the column, so `get_data` must
coerce the field to `[]` before constructing `LiteLLM_VerificationTokenView`
(the field is `List`, not `Optional[List]`, so passing `None` would raise).
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.utils import PrismaClient, ProxyLogging


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma so this test doesn't require generated Prisma binaries."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield


@pytest.fixture
def mock_proxy_logging():
    proxy_logging = AsyncMock(spec=ProxyLogging)
    proxy_logging.failure_handler = AsyncMock()
    return proxy_logging


def _minimal_combined_view_row(**overrides):
    """A combined_view row stripped to only what the null-coerce + view-construction path touches."""
    row = {
        "token": "hashed-token",
        "user_id": None,
        "team_id": None,
        "expires": None,
        "team_models": None,
        "team_access_group_ids": None,
        "team_blocked": None,
        "team_members_with_roles": None,
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_get_data_combined_view_coerces_null_team_access_group_ids_to_empty_list(
    mock_proxy_logging,
):
    """
    Regression: teams without `access_group_ids` set return SQL NULL from
    combined_view. Without the coerce in `get_data`, constructing
    `LiteLLM_VerificationTokenView(team_access_group_ids=None, ...)` raises a
    Pydantic validation error because the field type is `List`, not `Optional`.
    """
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client._query_first_with_cached_plan_fallback = AsyncMock(
        return_value=_minimal_combined_view_row()
    )

    result = await client.get_data(
        token="sk-test-token",
        table_name="combined_view",
        query_type="find_unique",
    )

    assert result is not None
    assert result.team_access_group_ids == []
    assert result.team_models == []
    assert result.team_blocked is False


@pytest.mark.asyncio
async def test_get_data_combined_view_preserves_non_null_team_access_group_ids(
    mock_proxy_logging,
):
    """When the team has DB access groups assigned, the IDs flow through to the view unchanged."""
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client._query_first_with_cached_plan_fallback = AsyncMock(
        return_value=_minimal_combined_view_row(
            team_access_group_ids=["ag-prod", "ag-finance"],
        )
    )

    result = await client.get_data(
        token="sk-test-token",
        table_name="combined_view",
        query_type="find_unique",
    )

    assert result is not None
    assert result.team_access_group_ids == ["ag-prod", "ag-finance"]
