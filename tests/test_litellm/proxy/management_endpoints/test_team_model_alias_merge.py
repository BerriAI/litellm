"""
Tests for _update_model_table alias merge behavior.

Regression tests for https://github.com/BerriAI/litellm/issues/22594
When a team already has a model_aliases row, adding a new BYOK model must
merge the new alias into existing aliases, not replace them.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)

from litellm.proxy._types import (
    LiteLLM_ModelTable,
    UpdateTeamRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.team_endpoints import _update_model_table


class MockModelTableRow:
    def __init__(self, id: int, model_aliases):
        self.id = id
        self.model_aliases = model_aliases


class TestUpdateModelTableMergesAliases:

    @pytest.mark.asyncio
    async def test_existing_row_uses_atomic_merge(self):
        """When model_id is set, execute_raw is called with jsonb || for atomic merge."""
        prisma_client = MagicMock()
        prisma_client.db.execute_raw = AsyncMock(return_value=None)

        result_id = await _update_model_table(
            data=UpdateTeamRequest(
                team_id="team-1",
                model_aliases={"model-b": "internal_b"},
            ),
            model_id=42,
            prisma_client=prisma_client,
            user_api_key_dict=UserAPIKeyAuth(user_id="test_user"),
            litellm_proxy_admin_name="proxy_admin",
        )

        prisma_client.db.execute_raw.assert_called_once()
        call_args = prisma_client.db.execute_raw.call_args
        sql = call_args[0][0]
        assert "COALESCE(aliases, '{}'::jsonb) || $1::jsonb" in sql
        assert json.loads(call_args[0][1]) == {"model-b": "internal_b"}
        assert call_args[0][2] == "test_user"  # updated_by
        assert call_args[0][3] == 42  # model_id
        assert result_id == 42

    @pytest.mark.asyncio
    async def test_no_existing_row_creates_normally(self):
        """When model_id is None, a new row is created via Prisma create."""
        mock_litellm_modeltable = MagicMock()
        mock_litellm_modeltable.create = AsyncMock(
            return_value=MockModelTableRow(id=1, model_aliases="{}")
        )

        prisma_client = MagicMock()
        prisma_client.db.litellm_modeltable = mock_litellm_modeltable

        result_id = await _update_model_table(
            data=UpdateTeamRequest(
                team_id="team-1",
                model_aliases={"model-a": "internal_a"},
            ),
            model_id=None,
            prisma_client=prisma_client,
            user_api_key_dict=UserAPIKeyAuth(user_id="test_user"),
            litellm_proxy_admin_name="proxy_admin",
        )

        mock_litellm_modeltable.create.assert_called_once()
        create_data = mock_litellm_modeltable.create.call_args[1]["data"]
        assert json.loads(create_data["model_aliases"]) == {"model-a": "internal_a"}
        assert result_id == 1

    @pytest.mark.asyncio
    async def test_returns_none_when_no_aliases(self):
        """When model_aliases is None, nothing happens and model_id is returned as-is."""
        prisma_client = MagicMock()

        result_id = await _update_model_table(
            data=UpdateTeamRequest(team_id="team-1"),
            model_id=42,
            prisma_client=prisma_client,
            user_api_key_dict=UserAPIKeyAuth(user_id="test_user"),
            litellm_proxy_admin_name="proxy_admin",
        )

        assert result_id == 42
        prisma_client.db.execute_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_litellm_proxy_admin_name_when_no_user_id(self):
        """When user_id is None, updated_by falls back to litellm_proxy_admin_name."""
        prisma_client = MagicMock()
        prisma_client.db.execute_raw = AsyncMock(return_value=None)

        await _update_model_table(
            data=UpdateTeamRequest(
                team_id="team-1",
                model_aliases={"model-a": "internal_a"},
            ),
            model_id=42,
            prisma_client=prisma_client,
            user_api_key_dict=UserAPIKeyAuth(user_id=None),
            litellm_proxy_admin_name="proxy_admin",
        )

        call_args = prisma_client.db.execute_raw.call_args[0]
        assert call_args[2] == "proxy_admin"
