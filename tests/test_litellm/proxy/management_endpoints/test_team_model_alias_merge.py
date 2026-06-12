"""
Tests for atomic team model operations during BYOK model creation.

Regression tests for https://github.com/BerriAI/litellm/issues/22594
Concurrent BYOK model creates must not overwrite each other's entries
in team.models.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import (
    LitellmUserRoles,
    TeamModelAddRequest,
    UserAPIKeyAuth,
)


class TestTeamModelAddAtomicAppend:
    """Verify team_model_add uses atomic SQL for the models array append."""

    @pytest.mark.asyncio
    async def test_uses_atomic_array_append_with_dedup(self):
        """team_model_add must call execute_raw with DISTINCT unnest SQL."""
        from unittest.mock import patch

        from litellm.proxy.management_endpoints.team_endpoints import team_model_add

        mock_request = MagicMock()
        mock_user = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user"
        )

        existing_team = MagicMock()
        existing_team.model_dump.return_value = {
            "team_id": "team-1",
            "models": ["existing-model"],
        }

        updated_team = MagicMock()
        updated_team.team_id = "team-1"
        updated_team.model_dump.return_value = {
            "team_id": "team-1",
            "models": ["existing-model", "new-model"],
        }

        with (
            patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
            patch(
                "litellm.proxy.management_endpoints.team_endpoints._cache_team_object",
                new_callable=AsyncMock,
            ),
            patch("litellm.proxy.proxy_server.user_api_key_cache"),
            patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        ):
            mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
                return_value=existing_team
            )
            mock_prisma.db.execute_raw = AsyncMock(return_value=None)
            mock_prisma.db.litellm_teamtable.update = AsyncMock(
                return_value=updated_team
            )

            await team_model_add(
                data=TeamModelAddRequest(team_id="team-1", models=["new-model"]),
                http_request=mock_request,
                user_api_key_dict=mock_user,
            )

            mock_prisma.db.execute_raw.assert_called_once()
            sql = mock_prisma.db.execute_raw.call_args[0][0]
            assert "DISTINCT unnest" in sql
            assert "all-proxy-models" in sql
            assert mock_prisma.db.execute_raw.call_args[0][1] == ["new-model"]
            assert mock_prisma.db.execute_raw.call_args[0][2] == "team-1"

            # Should use write-routed update to re-fetch, not find_unique
            mock_prisma.db.litellm_teamtable.update.assert_called_once()
