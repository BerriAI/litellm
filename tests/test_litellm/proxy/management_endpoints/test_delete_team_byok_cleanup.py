"""
Tests for team BYOK model delete cleanup.

Regression tests for https://github.com/BerriAI/litellm/issues/22594
Deleting a team BYOK model must remove the public model name from
team.models and the alias map, even when the alias was previously
overwritten or never populated.
"""

import json
import os
import sys
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)

from litellm.proxy._types import (
    LiteLLM_ModelTable,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo


def _make_deployment(
    model_name: str,
    model_id: str,
    team_id: str,
    team_public_model_name: str,
) -> Deployment:
    return Deployment(
        model_name=model_name,
        litellm_params=LiteLLM_Params(model="openai/fake"),
        model_info=ModelInfo(
            id=model_id,
            team_id=team_id,
            team_public_model_name=team_public_model_name,
        ),
    )


class MockTeamRow:
    def __init__(self, team_id: str, models: List[str], model_id: Optional[int] = None):
        self.team_id = team_id
        self.models = list(models)
        self.model_id = model_id

    def model_dump(self):
        return {
            "team_id": self.team_id,
            "models": self.models,
            "model_id": self.model_id,
        }


class TestResolveTeamPublicModelName:

    def test_returns_team_public_model_name_when_present(self):
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _resolve_team_public_model_name,
        )
        deployment = _make_deployment(
            model_name="model_name_team1_uuid1",
            model_id="uuid1",
            team_id="team1",
            team_public_model_name="my-gpt4",
        )
        assert _resolve_team_public_model_name(deployment) == "my-gpt4"

    def test_falls_back_to_model_name(self):
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _resolve_team_public_model_name,
        )
        deployment = Deployment(
            model_name="plain-model",
            litellm_params=LiteLLM_Params(model="openai/fake"),
            model_info=ModelInfo(id="uuid1", team_id="team1"),
        )
        assert _resolve_team_public_model_name(deployment) == "plain-model"


class TestCleanupTeamModelReferences:

    @pytest.mark.asyncio
    async def test_removes_public_name_from_team_models(self):
        """Public name must be removed from team.models even when alias map is empty."""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _cleanup_team_model_references,
        )
        team_row = MockTeamRow(
            team_id="team1",
            models=["my-gpt4", "other-model"],
            model_id=1,
        )

        team_update_calls = []

        mock_team_table = AsyncMock()
        mock_team_table.find_unique = AsyncMock(return_value=team_row)
        mock_team_table.update = AsyncMock(side_effect=lambda **kw: team_update_calls.append(kw))

        mock_model_row = MagicMock()
        mock_model_row.model_aliases = json.dumps({})
        mock_model_table = AsyncMock()
        mock_model_table.find_unique = AsyncMock(return_value=mock_model_row)
        mock_model_table.update = AsyncMock()

        prisma_client = MagicMock()
        prisma_client.db.litellm_teamtable = mock_team_table
        prisma_client.db.litellm_modeltable = mock_model_table

        await _cleanup_team_model_references(
            team_id="team1",
            internal_model_name="model_name_team1_uuid1",
            public_model_name="my-gpt4",
            prisma_client=prisma_client,
        )

        assert len(team_update_calls) == 1
        updated_models = team_update_calls[0]["data"]["models"]
        assert "my-gpt4" not in updated_models
        assert "other-model" in updated_models

    @pytest.mark.asyncio
    async def test_removes_alias_entry_by_internal_name(self):
        """If the alias map has an entry pointing to the internal name, remove it."""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _cleanup_team_model_references,
        )
        team_row = MockTeamRow(
            team_id="team1",
            models=["my-gpt4", "other-model"],
            model_id=1,
        )

        team_update_calls = []
        alias_update_calls = []

        mock_team_table = AsyncMock()
        mock_team_table.find_unique = AsyncMock(return_value=team_row)
        mock_team_table.update = AsyncMock(side_effect=lambda **kw: team_update_calls.append(kw))

        mock_model_row = MagicMock()
        mock_model_row.model_aliases = json.dumps({
            "my-gpt4": "model_name_team1_uuid1",
            "other-model": "model_name_team1_uuid2",
        })
        mock_model_table = AsyncMock()
        mock_model_table.find_unique = AsyncMock(return_value=mock_model_row)
        mock_model_table.update = AsyncMock(side_effect=lambda **kw: alias_update_calls.append(kw))

        prisma_client = MagicMock()
        prisma_client.db.litellm_teamtable = mock_team_table
        prisma_client.db.litellm_modeltable = mock_model_table

        await _cleanup_team_model_references(
            team_id="team1",
            internal_model_name="model_name_team1_uuid1",
            public_model_name="my-gpt4",
            prisma_client=prisma_client,
        )

        assert len(alias_update_calls) == 1
        updated_aliases = json.loads(alias_update_calls[0]["data"]["model_aliases"])
        assert "my-gpt4" not in updated_aliases
        assert updated_aliases == {"other-model": "model_name_team1_uuid2"}

    @pytest.mark.asyncio
    async def test_no_update_when_nothing_to_remove(self):
        """If public name is not in team.models, no team update should happen."""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _cleanup_team_model_references,
        )
        team_row = MockTeamRow(
            team_id="team1",
            models=["other-model"],
            model_id=1,
        )

        team_update_calls = []

        mock_team_table = AsyncMock()
        mock_team_table.find_unique = AsyncMock(return_value=team_row)
        mock_team_table.update = AsyncMock(side_effect=lambda **kw: team_update_calls.append(kw))

        mock_model_row = MagicMock()
        mock_model_row.model_aliases = json.dumps({})
        mock_model_table = AsyncMock()
        mock_model_table.find_unique = AsyncMock(return_value=mock_model_row)
        mock_model_table.update = AsyncMock()

        prisma_client = MagicMock()
        prisma_client.db.litellm_teamtable = mock_team_table
        prisma_client.db.litellm_modeltable = mock_model_table

        await _cleanup_team_model_references(
            team_id="team1",
            internal_model_name="model_name_team1_uuid1",
            public_model_name="not-in-team",
            prisma_client=prisma_client,
        )

        assert len(team_update_calls) == 0

    @pytest.mark.asyncio
    async def test_returns_none_when_team_not_found(self):
        """If the team doesn't exist, return None."""
        from litellm.proxy.management_endpoints.model_management_endpoints import (
            _cleanup_team_model_references,
        )

        mock_team_table = AsyncMock()
        mock_team_table.find_unique = AsyncMock(return_value=None)

        prisma_client = MagicMock()
        prisma_client.db.litellm_teamtable = mock_team_table

        result = await _cleanup_team_model_references(
            team_id="nonexistent",
            internal_model_name="internal",
            public_model_name="public",
            prisma_client=prisma_client,
        )

        assert result is None
