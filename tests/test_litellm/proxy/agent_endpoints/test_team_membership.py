from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy.agent_endpoints.identity import preserve_identity
from litellm.proxy.agent_endpoints.team_membership import assigned_agent_team, require_assigned_team, validate_agent_team
from litellm.types.agents import AgentResponse


@pytest.mark.parametrize("params", [None, {}, {"model": "runtime"}])
def test_legacy_agents_keep_claim_team_resolution(params) -> None:
    assert require_assigned_team(params) is None


@pytest.mark.parametrize("value", [None, "", " ", 123, []])
def test_removed_or_invalid_membership_denies_instead_of_falling_back(value) -> None:
    with pytest.raises(HTTPException) as failure:
        require_assigned_team({"team_id": value})
    assert failure.value.status_code == 403


def test_runtime_update_preserves_team_assignment_and_explicit_removal() -> None:
    assert require_assigned_team({"team_id": "assigned-team"}) == "assigned-team"
    assert preserve_identity({"model": "updated"}, {"team_id": "assigned-team"})["team_id"] == "assigned-team"
    assert preserve_identity({"model": "updated"}, {"team_id": None})["team_id"] is None
    assert preserve_identity({"team_id": None}, {"team_id": "assigned-team"})["team_id"] is None


@pytest.mark.asyncio
async def test_membership_reads_current_database_assignment_over_stale_registry() -> None:
    agent: Final = AgentResponse(agent_id="agent", agent_name="Agent", agent_card_params={}, litellm_params={"team_id": "old-team"})
    find: Final = AsyncMock(return_value=SimpleNamespace(litellm_params={"team_id": "new-team"}))
    db: Final = SimpleNamespace(db=SimpleNamespace(litellm_agentstable=SimpleNamespace(find_unique=find)))
    assert await assigned_agent_team(agent, db) == "new-team"
    find.assert_awaited_once_with(where={"agent_id": "agent"})
    find.return_value = SimpleNamespace(litellm_params={"team_id": None})
    with pytest.raises(HTTPException) as failure:
        await assigned_agent_team(agent, db)
    assert failure.value.status_code == 403
    assert await assigned_agent_team(None, db) is None
    assert await assigned_agent_team(agent, None) == "old-team"
    find.return_value = None
    with pytest.raises(HTTPException) as deleted:
        await assigned_agent_team(agent, db)
    assert deleted.value.status_code == 403
    legacy: Final = agent.model_copy(update={"litellm_params": {}})
    assert await assigned_agent_team(legacy, db) is None


@pytest.mark.asyncio
async def test_assignment_requires_existing_team_and_accepts_removal() -> None:
    find: Final = AsyncMock(return_value=SimpleNamespace(team_id="team"))
    db: Final = SimpleNamespace(db=SimpleNamespace(litellm_teamtable=SimpleNamespace(find_unique=find)))
    await validate_agent_team({"team_id": "team"}, db)
    find.assert_awaited_once_with(where={"team_id": "team"})
    for params in (None, {}, {"team_id": None}):
        await validate_agent_team(params, db)
    for value in ("", " ", 123):
        with pytest.raises(HTTPException) as invalid:
            await validate_agent_team({"team_id": value}, db)
        assert invalid.value.status_code == 400
    find.return_value = None
    with pytest.raises(HTTPException) as missing:
        await validate_agent_team({"team_id": "missing"}, db)
    assert missing.value.status_code == 404
