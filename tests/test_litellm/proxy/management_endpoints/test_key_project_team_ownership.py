"""A project may only be attached to keys of the team that owns it (#41089).

A project is created under exactly one team, and its budget and models are that
team's. Nothing checked that the key's team matched, so an admin of team-a — a
member of no other team — could issue a key on their own team pointing at a
team-b project, and team-b would see the spend under a project they never
granted anything on. Only the project id was needed.

The negative controls are the point: a project with no owning team is left
alone, and a key on the owning team still passes, or the check would be a wall
rather than a boundary.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.models.project import LiteLLM_ProjectTable
from litellm.proxy._types import GenerateKeyRequest, UpdateKeyRequest


def _project(team_id, models=None, project_id="proj-1"):
    return LiteLLM_ProjectTable(
        project_id=project_id,
        team_id=team_id,
        models=models or [],
    )


async def _check(project, data, key_team_id):
    """Drive _check_project_key_limits with the project the store would return."""
    from litellm.proxy.management_endpoints import key_management_endpoints as kme

    original = kme.get_project_object
    kme.get_project_object = AsyncMock(return_value=project)
    try:
        await kme._check_project_key_limits(
            project_id=project.project_id,
            data=data,
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
            key_team_id=key_team_id,
        )
    finally:
        kme.get_project_object = original


@pytest.mark.asyncio
async def test_a_key_may_not_point_at_another_teams_project():
    with pytest.raises(HTTPException) as exc:
        await _check(
            _project(team_id="team-b"),
            GenerateKeyRequest(team_id="team-a"),
            key_team_id="team-a",
        )
    assert exc.value.status_code == 403
    detail = str(exc.value.detail)
    assert "team-b" in detail and "team-a" in detail


@pytest.mark.asyncio
async def test_a_key_with_no_team_may_not_point_at_a_teams_project():
    # The issue's step 5: no team at all still charges a team's project.
    with pytest.raises(HTTPException) as exc:
        await _check(
            _project(team_id="team-b"),
            GenerateKeyRequest(),
            key_team_id=None,
        )
    assert exc.value.status_code == 403
    assert "no team" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_a_key_on_the_owning_team_is_accepted():
    await _check(
        _project(team_id="team-b"),
        GenerateKeyRequest(team_id="team-b"),
        key_team_id="team-b",
    )


@pytest.mark.asyncio
async def test_a_project_with_no_team_is_left_alone():
    # Nobody owns it, so there is no boundary to cross — rejecting here would
    # break every project created outside a team.
    await _check(_project(team_id=None), GenerateKeyRequest(team_id="team-a"), key_team_id="team-a")
    await _check(_project(team_id=None), GenerateKeyRequest(), key_team_id=None)


@pytest.mark.asyncio
async def test_the_ownership_check_runs_before_the_model_check():
    # A foreign project must be refused as foreign, not as "model not allowed":
    # the 400 would read as a configuration problem and hide the tenancy one.
    with pytest.raises(HTTPException) as exc:
        await _check(
            _project(team_id="team-b", models=["gpt-4o-mini"]),
            GenerateKeyRequest(team_id="team-a", models=["gpt-4o"]),
            key_team_id="team-a",
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_an_update_carries_the_keys_existing_team():
    # /key/update sends no team_id when only the project changes, so the check
    # has to use the key's stored team rather than treating it as absent.
    with pytest.raises(HTTPException) as exc:
        await _check(
            _project(team_id="team-b"),
            UpdateKeyRequest(key="sk-x", project_id="proj-1"),
            key_team_id="team-a",
        )
    assert exc.value.status_code == 403
