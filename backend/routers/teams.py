from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from scim2_models import Group as ScimGroup
from scim2_models import GroupMember

from litellm.proxy.auth_v2 import Role

from .dependencies import require_roles, team_store

ADMIN_PREFIX = "/admin"

router = APIRouter(prefix=f"{ADMIN_PREFIX}/teams", tags=["admin"])

_team_admin = require_roles(Role.ORG_ADMIN, Role.PLATFORM_ADMIN)
_protected = [Depends(_team_admin)]


class TeamUpsert(BaseModel):
    name: str
    members: List[str] = Field(default_factory=list)


class TeamView(BaseModel):
    id: str
    name: str
    members: List[str]


def _to_scim(team_id: Optional[str], body: TeamUpsert) -> ScimGroup:
    group = ScimGroup(
        display_name=body.name,
        members=[GroupMember(value=user_id) for user_id in body.members] or None,
    )
    if team_id is not None:
        group.id = team_id
    return group


def _to_view(group: ScimGroup) -> TeamView:
    return TeamView(
        id=group.id,
        name=group.display_name,
        members=[m.value for m in (group.members or []) if m.value],
    )


@router.post("", status_code=status.HTTP_201_CREATED, dependencies=_protected)
async def create_team(body: TeamUpsert, request: Request) -> TeamView:
    stored = await team_store(request).upsert_group(_to_scim(None, body))
    return _to_view(stored)


@router.get("", dependencies=_protected)
async def list_teams(request: Request) -> List[TeamView]:
    groups = await team_store(request).list_groups(None)
    return [_to_view(group) for group in groups]


@router.get("/{team_id}", dependencies=_protected)
async def get_team(team_id: str, request: Request) -> TeamView:
    group = await team_store(request).get_group(team_id)
    if group is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"team {team_id} not found"
        )
    return _to_view(group)


@router.put("/{team_id}", dependencies=_protected)
async def upsert_team(team_id: str, body: TeamUpsert, request: Request) -> TeamView:
    stored = await team_store(request).upsert_group(_to_scim(team_id, body))
    return _to_view(stored)


@router.delete(
    "/{team_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=_protected
)
async def delete_team(team_id: str, request: Request) -> Response:
    store = team_store(request)
    if await store.get_group(team_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"team {team_id} not found"
        )
    await store.delete_group(team_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
