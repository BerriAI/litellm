from collections.abc import Mapping
from typing import Final, TypedDict

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import ReadOnly

from litellm.proxy.agent_endpoints.agent_registry import agents_table
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import TeamRepository
from litellm.types.agents import AgentResponse


class AgentIdFilter(TypedDict):
    agent_id: ReadOnly[str]


class TeamIdFilter(TypedDict):
    team_id: ReadOnly[str]


class AgentTeamAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team_id: str | None = Field(min_length=1)


async def validate_agent_team(params: Mapping[str, object] | None, prisma_client: PrismaClient) -> None:
    if not params or "team_id" not in params:
        return
    value: Final = params["team_id"]
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(400, "Agent team_id must be a non-empty string or null")
    query: Final[TeamIdFilter] = {"team_id": value}
    team: Final = await TeamRepository(prisma_client).table.find_unique(where=query)
    if team is None:
        raise HTTPException(404, "The assigned team does not exist")


def require_assigned_team(params: Mapping[str, object] | None) -> str | None:
    if not params or "team_id" not in params:
        return None
    value: Final = params["team_id"]
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(403, "This agent has no assigned team. Add it to a team before making requests")
    return value


async def assigned_agent_team(agent: AgentResponse | None, prisma_client: PrismaClient | None) -> str | None:
    if agent is None:
        return None
    query: Final[AgentIdFilter] = {"agent_id": agent.agent_id}
    stored: Final = await agents_table(prisma_client).find_unique(where=query) if prisma_client is not None else None
    if stored is None and prisma_client is not None and agent.litellm_params and "team_id" in agent.litellm_params:
        raise HTTPException(403, "The assigned agent no longer exists in the database")
    return require_assigned_team(stored.litellm_params if stored is not None else agent.litellm_params)
