from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Annotated, Final, TypeAlias

from fastapi import Depends, HTTPException

from litellm.integrations.lens.generated.types import TraceScope
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

MAX_WORKSPACE_TRACE_KEYS: Final = 2048


@dataclass(frozen=True, slots=True)
class TraceWorkspaceFilters:
    organization_id: str | None = None
    team_id: str | None = None
    project_id: str | None = None
    api_key_hash: str | None = None

    @property
    def active(self) -> bool:
        return bool(self.organization_id or self.team_id or self.project_id or self.api_key_hash)


WorkspaceKeyLookup: TypeAlias = Callable[[TraceWorkspaceFilters], Awaitable[tuple[str, ...]]]


def _check_workspace_size(count: int) -> None:
    if count > MAX_WORKSPACE_TRACE_KEYS:
        raise HTTPException(status_code=413, detail="Workspace is too large. Select a team, project, or key.")


async def load_workspace_key_hashes(
    filters: TraceWorkspaceFilters, *, prisma_client: PrismaClient | None
) -> tuple[str, ...]:
    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Workspace trace filters require a configured database")
    teams: Final = (
        await TeamRepository(prisma_client).table.find_many(
            where={
                "organization_id": filters.organization_id,
                **({"team_id": filters.team_id} if filters.team_id else {}),
            },
            take=MAX_WORKSPACE_TRACE_KEYS + 1,
        )
        if filters.organization_id
        else ()
    )
    _check_workspace_size(len(teams))
    organization: Final[Mapping[str, object] | None] = (
        {
            "OR": (
                {"organization_id": filters.organization_id},
                {"organization_id": None, "team_id": {"in": tuple(team.team_id for team in teams)}},
            )
        }
        if filters.organization_id
        else None
    )
    clauses: Final = (
        organization,
        {"project_id": filters.project_id} if filters.project_id else None,
        {"team_id": filters.team_id} if filters.team_id else None,
        {"token": filters.api_key_hash} if filters.api_key_hash else None,
    )
    keys: Final = await VerificationTokenRepository(prisma_client).table.find_many(
        where={"AND": tuple(clause for clause in clauses if clause is not None)},
        take=MAX_WORKSPACE_TRACE_KEYS + 1,
        order={"token": "asc"},
    )
    _check_workspace_size(len(keys))
    return tuple(key.token for key in keys)


async def get_workspace_key_lookup() -> WorkspaceKeyLookup:
    from litellm.proxy.proxy_server import prisma_client

    return partial(load_workspace_key_hashes, prisma_client=prisma_client)


WorkspaceKeyLookupDependency: TypeAlias = Annotated[WorkspaceKeyLookup, Depends(get_workspace_key_lookup)]


async def workspace_trace_scope(
    scope: TraceScope, filters: TraceWorkspaceFilters, key_lookup: WorkspaceKeyLookup
) -> TraceScope:
    if not filters.active:
        return scope
    selected: Final[TraceScope] = {
        **scope,
        "filter_team_id": filters.team_id or "",
        "filter_api_key_hash": filters.api_key_hash or "",
    }
    if not filters.organization_id and not filters.project_id:
        return selected
    return {
        **selected,
        "filter_api_key_hashes_mode": "only",
        "filter_api_key_hashes": await key_lookup(filters),
    }
