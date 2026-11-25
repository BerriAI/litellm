from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict
from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from litellm.proxy.control_plane.store.postgres import PostgresTransaction
from litellm.proxy.control_plane.store.schema import fleet_team_dataplanes as mappings
from litellm.proxy.control_plane.store.schema import fleet_teams as teams
from litellm.types.proxy.control_plane.fleet import FleetTeamBudget, ManagedFleetTeam


class TeamMapping(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataplane_id: str
    dataplane_url: str
    team_id: str


class MappedTeam(TeamMapping, ManagedFleetTeam):
    applied_revision: str | None
    attempted_revision: str | None
    last_error: str | None
    unreachable: bool
    updated_by: str


class MappingKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    fleet_team_id: str
    dataplane_id: str


_JOIN: Final = teams.join(mappings)
_SELECT: Final = select(
    teams.c.id.label("fleet_team_id"),
    teams.c.team_alias,
    teams.c.desired,
    teams.c.revision,
    teams.c.updated_by,
    mappings.c.dataplane_id,
    mappings.c.dataplane_url,
    mappings.c.team_id,
    mappings.c.applied_revision,
    mappings.c.attempted_revision,
    mappings.c.last_error,
    mappings.c.unreachable,
).select_from(_JOIN)
_PENDING: Final = teams.c.desired.is_not(None) & mappings.c.applied_revision.is_distinct_from(teams.c.revision)


async def linked_mappings(connection: AsyncConnection, fleet_team_id: str) -> tuple[MappedTeam, ...]:
    rows: Final = await connection.execute(_SELECT.where(teams.c.id == fleet_team_id).order_by(mappings.c.dataplane_id))
    return tuple(MappedTeam.model_validate(row) for row in rows.mappings())


async def already_linked(connection: AsyncConnection, mapping: TeamMapping) -> bool:
    return bool(
        await connection.scalar(
            select(
                exists().where(mappings.c.dataplane_id == mapping.dataplane_id, mappings.c.team_id == mapping.team_id)
            )
        )
    )


async def adopt(
    connection: AsyncConnection,
    team: ManagedFleetTeam,
    targets: tuple[TeamMapping, ...],
    actor: str,
) -> None:
    await connection.execute(
        teams.insert().values(
            id=team.fleet_team_id, team_alias=team.team_alias, revision=team.revision, updated_by=actor
        )
    )
    await connection.execute(
        mappings.insert(),
        [{**target.model_dump(), "fleet_team_id": team.fleet_team_id} for target in targets],
    )


async def set_desired(
    connection: AsyncConnection, fleet_team_id: str, desired: FleetTeamBudget, revision: str, actor: str
) -> None:
    await connection.execute(
        teams.update()
        .where(teams.c.id == fleet_team_id)
        .values(desired=desired.model_dump(exclude_unset=True), revision=revision, updated_by=actor)
    )


async def refresh_url(connection: AsyncConnection, fleet_team_id: str, mapping: TeamMapping) -> None:
    await connection.execute(
        mappings.update()
        .where(
            mappings.c.fleet_team_id == fleet_team_id,
            mappings.c.dataplane_id == mapping.dataplane_id,
            mappings.c.team_id == mapping.team_id,
        )
        .values(
            dataplane_url=mapping.dataplane_url,
            applied_revision=None,
            attempted_revision=None,
            last_error=None,
            unreachable=False,
        )
    )


async def record_attempt(
    connection: AsyncConnection, target: MappedTeam, error: str | None, *, unreachable: bool
) -> None:
    await connection.execute(
        mappings.update()
        .where(
            mappings.c.fleet_team_id == target.fleet_team_id,
            mappings.c.dataplane_id == target.dataplane_id,
            exists().where(teams.c.id == target.fleet_team_id, teams.c.revision == target.revision),
        )
        .values(
            applied_revision=target.revision if error is None else target.applied_revision,
            attempted_revision=target.revision,
            last_error=error,
            unreachable=unreachable,
        )
    )


@dataclass(frozen=True, slots=True)
class FleetTeamStore:
    engine: AsyncEngine

    async def pending(self) -> tuple[MappingKey, ...]:
        async with self.engine.connect() as connection:
            rows: Final = await connection.execute(
                select(mappings.c.fleet_team_id, mappings.c.dataplane_id)
                .select_from(_JOIN)
                .where(_PENDING)
                .order_by(mappings.c.fleet_team_id, mappings.c.dataplane_id)
            )
            return tuple(MappingKey.model_validate(row) for row in rows.mappings())

    @asynccontextmanager
    async def claim(self, key: MappingKey) -> AsyncGenerator[tuple[PostgresTransaction, MappedTeam] | None]:
        async with self.engine.begin() as connection:
            # Pin the desired revision before claiming the mapping, while allowing sibling mappings to reconcile.
            parent: Final = await connection.scalar(
                select(teams.c.id).where(teams.c.id == key.fleet_team_id).with_for_update(read=True, skip_locked=True)
            )
            if parent is None:
                yield None
                return
            row: Final = (
                (
                    await connection.execute(
                        _SELECT.where(
                            _PENDING, teams.c.id == key.fleet_team_id, mappings.c.dataplane_id == key.dataplane_id
                        ).with_for_update(of=mappings, skip_locked=True)
                    )
                )
                .mappings()
                .one_or_none()
            )
            yield (PostgresTransaction(connection), MappedTeam.model_validate(row)) if row is not None else None
