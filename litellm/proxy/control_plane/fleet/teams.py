import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Final
from uuid import uuid4

from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from litellm.constants import CONTROL_PLANE_FLEET_CONCURRENCY, CONTROL_PLANE_FLEET_RECONCILE_SECONDS
from litellm.proxy.common_utils.timezone_utils import budget_duration_error
from litellm.proxy.control_plane.audit.log import AuditLog
from litellm.proxy.control_plane.audit.pipeline import Reservation
from litellm.proxy.control_plane.auth.rbac import Principal, Scope
from litellm.proxy.control_plane.common.exceptions import (
    ControlPlaneException,
    bad_request,
    error_body,
    forbidden,
    not_found,
    service_unavailable,
)
from litellm.proxy.control_plane.common.fanout import Failure, Fanout
from litellm.proxy.control_plane.dataplanes.client import DataplaneClient
from litellm.proxy.control_plane.dataplanes.registry import Dataplane, DataplaneRegistry
from litellm.proxy.control_plane.fleet.team_dataplanes import (
    BudgetUnreachable,
    differences,
    lookup_alias,
    service_dataplane,
    write_budget,
)
from litellm.proxy.control_plane.store import fleet_teams
from litellm.proxy.control_plane.store.fleet_teams import FleetTeamStore, MappedTeam, MappingKey, TeamMapping
from litellm.proxy.control_plane.store.models import Change
from litellm.proxy.control_plane.store.postgres import PostgresStore, PostgresTransaction
from litellm.proxy.control_plane.store.runtime import NOT_CONFIGURED, RuntimeConfig
from litellm.proxy.control_plane.store.state import InvalidSnapshot, Snapshot
from litellm.proxy.control_plane.store.stream import ConfigStream
from litellm.types.proxy.control_plane.fleet import (
    FleetTeamAdoption,
    FleetTeamAdoptionConflict,
    FleetTeamAdoptionProblem,
    FleetTeamBudget,
    FleetTeamDataplaneStatus,
    FleetTeamStatus,
    ManagedFleetTeam,
)

LOGGER: Final = logging.getLogger("litellm.control_plane.fleet")
_STORE_ERRORS: Final = (SQLAlchemyError, RedisError, OSError)


def _require_scope(snapshot: Snapshot, principal: Principal, scope: Scope) -> None:
    granted: Final = principal.token_kind == "master_key" or scope in snapshot.policy.grant(principal.roles)
    if not principal.has_scope(scope) or not granted:
        raise ControlPlaneException(forbidden(f"Missing scope in the committed policy: {scope.value}."))


def _can_access(principal: Principal, targets: tuple[str, ...]) -> bool:
    return all(principal.dataplanes.allows(dataplane_id) for dataplane_id in targets)


def _authorize_adoption(snapshot: Snapshot, principal: Principal, targets: tuple[str, ...]) -> None:
    _require_scope(snapshot, principal, Scope.FLEET_WRITE)
    if not _can_access(principal, targets):
        raise ControlPlaneException(forbidden("Access to every target dataplane is required."))


async def _linked_team(
    transaction: PostgresTransaction, snapshot: Snapshot, fleet_team_id: str, principal: Principal, scope: Scope
) -> tuple[MappedTeam, ...]:
    _require_scope(snapshot, principal, scope)
    targets: Final = await fleet_teams.linked_mappings(transaction.connection, fleet_team_id)
    if not targets or not _can_access(principal, tuple(target.dataplane_id for target in targets)):
        raise ControlPlaneException(not_found("Fleet team not found."))
    return targets


def _target(
    registry: DataplaneRegistry, dataplane_id: str, pinned_url: str | None = None
) -> Dataplane | BudgetUnreachable:
    dataplane: Final = registry.get(dataplane_id)
    if dataplane is None or (pinned_url is not None and dataplane.url != pinned_url):
        return BudgetUnreachable("Dataplane is missing or its registered URL changed.")
    return service_dataplane(dataplane) or BudgetUnreachable("Dataplane requires a configured service token.")


def _reject_adoption(conflicts: tuple[FleetTeamAdoptionConflict, ...]) -> None:
    if conflicts:
        problem: Final = error_body("fleet-team-adoption-conflict", "Conflict", 409, "Fleet team adoption failed.")
        raise ControlPlaneException(
            FleetTeamAdoptionProblem(
                type=problem.type,
                title=problem.title,
                status=problem.status,
                detail=problem.detail,
                conflicts=tuple(conflicts),
            )
        )


@dataclass(slots=True)
class FleetTeams:
    runtime: RuntimeConfig
    client: DataplaneClient
    fanout: Fanout
    audit: AuditLog
    task: asyncio.Task[None] | None = None
    _status_gate: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(CONTROL_PLANE_FLEET_CONCURRENCY),
        init=False,
        repr=False,
        compare=False,
    )

    def _config(self) -> tuple[PostgresStore, ConfigStream]:
        store, stream = self.runtime.store, self.runtime.stream
        if not self.runtime.ready or not isinstance(store, PostgresStore) or stream is None:
            raise ControlPlaneException(service_unavailable(NOT_CONFIGURED))
        return store, stream

    async def _snapshot(self, transaction: PostgresTransaction) -> Snapshot:
        snapshot: Final = self.runtime.state.build_committed(await transaction.load(), {})
        if isinstance(snapshot, InvalidSnapshot):
            raise ControlPlaneException(service_unavailable("Runtime config snapshot is invalid."))
        return snapshot

    async def adopt(self, data: FleetTeamAdoption, principal: Principal) -> ManagedFleetTeam:
        store, stream = self._config()
        try:
            async with store.transaction() as transaction:
                snapshot: Final = await self._snapshot(transaction)
                _authorize_adoption(snapshot, principal, data.dataplanes)
                if data.fleet_team_id is not None:
                    await _linked_team(transaction, snapshot, data.fleet_team_id, principal, Scope.FLEET_WRITE)
            targets: Final[list[Dataplane]] = []  # mutable-ok: collect valid fanout targets
            conflicts: Final[list[FleetTeamAdoptionConflict]] = []  # mutable-ok: collect all adoption conflicts
            for dataplane_id in data.dataplanes:
                match _target(snapshot.registry, dataplane_id):
                    case BudgetUnreachable(reason=reason):
                        conflicts.append(FleetTeamAdoptionConflict(dataplane_id=dataplane_id, reason=reason))
                    case Dataplane() as target:
                        targets.append(target)
            found: Final = await self.fanout.gather(
                tuple(targets), lambda dataplane: lookup_alias(self.client, dataplane, data.team_alias)
            )
            conflicts.extend(
                FleetTeamAdoptionConflict(dataplane_id=item.dataplane.id, reason=item.reason) for item in found.missed
            )
            matched: Final = tuple(item.value for item in found.reached)
            async with store.transaction(fleet_team_id=data.fleet_team_id) as transaction:
                current: Final = await self._snapshot(transaction)
                _authorize_adoption(current, principal, data.dataplanes)
                for mapping in matched:
                    match _target(current.registry, mapping.dataplane_id, mapping.dataplane_url):
                        case BudgetUnreachable(reason=reason):
                            conflicts.append(
                                FleetTeamAdoptionConflict(dataplane_id=mapping.dataplane_id, reason=reason)
                            )
                        case Dataplane():
                            if data.fleet_team_id is None and await fleet_teams.already_linked(
                                transaction.connection, mapping
                            ):
                                conflicts.append(
                                    FleetTeamAdoptionConflict(
                                        dataplane_id=mapping.dataplane_id, reason="Team is already mapped."
                                    )
                                )
                _reject_adoption(tuple(conflicts))
                if data.fleet_team_id is not None:
                    return await self._refresh_adoption(transaction, stream, data, matched, principal, current)
                fleet_team_id: Final = str(uuid4())
                revision: Final = await stream.append(
                    Change(op="fleet_team", name=fleet_team_id, actor=principal.subject)
                )
                team: Final = ManagedFleetTeam(
                    fleet_team_id=fleet_team_id, team_alias=data.team_alias, revision=revision, desired=None
                )
                await fleet_teams.adopt(transaction.connection, team, matched, principal.subject)
        except _STORE_ERRORS as error:
            raise ControlPlaneException(service_unavailable("Fleet team persistence failed.")) from error
        return team

    async def _refresh_adoption(
        self,
        transaction: PostgresTransaction,
        stream: ConfigStream,
        data: FleetTeamAdoption,
        matched: tuple[TeamMapping, ...],
        principal: Principal,
        snapshot: Snapshot,
    ) -> ManagedFleetTeam:
        assert data.fleet_team_id is not None
        linked: Final = await _linked_team(transaction, snapshot, data.fleet_team_id, principal, Scope.FLEET_WRITE)
        team: Final = linked[0]
        by_dataplane: Final = {mapping.dataplane_id: mapping for mapping in linked}
        conflicts: Final = tuple(
            FleetTeamAdoptionConflict(dataplane_id=mapping.dataplane_id, reason="Existing team link does not match.")
            for mapping in matched
            if data.team_alias != team.team_alias
            or mapping.dataplane_id not in by_dataplane
            or by_dataplane[mapping.dataplane_id].team_id != mapping.team_id
        )
        _reject_adoption(conflicts)
        refreshed: Final = tuple(
            mapping for mapping in matched if by_dataplane[mapping.dataplane_id].dataplane_url != mapping.dataplane_url
        )
        if refreshed:
            await stream.append(Change(op="fleet_team", name=data.fleet_team_id, actor=principal.subject))
            for mapping in refreshed:
                await fleet_teams.refresh_url(transaction.connection, data.fleet_team_id, mapping)
        return ManagedFleetTeam(
            fleet_team_id=team.fleet_team_id, team_alias=team.team_alias, revision=team.revision, desired=team.desired
        )

    async def put(self, fleet_team_id: str, data: FleetTeamBudget, principal: Principal) -> ManagedFleetTeam:
        error_detail: Final = budget_duration_error(data.budget_duration)
        if error_detail is not None:
            raise ControlPlaneException(bad_request(error_detail))
        store, stream = self._config()
        try:
            async with store.transaction(fleet_team_id=fleet_team_id) as transaction:
                target, _ = await self._authorized_team(transaction, fleet_team_id, principal, Scope.FLEET_WRITE)
                revision: Final = await stream.append(
                    Change(op="fleet_team", name=fleet_team_id, actor=principal.subject)
                )
                await fleet_teams.set_desired(transaction.connection, fleet_team_id, data, revision, principal.subject)
            self.runtime.fleet_changed.set()
            return ManagedFleetTeam(
                fleet_team_id=fleet_team_id, team_alias=target[0].team_alias, revision=revision, desired=data
            )
        except _STORE_ERRORS as error:
            raise ControlPlaneException(service_unavailable("Fleet team persistence failed.")) from error

    async def _authorized_team(
        self, transaction: PostgresTransaction, fleet_team_id: str, principal: Principal, scope: Scope
    ) -> tuple[tuple[MappedTeam, ...], Snapshot]:
        snapshot: Final = await self._snapshot(transaction)
        targets: Final = await _linked_team(transaction, snapshot, fleet_team_id, principal, scope)
        return targets, snapshot

    async def status(self, fleet_team_id: str, principal: Principal) -> FleetTeamStatus:
        store, _ = self._config()
        try:
            async with store.transaction() as transaction:
                targets, snapshot = await self._authorized_team(transaction, fleet_team_id, principal, Scope.FLEET_READ)
        except _STORE_ERRORS as error:
            raise ControlPlaneException(service_unavailable("Fleet team status unavailable.")) from error
        statuses: Final = await asyncio.gather(*(self._status(target, snapshot.registry) for target in targets))
        team: Final = targets[0]
        return FleetTeamStatus(
            fleet_team_id=team.fleet_team_id,
            team_alias=team.team_alias,
            revision=team.revision,
            desired=team.desired,
            dataplanes=tuple(statuses),
        )

    async def _status(self, target: MappedTeam, registry: DataplaneRegistry) -> FleetTeamDataplaneStatus:
        desired_revision: Final = target.revision if target.desired is not None else None
        last_error: Final = target.last_error if target.attempted_revision == desired_revision else None
        base_state: Final = "unreachable" if last_error is not None and target.unreachable else "pending"
        if target.desired is not None and target.applied_revision == desired_revision:
            dataplane: Final = _target(registry, target.dataplane_id, target.dataplane_url)
            async with self._status_gate:
                result: Final = (
                    dataplane
                    if isinstance(dataplane, Failure)
                    else await differences(self.client, dataplane, target.team_id, target.desired)
                )
            observation: Final = (
                "unreachable"
                if isinstance(result, BudgetUnreachable)
                else "pending"
                if isinstance(result, Failure)
                else "drifted"
                if result
                else "applied"
            )
            return FleetTeamDataplaneStatus(
                dataplane_id=target.dataplane_id,
                team_id=target.team_id,
                desired_revision=desired_revision,
                applied_revision=target.applied_revision,
                last_error=result.reason if isinstance(result, Failure) else last_error,
                state=observation,
            )
        return FleetTeamDataplaneStatus(
            dataplane_id=target.dataplane_id,
            team_id=target.team_id,
            desired_revision=desired_revision,
            applied_revision=target.applied_revision,
            last_error=last_error,
            state=base_state,
        )

    def start(self) -> None:
        store: Final = self.runtime.store
        if isinstance(store, PostgresStore) and self.runtime.stream is not None:
            self.task = asyncio.create_task(self._run(FleetTeamStore(store.engine)), name="control-plane-fleet-teams")

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task

    async def _run(self, store: FleetTeamStore) -> None:
        wake: Final = self.runtime.fleet_changed
        gate: Final = asyncio.Semaphore(CONTROL_PLANE_FLEET_CONCURRENCY)

        async def reconcile_key(key: MappingKey) -> None:
            try:
                async with gate:
                    await self._reconcile(store, key)
            except Exception:
                LOGGER.warning(
                    "Fleet team %s reconciliation failed for dataplane %s.",
                    key.fleet_team_id,
                    key.dataplane_id,
                    exc_info=True,
                )

        async def reconcile_pending() -> None:
            keys: Final = await store.pending()
            await asyncio.gather(*(reconcile_key(key) for key in keys))

        while True:
            wake.clear()
            try:
                if self.runtime.ready:
                    await reconcile_pending()
            except Exception:
                LOGGER.warning("Fleet team reconciliation temporarily unavailable.", exc_info=True)
            with suppress(TimeoutError):
                await asyncio.wait_for(wake.wait(), timeout=CONTROL_PLANE_FLEET_RECONCILE_SECONDS)

    async def _reconcile(self, store: FleetTeamStore, key: MappingKey) -> None:
        async with store.claim(key) as claim:
            if claim is None:
                return
            transaction, target = claim
            snapshot: Final = await self._snapshot(transaction)
            dataplane: Final = _target(snapshot.registry, target.dataplane_id, target.dataplane_url)
            if isinstance(dataplane, Failure):
                await fleet_teams.record_attempt(transaction.connection, target, dataplane.reason, unreachable=True)
                self._audit_attempt(target, dataplane)
                return
            if target.desired is None:
                return
            observed: Final = await differences(self.client, dataplane, target.team_id, target.desired)
            if isinstance(observed, Failure):
                await fleet_teams.record_attempt(
                    transaction.connection, target, observed.reason, unreachable=isinstance(observed, BudgetUnreachable)
                )
                self._audit_attempt(target, observed)
                return
            reservation: Final = self.audit.reserve() if observed else None
            if observed and reservation is None:
                await fleet_teams.record_attempt(
                    transaction.connection, target, "Audit sink backpressure.", unreachable=False
                )
                return
            try:
                outcome: Final = (
                    await write_budget(self.client, dataplane, target.team_id, observed) if observed else None
                )
                self._audit_attempt(target, outcome, reservation)
                await fleet_teams.record_attempt(
                    transaction.connection,
                    target,
                    outcome.reason if outcome is not None else None,
                    unreachable=isinstance(outcome, BudgetUnreachable),
                )
            finally:
                if reservation is not None:
                    reservation.release()

    def _audit_attempt(
        self, target: MappedTeam, failure: Failure | None, reservation: Reservation | None = None
    ) -> None:
        self.audit.record(
            target.updated_by,
            None,
            target.dataplane_id,
            "POST",
            "/team/update",
            "allowed" if failure is None else "unreachable" if isinstance(failure, BudgetUnreachable) else "denied",
            detail=f"Fleet team {target.fleet_team_id}, revision {target.revision}: "
            + (failure.reason if failure is not None else "budget applied or already matched"),
            credential="service_token",
            reservation=reservation,
        )
