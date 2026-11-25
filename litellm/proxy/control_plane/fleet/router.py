from typing import Annotated, Final, TypeAlias, TypeVar

from fastapi import APIRouter, Depends, Security

from litellm.proxy.control_plane.auth.dependencies import require_scopes
from litellm.proxy.control_plane.auth.rbac import Principal, Scope
from litellm.proxy.control_plane.common.exceptions import ControlPlaneException, service_unavailable
from litellm.proxy.control_plane.common.fanout import Fanout, Gathered, describe_missed
from litellm.proxy.control_plane.dataplanes.client import DataplaneClient
from litellm.proxy.control_plane.dataplanes.credentials import Caller
from litellm.proxy.control_plane.dataplanes.dependencies import (
    get_caller,
    get_dataplane_client,
    get_fanout,
    selected_dataplanes,
)
from litellm.proxy.control_plane.dataplanes.registry import Dataplane
from litellm.proxy.control_plane.fleet.service import fetch_models, fetch_teams, group_models
from litellm.proxy.control_plane.fleet.teams import FleetTeams
from litellm.proxy.control_plane.store.dependencies import audited_fleet_write
from litellm.types.proxy.control_plane.dataplanes import UnreachableDataplane
from litellm.types.proxy.control_plane.fleet import (
    FleetModel,
    FleetModelsResponse,
    FleetTeam,
    FleetTeamAdoption,
    FleetTeamAdoptionProblem,
    FleetTeamBudget,
    FleetTeamsResponse,
    FleetTeamStatus,
    ManagedFleetTeam,
)

T = TypeVar("T")

router: Final = APIRouter(prefix="/fleet", tags=["fleet"])

FleetRead: TypeAlias = Annotated[Principal, Security(require_scopes, scopes=[Scope.FLEET_READ.value])]
FleetWrite: TypeAlias = Annotated[Principal, Security(audited_fleet_write, scopes=[Scope.FLEET_WRITE.value])]
Selected: TypeAlias = Annotated[tuple[Dataplane, ...], Depends(selected_dataplanes)]


def get_fleet_teams() -> FleetTeams:
    raise RuntimeError("get_fleet_teams is not bound, build the app with create_app")


def _or_503(gathered: Gathered[T]) -> Gathered[T]:
    if gathered.nothing_answered:
        raise ControlPlaneException(service_unavailable(f"No dataplane answered: {describe_missed(gathered.missed)}"))
    return gathered


def _flatten(gathered: Gathered[tuple[T, ...]]) -> tuple[T, ...]:
    return tuple(item for reached in gathered.reached for item in reached.value)  # comprehension-ok: flatten


def _unreachable(gathered: Gathered[T]) -> tuple[UnreachableDataplane, ...]:
    return tuple(UnreachableDataplane(id=miss.dataplane.id, reason=miss.reason) for miss in gathered.missed)


@router.get("/models", response_model=FleetModelsResponse)
async def fleet_models(
    _principal: FleetRead,
    dataplanes: Selected,
    caller: Annotated[Caller, Depends(get_caller)],
    client: Annotated[DataplaneClient, Depends(get_dataplane_client)],
    fanout: Annotated[Fanout, Depends(get_fanout)],
) -> FleetModelsResponse:
    """Every deployment across the fleet, plus which dataplanes serve each public model name."""
    gathered: Final[Gathered[tuple[FleetModel, ...]]] = _or_503(
        await fanout.gather(dataplanes, lambda dataplane: fetch_models(client, dataplane, caller))
    )
    items: Final = _flatten(gathered)
    return FleetModelsResponse(
        items=items, groups=group_models(items), partial=gathered.partial, unreachable=_unreachable(gathered)
    )


@router.get("/teams", response_model=FleetTeamsResponse)
async def fleet_teams(
    _principal: FleetRead,
    dataplanes: Selected,
    caller: Annotated[Caller, Depends(get_caller)],
    client: Annotated[DataplaneClient, Depends(get_dataplane_client)],
    fanout: Annotated[Fanout, Depends(get_fanout)],
) -> FleetTeamsResponse:
    """Every team the caller may see on each dataplane."""
    gathered: Final[Gathered[tuple[FleetTeam, ...]]] = _or_503(
        await fanout.gather(dataplanes, lambda dataplane: fetch_teams(client, dataplane, caller))
    )
    return FleetTeamsResponse(items=_flatten(gathered), partial=gathered.partial, unreachable=_unreachable(gathered))


@router.post(
    "/teams/adopt",
    response_model=ManagedFleetTeam,
    response_model_exclude_unset=True,
    responses={409: {"model": FleetTeamAdoptionProblem}},
)
async def adopt_team(
    data: FleetTeamAdoption, principal: FleetWrite, service: Annotated[FleetTeams, Depends(get_fleet_teams)]
) -> ManagedFleetTeam:
    """Link existing teams by exact alias. Supply fleet_team_id to refresh URLs for existing links only."""
    return await service.adopt(data, principal)


@router.put("/teams/{fleet_team_id}", response_model=ManagedFleetTeam, response_model_exclude_unset=True)
async def put_team(
    fleet_team_id: str,
    data: FleetTeamBudget,
    principal: FleetWrite,
    service: Annotated[FleetTeams, Depends(get_fleet_teams)],
) -> ManagedFleetTeam:
    """Replace the managed budget fields. Omit a field to leave it unmanaged, or use null to clear it.

    Reconciliation reads before writing. If a worker crashes after a regional commit but before saving
    progress, a retry skips matching values. A local edit in that interval can be overwritten by the retry.
    """
    return await service.put(fleet_team_id, data, principal)


@router.get("/teams/{fleet_team_id}/status", response_model=FleetTeamStatus, response_model_exclude_unset=True)
async def team_status(
    fleet_team_id: str, principal: FleetRead, service: Annotated[FleetTeams, Depends(get_fleet_teams)]
) -> FleetTeamStatus:
    """Compare applied budgets with current regional settings. Drift is reported, not overwritten."""
    return await service.status(fleet_team_id, principal)
