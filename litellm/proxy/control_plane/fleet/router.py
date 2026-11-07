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
from litellm.types.proxy.control_plane.dataplanes import UnreachableDataplane
from litellm.types.proxy.control_plane.fleet import FleetModel, FleetModelsResponse, FleetTeam, FleetTeamsResponse

T: Final = TypeVar("T")

router: Final = APIRouter(prefix="/fleet", tags=["fleet"])

FleetRead: TypeAlias = Annotated[Principal, Security(require_scopes, scopes=[Scope.FLEET_READ.value])]
Selected: TypeAlias = Annotated[tuple[Dataplane, ...], Depends(selected_dataplanes)]


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
