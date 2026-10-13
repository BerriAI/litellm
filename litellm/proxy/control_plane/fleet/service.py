"""Fleet inventory: the union of what every dataplane serves, read through the caller's own identity."""

from http import HTTPStatus
from itertools import groupby
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.proxy.control_plane.common.fanout import Failure, parse_reply
from litellm.proxy.control_plane.dataplanes.client import CallResult, DataplaneClient, DataplaneReply
from litellm.proxy.control_plane.dataplanes.credentials import Caller
from litellm.proxy.control_plane.dataplanes.registry import Dataplane
from litellm.types.proxy.control_plane.fleet import FleetModel, FleetModelGroup, FleetTeam

MODEL_INFO_PATH: Final = "/model/info"
TEAM_LIST_PATH: Final = "/team/list"


class _Lenient(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _LitellmParams(_Lenient):
    model: str | None = None
    custom_llm_provider: str | None = None


class _ModelInfo(_Lenient):
    id: str | None = None
    litellm_provider: str | None = None
    mode: str | None = None


class _ModelEntry(_Lenient):
    model_name: str
    litellm_params: _LitellmParams = _LitellmParams()
    model_info: _ModelInfo = _ModelInfo()


class _ModelInfoBody(_Lenient):
    data: tuple[_ModelEntry, ...] = ()


class _TeamEntry(_Lenient):
    team_id: str
    team_alias: str | None = None
    organization_id: str | None = None
    spend: float | None = None
    max_budget: float | None = None
    members_with_roles: tuple[object, ...] = ()
    models: tuple[str, ...] = ()
    blocked: bool = False


_MODELS: Final = TypeAdapter(_ModelInfoBody)
_TEAMS: Final = TypeAdapter(tuple[_TeamEntry, ...])


async def fetch_models(
    client: DataplaneClient, dataplane: Dataplane, caller: Caller
) -> tuple[FleetModel, ...] | Failure:
    body: Final = parse_reply(await client.get(dataplane, MODEL_INFO_PATH, caller), _MODELS)
    if isinstance(body, Failure):
        return body
    return tuple(
        FleetModel(
            dataplane_id=dataplane.id,
            model_name=entry.model_name,
            litellm_model=entry.litellm_params.model,
            provider=entry.litellm_params.custom_llm_provider or entry.model_info.litellm_provider,
            model_id=entry.model_info.id,
            mode=entry.model_info.mode,
        )
        for entry in body.data
    )


NOT_ALLOWED_TO_LIST_ALL: Final = frozenset({HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN})


def _refused(result: CallResult) -> bool:
    return isinstance(result, DataplaneReply) and result.status in NOT_ALLOWED_TO_LIST_ALL


async def _list_teams(client: DataplaneClient, dataplane: Dataplane, caller: Caller) -> CallResult:
    """Every team when the dataplane lets the caller see them all, else the teams the caller belongs to."""
    everything: Final = await client.get(dataplane, TEAM_LIST_PATH, caller)
    if not _refused(everything):
        return everything
    own: Final = MappingProxyType({"user_id": caller.principal.subject})
    return await client.get(dataplane, TEAM_LIST_PATH, caller, params=own)


async def fetch_teams(client: DataplaneClient, dataplane: Dataplane, caller: Caller) -> tuple[FleetTeam, ...] | Failure:
    teams: Final = parse_reply(await _list_teams(client, dataplane, caller), _TEAMS)
    if isinstance(teams, Failure):
        return teams
    return tuple(
        FleetTeam(
            dataplane_id=dataplane.id,
            team_id=team.team_id,
            team_alias=team.team_alias,
            organization_id=team.organization_id,
            spend=team.spend or 0.0,
            max_budget=team.max_budget,
            members=len(team.members_with_roles),
            models=team.models,
            blocked=team.blocked,
        )
        for team in teams
    )


def group_models(models: tuple[FleetModel, ...]) -> tuple[FleetModelGroup, ...]:
    by_name: Final = sorted(models, key=_model_name)
    return tuple(
        FleetModelGroup(model_name=name, dataplanes=tuple(sorted({model.dataplane_id for model in served})))
        for name, served in groupby(by_name, key=_model_name)
    )


def _model_name(model: FleetModel) -> str:
    return model.model_name
