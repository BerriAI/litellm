"""Email team admins about the deprecating models their team can reach"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.auth_checks import can_team_access_model
from litellm.types.proxy.model_deprecation import ModelDeprecationInfo

if TYPE_CHECKING:
    from litellm.router import Router


@dataclass(frozen=True, slots=True)
class AffectedModel:
    info: ModelDeprecationInfo
    display_name: str
    milestone: int


@dataclass(frozen=True, slots=True)
class _DeploymentOwner:
    model_name: str
    team_id: str
    public_name: str


def select_milestone(days_until: int, thresholds: Sequence[int]) -> int | None:
    """The most urgent threshold already reached, None while the first one is still ahead"""
    return min((threshold for threshold in thresholds if days_until <= threshold), default=None)


def _reached(info: ModelDeprecationInfo, thresholds: Sequence[int]) -> tuple[ModelDeprecationInfo, int] | None:
    milestone: Final = select_milestone(info.days_until_deprecation, thresholds)
    return None if milestone is None else (info, milestone)


def _owner_of(deployment: Mapping[str, object]) -> _DeploymentOwner | None:
    model_name: Final = deployment.get("model_name")
    model_info: Final = deployment.get("model_info")
    if not isinstance(model_name, str) or not isinstance(model_info, Mapping):
        return None
    team_id: Final = model_info.get("team_id")
    if not isinstance(team_id, str) or not team_id:
        return None
    public_name: Final = model_info.get("team_public_model_name")
    return _DeploymentOwner(
        model_name=model_name,
        team_id=team_id,
        public_name=public_name if isinstance(public_name, str) and public_name else model_name,
    )


def _deployment_owners(llm_router: Router) -> Mapping[str, _DeploymentOwner]:
    """Team-scoped deployments keyed by their internal model name, with the name the team knows"""
    owners: Final = (_owner_of(deployment) for deployment in llm_router.get_model_list() or ())
    return MappingProxyType({owner.model_name: owner for owner in owners if owner is not None})


async def _team_can_access(model_name: str, team: LiteLLM_TeamTable, llm_router: Router) -> bool:
    """Reuse the auth check so wildcards, access groups and the empty-list rule match real requests"""
    try:
        await can_team_access_model(model=model_name, team_object=team, llm_router=llm_router)
    except ProxyException:
        return False
    return True


async def _display_name_for(
    info: ModelDeprecationInfo, team: LiteLLM_TeamTable, owner: _DeploymentOwner | None, llm_router: Router
) -> str | None:
    if owner is not None:
        return owner.public_name if owner.team_id == team.team_id else None
    if await _team_can_access(info.model_name, team, llm_router):
        return info.model_name
    return None


async def _affected_model(
    reached: tuple[ModelDeprecationInfo, int],
    team: LiteLLM_TeamTable,
    owners: Mapping[str, _DeploymentOwner],
    llm_router: Router,
) -> AffectedModel | None:
    info, milestone = reached
    display_name: Final = await _display_name_for(info, team, owners.get(info.model_name), llm_router)
    return None if display_name is None else AffectedModel(info=info, display_name=display_name, milestone=milestone)


async def _affected_for_team(
    team: LiteLLM_TeamTable,
    reached: Sequence[tuple[ModelDeprecationInfo, int]],
    owners: Mapping[str, _DeploymentOwner],
    llm_router: Router,
) -> tuple[AffectedModel, ...]:
    candidates: Final = tuple([await _affected_model(pair, team, owners, llm_router) for pair in reached])
    return tuple(model for model in candidates if model is not None)


async def resolve_affected_teams(
    infos: Sequence[ModelDeprecationInfo],
    llm_router: Router,
    teams: Sequence[LiteLLM_TeamTable],
    thresholds: Sequence[int],
) -> Mapping[str, tuple[AffectedModel, ...]]:
    """Per team id, the deprecating models it can reach that have crossed a threshold"""
    reached: Final = tuple(pair for pair in (_reached(info, thresholds) for info in infos) if pair is not None)
    owners: Final = _deployment_owners(llm_router)
    per_team: Final = MappingProxyType(
        {
            team.team_id: await _affected_for_team(team, reached, owners, llm_router)
            for team in teams
            if not team.blocked
        }
    )
    return MappingProxyType({team_id: models for team_id, models in per_team.items() if models})
