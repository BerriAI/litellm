"""Per-team PTU shares on a shared Azure provisioned deployment.

A deployment declaring ``model_info.ptu_shares`` is served only to the teams named in it,
and each team's share converts to a normalized-tokens-per-minute ceiling through the
model's Azure sizing row, so the proxy enforces the split instead of an operator hand
converting PTUs to ``model_tpm_limit``.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final, Generic, TypeAlias, TypeVar

from litellm.litellm_core_utils.ptu_pricing import parsed_ptu_shares, ptu_terms
from litellm.llms.azure.ptu_capacity import (
    PTUCapacity,
    deployment_ptu_capacity,
    is_azure_deployment,
    is_str_keyed_mapping,
)
from litellm.router_utils.common_utils import get_request_team_id, team_may_use_deployment

_DeploymentT: Final = TypeVar("_DeploymentT", bound=Mapping[str, object])
WildcardRoute: TypeAlias = Callable[[str], Sequence[Mapping[str, object]]]


@dataclass(frozen=True, slots=True)
class PTUTeamCeiling:
    model_group: str
    tpm_limit: int
    output_to_input_ratio: float
    cached_input_ratio: float

    def raw_output_token_limit(self) -> int:
        """The ceiling expressed in unweighted tokens: what fits under it when every token is output."""
        return max(1, int(self.tpm_limit / max(self.output_to_input_ratio, 1.0)))


@dataclass(frozen=True, slots=True)
class PTUShareFilterResult(Generic[_DeploymentT]):
    deployments: tuple[_DeploymentT, ...]
    withheld: bool


def _deployment_shares(deployment: Mapping[str, object]) -> Mapping[str, int] | None:
    """The teams a deployment is declared split across, else None.

    The declared map is the access rule on its own; the pricing terms are read separately, so a
    split whose terms are missing still serves only the teams it names rather than everyone.
    """
    model_info: Final = deployment.get("model_info")
    if not is_str_keyed_mapping(model_info):
        return None
    return parsed_ptu_shares(model_info.get("ptu_shares"))


def filter_ptu_shared_deployments(
    healthy_deployments: Sequence[_DeploymentT], request_team_id: str | None
) -> PTUShareFilterResult[_DeploymentT]:
    """Keep only the deployments ``request_team_id`` holds a share on when it holds one, else only
    the ones split into no shares, so a team's ceiling and PTU-hours count just the traffic its
    reservation serves, and an open deployment beside a shared one serves the teams holding none.

    A caller with no team, the master key included, holds no share, the same way an access
    window reserves a deployment away from it.
    """
    team_id: Final = request_team_id or ""
    checks: Final = tuple((deployment, _deployment_shares(deployment)) for deployment in healthy_deployments)
    holds_share: Final = any(shares is not None and team_id in shares for _, shares in checks)
    kept: Final = tuple(
        deployment for deployment, shares in checks if (team_id in shares if shares is not None else not holds_share)
    )
    return PTUShareFilterResult(deployments=kept, withheld=len(kept) < len(checks))


def ptu_share_teams(deployments: Sequence[Mapping[str, object]]) -> frozenset[str]:
    """Every team holding a share on any of ``deployments``."""
    return frozenset[str]().union(
        *(shares for deployment in deployments if (shares := _deployment_shares(deployment)) is not None)
    )


def share_holder_request_kwargs(
    request_kwargs: Mapping[str, object], share_teams: frozenset[str]
) -> Mapping[str, object] | None:
    """``request_kwargs`` when the caller's team is in ``share_teams``, else None."""
    return request_kwargs if get_request_team_id(request_kwargs) in share_teams else None


def deployments_shared_with(deployments: Sequence[Mapping[str, object]], team_id: str) -> tuple[str, ...]:
    """The ids of the deployments split into PTU shares ``team_id`` holds one of, each once."""
    return tuple(
        sorted(
            frozenset(
                str(_deployment_id(deployment))
                for deployment in deployments
                if (shares := _deployment_shares(deployment)) is not None and team_id in shares
            )
        )
    )


def team_ptu_ceiling(
    listed_rows: Sequence[Mapping[str, object]],
    deployments: Sequence[Mapping[str, object]],
    team_id: str,
    requested_model: str,
    wildcard_route: WildcardRoute,
) -> PTUTeamCeiling | None:
    """The per-minute normalized-token ceiling ``team_id``'s shares on the group serving
    ``requested_model`` add up to, else None when the team holds no share on a deployment with
    a known sizing row.

    ``listed_rows`` is every row the router lists a name under, alias and routing-group copies
    included, ``deployments`` is the router's own deployments, and ``wildcard_route`` gives the
    deployments the wildcard route serving a name routes it to. A name resolves to the
    deployments behind it, so a group, a routing group, a deployment id, a wildcard-matched
    name, and a provider model all count against the one ceiling of the group whose shared
    deployment the team can be served from.

    Two shared deployments of different models in one group are weighted by the larger
    output and cached-input ratios, which over-counts those tokens on the cheaper one rather
    than under-counting them on the dearer one.
    """
    model_group: Final = _model_group_of(listed_rows, deployments, team_id, requested_model, wildcard_route)
    priced: Final = tuple(
        (shares[team_id], capacity)
        for deployment in model_group_deployments(deployments, model_group)
        if (shares := _deployment_shares(deployment)) is not None
        and team_id in shares
        and (capacity := deployment_ptu_capacity(deployment)) is not None
    )
    if not priced:
        return None
    return PTUTeamCeiling(
        model_group=model_group,
        tpm_limit=sum(capacity.input_tpm_for(share) for share, capacity in priced),
        output_to_input_ratio=max(capacity.output_to_input_ratio for _, capacity in priced),
        cached_input_ratio=max(capacity.cached_input_ratio for _, capacity in priced),
    )


def model_group_deployments(deployments: Sequence[_DeploymentT], model_group: str) -> tuple[_DeploymentT, ...]:
    """Every deployment serving ``model_group``: by its own name, or by the public name a
    team-scoped deployment keeps in ``model_info.team_public_model_name`` after the router
    renames it to a unique internal one."""
    return tuple(
        deployment
        for deployment in deployments
        if deployment.get("model_name") == model_group
        or (
            is_str_keyed_mapping(model_info := deployment.get("model_info"))
            and model_info.get("team_public_model_name") == model_group
        )
    )


def _deployment_id(deployment: Mapping[str, object]) -> object:
    model_info: Final = deployment.get("model_info")
    return model_info.get("id") if is_str_keyed_mapping(model_info) else None


def _provider_model(deployment: Mapping[str, object]) -> object:
    litellm_params: Final = deployment.get("litellm_params")
    return litellm_params.get("model") if is_str_keyed_mapping(litellm_params) else None


def _deployment_model_group(deployment: Mapping[str, object]) -> str | None:
    model_info: Final = deployment.get("model_info")
    public_name: Final = model_info.get("team_public_model_name") if is_str_keyed_mapping(model_info) else None
    if isinstance(public_name, str):
        return public_name
    model_name: Final = deployment.get("model_name")
    return model_name if isinstance(model_name, str) else None


def _deployments_with_ids(
    deployments: Sequence[Mapping[str, object]], rows: Sequence[Mapping[str, object]]
) -> tuple[Mapping[str, object], ...]:
    ids: Final = frozenset(_deployment_id(row) for row in rows)
    return tuple(deployment for deployment in deployments if _deployment_id(deployment) in ids)


def routed_deployments(
    listed_rows: Sequence[Mapping[str, object]],
    deployments: Sequence[Mapping[str, object]],
    requested_model: str,
    wildcard_route: WildcardRoute,
) -> tuple[Mapping[str, object], ...]:
    """The router's own deployments behind ``requested_model``, in the order the router resolves a
    name: those of the rows listed under it when it names a group, else the one it names by id,
    else those of the one wildcard route that serves it, else the ones it names by provider model."""
    listed: Final = _deployments_with_ids(deployments, model_group_deployments(listed_rows, requested_model))
    if listed:
        return listed
    by_id: Final = tuple(deployment for deployment in deployments if _deployment_id(deployment) == requested_model)
    if by_id:
        return by_id
    wildcard: Final = _deployments_with_ids(deployments, wildcard_route(requested_model))
    if wildcard:
        return wildcard
    return tuple(deployment for deployment in deployments if _provider_model(deployment) == requested_model)


def _model_group_of(
    listed_rows: Sequence[Mapping[str, object]],
    deployments: Sequence[Mapping[str, object]],
    team_id: str,
    requested_model: str,
    wildcard_route: WildcardRoute,
) -> str:
    """The group of the deployment behind ``requested_model`` this team can be served from."""
    servable: Final = team_servable_deployments(
        routed_deployments(listed_rows, deployments, requested_model, wildcard_route), team_id
    )
    return next(
        (group for deployment in servable if (group := _deployment_model_group(deployment)) is not None),
        requested_model,
    )


def _deployment_owner(deployment: Mapping[str, object]) -> object:
    model_info: Final = deployment.get("model_info")
    return model_info.get("team_id") if is_str_keyed_mapping(model_info) else None


def team_servable_deployments(deployments: Sequence[_DeploymentT], team_id: str) -> tuple[_DeploymentT, ...]:
    """The deployments ``team_id`` can be served from: never one reserved for another team, and only
    the ones it holds a share on when it holds any."""
    shareable: Final = filter_ptu_shared_deployments(deployments, team_id).deployments
    return tuple(
        deployment for deployment in shareable if team_may_use_deployment(_deployment_owner(deployment), team_id)
    )


def model_group_ptu_capacity(deployments: Sequence[Mapping[str, object]], day: date) -> PTUCapacity | None:
    """The sizing row of the group's first deployment reserved on ``day``, single-team or shared,
    so a team's tokens on that group that day convert to PTU-hours."""
    return next(
        (
            capacity
            for deployment in deployments
            if is_str_keyed_mapping(model_info := deployment.get("model_info"))
            and (terms := ptu_terms(model_info)) is not None
            and terms.is_active_on(day)
            and (capacity := deployment_ptu_capacity(deployment)) is not None
        ),
        None,
    )


def ptu_capacity_warning(model_name: str, deployment: Mapping[str, object]) -> str | None:
    """Why this reserved deployment's tokens cannot be converted to PTUs, else None.

    Only a deployment that declares shares (whose ceilings need a sizing row) or one served by
    Azure (whose PTU hours need one) is worth warning about; a single-team reservation on another
    provider only ever used the flat-cost rollup, which needs no sizing.
    """
    model_info: Final = deployment.get("model_info")
    if not is_str_keyed_mapping(model_info) or ptu_terms(model_info) is None:
        return None
    if deployment_ptu_capacity(deployment) is not None:
        return None
    has_shares: Final = model_info.get("ptu_shares") is not None
    if has_shares:
        return (
            f"PTU deployment '{model_name}' has no Azure sizing row for its model, so its PTU shares set no "
            "team ceiling and its usage reports no PTU hours; set model_info.base_model to the Azure model name"
        )
    if is_azure_deployment(deployment):
        return (
            f"PTU deployment '{model_name}' has no Azure sizing row for its model, so its usage reports no "
            "PTU hours; set model_info.base_model to the Azure model name"
        )
    return None
