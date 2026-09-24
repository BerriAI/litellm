"""Per-team PTU shares on a shared Azure provisioned deployment.

A deployment declaring ``model_info.ptu_shares`` is served only to the teams named in it,
and each team's share converts to a normalized-tokens-per-minute ceiling through the
model's Azure sizing row, so the proxy enforces the split instead of an operator hand
converting PTUs to ``model_tpm_limit``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Generic, TypeVar

from litellm.litellm_core_utils.azure_ptu_capacity import PTUCapacity, deployment_ptu_capacity
from litellm.litellm_core_utils.ptu_pricing import parsed_ptu_shares, ptu_terms

_DeploymentT = TypeVar("_DeploymentT", bound=Mapping[str, object])


@dataclass(frozen=True, slots=True)
class PTUTeamCeiling:
    tpm_limit: int
    output_to_input_ratio: float


@dataclass(frozen=True, slots=True)
class PTUShareFilterResult(Generic[_DeploymentT]):
    deployments: tuple[_DeploymentT, ...]
    withheld: bool


def _deployment_shares(deployment: Mapping[str, object]) -> Mapping[str, int] | None:
    model_info: Final = deployment.get("model_info")
    if not isinstance(model_info, Mapping):
        return None
    return parsed_ptu_shares(model_info.get("ptu_shares"))


def filter_ptu_shared_deployments(
    healthy_deployments: Sequence[_DeploymentT], request_team_id: str | None
) -> PTUShareFilterResult[_DeploymentT]:
    """Drop every deployment split into PTU shares that ``request_team_id`` holds none of.

    A caller with no team, the master key included, holds no share, the same way an access
    window reserves a deployment away from it.
    """
    checks: Final = tuple((deployment, _deployment_shares(deployment)) for deployment in healthy_deployments)
    kept: Final = tuple(
        deployment for deployment, shares in checks if shares is None or (request_team_id or "") in shares
    )
    return PTUShareFilterResult(deployments=kept, withheld=len(kept) < len(checks))


def team_ptu_ceiling(deployments: Sequence[Mapping[str, object]], team_id: str) -> PTUTeamCeiling | None:
    """The per-minute normalized-token ceiling ``team_id``'s shares across ``deployments`` add
    up to, else None when the team holds no share on a deployment with a known sizing row.

    Two shared deployments of different models in one group are weighted by the larger
    output ratio, which over-counts output on the cheaper one rather than under-counting it
    on the dearer one.
    """
    priced: Final = tuple(
        (shares[team_id], capacity)
        for deployment in deployments
        if (shares := _deployment_shares(deployment)) is not None
        and team_id in shares
        and (capacity := deployment_ptu_capacity(deployment)) is not None
    )
    if not priced:
        return None
    return PTUTeamCeiling(
        tpm_limit=sum(share * capacity.input_tpm_per_ptu for share, capacity in priced),
        output_to_input_ratio=max(capacity.output_to_input_ratio for _, capacity in priced),
    )


def model_group_ptu_capacity(deployments: Sequence[Mapping[str, object]]) -> PTUCapacity | None:
    """The sizing row of the group's first reserved deployment, single-team or shared, so a
    team's tokens on that group convert to PTU-hours."""
    return next(
        (
            capacity
            for deployment in deployments
            if isinstance(model_info := deployment.get("model_info"), Mapping)
            and ptu_terms(model_info) is not None
            and (capacity := deployment_ptu_capacity(deployment)) is not None
        ),
        None,
    )


def ptu_capacity_warning(model_name: str, deployment: Mapping[str, object]) -> str | None:
    """Why this reserved deployment's tokens cannot be converted to PTUs, else None."""
    model_info: Final = deployment.get("model_info")
    if not isinstance(model_info, Mapping) or ptu_terms(model_info) is None:
        return None
    if deployment_ptu_capacity(deployment) is not None:
        return None
    return (
        f"PTU deployment '{model_name}' has no Azure sizing row for its model, so its PTU shares set no "
        "team ceiling and its usage reports no PTU hours; set model_info.base_model to the Azure model name"
    )
