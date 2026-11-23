"""Project the rows a caller belongs to onto the flat fields of ``UserAPIKeyAuth``.

Each row has one projection here, so every auth path that holds a project, an organization, a key's owner or
an end user stamps the same fields from it. A team's projection is ``team_grants``.
"""

from collections.abc import Mapping
from typing import Final

from typing_extensions import ReadOnly, TypedDict

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.models.user import LiteLLM_UserTable
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.router import AllowedModelRegion


class ProjectGrants(TypedDict):
    project_alias: ReadOnly[str | None]
    project_metadata: ReadOnly[Mapping[str, object] | None]


class OrganizationGrants(TypedDict):
    organization_alias: ReadOnly[str | None]
    organization_metadata: ReadOnly[Mapping[str, object] | None]
    organization_max_budget: ReadOnly[float | None]
    organization_tpm_limit: ReadOnly[int | None]
    organization_rpm_limit: ReadOnly[int | None]


class OwnerGrants(TypedDict):
    user_tpm_limit: ReadOnly[int | None]
    user_rpm_limit: ReadOnly[int | None]
    user_email: ReadOnly[str | None]
    user_spend: ReadOnly[float | None]
    user_max_budget: ReadOnly[float | None]
    user_model_max_budget: ReadOnly[Mapping[str, object] | None]


class EndUserGrants(TypedDict, total=False):
    """The fields one end user sets. A key that is absent leaves the token's own value."""

    end_user_id: ReadOnly[str]
    allowed_model_region: ReadOnly[AllowedModelRegion | None]
    end_user_tpm_limit: ReadOnly[int]
    end_user_rpm_limit: ReadOnly[int]
    end_user_tpd_limit: ReadOnly[int]
    end_user_max_budget: ReadOnly[float]
    end_user_model_max_budget: ReadOnly[Mapping[str, object]]
    end_user_model_tpm_limit: ReadOnly[Mapping[str, int]]
    end_user_model_rpm_limit: ReadOnly[Mapping[str, int]]


def project_grants(project: LiteLLM_ProjectTable) -> ProjectGrants:
    return {"project_alias": project.project_alias, "project_metadata": project.metadata}


def organization_grants(organization: LiteLLM_OrganizationTable) -> OrganizationGrants:
    budget: Final = organization.litellm_budget_table
    return {
        "organization_alias": organization.organization_alias,
        "organization_metadata": organization.metadata,
        "organization_max_budget": None if budget is None else budget.max_budget,
        "organization_tpm_limit": None if budget is None else budget.tpm_limit,
        "organization_rpm_limit": None if budget is None else budget.rpm_limit,
    }


def owner_grants(owner: LiteLLM_UserTable) -> OwnerGrants:
    return {
        "user_tpm_limit": owner.tpm_limit,
        "user_rpm_limit": owner.rpm_limit,
        "user_email": owner.user_email,
        "user_spend": owner.spend,
        "user_max_budget": owner.effective_max_budget(),
        "user_model_max_budget": owner.model_max_budget,
    }


def with_project(token: UserAPIKeyAuth, project: LiteLLM_ProjectTable | None) -> UserAPIKeyAuth:
    return token if project is None else token.model_copy(update=project_grants(project))


def with_owner_model_budget(token: UserAPIKeyAuth, owner: LiteLLM_UserTable | None) -> UserAPIKeyAuth:
    """The key with its owner's per-model budget alone, which the budget checks read before the key path ends
    and stamps the rest of the owner's values."""
    if owner is None:
        return token
    return token.model_copy(update={"user_model_max_budget": owner_grants(owner)["user_model_max_budget"]})


def stamp(token: UserAPIKeyAuth, grants: Mapping[str, object]) -> None:
    """Writes grants onto a token its caller already handed on, where a copy would be lost."""
    for name, value in grants.items():
        setattr(token, name, value)


def end_user_budget_grants(budget: LiteLLM_BudgetTable) -> EndUserGrants:
    """The limits a budget sets."""
    tpm: Final[EndUserGrants] = {} if budget.tpm_limit is None else {"end_user_tpm_limit": budget.tpm_limit}
    rpm: Final[EndUserGrants] = {} if budget.rpm_limit is None else {"end_user_rpm_limit": budget.rpm_limit}
    tpd: Final[EndUserGrants] = {} if budget.tpd_limit is None else {"end_user_tpd_limit": budget.tpd_limit}
    spend: Final[EndUserGrants] = {} if budget.max_budget is None else {"end_user_max_budget": budget.max_budget}
    model_spend: Final[EndUserGrants] = (
        {} if budget.model_max_budget is None else {"end_user_model_max_budget": budget.model_max_budget}
    )
    model_tpm: Final[EndUserGrants] = (
        {} if budget.model_tpm_limit is None else {"end_user_model_tpm_limit": budget.model_tpm_limit}
    )
    model_rpm: Final[EndUserGrants] = (
        {} if budget.model_rpm_limit is None else {"end_user_model_rpm_limit": budget.model_rpm_limit}
    )
    return {**tpm, **rpm, **tpd, **spend, **model_spend, **model_tpm, **model_rpm}


def end_user_grants(
    end_user_id: str, end_user: LiteLLM_EndUserTable | None, budget: LiteLLM_BudgetTable | None
) -> EndUserGrants:
    """What a loaded end user sets: its region when the row exists, and its budget's limits."""
    region: Final[EndUserGrants] = {} if end_user is None else {"allowed_model_region": end_user.allowed_model_region}
    limits: Final[EndUserGrants] = {} if budget is None else end_user_budget_grants(budget)
    return {"end_user_id": end_user_id, **region, **limits}


def with_end_user(token: UserAPIKeyAuth, grants: EndUserGrants) -> UserAPIKeyAuth:
    """The key with an end user's limits, except its max budget. A limit that is unset keeps the key's own."""
    return token.model_copy(
        update={
            **{name: value for name, value in grants.items() if value is not None and name != "end_user_max_budget"},
            "end_user_id": grants.get("end_user_id"),
        }
    )


def _end_user_limits_on(token: UserAPIKeyAuth) -> Mapping[str, object]:
    return {
        "end_user_max_budget": token.end_user_max_budget,
        "end_user_tpm_limit": token.end_user_tpm_limit,
        "end_user_rpm_limit": token.end_user_rpm_limit,
        "end_user_tpd_limit": token.end_user_tpd_limit,
        "end_user_model_max_budget": token.end_user_model_max_budget,
        "end_user_model_tpm_limit": token.end_user_model_tpm_limit,
        "end_user_model_rpm_limit": token.end_user_model_rpm_limit,
    }


def default_end_user_limits(
    token: UserAPIKeyAuth, budget: LiteLLM_BudgetTable, *, keep_token_limits: bool
) -> Mapping[str, object]:
    """The end-user limits a default budget gives a token: every one of them, a limit the budget lacks
    included, or with ``keep_token_limits`` only the ones the token left unset."""
    on_token: Final = _end_user_limits_on(token)
    kept: Final = {name: limit for name, limit in on_token.items() if limit is not None} if keep_token_limits else {}
    return {**dict.fromkeys(on_token), **end_user_budget_grants(budget), **kept}
