"""Project the rows a caller belongs to onto the flat fields of ``UserAPIKeyAuth``.

Each row has one projection here, so every auth path that holds a project, an organization or a key's owner
stamps the same fields from it. A team's projection is ``team_grants``.
"""

from collections.abc import Mapping
from typing import Final

from typing_extensions import ReadOnly, TypedDict

from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.models.user import LiteLLM_UserTable
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


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
