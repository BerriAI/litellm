"""Records the Python auth host answers with, and the conversions that stamp one row's values onto a key.
python-bridge reads the records by attribute, so a field rename here is a change to the Rust side too
(`python-bridge/src/auth/call.rs`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Final

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles
from litellm.types.router import AllowedModelRegion


@dataclass(frozen=True, slots=True)
class KeyFacts:
    token: str | None
    blocked: bool
    expires: float | None
    is_proxy_admin: bool
    user_id: str | None
    team_id: str | None
    team_object_permission_id: str | None
    project_id: str | None
    end_user_id: str | None
    last_refreshed_at: float | None


class EndUserParams(TypedDict, total=False):
    """The `UserAPIKeyAuth` fields one end user sets. A key that is absent leaves the token's own value."""

    end_user_id: ReadOnly[str]
    allowed_model_region: ReadOnly[AllowedModelRegion | None]
    end_user_tpm_limit: ReadOnly[int]
    end_user_rpm_limit: ReadOnly[int]
    end_user_tpd_limit: ReadOnly[int]
    end_user_max_budget: ReadOnly[float]
    end_user_model_max_budget: ReadOnly[Mapping[str, object]]
    end_user_model_tpm_limit: ReadOnly[Mapping[str, int]]
    end_user_model_rpm_limit: ReadOnly[Mapping[str, int]]


@dataclass(frozen=True, slots=True)
class EndUserRow:
    """The request's end user and the limits the key path stamps onto the token from it."""

    end_user_id: str | None
    end_user: LiteLLM_EndUserTable | None
    params: EndUserParams


def budget_params(budget: LiteLLM_BudgetTable) -> EndUserParams:
    """The limits a budget sets."""
    tpm: Final[EndUserParams] = {} if budget.tpm_limit is None else {"end_user_tpm_limit": budget.tpm_limit}
    rpm: Final[EndUserParams] = {} if budget.rpm_limit is None else {"end_user_rpm_limit": budget.rpm_limit}
    tpd: Final[EndUserParams] = {} if budget.tpd_limit is None else {"end_user_tpd_limit": budget.tpd_limit}
    spend: Final[EndUserParams] = {} if budget.max_budget is None else {"end_user_max_budget": budget.max_budget}
    model_spend: Final[EndUserParams] = (
        {} if budget.model_max_budget is None else {"end_user_model_max_budget": budget.model_max_budget}
    )
    model_tpm: Final[EndUserParams] = (
        {} if budget.model_tpm_limit is None else {"end_user_model_tpm_limit": budget.model_tpm_limit}
    )
    model_rpm: Final[EndUserParams] = (
        {} if budget.model_rpm_limit is None else {"end_user_model_rpm_limit": budget.model_rpm_limit}
    )
    return {**tpm, **rpm, **tpd, **spend, **model_spend, **model_tpm, **model_rpm}


def end_user_params(
    end_user_id: str, end_user: LiteLLM_EndUserTable | None, budget: LiteLLM_BudgetTable | None
) -> EndUserParams:
    """What a loaded end user sets: its region when the row exists, and its budget's limits."""
    region: Final[EndUserParams] = {} if end_user is None else {"allowed_model_region": end_user.allowed_model_region}
    limits: Final[EndUserParams] = {} if budget is None else budget_params(budget)
    return {"end_user_id": end_user_id, **region, **limits}


@dataclass(frozen=True, slots=True)
class TeamFacts:
    team_id: str
    models: Sequence[str] | None
    allowed_passthrough_routes: Sequence[str] | None
    denied_passthrough_routes: Sequence[str]
    members: Sequence[str]


@dataclass(frozen=True, slots=True)
class FoundTeam:
    facts: TeamFacts
    row: LiteLLM_TeamTable


_ROUTES: Final = TypeAdapter(tuple[str, ...] | None)


def team_facts(team: LiteLLM_TeamTable) -> FoundTeam:
    metadata: Final = team.metadata or {}
    return FoundTeam(
        facts=TeamFacts(
            team_id=team.team_id,
            models=tuple(team.models),
            allowed_passthrough_routes=_ROUTES.validate_python(metadata.get("allowed_passthrough_routes")),
            denied_passthrough_routes=_ROUTES.validate_python(metadata.get("denied_passthrough_routes")) or (),
            members=tuple(member.user_id for member in team.members_with_roles if member.user_id is not None),
        ),
        row=team,
    )


class TeamMissing(str, Enum):
    """Why `get_team_object` found no team: its `TeamNotFoundError`, or any other `HTTPException`."""

    ABSENT = "absent"
    UNREADABLE = "unreadable"


class AliasMissing(str, Enum):
    """Why `get_team_object_by_alias` found no team: its 404, its 400, or no database connected."""

    ABSENT = "absent"
    AMBIGUOUS = "ambiguous"
    NO_DB = "no_db"


@dataclass(frozen=True, slots=True)
class UserFacts:
    user_id: str
    user_email: str | None
    user_role: str | None
    teams: Sequence[str]
    deactivated: bool


@dataclass(frozen=True, slots=True)
class RegisteredAgentFacts:
    agent_id: str
    identity_managed: bool
    has_identity: bool
    has_legacy_identity: bool


def deactivated(user: LiteLLM_UserTable) -> bool:
    """Deactivated through SCIM."""
    return isinstance(user.metadata, dict) and user.metadata.get("scim_active") is False


def user_facts(user: LiteLLM_UserTable) -> UserFacts:
    return UserFacts(
        user_id=user.user_id,
        user_email=user.user_email,
        user_role=user.user_role,
        teams=tuple(user.teams or ()),
        deactivated=deactivated(user),
    )


def with_owner(token: UserAPIKeyAuth, owner: LiteLLM_UserTable | None) -> UserAPIKeyAuth:
    """The key with its owner's per-model budget, which the budget checks and the identity read."""
    return token if owner is None else token.model_copy(update={"user_model_max_budget": owner.model_max_budget})


def with_end_user(token: UserAPIKeyAuth, params: EndUserParams) -> UserAPIKeyAuth:
    """The key with an end user's limits, except its max budget. A limit that is unset keeps the key's own."""
    return token.model_copy(
        update={
            **{name: value for name, value in params.items() if value is not None and name != "end_user_max_budget"},
            "end_user_id": params.get("end_user_id"),
        }
    )


def with_team(
    token: UserAPIKeyAuth,
    object_permission: LiteLLM_ObjectPermissionTable | None,
    metadata: Mapping[str, object] | None,
) -> UserAPIKeyAuth:
    return token.model_copy(update={"team_object_permission": object_permission, "team_metadata": metadata})


def with_project(token: UserAPIKeyAuth, project: LiteLLM_ProjectTable | None) -> UserAPIKeyAuth:
    if project is None:
        return token
    return token.model_copy(update={"project_metadata": project.metadata, "project_alias": project.project_alias})


def _epoch(expires: str | datetime) -> float:
    expiry: Final = expires if isinstance(expires, datetime) else datetime.fromisoformat(expires)
    if expiry.tzinfo is None or expiry.tzinfo.utcoffset(expiry) is None:
        return expiry.replace(tzinfo=timezone.utc).timestamp()
    return expiry.timestamp()


def key_facts(token: UserAPIKeyAuth) -> KeyFacts:
    return KeyFacts(
        token=token.token,
        blocked=token.blocked is True,
        expires=None if token.expires is None else _epoch(token.expires),
        is_proxy_admin=token.user_role == LitellmUserRoles.PROXY_ADMIN,
        user_id=token.user_id,
        team_id=token.team_id,
        team_object_permission_id=token.team_object_permission_id,
        project_id=token.project_id,
        end_user_id=token.end_user_id,
        last_refreshed_at=token.last_refreshed_at,
    )
