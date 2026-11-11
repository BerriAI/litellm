"""Records the Python auth host answers with, and the conversions that stamp one row's values onto a key.
python-bridge reads the records by attribute, so a field rename here is a change to the Rust side too
(`python-bridge/src/auth/call.rs`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Final

from pydantic import TypeAdapter

from litellm.constants import EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles


@dataclass(frozen=True, slots=True)
class KeyFacts:
    token: str | None
    blocked: bool
    expires: float | None
    is_proxy_admin: bool
    is_session_token: bool
    user_id: str | None
    team_id: str | None
    team_object_permission_id: str | None
    project_id: str | None
    last_refreshed_at: float | None
    team_member_spend: float | None
    max_budget: float | None
    spend: float | None
    soft_budget: float | None
    key_alias: str | None
    key_name: str | None
    alert_email_lowest_percent: float | None
    alert_emails_configured: bool
    default_warning_share: float
    throttle_share: float | None


@dataclass(frozen=True, slots=True)
class EndUserRow:
    """The request's end user and the limits the key path stamps onto the token from it."""

    end_user_id: str | None
    end_user: LiteLLM_EndUserTable | None
    params: Mapping[str, object]


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


class TeamMissing(StrEnum):
    """Why `get_team_object` found no team: its `TeamNotFoundError`, or any other `HTTPException`."""

    ABSENT = "absent"
    UNREADABLE = "unreadable"


class AliasMissing(StrEnum):
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


def _lowest_percent(alert_emails: Mapping[str, object]) -> float | None:
    return min((float(percent) for percent in alert_emails if percent.isdigit()), default=None)


def key_facts(token: UserAPIKeyAuth) -> KeyFacts:
    from litellm.proxy.auth.auth_checks import key_alert_emails  # noqa: PLC0415  # auth_checks imports this module
    from litellm.proxy.auth.budget_throttle import (  # noqa: PLC0415  # budget_throttle reads litellm globals
        budget_throttle_percentage,
        should_throttle_budget_exceeded,
    )

    alert_emails: Final = key_alert_emails(token) if token.max_budget is not None else None
    return KeyFacts(
        token=token.token,
        blocked=token.blocked is True,
        expires=None if token.expires is None else _epoch(token.expires),
        is_proxy_admin=token.user_role == LitellmUserRoles.PROXY_ADMIN,
        is_session_token=token.is_session_token,
        user_id=token.user_id,
        team_id=token.team_id,
        team_object_permission_id=token.team_object_permission_id,
        project_id=token.project_id,
        last_refreshed_at=token.last_refreshed_at,
        team_member_spend=token.team_member_spend,
        max_budget=token.max_budget,
        spend=token.spend,
        soft_budget=token.soft_budget,
        key_alias=token.key_alias,
        key_name=token.key_name,
        alert_email_lowest_percent=None if not alert_emails else _lowest_percent(alert_emails),
        alert_emails_configured=bool(alert_emails),
        default_warning_share=EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE,
        throttle_share=budget_throttle_percentage() if should_throttle_budget_exceeded(token) else None,
    )
