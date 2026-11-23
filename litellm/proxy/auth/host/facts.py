"""Records the Python auth host answers with, and the conversions that stamp one row's values onto a key.
python-bridge reads the records by attribute, so a field rename here is a change to the Rust side too
(`python-bridge/src/auth/call.rs`)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Final

from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth.grants import EndUserGrants
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles


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
    last_refreshed_at: float | None


@dataclass(frozen=True, slots=True)
class EndUserRow:
    """The request's end user and the limits the key path stamps onto the token from it."""

    end_user_id: str | None
    end_user: LiteLLM_EndUserTable | None
    params: EndUserGrants


@dataclass(frozen=True, slots=True)
class TeamFacts:
    team_id: str
    models: Sequence[str] | None
    allowed_passthrough_routes: object
    denied_passthrough_routes: object
    members: Sequence[str]


@dataclass(frozen=True, slots=True)
class FoundTeam:
    facts: TeamFacts
    row: LiteLLM_TeamTable


def team_facts(team: LiteLLM_TeamTable) -> FoundTeam:
    metadata: Final = team.metadata or {}
    return FoundTeam(
        facts=TeamFacts(
            team_id=team.team_id,
            models=tuple(team.models),
            allowed_passthrough_routes=metadata.get("allowed_passthrough_routes"),
            denied_passthrough_routes=metadata.get("denied_passthrough_routes"),
            members=tuple(member.user_id for member in team.members_with_roles if member.user_id is not None),
        ),
        row=team,
    )


class TeamMissing(str, Enum):
    """Why `read_team` found no team: its `TeamNotFoundError`, or a read that failed with the database reachable."""

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
        last_refreshed_at=token.last_refreshed_at,
    )
