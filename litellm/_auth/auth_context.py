from dataclasses import dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.rust_core import encode_facts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class Fetch:
    team_id: str | None
    ui_team_id: str
    user_id: str | None
    project_id: str | None
    end_user_id: str | None


@dataclass(frozen=True, slots=True)
class Team:
    team_id: str | None
    ui_team_id: str
    failed: bool
    missing: bool
    recorded_models: tuple[str, ...]
    allow_unavailable: bool | None = None


@dataclass(frozen=True, slots=True)
class Admin:
    role: str | None
    user_id: str | None
    admin_id: str
    user_row: bool


@dataclass(frozen=True, slots=True)
class FetchPlan:
    team: bool
    user: bool
    project: bool
    end_user: bool


@dataclass(frozen=True, slots=True)
class TeamPlan:
    source: Literal["None", "Token", "Loaded", "ReadUnavailablePolicy", "Failure"]


@dataclass(frozen=True, slots=True)
class AdminPlan:
    force_admin: bool
    preserve_budget: bool


_FETCH: Final = TypeAdapter(FetchPlan)
_TEAM: Final = TypeAdapter(TeamPlan)
_ADMIN: Final = TypeAdapter(AdminPlan)


@native_boundary
def fetch(facts: Fetch) -> FetchPlan:
    return _FETCH.validate_python(_native.request_auth_context(encode_facts(facts)))


@native_boundary
def team(facts: Team) -> TeamPlan:
    return _TEAM.validate_python(_native.request_auth_context(encode_facts(facts)))


@native_boundary
def admin(facts: Admin) -> AdminPlan:
    return _ADMIN.validate_python(_native.request_auth_context(encode_facts(facts)))
