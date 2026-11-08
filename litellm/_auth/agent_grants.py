from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.rust_core import encode_facts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class Source:
    team_id: str | None
    grants: tuple[str, ...] | None
    unavailable: bool = False


@dataclass(frozen=True, slots=True)
class Resolve:
    managed: bool
    strict: bool
    role: str | None
    key_team_loaded: bool = False
    key_team: tuple[str, ...] | None = None
    default_deny: bool | None = None
    caller_loaded: bool = False
    own: tuple[str, ...] | None = None
    ceiling_loaded: bool = False
    ceiling: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Failure:
    strict: bool
    closed: bool


@dataclass(frozen=True, slots=True)
class OwnerTeam:
    strict: bool
    permission_id_present: bool
    permission_present: bool


@dataclass(frozen=True, slots=True)
class OwnerUnion:
    teams: tuple[Source, ...]
    withheld: tuple[Source, ...]
    unreadable: bool


@dataclass(frozen=True, slots=True)
class BillingTeam:
    sources: tuple[Source, ...]
    agent_id: str


@dataclass(frozen=True, slots=True)
class Human:
    sources: tuple[Source, ...]
    team_id: str | None


@dataclass(frozen=True, slots=True)
class Delegated:
    sources: tuple[Source, ...]
    agent_id: str
    team_id: str | None
    explicit: bool
    allowed: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class TeamLookup:
    subject: bool
    team_id: str | None
    database: bool
    strict: bool
    personal: bool | None = None
    loaded: bool = False
    row_present: bool = False


@dataclass(frozen=True, slots=True)
class ConfiguredAgent:
    agent_id: str
    groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupMatches:
    agents: tuple[ConfiguredAgent, ...]
    groups: tuple[str, ...]


Action: TypeAlias = Literal[
    "Complete",
    "ReadManaged",
    "ReadKeyTeam",
    "ReadDefault",
    "ReadCaller",
    "ReadCeiling",
    "ReadPersonalEligibility",
    "ReadOwnerTeams",
    "ReadTeam",
    "ReadDeclared",
    "Unavailable",
    "SelectTeam",
]


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    grants: tuple[str, ...] | None
    team_id: str | None


_DECISION: Final = TypeAdapter(Decision)

Facts: TypeAlias = (
    Resolve | Failure | OwnerTeam | OwnerUnion | BillingTeam | Human | Delegated | TeamLookup | GroupMatches
)


@native_boundary
def decide(facts: Facts) -> Decision:
    return _DECISION.validate_python(_native.request_auth_agent_grants(encode_facts(facts)))
