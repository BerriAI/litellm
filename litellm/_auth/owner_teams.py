from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.personal_budget import PersonalTeamBudgetFacts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class OwnerTeamLoad:
    present: bool
    failed: bool = False
    blocked: bool = False
    org: str | None = None


@dataclass(frozen=True, slots=True)
class OwnerTeamBudget:
    index: int
    facts: PersonalTeamBudgetFacts


@dataclass(frozen=True, slots=True)
class OwnerTeamFacts:
    purpose: Literal["Models", "Grants"]
    ids: tuple[str, ...]
    ui_team_id: str
    user_present: bool = True
    database_present: bool = True
    org: str | None = None
    loaded: bool = False
    loads: tuple[OwnerTeamLoad, ...] = ()
    budgets: tuple[OwnerTeamBudget, ...] = ()


@dataclass(frozen=True, slots=True)
class OwnerTeamDecision:
    kind: Literal["ReadTeams", "ReadBudgets", "Complete"]
    ids: tuple[str, ...] = ()
    indices: tuple[int, ...] = ()
    selected: tuple[int, ...] = ()
    withheld: tuple[int, ...] = ()
    unreadable: bool = False


_DECISION: Final = TypeAdapter(OwnerTeamDecision)


@native_boundary
def owner_teams(facts: OwnerTeamFacts) -> OwnerTeamDecision:
    return _DECISION.validate_python(_native.request_auth_owner_teams(asdict(facts)))
