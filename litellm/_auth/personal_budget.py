from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.common_budgets import BudgetFacts, MemberLimitFacts
from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class PersonalTeamBudgetFacts:
    user_present: bool = False
    team_id_present: bool = True
    check_db_only: bool = False
    token_org: str | None = None
    team_org: str | None = None
    source_ready: bool = False
    team: BudgetFacts | None = None
    team_alerted: bool = False
    window_count: int | None = None
    windows: tuple[BudgetFacts, ...] = ()
    membership_loaded: bool = False
    member_limit: MemberLimitFacts | None = None
    default_id: str | None = None
    default_loaded: bool = False
    member: BudgetFacts | None = None
    member_alerted: bool = False
    org_loaded: bool = False
    org_failed: bool = False
    org: BudgetFacts | None = None
    org_alerted: bool = False
    io_failed: bool = False


@dataclass(frozen=True, slots=True)
class PersonalTeamBudgetDecision:
    kind: Literal[
        "Complete",
        "Source",
        "ReadTeam",
        "AlertTeam",
        "ReadWindows",
        "ReadWindow",
        "ReadMembership",
        "ReadMemberLimit",
        "ReadDefault",
        "ReadMember",
        "AlertMember",
        "ReadOrg",
        "AlertOrg",
    ]
    allowed: bool = False
    log_failure: bool = False
    org_id: str | None = None
    id: str = ""
    index: int = 0
    limit: float | None = None


_DECISION: Final = TypeAdapter(PersonalTeamBudgetDecision)


@native_boundary
def personal_team_budget(facts: PersonalTeamBudgetFacts) -> PersonalTeamBudgetDecision:
    return _DECISION.validate_python(_native.request_auth_personal_budget(asdict(facts)))
