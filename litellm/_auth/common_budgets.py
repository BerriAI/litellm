from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native

BudgetScope: TypeAlias = Literal[
    "Team",
    "Key",
    "TeamWindow",
    "KeyWindow",
    "Project",
    "Organization",
    "Tag",
    "AccessGroup",
    "User",
    "TeamMember",
    "EndUser",
    "Proxy",
    "KeySoft",
    "TeamSoft",
    "ProjectSoft",
]
BudgetAlert: TypeAlias = Literal[
    "token_budget",
    "user_budget",
    "team_budget",
    "project_budget",
    "organization_budget",
    "team_member_threshold",
    "soft_budget",
]


@dataclass(frozen=True, slots=True)
class BudgetFacts:
    scope: BudgetScope
    limit: float | None
    spend: float | None = None
    token_present: bool = False
    team_key: bool = False
    apply_user_budget_to_team_keys: bool = False
    info_route: bool = False
    llm_route: bool = True
    model_list_route: bool = False
    skip: bool = False
    alert_recipients: tuple[str, ...] = ()
    throttle_percentage: float | None = None
    throttle_opt_in: bool = False
    tpm_limit: int | None = None
    rpm_limit: int | None = None


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    read_spend: bool = False
    denied: bool = False
    alert: BudgetAlert | None = None
    throttle_percentage: float | None = None


@dataclass(frozen=True, slots=True)
class AlertFacts:
    spend: float | None
    max_budget: float | None
    configured: bool = False
    minimum_percent: float | None = None
    default_ratio: float = 0.8
    team_member: bool = False


@dataclass(frozen=True, slots=True)
class MemberLimitFacts:
    member_limit: float | None
    default_limit: float | None
    temporary_increase: float | None
    expires_at: float | None
    now: float
    now_naive: bool = False


@dataclass(frozen=True, slots=True)
class WindowFacts:
    duration_seconds: float | None
    reset_at: float | None
    now: float


_FACTS: Final = TypeAdapter(BudgetFacts)
_DECISION: Final = TypeAdapter(BudgetDecision)
_ALERT: Final = TypeAdapter(AlertFacts)
_MEMBER: Final = TypeAdapter(MemberLimitFacts)
_WINDOW: Final = TypeAdapter(WindowFacts)


@native_boundary
def check(facts: BudgetFacts) -> BudgetDecision:
    return _DECISION.validate_python(_native.request_auth_common_budget(_FACTS.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes the validated frozen facts for the native boundary


@native_boundary
def alert(facts: AlertFacts) -> bool:
    return _native.request_auth_budget_alert(_ALERT.dump_python(facts))  # pyright: ignore[reportAny]  # Pydantic serializes the validated frozen facts for the native boundary


@native_boundary
def member_limit(facts: MemberLimitFacts) -> float | None:
    return _native.request_auth_member_limit(_MEMBER.dump_python(facts))  # pyright: ignore[reportAny]  # Pydantic serializes the validated frozen facts for the native boundary


@native_boundary
def window_start(facts: WindowFacts) -> float | None:
    return _native.request_auth_budget_window_start(_WINDOW.dump_python(facts))  # pyright: ignore[reportAny]  # Pydantic serializes the validated frozen facts for the native boundary
