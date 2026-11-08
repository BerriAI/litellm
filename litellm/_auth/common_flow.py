from dataclasses import asdict, dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class Facts:
    model_present: bool
    skip_budgets: bool
    team_present: bool
    user_present: bool
    token_present: bool
    user_id: str | None
    agent_id: str | None
    customer_present: bool
    customer_models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Plan:
    membership: bool
    team_models: bool
    member_models: bool
    user_models: bool
    customer_models: bool
    agent_trace: bool


@dataclass(frozen=True, slots=True)
class BudgetFacts:
    token_present: bool
    group_count: int
    customer_present: bool
    customer_budget_present: bool


BudgetScope: TypeAlias = Literal[
    "Team", "TeamWindows", "KeyWindows", "TeamSoft", "Organization", "Tags", "Groups", "User", "Member", "Customer"
]


@dataclass(frozen=True, slots=True)
class CustomFacts:
    end_user_id: str | None
    end_user_present: bool
    common_checks: bool
    model_present: bool = False
    router_present: bool = False


@dataclass(frozen=True, slots=True)
class CustomPlan:
    read_end_user: bool
    check_end_user: bool
    check_model_access: bool
    read_pricing: bool


_PLAN: Final = TypeAdapter(Plan)
_BUDGETS: Final = TypeAdapter(tuple[BudgetScope, ...])
_CUSTOM: Final = TypeAdapter(CustomPlan)


@native_boundary
def plan(facts: Facts) -> Plan:
    return _PLAN.validate_python(_native.request_auth_common_flow(asdict(facts)))


@native_boundary
def budgets(facts: BudgetFacts) -> tuple[BudgetScope, ...]:
    return _BUDGETS.validate_python(_native.request_auth_common_budget_scopes(asdict(facts)))


@native_boundary
def custom(facts: CustomFacts) -> CustomPlan:
    return _CUSTOM.validate_python(_native.request_auth_custom_flow(asdict(facts)))
