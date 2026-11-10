from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class UserAuthDisabled:
    pass


@dataclass(frozen=True, slots=True)
class MasterKeyOnlyRoute:
    pass


@dataclass(frozen=True, slots=True)
class TeamMemberBudgetExceeded:
    user_id: str
    team_id: str
    spend: float
    max_budget: float


@dataclass(frozen=True, slots=True)
class KeyBudgetExceeded:
    spend: float
    max_budget: float


@dataclass(frozen=True, slots=True)
class ModelBudgetExceeded:
    message: str
    current_cost: float
    max_budget: float
    entity_type: str | None
    entity_id: str | None


@dataclass(frozen=True, slots=True)
class WithinModelBudgets:
    pass


ModelBudgetResult: TypeAlias = WithinModelBudgets | ModelBudgetExceeded

Denial: TypeAlias = (
    UserAuthDisabled | MasterKeyOnlyRoute | TeamMemberBudgetExceeded | KeyBudgetExceeded | ModelBudgetExceeded
)


@dataclass(frozen=True, slots=True)
class Allow:
    budget_throttle_pct: float | None


@dataclass(frozen=True, slots=True)
class Deny:
    denial: Denial


Outcome: TypeAlias = Allow | Deny
