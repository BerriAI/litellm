from dataclasses import dataclass
from enum import Enum
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


@dataclass(frozen=True, slots=True)
class BudgetsSkipped:
    pass


ModelBudgetResult: TypeAlias = WithinModelBudgets | ModelBudgetExceeded

ModelBudgetProof: TypeAlias = WithinModelBudgets | BudgetsSkipped

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


class ExemptionReason(Enum):
    PUBLIC_ROUTE = "public_route"
    PASS_THROUGH_AUTH_OFF = "pass_through_auth_off"
    NO_AUTH_DEV_MODE = "no_auth_dev_mode"
    CUSTOM_AUTH = "custom_auth"


@dataclass(frozen=True, slots=True)
class Exempt:
    reason: ExemptionReason


@dataclass(frozen=True, slots=True)
class Checked:
    pass


RequestCheck: TypeAlias = Exempt | Checked
