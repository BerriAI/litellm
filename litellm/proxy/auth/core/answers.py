from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class AdminKey:
    expires: datetime | None


@dataclass(frozen=True, slots=True)
class NonAdminKey:
    pass


PresentedKey: TypeAlias = AdminKey | NonAdminKey


@dataclass(frozen=True, slots=True)
class TeamMember:
    user_id: str
    team_id: str


@dataclass(frozen=True, slots=True)
class KeyFacts:
    has_token: bool
    blocked: bool
    expires: datetime | None
    spend: float
    soft_budget: float | None
    max_budget: float | None
    user_id: str | None
    team_member: TeamMember | None
    budget_throttle_pct: float | None


@dataclass(frozen=True, slots=True)
class UserModelBudgets:
    limits: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class OwnerFacts:
    deactivated: bool
    model_budgets: UserModelBudgets | None


@dataclass(frozen=True, slots=True)
class BudgetExemption:
    route_skips_budgets: bool
    mcp_discovery: bool
    model_is_free: bool


@dataclass(frozen=True, slots=True)
class TeamMemberBudget:
    spend: float
    max_budget: float


@dataclass(frozen=True, slots=True)
class KeyModelBudgets:
    fallbacks: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ModelBudgetScope:
    models: tuple[str, ...]
    rewritable_model: str | None
    key: KeyModelBudgets | None
    user_has_model_budgets: bool


@dataclass(frozen=True, slots=True)
class TeamMemberModelBudgetScope:
    models: tuple[str, ...]
    rewritable_model: str | None
    has_model_budgets: bool
    can_fall_back: bool


@dataclass(frozen=True, slots=True)
class FallbackTarget:
    requested: str
    model: str
    served_model: str


@dataclass(frozen=True, slots=True)
class Introspection:
    active: bool
    user_id: object
    role: object
    team_id: object


@dataclass(frozen=True, slots=True)
class EndUserScope:
    has_end_user: bool
    has_key_end_user_budget: bool


@dataclass(frozen=True, slots=True)
class TokenRole:
    is_proxy_admin: bool
    is_builtin_admin: bool


@dataclass(frozen=True, slots=True)
class MappedKey:
    key_user_id: str | None
    key_has_email: bool
    claim_user_id: str | None
    claim_has_email: bool


@dataclass(frozen=True, slots=True)
class PendingAutoRegister:
    pass


@dataclass(frozen=True, slots=True)
class NoKeyMapping:
    pass


KeyMapping: TypeAlias = MappedKey | PendingAutoRegister | NoKeyMapping


@dataclass(frozen=True, slots=True)
class JwtPolicy:
    is_proxy_admin: bool
    user_id: str | None
    owner_deactivated: bool
    user_model_budgets: UserModelBudgets | None


@dataclass(frozen=True, slots=True)
class Registered:
    links_existing_key: bool


@dataclass(frozen=True, slots=True)
class RegistrationFailed:
    reason: str


@dataclass(frozen=True, slots=True)
class NotRegistered:
    pass


AutoRegistration: TypeAlias = Registered | RegistrationFailed | NotRegistered
