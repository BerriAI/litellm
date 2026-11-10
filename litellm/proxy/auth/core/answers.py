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
class OwnerFacts:
    deactivated: bool


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
class SigningKey:
    key_id: str
    algorithm: str
    public_key_pem: str


@dataclass(frozen=True, slots=True)
class Introspection:
    active: bool
    user_id: object
    role: object
    team_id: object


@dataclass(frozen=True, slots=True)
class TeamFacts:
    blocked: bool
    models: tuple[str, ...]
    spend: float
    max_budget: float | None


@dataclass(frozen=True, slots=True)
class OrganizationFacts:
    models: tuple[str, ...]
    spend: float
    max_budget: float | None
