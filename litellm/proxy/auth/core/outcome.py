from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class NoMasterKey:
    token: str | None


@dataclass(frozen=True, slots=True)
class UserAuthRoute:
    pass


@dataclass(frozen=True, slots=True)
class MasterKey:
    pass


@dataclass(frozen=True, slots=True)
class CachedAdminKey:
    pass


@dataclass(frozen=True, slots=True)
class VirtualKey:
    budget_throttle_pct: float | None


@dataclass(frozen=True, slots=True)
class WithinModelBudgets:
    pass


Grant: TypeAlias = NoMasterKey | UserAuthRoute | MasterKey | CachedAdminKey | VirtualKey


@dataclass(frozen=True, slots=True)
class MissingKey:
    pass


@dataclass(frozen=True, slots=True)
class MalformedKey:
    pass


@dataclass(frozen=True, slots=True)
class UserAuthDisabled:
    pass


@dataclass(frozen=True, slots=True)
class ExpiredKey:
    expires: datetime
    now: datetime
    key_label: str


@dataclass(frozen=True, slots=True)
class MasterKeyOnlyRoute:
    pass


@dataclass(frozen=True, slots=True)
class NoDatabase:
    pass


@dataclass(frozen=True, slots=True)
class NotAVirtualKey:
    pass


@dataclass(frozen=True, slots=True)
class KeyBlocked:
    pass


@dataclass(frozen=True, slots=True)
class OwnerDeactivated:
    user_id: str


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
class KeyWithoutToken:
    pass


@dataclass(frozen=True, slots=True)
class ModelBudgetExceeded:
    message: str
    current_cost: float
    max_budget: float
    entity_type: str | None
    entity_id: str | None


Denial: TypeAlias = (
    MissingKey
    | MalformedKey
    | UserAuthDisabled
    | ExpiredKey
    | MasterKeyOnlyRoute
    | NoDatabase
    | NotAVirtualKey
    | KeyBlocked
    | OwnerDeactivated
    | TeamMemberBudgetExceeded
    | KeyBudgetExceeded
    | KeyWithoutToken
    | ModelBudgetExceeded
)


@dataclass(frozen=True, slots=True)
class Allow:
    grant: Grant


@dataclass(frozen=True, slots=True)
class Deny:
    denial: Denial


Outcome: TypeAlias = Allow | Deny

ModelBudgetResult: TypeAlias = WithinModelBudgets | ModelBudgetExceeded


class AnonymousReason(Enum):
    PUBLIC_ROUTE = "public_route"
    PASS_THROUGH_AUTH_OFF = "pass_through_auth_off"
    NO_MASTER_KEY = "no_master_key"


@dataclass(frozen=True, slots=True)
class MasterKeyIdentity:
    pass


@dataclass(frozen=True, slots=True)
class KeyIdentity:
    hashed_token: str


@dataclass(frozen=True, slots=True)
class JwtIdentity:
    user_id: str | None
    team_id: str | None
    org_id: str | None
    end_user_id: str | None
    role: str | None
    scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OAuth2Identity:
    user_id: str | None
    team_id: str | None
    scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CustomIdentity:
    user_id: str | None
    team_id: str | None
    hashed_token: str | None


@dataclass(frozen=True, slots=True)
class Anonymous:
    reason: AnonymousReason


@dataclass(frozen=True, slots=True)
class InactiveToken:
    pass


OAuth2Denial: TypeAlias = MissingKey | InactiveToken


Identity: TypeAlias = MasterKeyIdentity | KeyIdentity | JwtIdentity | OAuth2Identity | CustomIdentity | Anonymous


class OAuth2Mode(Enum):
    INTROSPECTION = "introspection"
    PROXY = "proxy"


@dataclass(frozen=True, slots=True)
class KeyValidator:
    pass


@dataclass(frozen=True, slots=True)
class JwtValidator:
    pass


@dataclass(frozen=True, slots=True)
class OAuth2Validator:
    mode: OAuth2Mode


@dataclass(frozen=True, slots=True)
class CustomValidator:
    pass


@dataclass(frozen=True, slots=True)
class AnonymousValidator:
    reason: AnonymousReason


Validator: TypeAlias = KeyValidator | JwtValidator | OAuth2Validator | CustomValidator | AnonymousValidator
