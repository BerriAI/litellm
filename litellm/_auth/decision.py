from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class RequestFacts:
    route: str
    token: str | None
    is_llm_api_route: bool


@dataclass(frozen=True, slots=True)
class Settings:
    master_key: str | None
    allow_user_auth: bool
    has_db: bool
    master_key_only_routes: frozenset[str]
    proxy_max_budget: float


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
)


@dataclass(frozen=True, slots=True)
class Allow:
    grant: Grant


@dataclass(frozen=True, slots=True)
class Deny:
    denial: Denial


Outcome: TypeAlias = Allow | Deny


@dataclass(frozen=True, slots=True)
class AdminKey:
    expires: datetime | None


@dataclass(frozen=True, slots=True)
class NonAdminKey:
    pass


PresentedKey: TypeAlias = AdminKey | NonAdminKey


@dataclass(frozen=True, slots=True)
class KeyFacts:
    has_token: bool
    blocked: bool
    expires: datetime | None
    spend: float
    soft_budget: float | None
    max_budget: float | None
    team_member_spend: float | None
    user_id: str | None
    team_id: str | None
    budget_throttle_pct: float | None


@dataclass(frozen=True, slots=True)
class OwnerFacts:
    deactivated: bool


@dataclass(frozen=True, slots=True)
class TeamMemberBudget:
    spend: float
    max_budget: float


@dataclass(frozen=True, slots=True)
class ResolveEndUser:
    prefetch_key: bool
    resume: "Callable[[], Step]"


@dataclass(frozen=True, slots=True)
class ResolvePresentedKey:
    hashed_token: str
    resume: "Callable[[PresentedKey | None], Step]"


@dataclass(frozen=True, slots=True)
class EvictPresentedKey:
    hashed_token: str
    resume: "Callable[[], Step]"


@dataclass(frozen=True, slots=True)
class LoadKey:
    hashed_token: str
    resume: "Callable[[], Step]"


@dataclass(frozen=True, slots=True)
class PrepareKey:
    loaded_from_db: bool
    resume: "Callable[[], Step]"


@dataclass(frozen=True, slots=True)
class InspectKey:
    resume: "Callable[[KeyFacts], Step]"


@dataclass(frozen=True, slots=True)
class EnforceModelAccess:
    resume: "Callable[[], Step]"


@dataclass(frozen=True, slots=True)
class LoadKeyOwner:
    user_id: str
    resume: "Callable[[OwnerFacts | None], Step]"


@dataclass(frozen=True, slots=True)
class CheckBudgetExemption:
    resume: "Callable[[bool], Step]"


@dataclass(frozen=True, slots=True)
class ReadTeamMemberBudget:
    user_id: str
    team_id: str
    resume: "Callable[[TeamMemberBudget | None], Step]"


@dataclass(frozen=True, slots=True)
class AlertTeamMemberBudget:
    user_id: str
    team_id: str
    budget: TeamMemberBudget
    resume: "Callable[[], Step]"


@dataclass(frozen=True, slots=True)
class SendKeyBudgetAlerts:
    resume: "Callable[[], Step]"


@dataclass(frozen=True, slots=True)
class ReadKeySpend:
    max_budget: float
    resume: "Callable[[float], Step]"


@dataclass(frozen=True, slots=True)
class AlertKeyBudget:
    spend: float
    resume: "Callable[[], Step]"


@dataclass(frozen=True, slots=True)
class AlertKeySoftBudget:
    resume: "Callable[[], Step]"


@dataclass(frozen=True, slots=True)
class EnforceModelBudgets:
    resume: "Callable[[], Step]"


@dataclass(frozen=True, slots=True)
class LoadKeyContext:
    resume: "Callable[[], Step]"


@dataclass(frozen=True, slots=True)
class ReadProxySpend:
    resume: "Callable[[float | None], Step]"


@dataclass(frozen=True, slots=True)
class AlertProxyBudget:
    spend: float
    resume: "Callable[[], Step]"


IdentityEffect: TypeAlias = (
    ResolveEndUser
    | ResolvePresentedKey
    | EvictPresentedKey
    | LoadKey
    | PrepareKey
    | InspectKey
    | EnforceModelAccess
    | LoadKeyOwner
    | CheckBudgetExemption
    | LoadKeyContext
)
BudgetEffect: TypeAlias = (
    ReadTeamMemberBudget
    | AlertTeamMemberBudget
    | SendKeyBudgetAlerts
    | ReadKeySpend
    | AlertKeyBudget
    | AlertKeySoftBudget
    | EnforceModelBudgets
    | ReadProxySpend
    | AlertProxyBudget
)
Effect: TypeAlias = IdentityEffect | BudgetEffect
Step: TypeAlias = Outcome | Effect


class AuthCore(Protocol):
    def start(self, facts: RequestFacts, settings: Settings, now: datetime) -> Step: ...

    def start_resolved_key(self, facts: RequestFacts, settings: Settings, now: datetime) -> Step: ...
