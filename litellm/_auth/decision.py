from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class RequestFacts:
    route: str
    token: str | None


@dataclass(frozen=True, slots=True)
class Settings:
    master_key: str | None
    allow_user_auth: bool
    has_db: bool
    master_key_only_routes: frozenset[str]


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
    loaded_from_db: bool


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


@dataclass(frozen=True, slots=True)
class MasterKeyOnlyRoute:
    pass


@dataclass(frozen=True, slots=True)
class NoDatabase:
    pass


@dataclass(frozen=True, slots=True)
class NotAVirtualKey:
    pass


Denial: TypeAlias = (
    MissingKey | MalformedKey | UserAuthDisabled | ExpiredKey | MasterKeyOnlyRoute | NoDatabase | NotAVirtualKey
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


Effect: TypeAlias = ResolveEndUser | ResolvePresentedKey | EvictPresentedKey | LoadKey
Step: TypeAlias = Outcome | Effect


class AuthCore(Protocol):
    def start(self, facts: RequestFacts, settings: Settings, now: datetime) -> Step: ...
