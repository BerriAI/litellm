from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class RequestFacts:
    route: str
    token: str | None


@dataclass(frozen=True, slots=True)
class Settings:
    master_key: str | None
    allow_user_auth: bool


@dataclass(frozen=True, slots=True)
class NoMasterKey:
    token: str | None


@dataclass(frozen=True, slots=True)
class UserAuthRoute:
    pass


@dataclass(frozen=True, slots=True)
class MasterKey:
    pass


Grant: TypeAlias = NoMasterKey | UserAuthRoute | MasterKey


class DenyReason(Enum):
    MISSING_KEY = "missing_key"
    MALFORMED_KEY = "malformed_key"
    USER_AUTH_DISABLED = "user_auth_disabled"


@dataclass(frozen=True, slots=True)
class Allow:
    grant: Grant


@dataclass(frozen=True, slots=True)
class Deny:
    reason: DenyReason


@dataclass(frozen=True, slots=True)
class Defer:
    pass


Outcome: TypeAlias = Allow | Deny | Defer


class AuthCore(Protocol):
    def decide(self, facts: RequestFacts, settings: Settings) -> Outcome: ...
