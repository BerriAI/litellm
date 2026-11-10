from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TypeAlias

from litellm.proxy.auth.core.answers import KeyFacts


class AnonymousReason(Enum):
    PUBLIC_ROUTE = "public_route"
    PASS_THROUGH_AUTH_OFF = "pass_through_auth_off"


@dataclass(frozen=True, slots=True)
class Anonymous:
    reason: AnonymousReason


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
    key: KeyFacts


KeyIdentity: TypeAlias = NoMasterKey | UserAuthRoute | MasterKey | CachedAdminKey | VirtualKey


@dataclass(frozen=True, slots=True)
class JwtProxyAdmin:
    pass


@dataclass(frozen=True, slots=True)
class JwtUser:
    pass


@dataclass(frozen=True, slots=True)
class JwtKey:
    pass


JwtIdentity: TypeAlias = JwtProxyAdmin | JwtUser | JwtKey


@dataclass(frozen=True, slots=True)
class OAuth2Identity:
    token: str
    user_id: object
    role: object
    team_id: object


@dataclass(frozen=True, slots=True)
class OAuth2ProxyIdentity:
    user_id: str | None
    user_email: str | None
    team_id: str | None
    team_alias: str | None
    org_id: str | None
    models: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class CustomIdentity:
    user_id: str | None
    team_id: str | None
    hashed_token: str | None


Identity: TypeAlias = KeyIdentity | JwtIdentity | OAuth2Identity | OAuth2ProxyIdentity | CustomIdentity | Anonymous


@dataclass(frozen=True, slots=True)
class ReplacedKey:
    token: str


@dataclass(frozen=True, slots=True)
class NoCustomIdentity:
    pass


CustomAuthResult: TypeAlias = CustomIdentity | ReplacedKey | NoCustomIdentity


@dataclass(frozen=True, slots=True)
class MissingKey:
    pass


@dataclass(frozen=True, slots=True)
class MalformedKey:
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
class ExpiredKey:
    expires: datetime
    now: datetime
    key_label: str


@dataclass(frozen=True, slots=True)
class KeyWithoutToken:
    pass


KeyDenial: TypeAlias = (
    MissingKey
    | MalformedKey
    | NoDatabase
    | NotAVirtualKey
    | KeyBlocked
    | OwnerDeactivated
    | ExpiredKey
    | KeyWithoutToken
)

KeyAuthentication: TypeAlias = KeyIdentity | KeyDenial


@dataclass(frozen=True, slots=True)
class InactiveToken:
    pass


@dataclass(frozen=True, slots=True)
class IntrospectionNotConfigured:
    pass


OAuth2Denial: TypeAlias = InactiveToken | IntrospectionNotConfigured


@dataclass(frozen=True, slots=True)
class TrustedProxyRangesMissing:
    pass


@dataclass(frozen=True, slots=True)
class UntrustedProxy:
    client_ip: str | None


@dataclass(frozen=True, slots=True)
class ProxyMappingsMissing:
    pass


@dataclass(frozen=True, slots=True)
class DisallowedProxyMappings:
    fields: tuple[str, ...]


OAuth2ProxyDenial: TypeAlias = (
    TrustedProxyRangesMissing | UntrustedProxy | ProxyMappingsMissing | DisallowedProxyMappings
)


@dataclass(frozen=True, slots=True)
class ManagedAgentNeedsDirectJwt:
    pass


@dataclass(frozen=True, slots=True)
class JwtOwnerDeactivated:
    user_id: str | None


@dataclass(frozen=True, slots=True)
class JwtKeyMappingFailed:
    reason: str


JwtDenial: TypeAlias = ManagedAgentNeedsDirectJwt | JwtOwnerDeactivated | JwtKeyMappingFailed

JwtAuthentication: TypeAlias = JwtIdentity | JwtDenial

AuthDenial: TypeAlias = KeyDenial | JwtDenial | OAuth2Denial | OAuth2ProxyDenial
