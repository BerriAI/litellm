from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from litellm.proxy.auth.core.identity import AnonymousReason


class OAuth2Mode(Enum):
    INTROSPECTION = "introspection"
    PROXY = "proxy"


@dataclass(frozen=True, slots=True)
class KeyValidator:
    pass_through_key: bool


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


class PremiumFeature(Enum):
    OAUTH2 = "oauth2"
    JWT = "jwt"


@dataclass(frozen=True, slots=True)
class PremiumRequired:
    feature: PremiumFeature


KeyPlan: TypeAlias = KeyValidator | AnonymousValidator

CredentialPlan: TypeAlias = KeyPlan | JwtValidator | OAuth2Validator | PremiumRequired

Plan: TypeAlias = CredentialPlan | CustomValidator
