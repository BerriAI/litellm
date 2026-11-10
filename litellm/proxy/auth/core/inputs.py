from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import IPv4Network, IPv6Network
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class RequestFacts:
    route: str
    token: str | None
    is_llm_api_route: bool


@dataclass(frozen=True, slots=True)
class PlanFacts:
    request: RequestFacts
    is_info_route: bool
    unverified_claims: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class RoutingOverride:
    iss: tuple[str, ...] | None
    client_id: tuple[str, ...] | None
    scope: tuple[str, ...] | None
    aud: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class PlanSettings:
    custom_auth: bool
    premium_user: bool
    public_routes: frozenset[str]
    additional_public_routes: tuple[str, ...]
    metrics_require_auth: bool
    oauth2_introspection: bool
    oauth2_proxy: bool
    jwt_enabled: bool
    oauth2_routing_overrides: tuple[RoutingOverride, ...]
    unauthenticated_pass_through_routes: frozenset[str]


@dataclass(frozen=True, slots=True)
class KeySettings:
    master_key: str | None
    allow_user_auth: bool
    has_db: bool
    master_key_only_routes: frozenset[str]
    proxy_max_budget: float


@dataclass(frozen=True, slots=True)
class JwtSettings:
    audience: str | None
    leeway_seconds: int


@dataclass(frozen=True, slots=True)
class OAuth2IntrospectionSettings:
    endpoint_configured: bool
    requires_active: bool


TrustedNetwork: TypeAlias = IPv4Network | IPv6Network


@dataclass(frozen=True, slots=True)
class OAuth2ProxySettings:
    trusted_networks: tuple[TrustedNetwork, ...]
    header_mappings: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class AuthorizeSettings:
    admin_only_routes: frozenset[str]
    proxy_max_budget: float
