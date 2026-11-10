from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestFacts:
    route: str
    token: str | None
    is_llm_api_route: bool


@dataclass(frozen=True, slots=True)
class PlanSettings:
    custom_auth: bool
    jwt_enabled: bool
    oauth2_introspection: bool
    oauth2_proxy: bool
    public_routes: frozenset[str]


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
class OAuth2Settings:
    user_id_header: str | None
    team_id_header: str | None


@dataclass(frozen=True, slots=True)
class AuthorizeSettings:
    admin_only_routes: frozenset[str]
    proxy_max_budget: float
