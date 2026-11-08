from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class RouteCatalog:
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])
    viewer_blocked: tuple[str, ...] = ()
    viewer_write_blocked: tuple[str, ...] = ()
    self_credentials: tuple[tuple[str, str], ...] = ()
    managed_models: tuple[str, ...] = ()
    managed_paths: tuple[str, ...] = ()
    managed_realtime: tuple[str, ...] = ()
    managed_live: tuple[str, ...] = ()
    managed_mcp: tuple[str, ...] = ()
    unbilled: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RouteContext:
    route: str
    method: str | None = None
    catalog: RouteCatalog = field(default_factory=RouteCatalog)
    registered: bool = False
    registered_auth: bool | None = None
    configured_eu_jev: bool = False


@dataclass(frozen=True, slots=True)
class PassThroughFacts:
    key_allowed: tuple[str, ...] = ()
    team_allowed: tuple[str, ...] = ()
    denied: tuple[str, ...] = ()
    forwarded: tuple[str, ...] = ()
    jwt_team_routes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VirtualRouteFacts:
    context: RouteContext
    allowed_routes: tuple[str, ...] | None
    passthrough: PassThroughFacts = field(default_factory=PassThroughFacts)
    password_reset_required: bool = False


@dataclass(frozen=True, slots=True)
class ViewerFacts:
    context: RouteContext
    request_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NonAdminFacts:
    viewer: ViewerFacts
    role: str | None = None
    custom_admin_routes: tuple[str, ...] = ()
    premium: bool = False
    query_user_id: str | None = None
    token_user_id: str | None = None
    permissions: tuple[str, ...] = ()
    organization_ids: tuple[str, ...] = ()
    admin_organization_ids: tuple[str, ...] = ()
    team_service_account: bool = False
    allowed_routes: tuple[str, ...] | None = None
    passthrough: PassThroughFacts = field(default_factory=PassThroughFacts)


RouteDenial: TypeAlias = Literal[
    "VirtualRoute",
    "PasswordReset",
    "PassThrough",
    "DeniedPassThrough",
    "CustomAdmin",
    "UserInfo",
    "AdminOnly",
    "ViewerInference",
    "ViewerWrite",
    "ViewerParameter",
]


@dataclass(frozen=True, slots=True)
class RouteDecision:
    allowed: bool
    denial: RouteDenial | None = None
    detail: str | None = None


_DECISION: Final = TypeAdapter(RouteDecision)


@native_boundary
def match_route(
    route: str,
    patterns: tuple[str, ...],
    mode: Literal["Access", "Template", "Prefix", "Wildcard", "Denied", "Jwt"] = "Access",
) -> bool:
    return _DECISION.validate_python(
        _native.request_auth_route_access({"Match": {"route": route, "patterns": patterns, "mode": mode}})
    ).allowed


@native_boundary
def classify_route(
    context: RouteContext, kind: Literal["Llm", "Unbilled", "Discovery", "SelfCredential", "Azure", "Managed"]
) -> bool:
    return _DECISION.validate_python(
        _native.request_auth_route_access({"Classify": {"context": asdict(context), "kind": kind}})
    ).allowed


@native_boundary
def virtual_route(facts: VirtualRouteFacts) -> RouteDecision:
    return _DECISION.validate_python(_native.request_auth_route_access({"Virtual": asdict(facts)}))


@native_boundary
def non_admin_route(facts: NonAdminFacts) -> RouteDecision:
    return _DECISION.validate_python(_native.request_auth_route_access({"NonAdmin": asdict(facts)}))


@native_boundary
def viewer_route(facts: ViewerFacts) -> RouteDecision:
    return _DECISION.validate_python(_native.request_auth_route_access({"Viewer": asdict(facts)}))


@native_boundary
def passthrough_route(context: RouteContext, facts: PassThroughFacts, *, required: bool = False) -> RouteDecision:
    return _DECISION.validate_python(
        _native.request_auth_route_access(
            {"PassThrough": {"context": asdict(context), "facts": asdict(facts), "required": required}}
        )
    )


@native_boundary
def denied_passthrough_route(route: str, facts: PassThroughFacts) -> RouteDecision:
    return _DECISION.validate_python(
        _native.request_auth_route_access({"DeniedPassThrough": {"route": route, "facts": asdict(facts)}})
    )
