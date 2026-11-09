from dataclasses import asdict, dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.routes import RouteContext
from litellm.rust_bridge import _native

RouteFeature: TypeAlias = Literal["Admin", "Llm"]


@dataclass(frozen=True, slots=True)
class RouteSetting:
    environment_present: bool
    premium: bool
    secret: bool | None = None


@dataclass(frozen=True, slots=True)
class RouteEnableFacts:
    context: RouteContext
    imported: bool = False
    failed: Literal["Http", "Other"] | None = None
    admin: RouteSetting | None = None
    llm: RouteSetting | None = None
    exemptions: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class RouteEnableDecision:
    kind: Literal["Allow", "PropagateHttp", "ReadImport", "ReadExemptions", "ReadSettings", "ReadSecret", "Deny"]
    feature: RouteFeature | None = None
    reason: Literal["License", "Disabled"] | None = None


_DECISION: Final = TypeAdapter(RouteEnableDecision)


@native_boundary
def route_enable(facts: RouteEnableFacts) -> RouteEnableDecision:
    return _DECISION.validate_python(_native.request_auth_route_enable(asdict(facts)))
