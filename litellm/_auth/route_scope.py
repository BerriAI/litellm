from dataclasses import asdict, dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native

RouteScopeKind: TypeAlias = Literal["Proxy", "Admin"]


@dataclass(frozen=True, slots=True)
class RouteValues:
    text: str | None = None
    names: tuple[str, ...] = ()
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class RouteScopeFacts:
    kind: RouteScopeKind
    route: str
    present: bool
    premium: bool
    warned: bool = False
    values: RouteValues | None = None


@dataclass(frozen=True, slots=True)
class RouteScopeDecision:
    kind: Literal["Allow", "Deny", "Warn", "ReadValues", "Failure"]
    message: str | None = None


_DECISION: Final = TypeAdapter(RouteScopeDecision)


@native_boundary
def route_scope(facts: RouteScopeFacts) -> RouteScopeDecision:
    return _DECISION.validate_python(_native.request_auth_route_scope(asdict(facts)))
