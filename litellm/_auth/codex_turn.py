from dataclasses import asdict, dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.codex_models import CodexModelFacts
from litellm._auth.failure import native_boundary
from litellm._auth.route_enable import RouteEnableFacts
from litellm._auth.route_scope import RouteScopeFacts
from litellm._auth.routes import VirtualRouteFacts
from litellm.rust_bridge import _native

JwtRead: TypeAlias = Literal["Returned", "Exception"]


@dataclass(frozen=True, slots=True)
class CodexKeyFacts:
    proxy: RouteScopeFacts | None = None
    enabled: RouteEnableFacts | None = None
    virtual_route: VirtualRouteFacts | None = None
    role_loaded: bool = False
    role: str | None = None
    admin: RouteScopeFacts | None = None


@dataclass(frozen=True, slots=True)
class CodexAccessFacts:
    key: CodexKeyFacts
    jwt_token_present: bool
    jwt: JwtRead | None = None


@dataclass(frozen=True, slots=True)
class CodexCredentials:
    bearer_present: bool
    gateway_present: bool
    equal: bool


@dataclass(frozen=True, slots=True)
class CodexTurnFacts:
    model: str | None
    native: bool = False
    credentials: CodexCredentials | None = None
    access: CodexAccessFacts | None = None
    routing: CodexModelFacts | None = None
    catalog_loaded: bool = False
    slugs: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class CodexTurnDecision:
    kind: Literal[
        "Complete",
        "Failure",
        "ReadProxy",
        "ReadEnabled",
        "ReadVirtual",
        "ReadRole",
        "ReadAdmin",
        "ReadCredentials",
        "ReadAccess",
        "ReadRouting",
        "ReadJwt",
        "ReadCatalog",
    ]
    allowed: bool = False
    message: str | None = None


_DECISION: Final = TypeAdapter(CodexTurnDecision)


@native_boundary
def codex_key(facts: CodexKeyFacts) -> CodexTurnDecision:
    return _DECISION.validate_python(_native.request_auth_codex_turn({"Key": asdict(facts)}))


@native_boundary
def codex_access(facts: CodexAccessFacts) -> CodexTurnDecision:
    return _DECISION.validate_python(_native.request_auth_codex_turn({"Access": asdict(facts)}))


@native_boundary
def codex_turn(facts: CodexTurnFacts) -> CodexTurnDecision:
    return _DECISION.validate_python(_native.request_auth_codex_turn({"Turn": asdict(facts)}))


@native_boundary
def codex_jwt(read: JwtRead) -> CodexTurnDecision:
    return _DECISION.validate_python(_native.request_auth_codex_turn({"Jwt": read}))


@native_boundary
def codex_jwt_token(*, claims_present: bool, token_present: bool) -> CodexTurnDecision:
    return _DECISION.validate_python(
        _native.request_auth_codex_turn(
            {"JwtToken": {"claims_present": claims_present, "token_present": token_present}}
        )
    )
