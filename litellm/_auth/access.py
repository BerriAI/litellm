from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.rust_core import encode_facts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class Blocked:
    blocked: bool


@dataclass(frozen=True, slots=True)
class UserParamRequest:
    enabled: bool


@dataclass(frozen=True, slots=True)
class GuardrailRequest:
    requested_keys: tuple[str, ...]


TraceSource: TypeAlias = Literal["Gate", "Header", "Body", "Traceparent", "Selected", "Other"]


@dataclass(frozen=True, slots=True)
class TraceRequest:
    required: bool


@dataclass(frozen=True, slots=True)
class Search:
    name: str
    allowed: tuple[str, ...]
    strict: bool


@dataclass(frozen=True, slots=True)
class TraceGate:
    required: bool
    method: str
    llm_route: bool
    passthrough: bool
    telemetry: bool


@dataclass(frozen=True, slots=True)
class TraceStep:
    source: TraceSource
    value: str | None
    present: bool
    headers_only: bool


@dataclass(frozen=True, slots=True)
class UserParam:
    enabled: bool
    method: str | None
    llm_route: bool
    mcp_route: bool
    has_user: bool


@dataclass(frozen=True, slots=True)
class ClientTags:
    enabled: bool
    llm_route: bool
    has_tags: bool


@dataclass(frozen=True, slots=True)
class Guardrails:
    requested_keys: tuple[str, ...]
    modify_guardrails: bool | None


@dataclass(frozen=True, slots=True)
class AgentTrace:
    required: bool
    header: str | None


@dataclass(frozen=True, slots=True)
class ConcurrentFailures:
    failures: tuple[bool, ...]
    terminal: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class TeamTrace:
    required: bool
    method: str
    llm_route: bool
    passthrough: bool
    telemetry: bool
    headers_only: bool
    header: str | None
    body_trace: str | None
    body_trace_present: bool
    traceparent_id: str | None
    selected_trace: str | None
    selected_trace_present: bool
    other_trace: str | None


@dataclass(frozen=True, slots=True)
class Tools:
    requested: tuple[str, ...]
    key_allowed: tuple[str, ...] | None
    team_allowed: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class GrantLayers:
    deny_by_default: bool
    identity_present: bool
    master_key: bool
    dashboard: bool
    virtual_key: bool
    session_token: bool
    team_present: bool


@dataclass(frozen=True, slots=True)
class VectorStores:
    requested: tuple[str, ...]
    allowed: tuple[str, ...] | None
    strict: bool


@dataclass(frozen=True, slots=True)
class TeamLookup:
    dashboard: bool
    missing: bool
    recorded_models: tuple[str, ...]
    allow_unavailable: bool | None


AccessFacts: TypeAlias = (
    ConcurrentFailures
    | UserParamRequest
    | GuardrailRequest
    | TraceRequest
    | Search
    | Blocked
    | TraceGate
    | TraceStep
    | UserParam
    | ClientTags
    | Guardrails
    | AgentTrace
    | TeamTrace
    | Tools
    | GrantLayers
    | VectorStores
    | TeamLookup
)
AccessDenial: TypeAlias = Literal[
    "Search",
    "Blocked",
    "UserParam",
    "ClientTags",
    "Guardrails",
    "AgentTrace",
    "TeamTrace",
    "Tools",
    "VectorStores",
    "MissingVectorGrant",
    "TeamLookup",
]
GrantLayer: TypeAlias = Literal["key", "team", "user"]


@dataclass(frozen=True, slots=True)
class AccessDecision:
    read: Literal["UserParam", "GuardrailPermission", "UnavailablePolicy"] | None = None
    denial: AccessDenial | None = None
    rejected: tuple[str, ...] = ()
    layers: tuple[GrantLayer, ...] = ()
    rejected_index: int | None = None
    next_trace: TraceSource | None = None
    failure_index: int | None = None


_DECISION: Final = TypeAdapter(AccessDecision)


@native_boundary
def decide(facts: AccessFacts) -> AccessDecision:
    return _DECISION.validate_python(_native.request_auth_access(encode_facts(facts)))
