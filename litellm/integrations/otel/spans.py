"""Source of truth #2 for the LiteLLM OpenTelemetry instrumentation: the spans.

This module declares *every* span the instrumentation can emit and, via the
``parent`` field of each :class:`SpanSpec`, the *only* hierarchy it can produce.
It is free of any ``opentelemetry`` import; the span kind is expressed with the
local :class:`LiteLLMSpanKind` enum and mapped to the OTel ``SpanKind`` at emit
time. Span-name patterns live here as typed builder functions.

Canonical hierarchy::

    PROXY_REQUEST  (SERVER, root)
    ├── LLM_CALL   (CLIENT)
    ├── GUARDRAIL  (INTERNAL)   # falls back to PROXY_REQUEST when no LLM_CALL
    └── SERVICE    (INTERNAL)

    MANAGEMENT     (SERVER, root)   # admin endpoints, independent root
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from litellm.integrations.otel.payloads import (
        GuardrailSpanData,
        LLMCallSpanData,
        ManagementSpanData,
        ProxyRequestSpanData,
        ServiceSpanData,
    )


class SpanRole(str, Enum):
    """The closed set of spans this instrumentation emits."""

    PROXY_REQUEST = "proxy_request"
    LLM_CALL = "llm_call"
    GUARDRAIL = "guardrail"
    SERVICE = "service"
    MANAGEMENT = "management"


class LiteLLMSpanKind(str, Enum):
    """OTel span kinds, decoupled from the SDK so this module stays import-light."""

    SERVER = "server"
    CLIENT = "client"
    INTERNAL = "internal"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass(frozen=True)
class SpanSpec:
    """Identity + hierarchy of a single span role."""

    role: SpanRole
    kind: LiteLLMSpanKind
    parent: Optional[SpanRole]


SPAN_REGISTRY: Dict[SpanRole, SpanSpec] = {
    SpanRole.PROXY_REQUEST: SpanSpec(
        SpanRole.PROXY_REQUEST, LiteLLMSpanKind.SERVER, parent=None
    ),
    SpanRole.LLM_CALL: SpanSpec(
        SpanRole.LLM_CALL, LiteLLMSpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST
    ),
    SpanRole.GUARDRAIL: SpanSpec(
        SpanRole.GUARDRAIL, LiteLLMSpanKind.INTERNAL, parent=SpanRole.LLM_CALL
    ),
    SpanRole.SERVICE: SpanSpec(
        SpanRole.SERVICE, LiteLLMSpanKind.INTERNAL, parent=SpanRole.PROXY_REQUEST
    ),
    SpanRole.MANAGEMENT: SpanSpec(
        SpanRole.MANAGEMENT, LiteLLMSpanKind.SERVER, parent=None
    ),
}


# --- span name builders (the naming convention, per role) ------------------- #


def llm_call_span_name(data: "LLMCallSpanData") -> str:
    """``"{operation} {model}"`` e.g. ``"chat gpt-4o"`` (GenAI semconv)."""
    model = data.request_model or ""
    return f"{data.operation.value} {model}".strip()


def proxy_request_span_name(data: "ProxyRequestSpanData") -> str:
    """``"{method} {route}"`` (HTTP semconv)."""
    return f"{data.http_method} {data.route}".strip()


def guardrail_span_name(data: "GuardrailSpanData") -> str:
    return f"execute_guardrail {data.guardrail_name}".strip()


def service_span_name(data: "ServiceSpanData") -> str:
    return data.service_name


def management_span_name(data: "ManagementSpanData") -> str:
    return data.route


def root_roles() -> List[SpanRole]:
    """Roles that start a new trace (no in-process parent)."""
    return [role for role, spec in SPAN_REGISTRY.items() if spec.parent is None]


def child_roles(parent: SpanRole) -> List[SpanRole]:
    return [role for role, spec in SPAN_REGISTRY.items() if spec.parent == parent]


def validate_registry(
    registry: Optional[Dict[SpanRole, SpanSpec]] = None,
) -> None:
    """Fail loudly if the registry is internally inconsistent.

    Guarantees: every role has a spec keyed by itself, every declared ``parent``
    references a real role (no orphans / dangling parents), and every
    :class:`SpanRole` is present. ``registry`` defaults to :data:`SPAN_REGISTRY`
    and is parameterized so the invariants can be exercised directly.
    """
    reg = registry if registry is not None else SPAN_REGISTRY
    for role, spec in reg.items():
        if spec.role is not role:
            raise ValueError(f"SPAN_REGISTRY[{role}] has mismatched role {spec.role}")
        if spec.parent is not None and spec.parent not in reg:
            raise ValueError(f"span role {role} declares unknown parent {spec.parent}")
    missing = [role for role in SpanRole if role not in reg]
    if missing:
        raise ValueError(f"SPAN_REGISTRY is missing roles: {missing}")
