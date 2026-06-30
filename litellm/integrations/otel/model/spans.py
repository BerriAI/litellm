"""
This module declares every span the instrumentation can emit and the hierarchy.

Span-name patterns live here as typed builder functions.

Canonical hierarchy::

    PROXY_REQUEST  (SERVER, root)     # owned by the FastAPI instrumentor
    ├── SERVICE    (INTERNAL)         # auth phase span (live; see logger.phase_span)
    │   └── DB_CALL (CLIENT)          #   its key/user/team lookups nest here
    ├── GUARDRAIL  (INTERNAL)         # request-lifecycle hook, sibling of LLM_CALL
    ├── LLM_CALL   (CLIENT)
    └── DB_CALL    (CLIENT)           # e.g. the spend-log write

Guardrails parent to PROXY_REQUEST, not LLM_CALL: pre/during/post-call guardrail
hooks are orchestrated by the request lifecycle (a pre-call guardrail runs
before the LLM call even starts), so a guardrail is a sibling of the LLM call,
not a child of it. The emitter parents every span to the ambient OTel context
(the active server span), which matches this.

Not every service call becomes a span — :func:`span_role_for_service` decides:

- ``DB_CALL`` (CLIENT) — outbound datastores (redis, postgres,
  ``batch_write_to_db``), carrying ``db.*`` semconv.
- ``SERVICE`` (INTERNAL) — genuine internal work worth a span (background
  budget/reset jobs, pod-lock manager).
- ``None`` (metrics-only) — framework instrumentation that duplicates a gen-AI
  span (``self`` = the ``track_llm_api_timing`` wrapper, ``router``,
  ``proxy_pre_call``) or ``auth`` (which gets a live phase span instead). These
  still feed Prometheus/Datadog; they just never enter the trace.

``DB_CALL`` and ``SERVICE`` are built from the same ``ServiceSpanData``; only the
role (hence span kind and attribute vocabulary) differs. A service call can fire
outside any request (a background job), in which case it parents to no server
span and starts its own root trace rather than being dropped.

Management/admin endpoints are ordinary FastAPI routes — their SERVER spans are
owned by the instrumentor too, so they don't appear as a role here.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.integrations.otel.model.payloads import (
        GuardrailSpanData,
        LLMCallSpanData,
        MCPToolCallSpanData,
        ProxyRequestSpanData,
        ServiceSpanData,
    )


class SpanRole(str, Enum):
    PROXY_REQUEST = "proxy_request"
    LLM_CALL = "llm_call"
    MCP_TOOL_CALL = "mcp_tool_call"
    GUARDRAIL = "guardrail"
    DB_CALL = "db_call"
    SERVICE = "service"


class LiteLLMSpanKind(str, Enum):
    SERVER = "server"
    CLIENT = "client"
    INTERNAL = "internal"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass(frozen=True)
class SpanSpec:
    role: SpanRole
    kind: LiteLLMSpanKind
    parent: SpanRole | None


SPAN_REGISTRY: dict[SpanRole, SpanSpec] = {
    SpanRole.PROXY_REQUEST: SpanSpec(SpanRole.PROXY_REQUEST, LiteLLMSpanKind.SERVER, parent=None),
    SpanRole.LLM_CALL: SpanSpec(SpanRole.LLM_CALL, LiteLLMSpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST),
    # The proxy is an MCP client to the upstream server it dispatches the tool
    # call to, so this is a CLIENT span, sibling of the LLM call under the request.
    SpanRole.MCP_TOOL_CALL: SpanSpec(SpanRole.MCP_TOOL_CALL, LiteLLMSpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST),
    SpanRole.GUARDRAIL: SpanSpec(SpanRole.GUARDRAIL, LiteLLMSpanKind.INTERNAL, parent=SpanRole.PROXY_REQUEST),
    SpanRole.DB_CALL: SpanSpec(SpanRole.DB_CALL, LiteLLMSpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST),
    SpanRole.SERVICE: SpanSpec(SpanRole.SERVICE, LiteLLMSpanKind.INTERNAL, parent=SpanRole.PROXY_REQUEST),
}


# ``ServiceTypes`` value -> ``db.system.name``. These are outbound datastore
# calls and become CLIENT ``DB_CALL`` spans; ``redis_``-prefixed names cover the
# redis-backed spend queues. Any service not mapped here is litellm-internal work
# and stays an INTERNAL ``SERVICE`` span. This table is the single source of
# datastore knowledge — both the role classifier and the mapper read it.
_DB_SYSTEM_BY_SERVICE: dict[str, str] = {
    "redis": "redis",
    "postgres": "postgresql",
    "batch_write_to_db": "postgresql",
}


def db_system(service_name: str) -> str | None:
    """The ``db.system.name`` for a datastore service, else ``None``.

    ``None`` means the service is not an outbound datastore call. Redis-backed
    spend queues (``redis_*``) map to ``redis``.
    """
    if service_name in _DB_SYSTEM_BY_SERVICE:
        return _DB_SYSTEM_BY_SERVICE[service_name]
    if service_name.startswith("redis_"):
        return "redis"
    return None


# ``ServiceTypes`` values that are NOT emitted as spans — they are framework
# instrumentation that either duplicates a gen-AI span or has a better home as a
# Prometheus/Datadog metric. They still flow to those metric backends via their
# own hooks; the v2 logger just does not put them in the trace:
#
#   - ``self``           — ``track_llm_api_timing`` wraps the LLM call; the
#                          ``chat {model}`` CLIENT span already represents it.
#   - ``router``         — wraps the whole request; duplicates the server span.
#   - ``proxy_pre_call`` — per-callback pre-call timing; a guardrail's real span
#                          is ``execute_guardrail {name}``.
#   - ``auth``           — emitted instead as a live phase span (see
#                          ``logger.phase_span``) so its DB lookups nest under it,
#                          not as a flat post-hoc service span.
_METRICS_ONLY_SERVICES: frozenset[str] = frozenset({"self", "router", "proxy_pre_call", "auth"})


def span_role_for_service(service_name: str) -> SpanRole | None:
    """The span role for a service call, or ``None`` when it must not be a span.

    ``DB_CALL`` for outbound datastores, ``SERVICE`` for genuine internal work
    worth a span (background jobs), and ``None`` for framework instrumentation
    that duplicates a gen-AI span or belongs in metrics only
    (see ``_METRICS_ONLY_SERVICES``).
    """
    if service_name in _METRICS_ONLY_SERVICES:
        return None
    return SpanRole.DB_CALL if db_system(service_name) is not None else SpanRole.SERVICE


# --- span name builders (the naming convention, per role) ------------------- #


# The name the FastAPI instrumentor gives the root server span. V2 never creates
# this span (the instrumentor owns it), but it anchors request-level spans to it
# and tests assert against it by name, so the literal lives here with the rest of
# the span vocabulary rather than being duplicated at each call site.
LITELLM_PROXY_REQUEST_SPAN_NAME = "Received Proxy Server Request"


def llm_call_span_name(data: "LLMCallSpanData") -> str:
    """``"{operation} {model}"`` e.g. ``"chat gpt-4o"`` (GenAI semconv)."""
    model = data.request_model or ""
    return f"{data.operation.value} {model}".strip()


def mcp_tool_call_span_name(data: "MCPToolCallSpanData") -> str:
    """``"{mcp.method.name} {tool}"`` e.g. ``"tools/call get-weather"`` (MCP semconv)."""
    return f"{data.method} {data.tool_name}".strip()


def proxy_request_span_name(data: "ProxyRequestSpanData") -> str:
    """``"{method} {route}"`` (HTTP semconv)."""
    return f"{data.http_method} {data.route}".strip()


def guardrail_span_name(data: "GuardrailSpanData") -> str:
    return f"execute_guardrail {data.guardrail_name}".strip()


def service_span_name(data: "ServiceSpanData") -> str:
    """``"{service} {call_type}"`` e.g. ``"redis set"`` — service name alone when
    no call type is known, so identically-named calls stay distinguishable."""
    return f"{data.service_name} {data.call_type or ''}".strip()


def root_roles() -> list[SpanRole]:
    """Roles that start a new trace (no in-process parent)."""
    return [role for role, spec in SPAN_REGISTRY.items() if spec.parent is None]


def child_roles(parent: SpanRole) -> list[SpanRole]:
    return [role for role, spec in SPAN_REGISTRY.items() if spec.parent == parent]


def validate_registry(
    registry: dict[SpanRole, SpanSpec] | None = None,
) -> None:
    reg = registry if registry is not None else SPAN_REGISTRY
    for role, spec in reg.items():
        if spec.role is not role:
            raise ValueError(f"SPAN_REGISTRY[{role}] has mismatched role {spec.role}")
        if spec.parent is not None and spec.parent not in reg:
            raise ValueError(f"span role {role} declares unknown parent {spec.parent}")
    missing = [role for role in SpanRole if role not in reg]
    if missing:
        raise ValueError(f"SPAN_REGISTRY is missing roles: {missing}")
