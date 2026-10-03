"""
This module declares every span the instrumentation can emit and the hierarchy.

Span-name patterns live here as typed builder functions.

Canonical hierarchy::

    PROXY_REQUEST  (SERVER, root)     # owned by the FastAPI instrumentor
    ├── SERVICE    (INTERNAL)         # auth phase span (live; see logger.phase_span)
    │   └── DB_CALL (CLIENT)          #   its key/user/team lookups nest here
    ├── GUARDRAIL  (INTERNAL)         # request-lifecycle hook, sibling of LLM_CALL
    ├── LLM_CALL   (CLIENT)
    ├── MCP_TOOL_CALL  (CLIENT)       # nests under the POST carrying the message
    ├── MCP_LIST_TOOLS (CLIENT)       #   (client-propagated context is a span link)
    └── DB_CALL    (CLIENT)           # e.g. the spend-log write

Guardrails parent to PROXY_REQUEST, not LLM_CALL: pre/during/post-call guardrail
hooks are orchestrated by the request lifecycle (a pre-call guardrail runs
before the LLM call even starts), so a guardrail is a sibling of the LLM call,
not a child of it. The emitter parents every span to the ambient OTel context
(the active server span), which matches this.

MCP spans (``MCP_TOOL_CALL``, ``MCP_LIST_TOOLS``) are parented at emit time by
:func:`resolve_mcp_span_context`: they nest under the ``PROXY_REQUEST`` transport
span of the request carrying that message, so the tool call stays in one trace.
Trace context the client propagated in ``params._meta`` (SEP-414) is recorded as
a span *link*, never the parent — a remote parent would root the span in a trace
whose root never reaches the gateway's tracing backend. Links always target that
remote client context, never a registry role, so ``SpanSpec`` declares no link
field; the concrete transport parent is resolved per message at emit time.

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

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.integrations.otel.model.payloads import (
        GuardrailSpanData,
        LLMCallSpanData,
        MCPListToolsSpanData,
        MCPToolCallSpanData,
        ProxyRequestSpanData,
        ServiceSpanData,
    )


class SpanRole(str, Enum):
    PROXY_REQUEST = "proxy_request"
    LLM_CALL = "llm_call"
    MCP_TOOL_CALL = "mcp_tool_call"
    MCP_LIST_TOOLS = "mcp_list_tools"
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


SPAN_REGISTRY: Final[dict[SpanRole, SpanSpec]] = {
    SpanRole.PROXY_REQUEST: SpanSpec(SpanRole.PROXY_REQUEST, LiteLLMSpanKind.SERVER, parent=None),
    SpanRole.LLM_CALL: SpanSpec(SpanRole.LLM_CALL, LiteLLMSpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST),
    # The proxy is an MCP client to the upstream server, so MCP spans are CLIENT
    # spans. ``resolve_mcp_span_context`` nests them under the PROXY_REQUEST
    # transport span of the request carrying that message (resolved per message at
    # emit time), keeping the call in one trace. Trace context the client
    # propagated in ``params._meta`` becomes a span *link* to that remote context,
    # which is not a registry role, so ``SpanSpec`` has no link field.
    SpanRole.MCP_TOOL_CALL: SpanSpec(SpanRole.MCP_TOOL_CALL, LiteLLMSpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST),
    SpanRole.MCP_LIST_TOOLS: SpanSpec(SpanRole.MCP_LIST_TOOLS, LiteLLMSpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST),
    SpanRole.GUARDRAIL: SpanSpec(SpanRole.GUARDRAIL, LiteLLMSpanKind.INTERNAL, parent=SpanRole.PROXY_REQUEST),
    SpanRole.DB_CALL: SpanSpec(SpanRole.DB_CALL, LiteLLMSpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST),
    SpanRole.SERVICE: SpanSpec(SpanRole.SERVICE, LiteLLMSpanKind.INTERNAL, parent=SpanRole.PROXY_REQUEST),
}


# ``ServiceTypes`` value -> ``db.system.name``. These are outbound datastore
# calls and become CLIENT ``DB_CALL`` spans; ``redis_``-prefixed names cover the
# redis-backed spend queues. Any service not mapped here is litellm-internal work
# and stays an INTERNAL ``SERVICE`` span. This table is the single source of
# datastore knowledge — both the role classifier and the mapper read it.
POSTGRESQL: Final = "postgresql"

_DB_SYSTEM_BY_SERVICE: Final[dict[str, str]] = {
    "redis": "redis",
    "postgres": POSTGRESQL,
    "batch_write_to_db": POSTGRESQL,
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
_METRICS_ONLY_SERVICES: Final[frozenset[str]] = frozenset({"self", "router", "proxy_pre_call", "auth"})


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
LITELLM_PROXY_REQUEST_SPAN_NAME: Final = "Received Proxy Server Request"


def llm_call_span_name(data: "LLMCallSpanData") -> str:
    """``"{operation} {model}"`` e.g. ``"chat gpt-4o"`` (GenAI semconv)."""
    model: Final = data.request_model or ""
    return f"{data.operation.value} {model}".strip()


def mcp_tool_call_span_name(data: "MCPToolCallSpanData") -> str:
    """``"{mcp.method.name} {tool}"`` e.g. ``"tools/call get-weather"`` (MCP semconv)."""
    return f"{data.method} {data.tool_name}".strip()


def mcp_list_tools_span_name(data: "MCPListToolsSpanData") -> str:
    """``"{mcp.method.name}"`` i.e. ``"tools/list"`` — no low-cardinality target, so
    the method name alone names the span (MCP semconv)."""
    return data.method


def proxy_request_span_name(data: "ProxyRequestSpanData") -> str:
    """``"{method} {route}"`` (HTTP semconv)."""
    return f"{data.http_method} {data.route}".strip()


def guardrail_span_name(data: "GuardrailSpanData") -> str:
    return f"execute_guardrail {data.guardrail_name}".strip()


_SERVICE_VERB_BY_CALL_TYPE: Final[dict[str, str]] = {
    "get_cache": "get",
    "async_get_cache": "get",
    "batch_get_cache": "mget",
    "async_batch_get_cache": "mget",
    "set_cache": "set",
    "async_set_cache": "set",
    "async_set_cache_pipeline": "set",
    "async_set_cache_pipeline_with_ttls": "set",
    "async_set_cache_sadd": "sadd",
    "increment_cache": "incr",
    "async_increment": "incr",
    "async_increment_pipeline": "incr",
    "delete_cache": "delete",
    "async_delete_cache": "delete",
    "async_rpush": "rpush",
    "async_lpop": "lpop",
    "async_scan_iter": "scan",
    "async_lpop_pipeline": "lpop",
    "async_rpush_pipeline": "rpush",
    "async_rpush_and_trim": "rpush",
    "increment_cache_ttl": "ttl",
    "increment_cache_expire": "expire",
    "async_ping": "ping",
    "sync_ping": "ping",
    "redis_async_ping": "ping",
    "redis_sync_ping": "ping",
    "request_redis_batch": "pipeline",
    "post_call_redis_batch": "pipeline",
}


@dataclass(frozen=True, slots=True)
class PostgresOperation:
    """The SQL verb and primary table behind a Prisma helper, for ``postgres.{verb} {table}``;
    ``collection`` lists every relation on ``db.collection.name`` when one query joins several."""

    verb: str
    table: str | None
    collection: str | None = None


_POSTGRES_SERVICE: Final = "postgres"
PG_CATALOG: Final = "pg_catalog"
_PRISMA_VIEWS: Final[frozenset[str]] = frozenset(
    (
        "LiteLLM_VerificationTokenView",
        "MonthlyGlobalSpend",
        "Last30dKeysBySpend",
        "Last30dModelsBySpend",
        "MonthlyGlobalSpendPerKey",
        "MonthlyGlobalSpendPerUserPerKey",
        "Last30dTopEndUsersSpend",
        "DailyTagSpend",
    )
)
_PRISMA_MODELS: Final[frozenset[str]] = frozenset(
    (
        "LiteLLM_BudgetTable",
        "LiteLLM_CredentialsTable",
        "LiteLLM_ProxyModelTable",
        "LiteLLM_AgentsTable",
        "LiteLLM_AgentIdentity",
        "LiteLLM_RetiredAgentIdentity",
        "LiteLLM_RetiredAgent",
        "LiteLLM_VerifiedSubject",
        "LiteLLM_OrganizationTable",
        "LiteLLM_ModelTable",
        "LiteLLM_TeamTable",
        "LiteLLM_ProjectTable",
        "LiteLLM_DeletedTeamTable",
        "LiteLLM_UserTable",
        "LiteLLM_ObjectPermissionTable",
        "LiteLLM_MCPServerTable",
        "LiteLLM_MCPToolsetTable",
        "LiteLLM_MCPUserCredentials",
        "LiteLLM_MCPUserEnvVars",
        "LiteLLM_MCPServerOAuthClient",
        "LiteLLM_SSOIdentityAssertion",
        "LiteLLM_VerificationToken",
        "LiteLLM_JWTKeyMapping",
        "LiteLLM_DeprecatedVerificationToken",
        "LiteLLM_DeletedVerificationToken",
        "LiteLLM_EndUserTable",
        "LiteLLM_ModelAccessGroupBudgetTable",
        "LiteLLM_TagTable",
        "LiteLLM_Config",
        "LiteLLM_SpendLogs",
        "LiteLLM_BudgetWindowSpend",
        "LiteLLM_ErrorLogs",
        "LiteLLM_UserNotifications",
        "LiteLLM_TeamMembership",
        "LiteLLM_OrganizationMembership",
        "LiteLLM_InvitationLink",
        "LiteLLM_AuditLog",
        "LiteLLM_DailyUserSpend",
        "LiteLLM_DailyGlobalSpend",
        "LiteLLM_DailyOrganizationSpend",
        "LiteLLM_DailyEndUserSpend",
        "LiteLLM_DailyAgentSpend",
        "LiteLLM_DailyTeamSpend",
        "LiteLLM_DailyTagSpend",
        "LiteLLM_ProxyWorkerHeartbeat",
        "LiteLLM_CronJob",
        "LiteLLM_ManagedFileTable",
        "LiteLLM_ManagedObjectTable",
        "LiteLLM_ManagedFileContentTable",
        "LiteLLM_ManagedVectorStoreTable",
        "LiteLLM_ManagedVectorStoresTable",
        "LiteLLM_GuardrailsTable",
        "LiteLLM_DailyGuardrailMetrics",
        "LiteLLM_DailyGuardrailUsageUnits",
        "LiteLLM_DailyPolicyMetrics",
        "LiteLLM_SpendLogGuardrailIndex",
        "LiteLLM_SpendLogToolIndex",
        "LiteLLM_DailyToolSpend",
        "LiteLLM_DailyModelUsage",
        "LiteLLM_DailyGatewayRequests",
        "LiteLLM_PromptTable",
        "LiteLLM_HealthCheckTable",
        "LiteLLM_SearchToolsTable",
        "LiteLLM_SSOConfig",
        "LiteLLM_ManagedVectorStoreIndexTable",
        "LiteLLM_CacheConfig",
        "LiteLLM_UISettings",
        "LiteLLM_ConfigOverrides",
        "LiteLLM_SkillsTable",
        "LiteLLM_PolicyTable",
        "LiteLLM_PolicyAttachmentTable",
        "LiteLLM_ToolTable",
        "LiteLLM_AccessGroupTable",
        "LiteLLM_ClaudeCodePluginTable",
        "LiteLLM_MemoryTable",
        "LiteLLM_AdaptiveRouterState",
        "LiteLLM_AdaptiveRouterSession",
        "LiteLLM_AutoRouterBaselineComparison",
        "LiteLLM_AutoRouterBaselineObservation",
        "LiteLLM_AutoRouterSession",
        "LiteLLM_AutoRouterUserSession",
        "LiteLLM_AutoRouterDailySpend",
        "LiteLLM_ShadowEvalJob",
        "LiteLLM_ShadowEvalAttempt",
        "LiteLLM_ShadowEvalFunnel",
        "LiteLLM_WorkflowRun",
        "LiteLLM_WorkflowEvent",
        "LiteLLM_WorkflowMessage",
        "LiteLLM_Lens",
        "LiteLLM_LensRun",
        "LiteLLM_LensWorker",
    )
)
PRISMA_RELATIONS: Final[frozenset[str]] = _PRISMA_MODELS | _PRISMA_VIEWS
_TABLE_NAME_METADATA_KEY: Final = "table_name"

_PRISMA_MODEL_BY_TABLE_NAME: Final[Mapping[str, str]] = MappingProxyType(
    {
        "key": "LiteLLM_VerificationToken",
        "keys": "LiteLLM_VerificationToken",
        "combined_view": "LiteLLM_VerificationToken",
        "user": "LiteLLM_UserTable",
        "users": "LiteLLM_UserTable",
        "team": "LiteLLM_TeamTable",
        "config": "LiteLLM_Config",
        "spend": "LiteLLM_SpendLogs",
        "enduser": "LiteLLM_EndUserTable",
        "budget": "LiteLLM_BudgetTable",
        "user_notification": "LiteLLM_UserNotifications",
    }
)

_AUTH_OBJECT_RELATIONS: Final = ",".join(
    (
        "LiteLLM_UserTable",
        "LiteLLM_TeamTable",
        "LiteLLM_TeamMembership",
        "LiteLLM_OrganizationTable",
        "LiteLLM_OrganizationMembership",
        "LiteLLM_ProjectTable",
        "LiteLLM_ModelTable",
        "LiteLLM_BudgetTable",
        "LiteLLM_ObjectPermissionTable",
    )
)
_POSTGRES_OPERATION_BY_CALL_TYPE: Final[Mapping[str, PostgresOperation]] = MappingProxyType(
    {
        "get_data": PostgresOperation("select", None),
        "get_generic_data": PostgresOperation("select", None),
        "insert_data": PostgresOperation("insert", None),
        "update_data": PostgresOperation("update", None),
        "delete_data": PostgresOperation("delete", None),
        "get_key_object": PostgresOperation("select", "LiteLLM_VerificationToken"),
        "get_user_object": PostgresOperation("select", "LiteLLM_UserTable"),
        "get_org_object": PostgresOperation("select", "LiteLLM_OrganizationTable"),
        "get_org_object_by_alias": PostgresOperation("select", "LiteLLM_OrganizationTable"),
        "_get_team_db_check": PostgresOperation("select", "LiteLLM_TeamTable"),
        "get_team_object_by_alias": PostgresOperation("select", "LiteLLM_TeamTable"),
        "_fetch_team_membership_from_db": PostgresOperation("select", "LiteLLM_TeamMembership"),
        "get_team_member_default_budget": PostgresOperation("select", "LiteLLM_BudgetTable"),
        "get_end_user_object": PostgresOperation("select", "LiteLLM_EndUserTable"),
        "get_tag_object": PostgresOperation("select", "LiteLLM_TagTable"),
        "get_tag_objects_batch": PostgresOperation("select", "LiteLLM_TagTable"),
        "get_model_access_group_budgets_batch": PostgresOperation("select", "LiteLLM_ModelAccessGroupBudgetTable"),
        "get_access_object": PostgresOperation("select", "LiteLLM_AccessGroupTable"),
        "get_object_permission": PostgresOperation("select", "LiteLLM_ObjectPermissionTable"),
        "get_jwt_key_mapping_object": PostgresOperation("select", "LiteLLM_JWTKeyMapping"),
        "get_jwt_key_mapping_cache_keys_for_token": PostgresOperation("select", "LiteLLM_JWTKeyMapping"),
        "get_managed_vector_store_rows_by_uuids": PostgresOperation("select", "LiteLLM_ManagedVectorStoresTable"),
        "commit_spend_updates": PostgresOperation("update", None),
        "update_end_user_spend": PostgresOperation("upsert", None),
        "upsert_daily_spend": PostgresOperation("upsert", None),
        "insert_spend_logs": PostgresOperation("insert", None),
        "migrate_config_credentials": PostgresOperation("update", None),
        "migrate_sso_credentials": PostgresOperation("update", None),
        "backfill_mcp_oauth_issuer": PostgresOperation("update", None),
        "auto_register_jwt_mapping": PostgresOperation("insert", None),
        "delete_orphaned_jwt_key": PostgresOperation("delete", None),
        "save_email_settings": PostgresOperation("upsert", None),
        "reset_budget_cascade": PostgresOperation("transaction", None),
        "reset_spend_rows": PostgresOperation("update", None),
        "reset_budget_windows": PostgresOperation("select", None),
        "write_budget_windows": PostgresOperation("update", None),
        "roll_window_spend_row": PostgresOperation("update", None),
        "seed_window_spend": PostgresOperation("select", None),
        "select_window_spend_rows": PostgresOperation("select", None),
        "commit_window_spend_updates": PostgresOperation("upsert", None),
        "index_spend_log_tools": PostgresOperation("insert", None),
        "commit_daily_tool_spend": PostgresOperation("upsert", None),
        "flush_shadow_eval_funnel": PostgresOperation("upsert", None),
        "commit_gateway_requests": PostgresOperation("upsert", None),
        "cleanup_expired_rows": PostgresOperation("delete", None),
        "count_expired_rows": PostgresOperation("select", None),
        "check_spend_log_partitioning": PostgresOperation("select", None),
        "list_spend_log_partitions": PostgresOperation("select", None),
        "create_spend_log_partition": PostgresOperation("ddl", None),
        "proxy_worker_heartbeat": PostgresOperation("upsert", None),
        "prune_proxy_worker_heartbeats": PostgresOperation("delete", None),
        "deregister_proxy_worker": PostgresOperation("delete", None),
        "count_live_proxy_workers": PostgresOperation("select", None),
        "recover_key_metadata": PostgresOperation("select", None),
        "recover_user_details": PostgresOperation("select", None),
        "sync_team_access_group_membership": PostgresOperation("transaction", None),
        "latest_health_checks": PostgresOperation("select", None),
        "prefetch_auth_objects": PostgresOperation("select", "auth_objects", _AUTH_OBJECT_RELATIONS),
        "baseline_accounting": PostgresOperation("transaction", "LiteLLM_AutoRouterBaselineComparison"),
        "write_autorouter_turn": PostgresOperation("upsert", None),
        "team_user_spend": PostgresOperation("select", "LiteLLM_SpendLogs"),
        "daily_activity_query": PostgresOperation("select", None),
        "auto_router_report_query": PostgresOperation("select", None),
        "create_view": PostgresOperation("ddl", None),
        "health_check": PostgresOperation("ping", None),
        "db_health_watchdog": PostgresOperation("ping", None),
        "find_unique": PostgresOperation("select", None),
        "find_first": PostgresOperation("select", None),
        "find_many": PostgresOperation("select", None),
        "count": PostgresOperation("select", None),
        "group_by": PostgresOperation("select", None),
        "create": PostgresOperation("insert", None),
        "create_many": PostgresOperation("insert", None),
        "update": PostgresOperation("update", None),
        "update_many": PostgresOperation("update", None),
        "delete": PostgresOperation("delete", None),
        "delete_many": PostgresOperation("delete", None),
        "upsert": PostgresOperation("upsert", None),
    }
)
_RAW_PRISMA_CALL_TYPES: Final[frozenset[str]] = frozenset(("query_raw", "execute_raw"))
_DB_OPERATION_METADATA_KEY: Final = "db_operation"
_POSTGRES_VERBS: Final[frozenset[str]] = frozenset(
    ("select", "insert", "update", "delete", "upsert", "ddl", "set", "ping")
)
_TARGETLESS_VERBS: Final[frozenset[str]] = frozenset(("ping",))
_SETTING_NAME: Final = re.compile(r"[a-z_][a-z0-9_.]*")


def _postgres_table_from_metadata(data: "ServiceSpanData", verb: str) -> str | None:
    """The relation named by the event's ``table_name`` metadata, or ``None``.

    Only the short ``PrismaClient`` literals, the relations declared in ``schema.prisma``
    (plus the spend views), ``pg_catalog`` and, for a ``set`` verb, a Postgres setting name
    resolve, so a free-form string can never become a span-name cardinality."""
    table_name: Final = data.event_metadata.get(_TABLE_NAME_METADATA_KEY)
    if not isinstance(table_name, str):
        return None
    if table_name in PRISMA_RELATIONS or table_name == PG_CATALOG:
        return table_name
    if verb == "set":
        return table_name if _SETTING_NAME.fullmatch(table_name) else None
    return _PRISMA_MODEL_BY_TABLE_NAME.get(table_name)


def _postgres_verb_from_metadata(data: "ServiceSpanData") -> str | None:
    """The SQL verb a raw-statement producer declared on ``db_operation``, bounded to the known verbs."""
    verb: Final = data.event_metadata.get(_DB_OPERATION_METADATA_KEY)
    return verb if isinstance(verb, str) and verb in _POSTGRES_VERBS else None


def postgres_operation(data: "ServiceSpanData") -> PostgresOperation | None:
    """The verb and table behind a ``postgres`` service event, else ``None``.

    ``None`` for every other service (Redis keeps its own verb table) and for a
    Postgres call type this module does not know, which stays ``postgres {call_type}``."""
    if data.service_name != _POSTGRES_SERVICE or not data.call_type:
        return None
    if data.call_type in _RAW_PRISMA_CALL_TYPES:
        verb: Final = _postgres_verb_from_metadata(data)
        return PostgresOperation(verb, _postgres_table_from_metadata(data, verb)) if verb is not None else None
    operation: Final = _POSTGRES_OPERATION_BY_CALL_TYPE.get(data.call_type)
    if operation is None:
        return None
    if operation.table is not None:
        return operation
    return PostgresOperation(operation.verb, _postgres_table_from_metadata(data, operation.verb))


def service_operation(data: "ServiceSpanData") -> str | None:
    """``"redis.get"`` when the call type is a known datastore verb, else ``None``."""
    if not data.call_type:
        return None
    verb: Final = _SERVICE_VERB_BY_CALL_TYPE.get(data.call_type)
    if verb is None:
        return None
    return f"{data.service_name}.{verb}"


def service_span_name(data: "ServiceSpanData") -> str:
    """``"{service}.{verb} {target}"`` (``"redis.get llm_response"``) for a known datastore
    verb, ``"{service}.{verb}"`` (``"redis.pipeline"``) when the producer declared no
    target, ``"postgres.{verb} {table}"`` (``"postgres.select LiteLLM_UserTable"``) for a
    known Prisma helper whose table resolved (from the helper or the event's ``table_name``,
    never from the ambient ``service_target``, which names a cache key family), else
    ``"{service} {call_type}"`` (``"postgres some_helper"``, and a known helper whose table
    did not resolve, so a half-named ``postgres.select`` never ships) — service name alone
    when no call type is known, so identically-named calls stay distinguishable."""
    postgres: Final = postgres_operation(data)
    if postgres is not None and postgres.table is not None:
        return f"{data.service_name}.{postgres.verb} {postgres.table}"
    if postgres is not None and postgres.verb in _TARGETLESS_VERBS:
        return f"{data.service_name}.{postgres.verb}"
    if postgres is not None:
        return f"{data.service_name} {data.call_type}"
    operation: Final = service_operation(data)
    if operation is None:
        return f"{data.service_name} {data.call_type or ''}".strip()
    return f"{operation} {data.target}" if data.target else operation


def root_roles() -> list[SpanRole]:
    """Roles with no in-process parent, i.e. they start a new trace (only the
    instrumentor-owned ``PROXY_REQUEST`` server span today)."""
    return [role for role, spec in SPAN_REGISTRY.items() if spec.parent is None]


def child_roles(parent: SpanRole) -> list[SpanRole]:
    return [role for role, spec in SPAN_REGISTRY.items() if spec.parent == parent]


def validate_registry(
    registry: dict[SpanRole, SpanSpec] | None = None,
) -> None:
    reg: Final = registry if registry is not None else SPAN_REGISTRY
    for role, spec in reg.items():
        if spec.role is not role:
            raise ValueError(f"SPAN_REGISTRY[{role}] has mismatched role {spec.role}")
        if spec.parent is not None and spec.parent not in reg:
            raise ValueError(f"span role {role} declares unknown parent {spec.parent}")
    missing: Final = [role for role in SpanRole if role not in reg]
    if missing:
        raise ValueError(f"SPAN_REGISTRY is missing roles: {missing}")
