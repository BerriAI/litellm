import enum
import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Json,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import NotRequired, ReadOnly, Required, TypedDict

from litellm._uuid import uuid
from litellm.constants import DEFAULT_STAGGER_WINDOW_SECONDS, MCP_STDIO_ALLOWED_COMMANDS
from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    validate_langfuse_environment_value,
    validate_no_callback_env_reference,
)
from litellm.types.integrations.compression_interception import (
    CompressionSavingsMetadata,
)
from litellm.types.integrations.slack_alerting import AlertType
from litellm.types.llms.openai import (
    AllMessageValues,
    ResponsesAPIResponse,
)
from litellm.types.mcp import (
    MCPAuth,
    MCPAuthType,
    MCPCredentials,
    MCPTransport,
    MCPTransportType,
)
from litellm.types.mcp_server.mcp_server_manager import MCPInfo
from litellm.types.proxy.control_plane_endpoints import WorkerRegistryEntry
from litellm.types.router import RouterErrors, UpdateRouterConfig
from litellm.types.secret_managers.main import KeyManagementSystem
from litellm.types.utils import (
    CallTypes,
    CostBreakdown,
    EmbeddingResponse,
    GenericBudgetConfigType,
    ImageResponse,
    InternalCallOrigin,
    LiteLLMPydanticObjectBase,
    ModelResponse,
    ProviderField,
    StandardCallbackDynamicParams,
    StandardLoggingGuardrailInformation,
    StandardLoggingMCPToolCall,
    StandardLoggingModelInformation,
    StandardLoggingPayloadErrorInformation,
    StandardLoggingPayloadStatus,
    StandardLoggingRoutingDecision,
    StandardLoggingVectorStoreRequest,
    StandardPassThroughResponseObject,
    TextCompletionResponse,
)
from litellm.types.videos.main import VideoObject

from .types_utils.utils import get_instance_fn, validate_custom_validate_return_type

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span | Any
else:
    Span = Any


class ReconcileOutcome(NamedTuple):
    """What a model reconcile observed, captured while it still held the reconcile
    lock.

    Both fields have to be read under that lock to be worth anything. ``live_after``
    in particular is the router's serving state the instant this reconcile finished,
    which is NOT the same as what a later snapshot would see: any other model write
    admitted in between briefly un-serves every db model (see ``clear_cache``), so a
    caller that re-snapshots at verdict time can observe that hole and blame its own
    reload for it.

    - ``still_desired``: the db + config ids the reconcile reconciled against, or None
      when no reconcile ran and the desired set is therefore unknown.
    - ``live_after``: the ids the router served immediately after the reconcile, or
      None when no reconcile ran.
    """

    still_desired: frozenset[str] | None
    live_after: frozenset[str] | None


class SupportedDBObjectType(str, enum.Enum):
    """
    Supported database object types for fine-grained DB storage control.
    Use in general_settings.supported_db_objects to specify which objects to load from DB.
    """

    MODELS = "models"
    MCP = "mcp"
    GUARDRAILS = "guardrails"
    POLICIES = "policies"
    VECTOR_STORES = "vector_stores"
    PASS_THROUGH_ENDPOINTS = "pass_through_endpoints"
    PROMPTS = "prompts"
    MODEL_COST_MAP = "model_cost_map"
    TOOLS = "tools"
    CONFIG_OVERRIDES = "config_overrides"

    def __str__(self):
        return str(self.value)


class LiteLLMTeamRoles(enum.Enum):
    # team admin
    TEAM_ADMIN = "admin"
    # team member
    TEAM_MEMBER = "user"


class LitellmUserRoles(str, enum.Enum):
    """
    Admin Roles:
    PROXY_ADMIN: admin over the platform
    PROXY_ADMIN_VIEW_ONLY: can login, view all own keys, view all spend
    ORG_ADMIN: admin over a specific organization, can create teams, users only within their organization

    Internal User Roles:
    INTERNAL_USER: can login, view/create/delete their own keys, view their spend
    INTERNAL_USER_VIEW_ONLY: can login, view their own keys, view their own spend


    Team Roles:
    TEAM: used for JWT auth


    Customer Roles:
    CUSTOMER: External users -> these are customers

    """

    # Admin Roles
    PROXY_ADMIN = "proxy_admin"
    PROXY_ADMIN_VIEW_ONLY = "proxy_admin_viewer"

    # Organization admins
    ORG_ADMIN = "org_admin"

    # Internal User Roles
    INTERNAL_USER = "internal_user"
    INTERNAL_USER_VIEW_ONLY = "internal_user_viewer"

    # Team Roles
    TEAM = "team"

    # Customer Roles - External users of proxy
    CUSTOMER = "customer"

    def __str__(self):
        return str(self.value)

    def values(self) -> list[str]:
        return list(self.__annotations__.keys())

    @property
    def description(self):
        """
        Descriptions for the enum values
        """
        descriptions: Final = {
            "proxy_admin": "admin over litellm proxy, has all permissions",
            "proxy_admin_viewer": "view all keys, view all spend",
            "internal_user": "view/create/delete their own keys, view their own spend",
            "internal_user_viewer": "view their own keys, view their own spend",
            "team": "team scope used for JWT auth",
            "customer": "customer",
        }
        return descriptions.get(self.value, "")

    @property
    def ui_label(self):
        """
        UI labels for the enum values
        """
        ui_labels: Final = {
            "proxy_admin": "Admin (All Permissions)",
            "proxy_admin_viewer": "Admin (View Only)",
            "internal_user": "Internal User (Create/Delete/View)",
            "internal_user_viewer": "Internal User (View Only)",
            "team": "Team",
            "customer": "Customer",
        }
        return ui_labels.get(self.value, "")

    @property
    def is_internal_user_role(self) -> bool:
        """returns true if this role is an `internal_user` or `internal_user_viewer` role"""
        return self.value in [
            self.INTERNAL_USER,
            self.INTERNAL_USER_VIEW_ONLY,
        ]


class LitellmTableNames(str, enum.Enum):
    """
    Enum for Table Names used by LiteLLM
    """

    TEAM_TABLE_NAME = "LiteLLM_TeamTable"
    USER_TABLE_NAME = "LiteLLM_UserTable"
    KEY_TABLE_NAME = "LiteLLM_VerificationToken"
    PROXY_MODEL_TABLE_NAME = "LiteLLM_ProxyModelTable"
    MANAGED_FILE_TABLE_NAME = "LiteLLM_ManagedFileTable"
    TOOL_TABLE_NAME = "LiteLLM_ToolTable"
    CACHE_CONFIG_TABLE_NAME = "LiteLLM_CacheConfig"
    CONFIG_OVERRIDES_TABLE_NAME = "LiteLLM_ConfigOverrides"
    CONFIG_TABLE_NAME = "LiteLLM_Config"
    SSO_CONFIG_TABLE_NAME = "LiteLLM_SSOConfig"
    UI_SETTINGS_TABLE_NAME = "LiteLLM_UISettings"


class Litellm_EntityType(enum.Enum):
    """
    Enum for types of entities on litellm

    This enum allows specifying the type of entity that is being tracked in the database.
    """

    KEY = "key"
    USER = "user"
    END_USER = "end_user"
    TEAM = "team"
    TEAM_MEMBER = "team_member"
    ORGANIZATION = "organization"
    PROJECT = "project"
    TAG = "tag"
    AGENT = "agent"

    # global proxy level entity
    PROXY = "proxy"


def hash_token(token: str):
    import hashlib

    # Hash the string using SHA-256
    hashed_token: Final = hashlib.sha256(token.encode()).hexdigest()

    return hashed_token


class KeyManagementRoutes(str, enum.Enum):
    """
    Enum for key management routes
    """

    # write routes
    KEY_GENERATE = "/key/generate"
    KEY_UPDATE = "/key/update"
    KEY_DELETE = "/key/delete"
    KEY_REGENERATE = "/key/regenerate"
    KEY_GENERATE_SERVICE_ACCOUNT = "/key/service-account/generate"
    KEY_REGENERATE_WITH_PATH_PARAM = "/key/{key_id}/regenerate"
    KEY_BLOCK = "/key/block"
    KEY_UNBLOCK = "/key/unblock"
    KEY_BULK_UPDATE = "/key/bulk_update"
    TEAM_KEY_BULK_UPDATE = "/team/key/bulk_update"
    KEY_RESET_SPEND = "/key/{key_id}/reset_spend"

    # Field-level opt-in permission (not a real HTTP route). When present in a
    # team's `team_member_permissions`, non-admin members of that team may set
    # `access_group_ids` on keys they create/update. Default-deny.
    KEY_ACCESS_GROUP_ASSIGNMENT = "/key/access_group_assignment"

    # info and health routes
    KEY_INFO = "/key/info"
    KEY_HEALTH = "/key/health"

    # list routes
    KEY_LIST = "/key/list"
    KEY_ALIASES = "/key/aliases"

    # team usage routes
    TEAM_DAILY_ACTIVITY = "/team/daily/activity"
    TEAM_DAILY_ACTIVITY_AGGREGATED = "/team/daily/activity/aggregated"

    # team spend-log viewing
    SPEND_LOGS = "/spend/logs"
    SPEND_LOGS_V2 = "/spend/logs/v2"


class LiteLLMRoutes(enum.Enum):
    openai_route_names = [
        "chat_completion",
        "completion",
        "embeddings",
        "image_generation",
        "video_generation",
        "audio_transcriptions",
        "moderations",
        "model_list",  # OpenAI /v1/models route
    ]
    openai_routes = [
        # chat completions
        "/engines/{model}/chat/completions",
        "/openai/deployments/{model}/chat/completions",
        "/chat/completions",
        "/v1/chat/completions",
        "/cursor/chat/completions",
        "/cursor/models",
        "/cursor/v1/models",
        # completions
        "/engines/{model}/completions",
        "/openai/deployments/{model}/completions",
        "/completions",
        "/v1/completions",
        # embeddings
        "/engines/{model}/embeddings",
        "/openai/deployments/{model}/embeddings",
        "/embeddings",
        "/v1/embeddings",
        # image generation
        "/images/generations",
        "/v1/images/generations",
        # image edit
        "/images/edits",
        "/v1/images/edits",
        # video generation
        "/videos",
        "/v1/videos",
        "/videos/{video_id}",
        "/v1/videos/{video_id}",
        "/videos/{video_id}/content",
        "/v1/videos/{video_id}/content",
        "/videos/{video_id}/remix",
        "/v1/videos/{video_id}/remix",
        # audio transcription
        "/audio/transcriptions",
        "/v1/audio/transcriptions",
        # audio Speech
        "/audio/speech",
        "/v1/audio/speech",
        # moderations
        "/moderations",
        "/v1/moderations",
        # batches
        "/v1/batches",
        "/batches",
        "/v1/batches/{batch_id}",
        "/batches/{batch_id}",
        "/v1/batches/{batch_id}/cancel",
        "/batches/{batch_id}/cancel",
        # files
        "/v1/files",
        "/files",
        "/v1/files/{file_id}",
        "/files/{file_id}",
        "/v1/files/{file_id}/content",
        "/files/{file_id}/content",
        # fine_tuning
        "/fine_tuning/jobs",
        "/v1/fine_tuning/jobs",
        "/fine_tuning/jobs/{fine_tuning_job_id}/cancel",
        "/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel",
        # assistants-related routes
        "/assistants",
        "/v1/assistants",
        "/v1/assistants/{assistant_id}",
        "/assistants/{assistant_id}",
        "/threads",
        "/v1/threads",
        "/threads/{thread_id}",
        "/v1/threads/{thread_id}",
        "/threads/{thread_id}/messages",
        "/v1/threads/{thread_id}/messages",
        "/threads/{thread_id}/runs",
        "/v1/threads/{thread_id}/runs",
        # models
        "/models",
        "/v1/models",
        # token counter
        "/utils/token_counter",
        "/utils/transform_request",
        # rerank
        "/rerank",
        "/v1/rerank",
        "/v2/rerank",
        # realtime
        "/realtime",
        "/v1/realtime",
        "/openai/v1/realtime",
        "/realtime?{model}",
        "/v1/realtime?{model}",
        "/openai/v1/realtime?{model}",
        # realtime (GA WebRTC HTTP routes)
        "/realtime/client_secrets",
        "/v1/realtime/client_secrets",
        "/openai/v1/realtime/client_secrets",
        "/realtime/calls",
        "/v1/realtime/calls",
        "/openai/v1/realtime/calls",
        "/realtime/transcription_sessions",
        "/v1/realtime/transcription_sessions",
        "/openai/v1/realtime/transcription_sessions",
        # responses API
        "/responses",
        "/v1/responses",
        "/openai/v1/responses",
        "/responses/{response_id}",
        "/v1/responses/{response_id}",
        "/openai/v1/responses/{response_id}",
        "/responses/{response_id}/input_items",
        "/v1/responses/{response_id}/input_items",
        "/openai/v1/responses/{response_id}/input_items",
        "/responses/{response_id}/cancel",
        "/v1/responses/{response_id}/cancel",
        "/openai/v1/responses/{response_id}/cancel",
        # vector stores
        "/vector_stores",
        "/v1/vector_stores",
        "/vector_stores/{vector_store_id}",
        "/v1/vector_stores/{vector_store_id}",
        "/vector_stores/{vector_store_id}/search",
        "/v1/vector_stores/{vector_store_id}/search",
        "/vector_stores/{vector_store_id}/files",
        "/v1/vector_stores/{vector_store_id}/files",
        "/vector_stores/{vector_store_id}/files/{file_id}",
        "/v1/vector_stores/{vector_store_id}/files/{file_id}",
        "/vector_stores/{vector_store_id}/files/{file_id}/content",
        "/v1/vector_stores/{vector_store_id}/files/{file_id}/content",
        "/vector_store/list",
        "/v1/vector_store/list",
        # search
        "/search",
        "/v1/search",
        "/search/{search_tool_name}",
        "/v1/search/{search_tool_name}",
        # OCR
        "/ocr",
        "/v1/ocr",
        # containers API
        "/containers",
        "/v1/containers",
        "/containers/*",
        "/v1/containers/*",
    ]

    mapped_pass_through_routes = [
        "/bedrock",
        "/comprehendmedical",
        "/vertex-ai",
        "/vertex_ai",
        "/cohere",
        "/cursor",
        "/gemini",
        "/anthropic",
        "/langfuse",
        "/azure",
        "/azure_ai",
        "/openai",
        "/openai_passthrough",
        "/assemblyai",
        "/eu.assemblyai",
        "/vllm",
        "/mistral",
        "/milvus",
        "/watsonx",
    ]

    #########################################################
    # e.g /vllm/*, anthropic/*, etc.
    # allows using /anthropic/v1/messages, /vllm/v1/chat/completions, etc.
    #########################################################
    passthrough_routes_wildcard = [f"{route}/*" for route in mapped_pass_through_routes]

    litellm_native_routes = [
        "/rag/ingest",
        "/v1/rag/ingest",
        "/rag/query",
        "/v1/rag/query",
    ]

    anthropic_routes = [
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/skills",
        "/v1/skills/{skill_id}",
    ]

    # MCP tool-call / passthrough routes — data-plane. Gated by DISABLE_LLM_API_ENDPOINTS.
    mcp_inference_routes = [
        "/mcp",
        "/mcp/",
        "/mcp/{subpath}",
        "/mcp/tools",
        "/mcp/tools/list",
        "/mcp/tools/call",
        "/mcp-rest/tools/list",
        "/mcp-rest/tools/call",
        "/v1/mcp/tools",
    ]

    # MCP server CRUD routes — control-plane. Gated by DISABLE_ADMIN_ENDPOINTS.
    mcp_management_routes = [
        "/v1/mcp/server",
        "/v1/mcp/server/{path:path}",
    ]

    # Backwards-compat union — virtual keys may be configured with
    # allowed_routes=["mcp_routes"], which should cover both halves.
    mcp_routes = mcp_inference_routes + mcp_management_routes

    # A2A agent invocation / discovery routes — data-plane. Gated by DISABLE_LLM_API_ENDPOINTS.
    agent_inference_routes = (
        "/agents",
        "/a2a/{agent_id}",
        "/a2a/{agent_id}/message/send",
        "/a2a/{agent_id}/message/stream",
        "/a2a/{agent_id}/.well-known/agent-card.json",
    )

    # Agent registry CRUD routes — control-plane. Gated by DISABLE_ADMIN_ENDPOINTS.
    # The handlers in agent_endpoints/endpoints.py enforce proxy-admin on writes and
    # scope reads by role, so these also appear in self_managed_routes.
    agent_management_routes = (
        "/v1/agents",
        "/v1/agents/{agent_id}",
        "/v1/agents/make_public",
        "/v1/agents/{agent_id}/make_public",
    )

    # Backwards-compat union — virtual keys may be configured with
    # allowed_routes=["agent_routes"], which should cover both halves.
    agent_routes = agent_inference_routes + agent_management_routes

    google_routes = [
        "/v1beta/models/{model_name:path}:countTokens",
        "/v1beta/models/{model_name:path}:generateContent",
        "/v1beta/models/{model_name:path}:streamGenerateContent",
        "/models/{model_name:path}:countTokens",
        "/models/{model_name:path}:generateContent",
        "/models/{model_name:path}:streamGenerateContent",
        # Google Interactions API
        "/interactions",
        "/v1beta/interactions",
        "/interactions/{interaction_id}",
        "/v1beta/interactions/{interaction_id}",
        "/interactions/{interaction_id}/cancel",
        "/v1beta/interactions/{interaction_id}/cancel",
        # Google Managed Agents API
        "/v1beta/agents",
        "/v1beta/agents/{name}",
        "/v1beta/agents/{name}/versions",
    ]

    apply_guardrail_routes = [
        "/guardrails/apply_guardrail",
    ]

    model_info_routes = [
        "/model/info",
        "/v1/model/info",
    ]

    llm_api_routes = (
        openai_routes
        + anthropic_routes
        + google_routes
        + mapped_pass_through_routes
        + passthrough_routes_wildcard
        + apply_guardrail_routes
        + mcp_inference_routes
        + litellm_native_routes
        + list(agent_inference_routes)
        + model_info_routes
    )
    info_routes = [
        "/key/info",
        "/key/health",
        "/team/info",
        "/team/list",
        "/v2/team/list",
        "/organization/list",
        "/team/available",
        "/team/metadata_schema",
        "/user/info",
        "/v2/user/info",
        "/model/info",
        "/v1/model/info",
        "/v2/model/info",
        "/v2/key/info",
        "/model_group/info",
        "/health",
        "/health/services",
        "/key/list",
        "/user/filter/ui",
        "/models",
        "/v1/models",
        "/sso/get/ui_settings",
        "/get/user_banner",
    ]

    # NOTE: ROUTES ONLY FOR MASTER KEY - only the Master Key should be able to Reset Spend
    master_key_only_routes = [
        "/global/spend/reset",
        "/memory-usage-in-mem-cache",
        "/memory-usage-in-mem-cache-items",
    ]

    key_management_routes = [
        KeyManagementRoutes.KEY_GENERATE.value,
        KeyManagementRoutes.KEY_UPDATE.value,
        KeyManagementRoutes.KEY_DELETE.value,
        KeyManagementRoutes.KEY_INFO.value,
        KeyManagementRoutes.KEY_REGENERATE.value,
        KeyManagementRoutes.KEY_GENERATE_SERVICE_ACCOUNT.value,
        KeyManagementRoutes.KEY_REGENERATE_WITH_PATH_PARAM.value,
        KeyManagementRoutes.KEY_LIST.value,
        KeyManagementRoutes.KEY_BLOCK.value,
        KeyManagementRoutes.KEY_UNBLOCK.value,
        KeyManagementRoutes.KEY_BULK_UPDATE.value,
        KeyManagementRoutes.TEAM_KEY_BULK_UPDATE.value,
        KeyManagementRoutes.TEAM_DAILY_ACTIVITY.value,
        KeyManagementRoutes.TEAM_DAILY_ACTIVITY_AGGREGATED.value,
        KeyManagementRoutes.SPEND_LOGS.value,
        KeyManagementRoutes.SPEND_LOGS_V2.value,
        KeyManagementRoutes.KEY_RESET_SPEND.value,
        KeyManagementRoutes.KEY_ALIASES.value,
        KeyManagementRoutes.KEY_ACCESS_GROUP_ASSIGNMENT.value,
    ]

    management_routes = (
        [
            # user
            "/user/new",
            "/user/update",
            "/user/bulk_update",
            "/user/delete",
            "/user/info",
            "/user/list",
            "/user/daily/activity",
            "/user/daily/activity/aggregated",
            # team
            "/team/new",
            "/team/update",
            "/team/{team_id}",
            "/team/delete",
            "/team/list",
            "/v2/team/list",
            "/team/info",
            "/team/block",
            "/team/unblock",
            "/team/available",
            "/team/metadata_schema",
            "/team/permissions_list",
            "/team/permissions_update",
            "/team/permissions_bulk_update",
            "/team/daily/activity",
            "/team/daily/activity/aggregated",
            # gateway request counts (SGR); deployment-wide, admin-only
            "/gateway/daily/activity",
            # model
            "/model/new",
            "/model/update",
            "/model/delete",
            "/model/info",
            "/jwt/key/mapping/new",
            "/jwt/key/mapping/update",
            "/jwt/key/mapping/delete",
            "/jwt/key/mapping/list",
            "/jwt/key/mapping/info",
        ]
        + key_management_routes
        + mcp_management_routes
        + list(agent_management_routes)
    )

    spend_tracking_routes = [
        # spend
        "/spend/keys",
        "/spend/users",
        "/spend/tags",
        "/spend/calculate",
        "/spend/logs",
        "/spend/logs/v2",
        "/spend/logs/ui",
        "/spend/logs/session/ui",
        "/key/spend/report",
        "/user/spend/report",
        "/team/spend/report",
        "/organization/spend/report",
        # Reads end users out of spend logs, scoped to the caller's own rows and
        # permitted teams exactly like /spend/logs/ui — it belongs to the same
        # access tier, not to customer management.
        "/management/v1/spend_logs/end_users",
        "/management/v1/spend_logs/users",
        "/cost/estimate",
    ]

    global_spend_tracking_routes = [
        # global spend
        "/global/spend/logs",
        "/global/spend",
        "/global/spend/keys",
        "/global/spend/teams",
        "/global/spend/end_users",
        "/global/spend/models",
        "/global/predict/spend/logs",
        "/global/spend/report",
        "/global/spend/provider",
        "/global/spend/tags",
        "/global/spend/all_tag_names",
    ]

    public_routes = frozenset(
        (
            "/routes",
            "/",
            "/health/liveliness",
            "/health/liveness",
            "/test",
            "/config/yaml",
            "/litellm/.well-known/litellm-ui-config",
            "/.well-known/litellm-ui-config",
            "/public/model_hub",
            "/public/model_hub/info",
            "/public/agent_hub",
            "/public/mcp_hub",
            "/public/skill_hub",
            "/public/litellm_model_cost_map",
        )
    )

    # Retained for backwards compatibility with JWT auth configs that reference
    # "ui_routes" in admin_allowed_routes. Not used by the proxy's own route
    # authorization — UI tokens now go through the same RBAC path as API tokens.
    ui_routes = [
        "/sso",
        "/sso/get/ui_settings",
        "/get/ui_settings",
        "/login",
        "/key/info",
        "/config",
        "/spend",
        "/model/info",
        "/v2/model/info",
        "/v2/key/info",
        "/models",
        "/v1/models",
        "/global/spend",
        "/global/spend/logs",
        "/global/spend/keys",
        "/global/spend/models",
        "/global/spend/tags",
        "/global/predict/spend/logs",
        "/global/activity",
        "/gateway/daily/activity",
        "/health/services",
    ] + info_routes

    # Stateless validators on caller-supplied log data; source logs are
    # already accessible via spend_tracking_routes, so no scope expansion.
    compliance_check_routes = [
        "/compliance/eu-ai-act",
        "/compliance/gdpr",
    ]

    # Routes in `global_spend_tracking_routes` return proxy-wide spend across
    # every team, customer, and api_key. They are intentionally NOT included
    # here — non-admin roles must not see other tenants' spend. Admin roles go
    # through their own branches in `route_checks.py`, and a key minted with
    # the `get_spend_routes` permission retains explicit opt-in access.
    internal_user_routes = (
        [
            "/global/activity",
            "/global/activity/model",
            "/global/activity/cache_hits",
            # Tag usage endpoints scope internal users to tags produced by
            # their own keys in tag_management_endpoints.py.
            "/tag/daily/activity",
            "/tag/list",
            "/v1/models/{model_id}",
            "/models/{model_id}",
            "/guardrails/list",
            "/v2/guardrails/list",
            "/project/list",
            "/project/info",
            # Read-only search tool routes power the Search Tools UI page.
            # Create/update/delete and test_connection stay admin-only.
            "/search_tools/list",
            "/search_tools/ui/available_providers",
        ]
        + spend_tracking_routes
        + key_management_routes
        + compliance_check_routes
    )

    internal_user_view_only_routes = (
        spend_tracking_routes
        + compliance_check_routes
        + [
            # Tag usage endpoints scope internal viewers to tags produced by
            # their own keys in tag_management_endpoints.py.
            "/tag/daily/activity",
            "/tag/list",
        ]
    )

    self_managed_routes = [
        "/team/member_add",
        "/team/member_delete",
        "/team/member_update",
        "/team/{team_id}/member/{user_id}/reset_spend",
        "/team/permissions_list",
        "/team/permissions_update",
        "/team/daily/activity",
        "/team/daily/activity/aggregated",
        "/team/{team_id}/members/me",
        "/model/new",
        "/model/update",
        "/model/delete",
        "/user/daily/activity",
        "/user/daily/activity/aggregated",
        # Endpoint restricts results to organizations the caller is ORG_ADMIN
        # of; a caller who administers none gets an empty result set.
        "/organization/daily/activity",
        "/user/available_roles",  # read-only role metadata; any authenticated user may read
        "/user/list",  # org admins checked in endpoint; non-admins get 403
        "/model/{model_id}/update",
        "/prompt/list",
        "/prompt/info",
        # Project read routes - endpoint scopes results to caller's teams (non-admin)
        "/project/list",
        "/project/info",
        # Endpoint enforces proxy-admin vs team-admin model access itself.
        "/health/test_connection",
        # Invitation routes - org/team admins checked in endpoint via _user_has_admin_privileges
        "/invitation/new",
        "/invitation/delete",
        # Team guardrail submission - requires team-scoped key; endpoint enforces team_id
        "/guardrails/register",
        # Team guardrail submissions - endpoint scopes results to caller's teams (non-admin)
        "/guardrails/submissions",
        "/guardrails/submissions/{guardrail_id}",
        # Auto-router dry runs - both gate like the /model/new write they rehearse:
        # proxy admin, or team admin naming their own team via team_id
        "/auto_router/test_routing",
        "/auto_router/validate_complexity_router_config",
        # Agent registry - reads are role-scoped and writes are proxy-admin-gated
        # inside agent_endpoints/endpoints.py
        *agent_management_routes,
    ]  # routes that manage their own allowed/disallowed logic

    ## Org Admin Routes ##

    # Routes only an Org Admin Can Access
    org_admin_only_routes = [
        "/organization/info",
        "/organization/delete",
        "/organization/member_add",
        "/organization/member_update",
        # member_delete is equally destructive as member_add / member_update
        # and must be scoped the same way — otherwise it falls through to
        # the management_routes / self_managed_routes path and lets any
        # non-PROXY_ADMIN caller that reaches the route delete arbitrary
        # org memberships without the organization_role_based_access_check
        # that member_add / member_update trigger.
        "/organization/member_delete",
    ]

    # Routes accessible by Admin Viewer (read-only admin access).
    #
    # Admin Viewer follows a read-parity-with-Proxy-Admin rule: anything Proxy
    # Admin can read/list/get, Admin Viewer can too (no writes, no cost-incurring
    # actions).
    #
    # NOTE: This list is no longer the primary mechanism for granting access —
    # `_check_proxy_admin_viewer_access()` in route_checks.py default-allows
    # any safe HTTP method (GET/HEAD/OPTIONS) on non-inference routes. This
    # list now matters only for non-GET routes that are semantically reads
    # (e.g. POST /spend/calculate). Adding a new GET endpoint does not require
    # updating this list — the default-allow behavior covers it automatically.
    admin_viewer_routes = (
        [
            "/user/list",
            "/user/available_users",
            "/user/available_roles",
            "/user/daily/activity",
            "/team/daily/activity",
            "/team/daily/activity/aggregated",
            "/tag/daily/activity",
            "/tag/list",
            "/audit",
            "/audit/{id}",
            "/global/activity",
            "/global/activity/model",
            "/global/activity/cache_hits",
            # Customer / end-user listing (handlers already gate on
            # PROXY_ADMIN_VIEW_ONLY — the route gate must match).
            "/customer/list",
            "/customer/info",
            # UI Logs page detail drawer (single + session) and the filter facets.
            # The list endpoint `/spend/logs/ui` is covered via
            # spend_tracking_routes below.
            "/spend/logs/ui/{logId}",
            "/spend/logs/session/ui",
            "/management/v1/spend_logs/end_users",
            "/management/v1/spend_logs/users",
            # Settings / observability read endpoints exposed in admin-only
            # sidebar groups (Logging & Alerts, Admin Settings, Budgets,
            # Invitations).
            "/callbacks/list",
            "/callbacks/configs",
            "/get/config/callbacks",
            "/alerting/settings",
            "/config/list",
            "/config/field/info",
            "/budget/list",
            "/management/v1/budgets",
            "/budget/settings",
            # Invitation viewing (admin viewer cannot create/delete; can read).
            "/invitation/info",
            # Guardrails / Policies pages (read-only views).
            "/guardrails/list",
            "/v2/guardrails/list",
            "/guardrails/submissions",
            "/guardrails/submissions/{guardrail_id}",
            "/guardrails/usage/overview",
            "/policies/attachments/list",
            # MCP semantic filter settings (read).
            "/get/mcp_semantic_filter_settings",
            # Model cost map maintenance views (read-only status / source).
            "/schedule/model_cost_map_reload/status",
            "/model/cost_map/source",
        ]
        # Spend tracking reads (/spend/logs, /spend/logs/ui, /spend/keys,
        # /spend/users, /spend/tags, /spend/calculate, /cost/estimate). Admin
        # Viewer can already read /global/spend/* via global_spend_tracking_routes;
        # the per-tenant /spend/* views were the missing peer.
        + spend_tracking_routes
        + info_routes
    )

    # All routes accesible by an Org Admin
    org_admin_allowed_routes = org_admin_only_routes + management_routes + self_managed_routes + admin_viewer_routes


class LiteLLMPromptInjectionParams(LiteLLMPydanticObjectBase):
    heuristics_check: bool = False
    vector_db_check: bool = False
    llm_api_check: bool = False
    llm_api_name: str | None = None
    llm_api_system_prompt: str | None = None
    llm_api_fail_call_string: str | None = None
    reject_as_response: bool | None = Field(
        default=False,
        description="Return rejected request error message as a string to the user. Default behaviour is to raise an exception.",
    )

    @model_validator(mode="before")
    @classmethod
    def check_llm_api_params(cls, values):
        llm_api_check: Final = values.get("llm_api_check")
        if llm_api_check is True:
            if "llm_api_name" not in values or not values["llm_api_name"]:
                raise ValueError("If llm_api_check is set to True, llm_api_name must be provided")
            if "llm_api_system_prompt" not in values or not values["llm_api_system_prompt"]:
                raise ValueError("If llm_api_check is set to True, llm_api_system_prompt must be provided")
            if "llm_api_fail_call_string" not in values or not values["llm_api_fail_call_string"]:
                raise ValueError("If llm_api_check is set to True, llm_api_fail_call_string must be provided")
        return values


######### Request Class Definition ######
class ProxyChatCompletionRequest(LiteLLMPydanticObjectBase):
    """
    Pydantic model for chat completion requests that includes both OpenAI standard fields
    and LiteLLM-specific parameters. This replaces the previous TypedDict version.
    """

    # Required fields (from ChatCompletionRequest)
    model: str
    messages: list[AllMessageValues]

    # Standard OpenAI completion parameters (all optional)
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    service_tier: str | None = None
    stop: str | list[str] | None = None
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    function_call: str | dict[str, Any] | None = None
    functions: list[dict[str, Any]] | None = None
    user: str | None = None
    stream: bool | None = None

    # LiteLLM-specific metadata param (from original ChatCompletionRequest)
    metadata: dict[str, Any] | None = None

    # Optional LiteLLM params
    guardrails: list[str] | None = None
    caching: bool | None = None
    num_retries: int | None = None
    context_window_fallback_dict: dict[str, str] | None = None
    fallbacks: list[str] | None = None


class ModelInfoDelete(LiteLLMPydanticObjectBase):
    id: str


class ModelInfo(LiteLLMPydanticObjectBase):
    id: str | None
    mode: Literal["embedding", "chat", "completion"] | None
    input_cost_per_token: float | None = 0.0
    output_cost_per_token: float | None = 0.0
    max_tokens: int | None = 2048  # assume 2048 if not set

    # for azure models we need users to specify the base model, one azure you can call deployments - azure/my-random-model
    # we look up the base model in model_prices_and_context_window.json
    base_model: (
        Literal[
            "gpt-4-1106-preview", "gpt-4-32k", "gpt-4", "gpt-3.5-turbo-16k", "gpt-3.5-turbo", "text-embedding-ada-002"
        ]
        | None
    )

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        if values.get("id") is None:
            values.update({"id": str(uuid.uuid4())})
        if values.get("mode") is None:
            values.update({"mode": None})
        if values.get("input_cost_per_token") is None:
            values.update({"input_cost_per_token": None})
        if values.get("output_cost_per_token") is None:
            values.update({"output_cost_per_token": None})
        if values.get("max_tokens") is None:
            values.update({"max_tokens": None})
        if values.get("base_model") is None:
            values.update({"base_model": None})
        return values


class ProviderInfo(LiteLLMPydanticObjectBase):
    name: str
    fields: list[ProviderField]


class BlockUsers(LiteLLMPydanticObjectBase):
    user_ids: list[str]  # required


class ModelParams(LiteLLMPydanticObjectBase):
    model_name: str
    litellm_params: dict
    model_info: ModelInfo

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        if values.get("model_info") is None:
            values.update({"model_info": ModelInfo(id=None, mode="chat", base_model=None)})
        return values


class LiteLLM_ObjectPermissionBase(LiteLLMPydanticObjectBase):
    mcp_servers: list[str] | None = None
    mcp_access_groups: list[str] | None = None
    mcp_tool_permissions: dict[str, list[str]] | None = None
    mcp_toolsets: list[str] | None = None
    blocked_tools: list[str] | None = None
    vector_stores: list[str] | None = None
    agents: list[str] | None = None
    agent_access_groups: list[str] | None = None
    models: list[str] | None = None
    search_tools: list[str] | None = None
    mcp_tool_search_enabled: bool | None = None


from litellm.models.team import BudgetLimitEntry as BudgetLimitEntry  # noqa: E402
from litellm.types.object_permission import (  # noqa: E402
    ObjectPermissionDict as ObjectPermissionDict,
)


class GenerateRequestBase(LiteLLMPydanticObjectBase):
    """
    Overlapping schema between key and user generate/update requests
    """

    key_alias: str | None = None
    duration: str | None = None
    models: list | None = []
    spend: float | None = 0
    max_budget: float | None = None
    user_id: str | None = None
    team_id: str | None = None
    agent_id: str | None = None
    max_parallel_requests: int | None = None
    metadata: dict | None = {}
    tpm_limit: int | None = None
    rpm_limit: int | None = None

    budget_duration: str | None = None
    budget_limits: list[BudgetLimitEntry] | None = None  # multiple concurrent budget windows
    allowed_cache_controls: list | None = []
    config: dict | None = {}
    permissions: dict | None = {}
    model_max_budget: dict | None = {}  # {"gpt-4": 5.0, "gpt-3.5-turbo": 5.0}, defaults to {}
    budget_fallbacks: dict[str, list[str]] | None = None

    model_config = ConfigDict(protected_namespaces=())
    model_rpm_limit: dict | None = None
    model_tpm_limit: dict | None = None
    mcp_rpm_limit: dict[str, int] | None = None
    tag_rpm_limit: dict[str, int] | None = None
    guardrails: list[str] | None = None
    policies: list[str] | None = None
    prompts: list[str] | None = None
    blocked: bool | None = None
    aliases: dict | None = {}
    object_permission: LiteLLM_ObjectPermissionBase | None = None

    @field_validator("max_budget", mode="before")
    @classmethod
    def check_max_budget(cls, v):
        if v == "":
            return None
        return v


class AllowedVectorStoreIndexItem(LiteLLMPydanticObjectBase):
    index_name: str
    index_permissions: list[Literal["read", "write"]]


class KeyRequestBase(GenerateRequestBase):
    key: str | None = None
    default_estimated_output_tokens: PositiveInt | None = None
    default_estimated_output_tokens_per_model: Mapping[str, PositiveInt] | None = None
    budget_id: str | None = None
    tags: list[str] | None = None
    disable_global_guardrails: bool | None = None
    enable_prompt_caching: bool | None = None
    throttle_on_budget_exceeded: bool | None = None
    enforced_params: list[str] | None = None
    allowed_routes: list | None = []
    allowed_passthrough_routes: list | None = None
    allowed_vector_store_indexes: list[AllowedVectorStoreIndexItem] | None = None
    rpm_limit_type: Literal["guaranteed_throughput", "best_effort_throughput", "dynamic"] | None = (
        None  # raise an error if 'guaranteed_throughput' is set and we're overallocating rpm
    )
    tpm_limit_type: Literal["guaranteed_throughput", "best_effort_throughput", "dynamic"] | None = (
        None  # raise an error if 'guaranteed_throughput' is set and we're overallocating tpm
    )
    router_settings: UpdateRouterConfig | None = None
    access_group_ids: list[str] | None = None


class LiteLLMKeyType(str, enum.Enum):
    """
    Enum for key types that determine what routes a key can access
    """

    LLM_API = "llm_api"  # Can call LLM API routes (chat/completions, embeddings, etc.)
    MANAGEMENT = "management"  # Can call management routes (user/team/key management)
    READ_ONLY = "read_only"  # Can only call info/read routes
    DEFAULT = "default"  # Uses default allowed routes


class GenerateKeyRequest(KeyRequestBase):
    soft_budget: float | None = None
    send_invite_email: bool | None = None
    key_type: LiteLLMKeyType | None = Field(
        default=LiteLLMKeyType.DEFAULT,
        description="Type of key that determines default allowed routes.",
    )
    auto_rotate: bool | None = Field(default=False, description="Whether this key should be automatically rotated")
    rotation_interval: str | None = Field(
        default=None,
        description="How often to rotate this key (e.g., '30d', '90d'). Required if auto_rotate=True",
    )
    organization_id: str | None = None
    project_id: str | None = None


class GenerateKeyResponse(KeyRequestBase):
    key: str
    key_name: str | None = None
    key_type: str | None = None
    expires: datetime | None = None
    user_id: str | None = None
    token_id: str | None = None
    organization_id: str | None = None
    project_id: str | None = None
    litellm_budget_table: Any | None = None
    token: str | None = None
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        if values.get("token") is not None:
            values.update({"key": values.get("token")})
        dict_fields: Final = [
            "metadata",
            "aliases",
            "config",
            "permissions",
            "model_max_budget",
            "budget_fallbacks",
            "router_settings",
            "budget_limits",
        ]
        for field in dict_fields:
            value = values.get(field)
            if value is not None and isinstance(value, str):
                try:
                    values[field] = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError(f"Field {field} should be a valid dictionary")

        return values


class UpdateKeyRequest(KeyRequestBase):
    # Note: the defaults of all Params here MUST BE NONE
    # else they will get overwritten
    duration: str | None = None
    spend: float | None = None
    metadata: dict | None = None
    temp_budget_increase: float | None = None
    temp_budget_expiry: datetime | None = None
    auto_rotate: bool | None = None
    rotation_interval: str | None = None
    organization_id: str | None = None

    @model_validator(mode="after")
    def validate_temp_budget(self) -> "UpdateKeyRequest":
        if self.temp_budget_increase is not None or self.temp_budget_expiry is not None:
            if self.temp_budget_increase is None or self.temp_budget_expiry is None:
                raise ValueError("temp_budget_increase and temp_budget_expiry must be set together")
        return self

    @model_validator(mode="after")
    def validate_key_identifier(self) -> "UpdateKeyRequest":
        if self.key is None and self.key_alias is None:
            raise ValueError("either key or key_alias must be provided")
        return self


class RegenerateKeyRequest(GenerateKeyRequest):
    # This needs to be different from UpdateKeyRequest, because "key" is optional for this
    key: str | None = None
    new_key: str | None = None
    duration: str | None = None
    spend: float | None = None
    metadata: dict | None = None
    new_master_key: str | None = None
    grace_period: str | None = None  # Duration to keep old key valid (e.g. "24h", "2d"); None = immediate revoke


class ResetSpendRequest(LiteLLMPydanticObjectBase):
    reset_to: float

    @field_validator("reset_to", mode="before")
    @classmethod
    def reject_bool_reset_to(cls, v):
        # bool is a subclass of int, so pydantic silently coerces True/False into
        # 1.0/0.0 for a `float` field: a caller who accidentally sends a boolean
        # would otherwise get an unintended spend reset instead of a 422.
        if isinstance(v, bool):
            raise ValueError("reset_to must be a number, not a boolean")  # noqa: TRY004  # pydantic needs ValueError
        return v


class KeyRequest(LiteLLMPydanticObjectBase):
    keys: list[str] | None = None
    key_aliases: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_at_least_one(cls, values):
        if not values.get("keys") and not values.get("key_aliases"):
            raise ValueError("At least one of 'keys' or 'key_aliases' must be provided.")
        return values


from litellm.models.model import (  # noqa: E402
    LiteLLM_ProxyModelTable as LiteLLM_ProxyModelTable,
)
from litellm.models.team import LiteLLM_ModelTable as LiteLLM_ModelTable  # noqa: E402


# MCP Types
class SpecialMCPServerName(str, enum.Enum):
    all_team_servers = "all-team-mcpservers"
    all_proxy_servers = "all-proxy-mcpservers"


class MCPApprovalStatus(str, enum.Enum):
    pending_review = "pending_review"
    active = "active"
    rejected = "rejected"
    # Short-lived row backing the admin OAuth "Authorize & Fetch Token" flow. Never served: the
    # registry loader and every listing exclude it, so it is reachable only by its own server_id.
    draft = "draft"


from litellm.models.mcp_server import (  # noqa: E402
    MCPEnvVar as MCPEnvVar,
)
from litellm.models.mcp_server import (  # noqa: E402
    MCPEnvVarScope as MCPEnvVarScope,
)


# MCP Proxy Request Types
def _dcr_bridge_auth_type_error(auth_type: object) -> ValueError:
    return ValueError(
        f"dcr_bridge is only supported for auth_type true_passthrough or oauth_delegate (got {auth_type!r}). "
        "The DCR bridge serves gateway-hosted OAuth discovery for the client-forwarded token modes; "
        "interactive oauth2 servers already run the gateway authorization-code flow."
    )


class NewMCPServerRequest(LiteLLMPydanticObjectBase):
    server_id: str | None = None
    server_name: str | None = None
    alias: str | None = None
    description: str | None = None
    transport: MCPTransportType = MCPTransport.sse
    auth_type: MCPAuthType | None = None
    credentials: MCPCredentials | None = None
    url: str | None = None
    spec_path: str | None = None
    mcp_info: MCPInfo | None = None
    mcp_access_groups: list[str] = Field(default_factory=list)
    allowed_tools: list[str] | None = None
    tool_name_to_display_name: dict[str, str] | None = None
    tool_name_to_description: dict[str, str] | None = None
    extra_headers: list[str] | None = None
    static_headers: dict[str, str] | None = None
    env_vars: list[MCPEnvVar] | None = None
    instructions: str | None = None
    # Stdio-specific fields
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    issuer: str | None = None
    authorization_url: str | None = None
    token_url: str | None = None
    registration_url: str | None = None
    oauth2_flow: Literal["client_credentials", "authorization_code"] | None = None
    # Token Exchange (OBO) fields — RFC 8693. These top-level fields are the
    # canonical shape; the same keys inside ``credentials`` are the legacy
    # pre-column REST shape and are lifted into these columns on write (an
    # explicit top-level value wins) and stripped from the stored blob.
    token_exchange_endpoint: str | None = None
    audience: str | None = None
    subject_token_type: str | None = None
    token_exchange_profile: str | None = None
    allow_all_keys: bool = False
    available_on_public_internet: bool = True
    delegate_auth_to_upstream: bool = False
    oauth_passthrough: bool = False
    dcr_bridge: bool | None = None
    is_byok: bool = False
    byok_description: list[str] = Field(default_factory=list)
    byok_api_key_help_url: str | None = None
    source_url: str | None = None
    timeout: float | None = None
    max_concurrent_requests: int | None = None
    # BYOM submission fields — set by the endpoint, not by the caller.
    # Any caller-provided values are silently overridden before persistence.
    approval_status: str | None = Field(
        None,
        description="Server-managed: set by the endpoint; caller values are overridden.",
    )
    submitted_by: str | None = Field(
        None,
        description="Server-managed: set by the endpoint; caller values are overridden.",
    )
    submitted_at: datetime | None = Field(
        None,
        description="Server-managed: set by the endpoint; caller values are overridden.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_transport_fields(cls, values):
        if isinstance(values, dict):
            transport: Final = values.get("transport")
            if transport == MCPTransport.stdio:
                if not values.get("command"):
                    raise ValueError("command is required for stdio transport")
                if not values.get("args"):
                    raise ValueError("args is required for stdio transport")
                # Validate command against allowlist to prevent arbitrary execution
                base_command: Final = os.path.basename(values["command"])
                if base_command not in MCP_STDIO_ALLOWED_COMMANDS:
                    raise ValueError(
                        f"Command '{values['command']}' is not in the allowed commands list "
                        f"for stdio transport. Allowed commands: {sorted(MCP_STDIO_ALLOWED_COMMANDS)}"
                    )
            elif transport in [MCPTransport.http, MCPTransport.sse]:
                if not values.get("url") and not values.get("spec_path"):
                    raise ValueError("url or spec_path is required for HTTP/SSE transport")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_credentials_requirements(cls, values):
        """Validate credentials when provided.

        auth_value is optional — users may configure it dynamically
        (e.g. via per-request headers or OAuth2 flows) instead of
        storing a static value at server creation time.
        """
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_dcr_bridge_auth_type(cls, values):
        if not isinstance(values, dict) or not values.get("dcr_bridge"):
            return values
        auth_type: Final = values.get("auth_type")
        if auth_type in (MCPAuth.true_passthrough, MCPAuth.oauth_delegate):
            return values
        raise _dcr_bridge_auth_type_error(auth_type)


class UpdateMCPServerRequest(LiteLLMPydanticObjectBase):
    server_id: str
    server_name: str | None = None
    alias: str | None = None
    description: str | None = None
    transport: MCPTransportType = MCPTransport.sse
    auth_type: MCPAuthType | None = None
    credentials: MCPCredentials | None = None
    url: str | None = None
    spec_path: str | None = None
    mcp_info: MCPInfo | None = None
    mcp_access_groups: list[str] = Field(default_factory=list)
    allowed_tools: list[str] | None = None
    tool_name_to_display_name: dict[str, str] | None = None
    tool_name_to_description: dict[str, str] | None = None
    extra_headers: list[str] | None = None
    static_headers: dict[str, str] | None = None
    env_vars: list[MCPEnvVar] | None = None
    instructions: str | None = None
    # Stdio-specific fields
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    issuer: str | None = None
    authorization_url: str | None = None
    token_url: str | None = None
    registration_url: str | None = None
    oauth2_flow: Literal["client_credentials", "authorization_code"] | None = None
    # Token Exchange (OBO) fields — RFC 8693. These top-level fields are the
    # canonical shape; the same keys inside ``credentials`` are the legacy
    # pre-column REST shape and are lifted into these columns on write (an
    # explicit top-level value wins) and stripped from the stored blob.
    token_exchange_endpoint: str | None = None
    audience: str | None = None
    subject_token_type: str | None = None
    token_exchange_profile: str | None = None
    allow_all_keys: bool = False
    available_on_public_internet: bool = True
    delegate_auth_to_upstream: bool = False
    oauth_passthrough: bool = False
    dcr_bridge: bool | None = None
    is_byok: bool = False
    byok_description: list[str] = Field(default_factory=list)
    byok_api_key_help_url: str | None = None
    source_url: str | None = None
    timeout: float | None = None
    max_concurrent_requests: int | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_transport_fields(cls, values):
        if isinstance(values, dict):
            transport: Final = values.get("transport")
            if transport == MCPTransport.stdio:
                if not values.get("command"):
                    raise ValueError("command is required for stdio transport")
                if not values.get("args"):
                    raise ValueError("args is required for stdio transport")
                # Validate command against allowlist to prevent arbitrary execution
                base_command: Final = os.path.basename(values["command"])
                if base_command not in MCP_STDIO_ALLOWED_COMMANDS:
                    raise ValueError(
                        f"Command '{values['command']}' is not in the allowed commands list "
                        f"for stdio transport. Allowed commands: {sorted(MCP_STDIO_ALLOWED_COMMANDS)}"
                    )
            elif transport in [MCPTransport.http, MCPTransport.sse]:
                if not values.get("url") and not values.get("spec_path"):
                    raise ValueError("url or spec_path is required for HTTP/SSE transport")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_dcr_bridge_auth_type(cls, values):
        """Partial updates omit auth_type; that case is validated against the stored row by the
        update endpoint, which can read the database. This validator covers payloads that carry
        both fields."""
        if not isinstance(values, dict) or not values.get("dcr_bridge"):
            return values
        if "auth_type" not in values:
            return values
        auth_type: Final = values.get("auth_type")
        if auth_type in (MCPAuth.true_passthrough, MCPAuth.oauth_delegate):
            return values
        raise _dcr_bridge_auth_type_error(auth_type)


from litellm.models.mcp_server import (  # noqa: E402
    LiteLLM_MCPServerTable as LiteLLM_MCPServerTable,
)


class MakeMCPServersPublicRequest(LiteLLMPydanticObjectBase):
    mcp_server_ids: list[str]


class MCPUserCredentialRequest(LiteLLMPydanticObjectBase):
    credential: str
    save: bool = True


class MCPUserCredentialResponse(LiteLLMPydanticObjectBase):
    server_id: str
    has_credential: bool


class MCPOAuthUserCredentialRequest(LiteLLMPydanticObjectBase):
    """Stores a user's OAuth2 token for an OpenAPI MCP server."""

    access_token: str
    refresh_token: str | None = None
    expires_in: int | None = None  # seconds until expiry
    scopes: list[str] | None = None


class MCPOAuthUserCredentialStatus(LiteLLMPydanticObjectBase):
    """Describes whether the calling user has a stored OAuth credential."""

    server_id: str
    has_credential: bool
    expires_at: str | None = None  # ISO-8601
    is_expired: bool = False
    connected_at: str | None = None  # ISO-8601


class MCPUserCredentialListItem(LiteLLMPydanticObjectBase):
    """One entry in the /user-credentials list."""

    server_id: str
    server_name: str | None = None
    alias: str | None = None
    credential_type: str  # "oauth2" or "byok"
    has_credential: bool
    expires_at: str | None = None  # ISO-8601; None means non-expiring
    connected_at: str | None = None  # ISO-8601


class MCPUserEnvVarsRequest(LiteLLMPydanticObjectBase):
    """Payload for storing the calling user's per-user env var values."""

    values: dict[str, str]


class MCPUserEnvVarSpec(LiteLLMPydanticObjectBase):
    """Describes one per-user env var slot for the calling user.

    Stored values are write-only: the status only reports whether a value
    ``is_set`` and never echoes the decrypted secret back to the client.
    """

    name: str
    description: str | None = None
    is_set: bool = False


class MCPUserEnvVarsStatus(LiteLLMPydanticObjectBase):
    """Per-user env var status for a single MCP server."""

    server_id: str
    server_name: str | None = None
    alias: str | None = None
    required: list[MCPUserEnvVarSpec] = Field(default_factory=list)
    missing_count: int = 0
    setup_url: str | None = None  # frontend URL where the user can fill these in


class RejectMCPServerRequest(LiteLLMPydanticObjectBase):
    review_notes: str | None = None


class MCPSubmissionsSummary(LiteLLMPydanticObjectBase):
    total: int
    pending_review: int
    active: int
    rejected: int
    items: list["LiteLLM_MCPServerTable"]


######## Skills API Types ########


class NewSkillRequest(LiteLLMPydanticObjectBase):
    """Request to create a new skill in LiteLLM database"""

    display_title: str | None = None
    description: str | None = None
    instructions: str | None = None
    file_content: bytes | None = None  # Binary content of skill files (zip)
    file_name: str | None = None  # Original filename
    file_type: str | None = None  # MIME type (e.g., "application/zip")
    metadata: dict[str, Any] | None = None
    authorization_url: str | None = None
    token_url: str | None = None
    registration_url: str | None = None


class UpdateSkillRequest(LiteLLMPydanticObjectBase):
    """Request to update an existing skill"""

    skill_id: str
    display_title: str | None = None
    description: str | None = None
    instructions: str | None = None
    file_content: bytes | None = None  # Binary content of skill files (zip)
    file_name: str | None = None  # Original filename
    file_type: str | None = None  # MIME type
    metadata: dict[str, Any] | None = None


from litellm.models.skills import (  # noqa: E402
    LiteLLM_SkillsTable as LiteLLM_SkillsTable,
)


class ListSkillsRequest(LiteLLMPydanticObjectBase):
    """Request to list skills from LiteLLM database"""

    limit: int | None = 20
    offset: int | None = 0


class NewUserRequestTeam(LiteLLMPydanticObjectBase):
    team_id: str
    max_budget_in_team: float | None = None
    user_role: Literal["user", "admin"] = "user"


class NewUserRequest(GenerateRequestBase):
    max_budget: float | None = None
    user_email: str | None = None
    user_alias: str | None = None
    user_role: (
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
        | None
    ) = None
    teams: list[str] | list[NewUserRequestTeam] | None = None
    auto_create_key: bool = True  # flag used for returning a key as part of the /user/new response
    send_invite_email: bool | None = None
    sso_user_id: str | None = None
    organizations: list[str] | None = None


class NewUserResponse(GenerateKeyResponse):
    max_budget: float | None = None
    user_email: str | None = None
    user_role: (
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
        | None
    ) = None
    teams: list | None = None
    user_alias: str | None = None
    model_max_budget: dict | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class UpdateUserRequestNoUserIDorEmail(GenerateRequestBase):  # shared with BulkUpdateUserRequest
    password: str | None = None
    spend: float | None = None
    metadata: dict | None = None
    user_alias: str | None = None
    user_role: (
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
        | None
    ) = None
    max_budget: float | None = None


class UpdateUserRequest(UpdateUserRequestNoUserIDorEmail):
    # Note: the defaults of all Params here MUST BE NONE
    # else they will get overwritten
    user_id: str | None = None
    user_email: str | None = None

    @model_validator(mode="before")
    @classmethod
    def check_user_info(cls, values):
        if values.get("user_id") is None and values.get("user_email") is None:
            raise ValueError("Either user id or user email must be provided")
        return values


class DeleteUserRequest(LiteLLMPydanticObjectBase):
    user_ids: list[str]  # required


AllowedModelRegion = Literal["eu", "us"]


class BudgetNewRequest(LiteLLMPydanticObjectBase):
    budget_id: str | None = Field(default=None, description="The unique budget id.")
    max_budget: float | None = Field(
        default=None,
        description="Requests will fail if this budget (in USD) is exceeded.",
    )
    soft_budget: float | None = Field(
        default=None,
        description="Requests will NOT fail if this is exceeded. Will fire alerting though.",
    )
    max_parallel_requests: int | None = Field(
        default=None, description="Max concurrent requests allowed for this budget id."
    )
    tpm_limit: int | None = Field(default=None, description="Max tokens per minute, allowed for this budget id.")
    rpm_limit: int | None = Field(default=None, description="Max requests per minute, allowed for this budget id.")
    budget_duration: str | None = Field(
        default=None,
        description="Max duration budget should be set for (e.g. '1hr', '1d', '28d')",
    )
    model_max_budget: GenericBudgetConfigType | None = Field(
        default=None,
        description="Max budget for each model (e.g. {'gpt-4o': {'max_budget': '0.0000001', 'budget_duration': '1d', 'tpm_limit': 1000, 'rpm_limit': 1000}})",
    )
    budget_reset_at: datetime | None = Field(
        default=None,
        description="Datetime when the budget is reset",
    )


class BudgetRequest(LiteLLMPydanticObjectBase):
    budgets: list[str]


class BudgetDeleteRequest(LiteLLMPydanticObjectBase):
    id: str


class CustomerBase(LiteLLMPydanticObjectBase):
    user_id: str
    alias: str | None = None
    spend: float = 0.0
    allowed_model_region: AllowedModelRegion | None = None
    default_model: str | None = None
    budget_id: str | None = None
    litellm_budget_table: BudgetNewRequest | None = None
    blocked: bool = False


class NewCustomerRequest(BudgetNewRequest):
    """
    Create a new customer, allocate a budget to them
    """

    user_id: str
    alias: str | None = None  # human-friendly alias
    blocked: bool = False  # allow/disallow requests for this end-user
    budget_id: str | None = None  # give either a budget_id or max_budget
    spend: float | None = None
    allowed_model_region: AllowedModelRegion | None = (
        None  # require all user requests to use models in this specific region
    )
    default_model: str | None = None  # if no equivalent model in allowed region - default all requests to this model
    object_permission: LiteLLM_ObjectPermissionBase | None = None

    @model_validator(mode="before")
    @classmethod
    def check_user_info(cls, values):
        if values.get("max_budget") is not None and values.get("budget_id") is not None:
            raise ValueError("Set either 'max_budget' or 'budget_id', not both.")

        return values


class UpdateCustomerRequest(LiteLLMPydanticObjectBase):
    """
    Update a Customer, use this to update customer budgets etc

    """

    user_id: str
    alias: str | None = None  # human-friendly alias
    blocked: bool = False  # allow/disallow requests for this end-user
    max_budget: float | None = None
    budget_id: str | None = None  # give either a budget_id or max_budget
    allowed_model_region: AllowedModelRegion | None = (
        None  # require all user requests to use models in this specific region
    )
    default_model: str | None = None  # if no equivalent model in allowed region - default all requests to this model
    object_permission: LiteLLM_ObjectPermissionBase | None = None


class DeleteCustomerRequest(LiteLLMPydanticObjectBase):
    """
    Delete multiple Customers
    """

    user_ids: list[str]


from litellm.models.team import Member as Member  # noqa: E402
from litellm.models.team import MemberBase as MemberBase  # noqa: E402


class OrgMember(MemberBase):
    role: Literal[
        LitellmUserRoles.ORG_ADMIN,
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
    ]


from litellm.models.team import TeamBase as TeamBase  # noqa: E402


class NewTeamRequest(TeamBase):
    model_aliases: dict | None = None
    tags: list | None = None
    guardrails: list[str] | None = None
    policies: list[str] | None = None
    prompts: list[str] | None = None
    object_permission: LiteLLM_ObjectPermissionBase | None = None
    allowed_passthrough_routes: list | None = None
    disable_global_guardrails: bool | None = None
    secret_manager_settings: dict | None = None
    model_rpm_limit: dict[str, int] | None = None
    rpm_limit_type: Literal["guaranteed_throughput", "best_effort_throughput"] | None = (
        None  # raise an error if 'guaranteed_throughput' is set and we're overallocating rpm
    )
    tpm_limit_type: Literal["guaranteed_throughput", "best_effort_throughput"] | None = (
        None  # raise an error if 'guaranteed_throughput' is set and we're overallocating tpm
    )

    model_tpm_limit: dict[str, int] | None = None
    default_estimated_output_tokens: PositiveInt | None = None
    default_estimated_output_tokens_per_model: Mapping[str, PositiveInt] | None = None
    mcp_rpm_limit: dict[str, int] | None = None
    team_member_budget: float | None = None  # allow user to set a budget for all team members
    team_member_rpm_limit: int | None = None  # allow user to set RPM limit for all team members
    team_member_tpm_limit: int | None = None  # allow user to set TPM limit for all team members
    team_member_key_duration: str | None = None  # e.g. "1d", "1w", "1m"
    team_member_budget_duration: str | None = None  # e.g. "30d", "1mo"
    allowed_vector_store_indexes: list[AllowedVectorStoreIndexItem] | None = None
    enforced_batch_output_expires_after: dict | None = None
    enforced_file_expires_after: dict | None = None

    model_config = ConfigDict(protected_namespaces=())


class GlobalEndUsersSpend(LiteLLMPydanticObjectBase):
    api_key: str | None = None
    startTime: datetime | None = None
    endTime: datetime | None = None


class UpdateTeamRequest(LiteLLMPydanticObjectBase):
    """
    UpdateTeamRequest, used by /team/update when you need to update a team

    team_id: str
    team_alias: Optional[str] = None
    organization_id: Optional[str] = None
    metadata: Optional[dict] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    max_budget: Optional[float] = None
    models: Optional[list] = None
    blocked: Optional[bool] = None
    budget_duration: Optional[str] = None
    guardrails: Optional[List[str]] = None
    policies: Optional[List[str]] = None
    """

    team_id: str  # required
    team_alias: str | None = None
    organization_id: str | None = None
    metadata: dict | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    max_budget: float | None = None
    soft_budget: float | None = None
    models: list | None = None
    blocked: bool | None = None
    budget_duration: str | None = None
    tags: list | None = None
    model_aliases: dict | None = None
    guardrails: list[str] | None = None
    policies: list[str] | None = None
    object_permission: LiteLLM_ObjectPermissionBase | None = None
    disable_global_guardrails: bool | None = None
    team_member_budget: float | None = None
    team_member_budget_duration: str | None = None
    team_member_rpm_limit: int | None = None
    team_member_tpm_limit: int | None = None
    team_member_key_duration: str | None = None
    allowed_passthrough_routes: list | None = None
    secret_manager_settings: dict | None = None
    prompts: list[str] | None = None
    model_rpm_limit: dict[str, int] | None = None
    model_tpm_limit: dict[str, int] | None = None
    default_estimated_output_tokens: PositiveInt | None = None
    default_estimated_output_tokens_per_model: Mapping[str, PositiveInt] | None = None
    mcp_rpm_limit: dict[str, int] | None = None
    allowed_vector_store_indexes: list[AllowedVectorStoreIndexItem] | None = None
    enforced_batch_output_expires_after: dict | None = None
    enforced_file_expires_after: dict | None = None
    router_settings: dict | None = None
    access_group_ids: list[str] | None = None
    budget_limits: list[BudgetLimitEntry] | None = None  # multiple concurrent budget windows
    default_team_member_models: list[str] | None = None  # default allowed_models seeded onto new team members


class PatchTeamRequest(UpdateTeamRequest):
    """
    Body of PATCH /team/{team_id}.

    Identical to UpdateTeamRequest except team_id is optional, because PATCH takes it
    from the path. A team_id in the body is still accepted when it matches the path.
    """

    team_id: str | None = None


class ResetTeamBudgetRequest(LiteLLMPydanticObjectBase):
    """
    internal type used to reset the budget on a team
    used by reset_budget()

    team_id: str
    spend: float
    budget_reset_at: datetime
    """

    team_id: str
    spend: float
    budget_reset_at: datetime
    updated_at: datetime


class DeleteTeamRequest(LiteLLMPydanticObjectBase):
    team_ids: list[str]  # required


class BlockTeamRequest(LiteLLMPydanticObjectBase):
    team_id: str  # required


class BlockKeyRequest(LiteLLMPydanticObjectBase):
    key: str  # required


class BlockModelRequest(LiteLLMPydanticObjectBase):
    model_id: str  # required


class AddTeamCallback(LiteLLMPydanticObjectBase):
    callback_name: str
    callback_type: Literal["success", "failure", "success_and_failure"] | None = "success_and_failure"
    callback_vars: dict[str, str]

    @model_validator(mode="before")
    @classmethod
    def validate_callback_vars(cls, values):
        callback_vars: Final = values.get("callback_vars", {})
        valid_keys: Final = set(StandardCallbackDynamicParams.__annotations__.keys())
        for key, value in callback_vars.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid callback variable: {key}. Must be one of {valid_keys}")
            callback_vars[key] = str(value)
            validate_no_callback_env_reference(key, callback_vars[key], source="key/team callback metadata")
            if key == "langfuse_environment":
                validate_langfuse_environment_value(callback_vars[key])
        return values


class TeamCallbackDeleteResponseData(LiteLLMPydanticObjectBase):
    team_id: str
    success_callbacks: tuple[str, ...]
    failure_callbacks: tuple[str, ...]


class TeamCallbackDeleteResponse(LiteLLMPydanticObjectBase):
    status: Literal["success"]
    message: str
    data: TeamCallbackDeleteResponseData


class TeamCallbackMetadata(LiteLLMPydanticObjectBase):
    success_callback: list[str] | None = []
    failure_callback: list[str] | None = []
    callbacks: list[str] | None = []
    # for now - only supported for langfuse
    callback_vars: dict[str, str] | None = {}

    @model_validator(mode="before")
    @classmethod
    def validate_callback_vars(cls, values):
        success_callback: Final = values.get("success_callback", [])
        if success_callback is None:
            values.pop("success_callback", None)
        failure_callback: Final = values.get("failure_callback", [])
        if failure_callback is None:
            values.pop("failure_callback", None)
        callbacks: Final = values.get("callbacks", [])
        if callbacks is None:
            values.pop("callbacks", None)

        callback_vars: Final = values.get("callback_vars", {})
        if callback_vars is None:
            values.pop("callback_vars", None)
        if all(val is None for val in values.values()):
            return {
                "success_callback": [],
                "failure_callback": [],
                "callbacks": [],
                "callback_vars": {},
            }
        valid_keys: Final = set(StandardCallbackDynamicParams.__annotations__.keys())
        if callback_vars is not None:
            for key in callback_vars:
                if key not in valid_keys:
                    raise ValueError(f"Invalid callback variable: {key}. Must be one of {valid_keys}")
        return values


from litellm.models.object_permission import (  # noqa: E402
    LiteLLM_ObjectPermissionTable as LiteLLM_ObjectPermissionTable,
)
from litellm.models.team import (  # noqa: E402
    LiteLLM_DeletedTeamTable as LiteLLM_DeletedTeamTable,
)
from litellm.models.team import LiteLLM_TeamTable as LiteLLM_TeamTable  # noqa: E402
from litellm.models.team import (  # noqa: E402
    LiteLLM_TeamTableCachedObj as LiteLLM_TeamTableCachedObj,
)


class TeamRequest(LiteLLMPydanticObjectBase):
    teams: list[str]


from litellm.models.budget import (  # noqa: E402
    LiteLLM_BudgetTable as LiteLLM_BudgetTable,
)
from litellm.models.budget import (  # noqa: E402
    LiteLLM_BudgetTableFull as LiteLLM_BudgetTableFull,
)
from litellm.models.budget import (  # noqa: E402
    LiteLLM_TeamMemberTable as LiteLLM_TeamMemberTable,
)


class NewOrganizationRequest(LiteLLM_BudgetTable):
    organization_id: str | None = None
    organization_alias: str
    models: list = []
    budget_id: str | None = None
    metadata: dict | None = None
    model_rpm_limit: dict[str, int] | None = None
    model_tpm_limit: dict[str, int] | None = None

    #########################################################
    # Object Permission - MCP, Vector Stores etc.
    #########################################################
    object_permission: LiteLLM_ObjectPermissionBase | None = None


class OrganizationRequest(LiteLLMPydanticObjectBase):
    organizations: list[str]


class DeleteOrganizationRequest(LiteLLMPydanticObjectBase):
    organization_ids: list[str]  # required


class TeamDefaultSettings(LiteLLMPydanticObjectBase):
    team_id: str

    model_config = ConfigDict(
        extra="allow"
    )  # allow params not defined here, these fall in litellm.completion(**kwargs)


class DynamoDBArgs(LiteLLMPydanticObjectBase):
    billing_mode: Literal["PROVISIONED_THROUGHPUT", "PAY_PER_REQUEST"]
    read_capacity_units: int | None = None
    write_capacity_units: int | None = None
    ssl_verify: bool | None = None
    region_name: str
    user_table_name: str = "LiteLLM_UserTable"
    key_table_name: str = "LiteLLM_VerificationToken"
    config_table_name: str = "LiteLLM_Config"
    spend_table_name: str = "LiteLLM_SpendLogs"
    aws_role_name: str | None = None
    aws_session_name: str | None = None
    aws_web_identity_token: str | None = None
    aws_provider_id: str | None = None
    aws_policy_arns: list[str] | None = None
    aws_policy: str | None = None
    aws_duration_seconds: int | None = None
    assume_role_aws_role_name: str | None = None
    assume_role_aws_session_name: str | None = None


class PassThroughGuardrailSettings(LiteLLMPydanticObjectBase):
    """
    Settings for a specific guardrail on a passthrough endpoint.

    Allows field-level targeting for guardrail execution.
    """

    request_fields: list[str] | None = Field(
        default=None,
        description="JSONPath expressions for input field targeting (pre_call). Examples: 'query', 'documents[*].text', 'messages[*].content'. If not specified, guardrail runs on entire request payload.",
    )
    response_fields: list[str] | None = Field(
        default=None,
        description="JSONPath expressions for output field targeting (post_call). Examples: 'results[*].text', 'output'. If not specified, guardrail runs on entire response payload.",
    )


# Type alias for the guardrails dict: guardrail_name -> settings (or None for defaults)
PassThroughGuardrailsConfig = dict[str, PassThroughGuardrailSettings | None]


class PassThroughGenericEndpoint(LiteLLMPydanticObjectBase):
    id: str | None = Field(
        default=None,
        description="Optional unique identifier for the pass-through endpoint. If not provided, endpoints will be identified by path for backwards compatibility.",
    )
    path: str = Field(description="The route to be added to the LiteLLM Proxy Server.")
    target: str = Field(description="The URL to which requests for this path should be forwarded.")
    headers: dict = Field(
        default={},
        description="Key-value pairs of headers to be forwarded with the request. You can set any key value pair here and it will be forwarded to your target endpoint",
    )
    default_query_params: dict = Field(
        default={},
        description="Key-value pairs of default query parameters to be sent with every request to this endpoint. These can be overridden by client-provided query parameters. For example: {'key': 'default_value', 'api_version': '2023-01'}",
    )
    include_subpath: bool = Field(
        default=False,
        description="If True, requests to subpaths of the path will be forwarded to the target endpoint. For example, if the path is /bria and include_subpath is True, requests to /bria/v1/text-to-image/base/2.3 will be forwarded to the target endpoint.",
    )
    cost_per_request: float = Field(
        default=0.0,
        description="The USD cost per request to the target endpoint. This is used to calculate the cost of the request to the target endpoint.",
    )
    timeout: float | None = Field(
        default=None,
        description="Upstream request timeout in seconds for this pass-through endpoint. If unset, uses general_settings.pass_through_request_timeout (default 600).",
    )
    auth: bool = Field(
        default=True,
        description="Whether authentication is required for the pass-through endpoint. Defaults to True so a pass-through silently created without an explicit value still requires a valid LiteLLM API key — set to False only if the endpoint is meant to be a public forwarder (e.g. an unauthenticated webhook target).",
    )
    guardrails: PassThroughGuardrailsConfig | None = Field(
        default=None,
        description="Guardrails configuration for this passthrough endpoint. Dict keys are guardrail names, values are optional settings for field targeting. When set, all org/team/key level guardrails will also execute. Defaults to None (no guardrails execute).",
    )
    is_from_config: bool = Field(
        default=False,
        description="True if this endpoint is defined in the config file, False if from DB. Config-defined endpoints cannot be edited via the UI.",
    )
    methods: list[str] | None = Field(
        default=None,
        description="List of HTTP methods this endpoint handles (e.g., ['GET', 'POST']). If None or empty, all methods (GET, POST, PUT, DELETE, PATCH) are supported for backward compatibility. This allows the same path to have different targets for different HTTP methods.",
    )


class PassThroughEndpointResponse(LiteLLMPydanticObjectBase):
    endpoints: list[PassThroughGenericEndpoint]


class ConfigFieldUpdate(LiteLLMPydanticObjectBase):
    field_name: str
    field_value: Any
    config_type: Literal["general_settings"]


class ConfigFieldDelete(LiteLLMPydanticObjectBase):
    config_type: Literal["general_settings"]
    field_name: str


class CallbackDelete(LiteLLMPydanticObjectBase):
    callback_name: str


class FieldDetail(BaseModel):
    field_name: str
    field_type: str
    field_description: str
    field_default_value: Any = None
    stored_in_db: bool | None


class ConfigList(LiteLLMPydanticObjectBase):
    field_name: str
    field_type: str
    field_description: str
    field_value: Any
    stored_in_db: bool | None
    field_default_value: Any
    premium_field: bool = False
    nested_fields: list[FieldDetail] | None = None  # For nested dictionary or Pydantic fields
    field_options: list[str] | None = None  # Allowed values, for field_type == "Select"
    field_tab: str | None = None  # Admin UI sub-tab this field renders under; None groups it with the rest


class UserHeaderMapping(LiteLLMPydanticObjectBase):
    """
    Map an incoming HTTP header to a LiteLLM user role.
    """

    header_name: str
    litellm_user_role: Literal[
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.CUSTOMER,
    ]

    model_config = {
        "extra": "forbid",
    }


UserMCPManagementMode = Literal["restricted", "view_all"]


class PluginConfig(LiteLLMPydanticObjectBase):
    """A single external service registered as an embeddable UI plugin."""

    name: str = Field(description="unique plugin identifier (kebab-case)")
    display_name: str | None = Field(None, description="human-readable label shown in the UI view switcher")
    url: str = Field(description="base URL of the plugin service")
    plugin_key: str | None = Field(
        None,
        description="plugin's own credential, injected as Bearer auth only on /plugin-proxy/<name>/* reverse-proxy calls",
    )


class CoordinationRedisNode(LiteLLMPydanticObjectBase):
    """A single startup node of a cluster-mode Redis used for proxy coordination."""

    host: str = Field(description="hostname of the cluster node")
    port: int = Field(description="port of the cluster node")


class CoordinationRedisParams(LiteLLMPydanticObjectBase):
    """
    Connection params for the proxy's coordination Redis (cross-pod tpm/rpm rate
    limits, spend tracking, pod lock manager, shared health checks), configured
    independently of the response-cache backend in `litellm_settings.cache_params`.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    host: str | None = Field(None, description="Redis hostname")
    port: int | None = Field(None, description="Redis port")
    password: str | None = Field(None, description="Redis password")
    username: str | None = Field(None, description="Redis username")
    url: str | None = Field(None, description="full Redis connection url, e.g. redis://:pass@host:6379")
    ssl: bool | None = Field(None, description="connect over TLS")
    startup_nodes: list[CoordinationRedisNode] | None = Field(
        None, description="cluster-mode startup nodes; when set a cluster client is used"
    )
    sentinel_nodes: list[list[str | int]] | None = Field(
        None, description="sentinel [host, port] pairs; when set a sentinel-managed client is used"
    )
    sentinel_password: str | None = Field(None, description="password for the sentinel nodes")
    service_name: str | None = Field(None, description="sentinel service name")

    def has_connection_target(self) -> bool:
        return any(value is not None for value in (self.host, self.url, self.startup_nodes, self.sentinel_nodes))


class ScheduledJobStaggerSettings(LiteLLMPydanticObjectBase):
    """
    Spreads the proxy's scheduled background jobs across a window instead of firing them
    all on one instant, on every replica, forever.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    enabled: bool = Field(default=True, description="apply deterministic phase offsets to scheduled background jobs")
    window_seconds: int = Field(
        default=DEFAULT_STAGGER_WINDOW_SECONDS,
        ge=0,
        description=(
            "width of the window jobs are spread over. An interval job is never offset by "
            "more than one of its own periods, so it is not delayed past the wait it already has"
        ),
    )
    identity: str | None = Field(
        default=None,
        description=(
            "replaces the POD_NAME/HOSTNAME-derived component of the offset hash. Set this "
            "when replicas share a hostname and would otherwise land on the same offset"
        ),
    )
    offsets: Mapping[str, int] = Field(
        default_factory=dict,
        description=(
            "explicit offset in seconds per scheduler job id, overriding the derived value. "
            "0 pins a job to its unshifted schedule"
        ),
    )


class ConfigGeneralSettings(LiteLLMPydanticObjectBase):
    """
    Documents all the fields supported by `general_settings` in config.yaml
    """

    completion_model: str | None = Field(None, description="proxy level default model for all chat completion calls")
    plugins: list[PluginConfig] | None = Field(
        None, description="external services registered as embeddable UI plugins"
    )
    key_management_system: KeyManagementSystem | None = Field(
        None, description="key manager to load keys from / decrypt keys with"
    )
    use_google_kms: bool | None = Field(None, description="decrypt keys with google kms")
    use_azure_key_vault: bool | None = Field(None, description="load keys from azure key vault")
    master_key: str | None = Field(None, description="require a key for all calls to proxy")
    coordination_redis: CoordinationRedisParams | None = Field(
        None,
        description=(
            "standalone Redis for cross-pod coordination (tpm/rpm rate limits, "
            "spend tracking, pod lock manager, shared health checks), configured "
            "independently of the response-cache backend; takes precedence over "
            "borrowing the `cache_params` Redis and over the REDIS_* env fallback"
        ),
    )
    control_plane_url: str | None = Field(
        None,
        description=(
            "Global Control Plane: URL of the control plane whose admin UI manages this instance. "
            "Enables /v3/login and /v3/login/exchange on this instance so that UI can authenticate "
            "against it cross-origin, and restricts the SSO return_to origin to that URL. "
            "No state is shared with the control plane"
        ),
    )
    allow_cli_sso_verification_uri_complete: bool | None = Field(
        None,
        description="opt-in to RFC 8628 verification_uri_complete for the CLI SSO device flow, pre-filling the user_code in the browser. Off by default; intended for same-host clients where the device that starts the flow and the browser run on the same machine",
    )
    database_url: str | None = Field(
        None,
        description="connect to a postgres db - needed for generating temporary keys + tracking spend / key",
    )
    database_connection_pool_limit: int | None = Field(
        10,
        description="default connection pool for prisma client connecting to postgres db",
    )
    database_connection_timeout: float | None = Field(
        60, description="default timeout for a connection to the database"
    )
    database_connect_timeout: float | None = Field(
        None,
        description=(
            "Prisma `connect_timeout` URL param (seconds). Bounds how long the "
            "engine waits to establish a new connection before failing. Defaults "
            "to Prisma's built-in value when unset."
        ),
    )
    database_socket_timeout: float | None = Field(
        None,
        description=(
            "Prisma `socket_timeout` URL param (seconds). When set, an idle/slow "
            "connection that has not produced data within this window is closed. "
            "This is the main knob for capping idle DB connections from LiteLLM."
        ),
    )
    database_extra_connection_params: dict[str, Any] | None = Field(
        None,
        description=(
            "Escape hatch: extra key/value pairs appended verbatim to the Prisma "
            "DATABASE_URL / DIRECT_URL query string (e.g. `sslmode`, `pgbouncer`, "
            "`statement_cache_size`). Keys here override any default LiteLLM sets."
        ),
    )
    database_disable_prepared_statements: bool | None = Field(
        None,
        description=(
            "Disable server-side prepared statements by setting Prisma's "
            "`pgbouncer=true` URL param. Use this for pgbouncer transaction-pooling "
            "deployments, or to prevent the 'cached plan must not change result "
            "type' error that pooled connections hit during rolling schema "
            "migrations. An explicit `pgbouncer` in `database_extra_connection_params` "
            "takes precedence."
        ),
    )
    database_type: Literal["dynamo_db"] | None = Field(None, description="to use dynamodb instead of postgres db")
    database_args: DynamoDBArgs | None = Field(
        None,
        description="custom args for instantiating dynamodb client - e.g. billing provision",
    )
    otel: bool | None = Field(
        None,
        description="[BETA] OpenTelemetry support - this might change, use with caution.",
    )
    custom_auth: str | None = Field(
        None,
        description="override user_api_key_auth with your own auth script - https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth",
    )
    max_parallel_requests: int | None = Field(
        None,
        description="maximum parallel requests for each api key",
    )
    global_max_parallel_requests: int | None = Field(
        None, description="global max parallel requests to allow for a proxy instance."
    )
    max_request_size_mb: int | None = Field(
        None,
        description="max request size in MB, if a request is larger than this size it will be rejected",
    )
    max_batch_file_size_mb: int | None = Field(
        None,
        description="max batch input file size in MB for /v1/files uploads with purpose=batch, if a file is larger than this size it will be rejected before being forwarded to the provider",
    )
    max_response_size_mb: int | None = Field(
        None,
        description="max response size in MB, if a response is larger than this size it will be rejected",
    )
    proxy_config_reload_interval_seconds: int = Field(
        30,
        gt=0,
        description="how often (in seconds) each pod reloads config-in-DB objects (models, credentials, guardrails, etc.) when store_model_in_db is enabled; lower values speed up multi-pod convergence at the cost of more DB load. Applied on proxy startup",
    )
    cancel_on_disconnect: bool | None = Field(
        None,
        description="cancel the in-flight upstream LLM request (non-streaming) when the client disconnects, freeing backend capacity (e.g. a vLLM GPU slot); the request is logged as a 499 failure",
    )
    infer_model_from_keys: bool | None = Field(
        None,
        description="for `/models` endpoint, infers available model based on environment keys (e.g. OPENAI_API_KEY)",
    )
    background_health_checks: bool | None = Field(None, description="run health checks in background")
    health_check_interval: int = Field(300, description="background health check interval in seconds")
    health_check_concurrency: int | None = Field(
        None,
        description=(
            "limit concurrent health checks per cycle; when unset, health checks run without a concurrency cap"
        ),
    )
    health_check_skip_disabled_background_models: bool = Field(
        False,
        description=(
            "When true, deployments with model_info.disable_background_health_check "
            "are skipped for on-demand GET /health as well as the background health loop."
        ),
    )
    background_health_check_model_groups: tuple[str, ...] | None = Field(
        None,
        description=(
            "Opt-in allowlist of model group names for background health checks and "
            "health-check routing. When set, the background loop probes only deployments "
            "whose model_name is listed, and enable_health_check_routing filters unhealthy "
            "deployments only within the listed groups; every other group, including newly "
            "added deployments, is skipped and keeps its configured routing strategy. "
            "When unset, all deployments participate (opt out per deployment via "
            "model_info.disable_background_health_check)."
        ),
    )
    model_list_healthy_only: bool | None = Field(
        None,
        description=(
            "When true, `/models`, `/v1/models/{id}` and `/model/info` hide models whose backing "
            "deployments are all unhealthy, for every caller, without needing `healthy_only=true` "
            "per request. Requires `background_health_checks: true`, and keeps deployment health "
            "state cached without turning on `enable_health_check_routing`, so routing is "
            "unaffected. With no health state nothing is hidden. Hiding is presentation-only, a "
            "hidden model can still be called."
        ),
    )
    alerting: list | None = Field(
        None,
        description="List of alerting integrations. Today, just slack - `alerting: ['slack']`",
    )
    alert_types: list[AlertType] | None = Field(
        None,
        description="List of alerting types. By default it is all alerts",
    )
    alert_to_webhook_url: dict | None = Field(
        None,
        description="Mapping of alert type to webhook url. e.g. `alert_to_webhook_url: {'budget_alerts': 'https://nothooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX'}`",
    )
    alerting_args: dict | None = Field(None, description="Controllable params for slack alerting - e.g. ttl in cache.")
    alerting_threshold: int | None = Field(
        None,
        description="sends alerts if requests hang for 5min+",
    )
    ui_access_mode: Literal["admin_only", "all"] | None = Field("all", description="Control access to the Proxy UI")
    allowed_routes: list | None = Field(None, description="Proxy API Endpoints you want users to be able to access")
    reject_clientside_metadata_tags: bool | None = Field(
        None,
        description="When set to True, rejects requests that contain client-side 'metadata.tags' to prevent users from influencing budgets by sending different tags. Tags can only be inherited from the API key metadata.",
    )
    enable_public_model_hub: bool = Field(
        default=False,
        description="Public model hub for users to see what models they have access to, supported openai params, etc.",
    )
    pass_through_request_timeout: float | None = Field(
        default=None,
        description="Default upstream request timeout in seconds for native and custom pass-through endpoints that use pass_through_request. Defaults to 600 when unset.",
    )
    pass_through_endpoints: list[PassThroughGenericEndpoint] | None = Field(
        default=None,
        description="Set-up pass-through endpoints for provider-specific endpoints. Docs - https://docs.litellm.ai/docs/proxy/pass_through",
    )
    user_header_name: str | None = Field(
        None,
        description="[DEPRECATED] Use 'user_header_mappings' instead. When set, the header value is treated as the end user id unless overridden by user_header_mappings.",
    )
    user_header_mappings: list[UserHeaderMapping] | None = None
    supported_db_objects: list[SupportedDBObjectType] | None = Field(
        None,
        description="Fine-grained control over which object types to load from the database when store_model_in_db is True. Available types: 'models', 'mcp', 'guardrails', 'vector_stores', 'pass_through_endpoints', 'prompts', 'model_cost_map', 'tools', 'config_overrides'. If not set, all objects are loaded (default behavior).",
    )
    user_mcp_management_mode: UserMCPManagementMode | None = Field(
        None,
        description="Controls how non-admin users interact with MCP servers in the dashboard. 'restricted' shows only accessible servers, 'view_all' lists every server in read-only mode.",
    )
    store_prompts_in_spend_logs: bool | None = Field(
        None,
        description="If True, stores request messages and responses in spend logs. Default is False.",
    )
    disable_auto_add_proxy_admin_to_teams: bool | None = Field(
        None,
        description="By default, the user calling /team/new is automatically added to the new team as a team admin. If True, proxy admins are no longer auto-added; members explicitly listed in members_with_roles are unaffected. Default is False.",
    )
    scheduled_job_stagger: ScheduledJobStaggerSettings | None = Field(
        None,
        description=(
            "Spreads the proxy's scheduled background jobs (spend flushes, budget resets, "
            "config reloads, exports) across a window instead of firing them together on "
            "every replica. On by default; set to tune the window, pin a job, or turn it off."
        ),
    )
    maximum_spend_logs_retention_period: str | None = Field(
        None,
        description="Maximum retention period for spend logs (e.g., '7d' for 7 days). Logs older than this will be deleted.",
    )
    maximum_autorouter_session_retention_period: str | None = Field(
        None,
        description="Maximum retention period for auto-router benchmark session rollup rows (e.g., '365d'). Rows whose last turn is older than this are deleted by the spend log cleanup job, on that job's schedule. Unset means rollup rows are never deleted.",
    )
    maximum_health_check_retention_period: str | None = Field(
        None,
        description=(
            "Maximum retention period for health-check rows (e.g., '30d'). Rows whose checked_at is older than this "
            "are deleted by the spend log cleanup job, on that job's schedule. Unset means rows are never deleted. "
            "Set this well above health_check_interval because /health and the UI read the latest row per model."
        ),
    )
    use_spend_logs_partitioning: bool | None = Field(
        None,
        description="If True and LiteLLM_SpendLogs has been converted to a range-partitioned table (db_scripts/partition_spend_logs.sql), retention cleanup drops expired partitions instead of deleting rows, and pre-creates upcoming partitions. Default is False.",
    )
    maximum_spend_logs_cleanup_batch_size: int | None = Field(
        None,
        description="Rows deleted per DELETE statement by the spend log cleanup job. Defaults to 1000.",
    )
    maximum_spend_logs_cleanup_max_batches: int | None = Field(
        None,
        description="Maximum DELETE statements the spend log cleanup job issues per table per run. Defaults to 500.",
    )
    maximum_spend_logs_cleanup_run_budget: str | None = Field(
        None,
        description="Wall-clock budget for one spend log cleanup run (e.g. '5m'), shared across every table it prunes. A run that hits the budget stops and the next run resumes from where it left off. Defaults to '5m'.",
    )
    maximum_spend_logs_cleanup_batch_timeout: str | None = Field(
        None,
        description="Postgres statement_timeout and lock_timeout applied to each spend log cleanup delete batch (e.g. '30s'), so cleanup cannot hold row locks or a connection indefinitely. Defaults to '30s'.",
    )
    mcp_internal_ip_ranges: list[str] | None = Field(
        None,
        description="Custom CIDR ranges that define internal/private networks for MCP access control. When set, only these ranges are treated as internal. Defaults to RFC 1918 private ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8).",
    )
    mcp_trusted_proxy_ranges: list[str] | None = Field(
        None,
        description="CIDR ranges of trusted reverse proxies. When set, X-Forwarded-For and X-Forwarded-* origin headers are only trusted from these IPs.",
    )
    mcp_xff_num_trusted_hops: int | None = Field(
        None,
        ge=1,
        description="Number of trusted reverse proxies/load balancers in front of the gateway that append to X-Forwarded-For. When set (and mcp_trusted_proxy_ranges validates the direct peer), the client IP for MCP access control is read this many entries from the right of the chain instead of the spoofable leftmost value, defeating append-style X-Forwarded-For forgery.",
    )
    trusted_proxy_ranges: list[str] | None = Field(
        None,
        description="CIDR ranges of trusted reverse proxies allowed to provide identity headers for header-based auth paths such as enable_oauth2_proxy_auth and custom_ui_sso_sign_in_handler.",
    )
    store_model_in_db: bool | None = Field(
        None,
        description="If True, models and config are stored in and loaded from the database. Default is False.",
    )
    forward_client_headers_to_llm_api: bool | None = Field(
        None,
        description="If True, forwards client headers (e.g. Authorization) to the LLM API. Required for Claude Code with Max subscription.",
    )
    mcp_required_fields: list[str] | None = Field(
        None,
        description="List of MCP server fields that must be filled in for a submission to pass standards checks (e.g. ['description', 'source_url', 'alias']).",
    )
    disable_budget_reservation: bool | None = Field(
        None,
        description=(
            "If True, disables the optimistic per-request budget reservation "
            "introduced in v1.84.0. "
            "WARNING: This weakens hard budget enforcement. Without the reservation, "
            "a burst of concurrent requests from a single key can each pass the "
            "read-time spend check before any of them is charged, allowing a "
            "configured budget to be exceeded under high concurrency. "
            "Budgets are still evaluated on every request at read time, so "
            "an already-exhausted budget is still rejected. "
            "Enable only if your deployment is experiencing phantom "
            "BudgetExceededError responses caused by leaked reservations "
            "(see GitHub issue #27639). "
            "A proxy-level WARNING is logged on every request while this flag "
            "is active as a reminder that hard enforcement is relaxed."
        ),
    )
    apply_user_budget_to_team_keys: bool | None = Field(
        None,
        description=(
            "If True, a user's personal max_budget is enforced on every request they "
            "make, including requests made with a team-scoped key. Defaults to False, "
            "where a team-scoped key is governed only by the team and team-member "
            "budgets and the key owner's personal max_budget does not apply "
            "(see GitHub issue #12905)."
        ),
    )
    user_url_validation: bool | None = Field(
        None,
        description=(
            "Master switch for the SSRF guard applied to user-supplied URLs "
            "(image_url, file_url, MCP/OpenAPI spec URLs, etc). Defaults to True. "
            "Set to False to disable DNS/IP validation entirely (not recommended)."
        ),
    )
    user_url_allowed_hosts: list[str] | None = Field(
        None,
        description=(
            "SSRF allowlist for user-supplied URLs. Entries are `hostname` or "
            "`hostname:port` (bracketed for IPv6, e.g. `[::1]:8080`). Allowlisted "
            "hosts skip the blocked-network check in validate_url() but still "
            "resolve DNS. Use this to permit legitimate internal targets, e.g. "
            "an internal OpenAPI/MCP server."
        ),
    )
    provider_url_destination_allowed_hosts: list[str] | None = Field(
        None,
        description="Allowlist of hosts a request may redirect a provider call's destination URL to.",
    )


class ConfigYAML(LiteLLMPydanticObjectBase):
    """
    Documents all the fields supported by the config.yaml
    """

    environment_variables: dict | None = Field(
        None,
        description="Object to pass in additional environment variables via POST request",
    )
    model_list: list[ModelParams] | None = Field(
        None,
        description="List of supported models on the server, with model-specific configs",
    )
    litellm_settings: dict | None = Field(
        None,
        description="litellm Module settings. See __init__.py for all, example litellm.drop_params=True, litellm.set_verbose=True, litellm.api_base, litellm.cache",
    )
    general_settings: ConfigGeneralSettings | None = None
    worker_registry: list[WorkerRegistryEntry] | None = Field(
        None,
        description=(
            "Global Control Plane: the independent proxy instances this instance's admin UI manages. "
            "Setting it makes this a control plane, which serves the UI and does not route LLM requests. "
            "Enterprise-only"
        ),
    )
    router_settings: UpdateRouterConfig | None = Field(
        None,
        description="litellm router object settings. See router.py __init__ for all, example router.num_retries=5, router.timeout=5, router.max_retries=5, router.retry_after=5",
    )

    model_config = ConfigDict(protected_namespaces=())


from litellm.models.verification_token import (  # noqa: E402
    LiteLLM_DeletedVerificationToken as LiteLLM_DeletedVerificationToken,
)
from litellm.models.verification_token import (  # noqa: E402
    LiteLLM_VerificationToken as LiteLLM_VerificationToken,
)


class LiteLLM_VerificationTokenView(LiteLLM_VerificationToken):
    """
    Combined view of litellm verification token + litellm team table (select values)
    """

    team_spend: float | None = None
    team_alias: str | None = None
    team_tpm_limit: int | None = None
    team_rpm_limit: int | None = None
    team_max_budget: float | None = None
    team_soft_budget: float | None = None
    team_models: list = []
    team_blocked: bool = False
    soft_budget: float | None = None
    team_model_aliases: dict | None = None
    team_member: Member | None = None
    team_metadata: dict | None = None
    team_object_permission_id: str | None = None

    # Team Member Specific Params
    team_member_spend: float | None = None
    team_member_tpm_limit: int | None = None
    team_member_rpm_limit: int | None = None

    # End User Params
    end_user_id: str | None = None
    end_user_tpm_limit: int | None = None
    end_user_rpm_limit: int | None = None
    end_user_max_budget: float | None = None
    end_user_model_max_budget: dict | None = None

    # Organization Params
    organization_alias: str | None = None
    organization_max_budget: float | None = None
    organization_tpm_limit: int | None = None
    organization_rpm_limit: int | None = None
    organization_metadata: dict | None = None

    # Project Params
    project_alias: str | None = None
    project_metadata: dict | None = None

    # Time stamps
    last_refreshed_at: float | None = None  # last time joint view was pulled from db

    def __init__(self, **kwargs):
        # Handle litellm_budget_table_* keys (budget table overrides when key value is None or empty)
        for key, value in list(kwargs.items()):
            if key.startswith("litellm_budget_table_") and value is not None:
                # Extract the corresponding attribute name
                attr_name = key.replace("litellm_budget_table_", "")
                # Use key's value from kwargs (from DB view), not class default
                current = kwargs.get(attr_name)
                if current is None:
                    current = getattr(self, attr_name, None)
                # Apply budget value when key has no value, or for model_max_budget when key has empty dict
                should_apply = current is None or (
                    attr_name == "model_max_budget" and isinstance(current, dict) and len(current) == 0
                )
                if should_apply:
                    kwargs[attr_name] = value
            if key == "end_user_id" and value is not None and isinstance(value, int):
                kwargs[key] = str(value)

        if kwargs.get("organization_id") is not None:
            kwargs["org_id"] = kwargs.pop("organization_id")
        # Initialize the superclass
        super().__init__(**kwargs)


class UserAPIKeyAuth(LiteLLM_VerificationTokenView):  # the expected response object for user api key auth
    """
    Return the row in the db
    """

    api_key: str | None = None
    user_role: LitellmUserRoles | None = None
    allowed_model_region: AllowedModelRegion | None = None
    parent_otel_span: Span | None = None
    rpm_limit_per_model: dict[str, int] | None = None
    tpm_limit_per_model: dict[str, int] | None = None
    user_tpm_limit: int | None = None
    user_rpm_limit: int | None = None
    user_email: str | None = None
    user_spend: float | None = None
    user_max_budget: float | None = None
    # Values stay `object` rather than BudgetConfig: this is the raw JSON column,
    # and validating it here would make one malformed row fail auth outright.
    # resolve_model_budget validates the single entry a request actually needs.
    user_model_max_budget: Mapping[str, object] | None = None
    request_route: str | None = None
    is_session_token: bool = False
    # Server-only marker set exclusively by the MCP gateway admission path
    # (reload_admitted_user) for a keyless user-subject admitted via a gateway DCR session
    # bearer or bridge envelope. Not a DB column and never populated from caller-controlled key
    # metadata or JWT claims, so it cannot be forged to gain the team-inherited MCP grant union
    # or to escape the caller-Authorization egress scrub. exclude=True keeps it out of serialization.
    mcp_admitted_user_subject: bool = Field(default=False, exclude=True)
    # team_id -> that team's mcp_rpm_limit map, for a keyless admitted subject that reaches MCP
    # servers through several teams at once and therefore has no single team_id for the limiter to
    # key off. Server-only and stripped from validated input for the same reason as the marker
    # above: a forged entry would let a caller pick which team's rpm bucket it is charged against.
    mcp_source_team_rpm_limits: dict[str, dict[str, int]] | None = Field(default=None, exclude=True)
    # The single MCP server_id a gateway session bearer was scoped to at authorize time (RFC 8707
    # resource), or None for an aggregate-scope session. A RESTRICTION intersected against the live
    # grant resolution, never a grant. Server-only, set exclusively by the MCP gateway admission
    # path via post-construction assignment and stripped from validated input like the markers
    # above; a forged value could at most narrow, but the stripping keeps the field's provenance
    # single-owner so its meaning stays trustworthy.
    mcp_session_resource_server_id: str | None = Field(default=None, exclude=True)
    via_virtual_key: bool = Field(
        default=False,
        exclude=True,
        description=(
            "Server-only marker set exclusively by the DB virtual-key and master-key auth paths via "
            "post-construction assignment. Stripped from validated input so custom auth handlers, JWT "
            "claims, or key metadata cannot forge it. Gates overwrite_user_with_key_hash stamping: only "
            "a credential the proxy itself validated as a key may be forwarded as the provider-facing "
            "user id."
        ),
    )
    budget_reservation: dict[str, Any] | None = Field(default=None, exclude=True)
    budget_throttle_pct: float | None = Field(default=None, exclude=True)
    user: Any | None = None  # Expanded user object when expand=user is used
    created_by_user: Any | None = None  # Expanded created_by user when expand=user is used
    end_user_object_permission: LiteLLM_ObjectPermissionTable | None = None
    # Team object_permission preloaded in auth (e.g. get_team_object) to avoid
    # per-request object_permission fetches in downstream checks (vector stores, etc.)
    team_object_permission: LiteLLM_ObjectPermissionTable | None = None
    # Decoded upstream IdP claims (groups, roles, etc.) propagated by JWT auth machinery
    # and forwarded into outbound tokens by guardrails such as MCPJWTSigner.
    jwt_claims: dict | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def check_api_key(cls, values):
        # If values is already an instance (not a dict), return it as-is
        if not isinstance(values, dict):
            return values
        # mcp_admitted_user_subject is a server-only marker, set ONLY by the MCP gateway admission
        # path via post-construction assignment. Strip it from any validated input (constructor
        # kwargs, model_validate, a JWT/key claim splat) so it can never be forged from caller data.
        values.pop("mcp_admitted_user_subject", None)
        values.pop("mcp_source_team_rpm_limits", None)
        values.pop("mcp_session_resource_server_id", None)
        values.pop("via_virtual_key", None)
        if values.get("api_key") is not None:
            values.update({"token": cls._safe_hash_litellm_api_key(values.get("api_key"))})
            if isinstance(values.get("api_key"), str):
                values.update({"api_key": cls._safe_hash_litellm_api_key(values.get("api_key"))})
        return values

    @classmethod
    def _safe_hash_litellm_api_key(cls, api_key: str) -> str:
        """
        Helper to ensure all logged keys are hashed
        Covers:
        1. Regular API keys from LiteLLM DB
        2. JWT tokens used for connecting to LiteLLM API
        """
        normalized = api_key
        if normalized[:7].lower() == "bearer ":
            normalized = normalized[7:]
        if normalized.startswith("sk-"):
            return hash_token(normalized)
        from litellm.proxy.auth.handle_jwt import JWTHandler

        if JWTHandler.is_jwt(token=normalized):
            return f"hashed-jwt-{hash_token(token=normalized)}"
        return normalized

    @classmethod
    def get_litellm_internal_health_check_user_api_key_auth(cls) -> "UserAPIKeyAuth":
        """
        Returns a `UserAPIKeyAuth` object for the litellm internal health check service account.

        This is used to track number of requests/spend for health check calls.
        """
        from litellm.constants import LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME

        return cls(
            api_key=LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
            team_id=LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
            key_alias=LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
            team_alias=LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
        )

    @classmethod
    def get_litellm_cli_user_api_key_auth(cls) -> "UserAPIKeyAuth":
        """
        Returns a `UserAPIKeyAuth` object for the litellm internal health check service account.

        This is used to track number of requests/spend for health check calls.
        """
        from litellm.constants import LITTELM_CLI_SERVICE_ACCOUNT_NAME

        return cls(
            api_key=LITTELM_CLI_SERVICE_ACCOUNT_NAME,
            team_id=LITTELM_CLI_SERVICE_ACCOUNT_NAME,
            key_alias=LITTELM_CLI_SERVICE_ACCOUNT_NAME,
            team_alias=LITTELM_CLI_SERVICE_ACCOUNT_NAME,
        )

    @classmethod
    def get_litellm_internal_jobs_user_api_key_auth(cls) -> "UserAPIKeyAuth":
        """
        Returns a `UserAPIKeyAuth` object for internal LiteLLM jobs like key rotation.

        This is used to track actions performed by automated system jobs.
        """
        from litellm.constants import LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME

        return cls(
            api_key=LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
            team_id="system",
            key_alias=LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
            team_alias="system",
            user_id="system",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )


def user_api_key_has_admin_view(user_api_key_dict: UserAPIKeyAuth) -> bool:
    """Return True if the caller's role grants unscoped read access to all
    tenant resources (managed files, batches, vector stores, spend rows, etc).

    Lives on _types.py so leaf modules (e.g. litellm.llms.base_llm.managed_resources)
    can use it without pulling in litellm.proxy.utils via management_endpoints.
    """
    return user_api_key_dict.user_role in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    )


class UserInfoResponse(LiteLLMPydanticObjectBase):
    user_id: str | None
    user_info: dict | BaseModel | None
    keys: list
    teams: list


class UserInfoV2Response(LiteLLMPydanticObjectBase):
    """
    Response model for GET /v2/user/info

    Returns ONLY the user object - no keys, no teams objects.
    This is a lightweight alternative to UserInfoResponse.
    """

    user_id: str
    user_email: str | None = None
    user_alias: str | None = None
    user_role: str | None = None
    spend: float = 0.0
    max_budget: float | None = None
    models: list[str] = []
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    metadata: dict | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    sso_user_id: str | None = None
    teams: list[str] = []  # Just team IDs, not full team objects
    object_permission: LiteLLM_ObjectPermissionTable | None = None
    model_max_budget: Mapping[str, object] | None = None
    model_max_budget_usage: Mapping[str, Mapping[str, object]] | None = None


from litellm.models.config import LiteLLM_Config as LiteLLM_Config  # noqa: E402
from litellm.models.organization_membership import (  # noqa: E402
    LiteLLM_OrganizationMembershipTable as LiteLLM_OrganizationMembershipTable,
)


class LiteLLM_OrganizationTableUpdate(LiteLLM_BudgetTable):
    """Represents user-controllable params for a LiteLLM_OrganizationTable record"""

    organization_id: str | None = None
    organization_alias: str | None = None
    budget_id: str | None = None
    spend: float | None = None
    metadata: dict | None = None
    models: list[str] | None = None
    updated_by: str | None = None
    object_permission: LiteLLM_ObjectPermissionBase | None = None
    model_tpm_limit: dict[str, int] | None = None
    model_rpm_limit: dict[str, int] | None = None

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        for field in LiteLLM_ManagementEndpoint_MetadataFields:
            if values.get(field) is not None:
                # add to metadata
                if values.get("metadata") is None:
                    values.update({"metadata": {}})
                values["metadata"][field] = values.get(field)
                values.pop(field)
        return values


class OrganizationUpdateRequestV2(LiteLLMPydanticObjectBase):
    """
    Typed PATCH body for ``/v2/organization/{organization_id}`` (RFC 7396 merge-patch).

    Presence is read from ``model_fields_set``, so a sent field is written and an omitted one is
    left untouched. ``extra="forbid"`` makes an unknown key a 422 rather than a silent no-op, since
    the contract hinges on which keys are present. See the endpoint for the per-field clear tokens.
    """

    model_config = ConfigDict(extra="forbid")

    organization_alias: str | None = None
    models: list[str] | None = None
    metadata: dict | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    max_budget: float | None = None
    soft_budget: float | None = None
    max_parallel_requests: int | None = None
    model_max_budget: dict | None = None
    budget_duration: str | None = None
    object_permission: LiteLLM_ObjectPermissionBase | None = None


from litellm.models.organization import (  # noqa: E402
    LiteLLM_OrganizationTable as LiteLLM_OrganizationTable,
)
from litellm.models.user import LiteLLM_UserTable as LiteLLM_UserTable  # noqa: E402


class LiteLLM_OrganizationTableWithMembers(LiteLLM_OrganizationTable):
    """Returned by the /organization/info endpoint and /organization/list endpoint"""

    members: list[LiteLLM_OrganizationMembershipTable] = []
    teams: list[LiteLLM_TeamTable] = []
    litellm_budget_table: LiteLLM_BudgetTable | None = None
    created_at: datetime
    updated_at: datetime


class NewOrganizationResponse(LiteLLM_OrganizationTable):
    organization_id: str
    created_at: datetime
    updated_at: datetime


### PROJECT MANAGEMENT TYPES ###


class ProjectBase(LiteLLMPydanticObjectBase):
    """Base fields shared by project create/update requests"""

    project_id: str | None = None
    project_alias: str | None = None
    team_id: str | None = None
    metadata: dict | None = None
    models: list[str] | None = None
    blocked: bool = False


class NewProjectRequest(LiteLLM_BudgetTable):
    """Request model for POST /project/new"""

    project_id: str | None = None
    project_alias: str | None = None
    description: str | None = None
    team_id: str
    budget_id: str | None = None
    metadata: dict | None = None
    tags: list[str] | None = None
    guardrails: list[str] | None = None
    policies: list[str] | None = None
    models: list[str] = []
    model_rpm_limit: dict | None = None
    model_tpm_limit: dict | None = None
    model_itpm_limit: Mapping[str, int] | None = None
    model_otpm_limit: Mapping[str, int] | None = None
    blocked: bool = False
    object_permission: LiteLLM_ObjectPermissionBase | None = None

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        if "tags" in values and values["tags"] is not None:
            if not isinstance(values["tags"], list):
                raise ValueError(f"tags must be a list of strings, got {type(values['tags']).__name__}")
        for field in LiteLLM_ManagementEndpoint_MetadataFields:
            if values.get(field) is not None:
                if values.get("metadata") is None:
                    values.update({"metadata": {}})
                values["metadata"][field] = values.get(field)
                values.pop(field)
        return values


class UpdateProjectRequest(LiteLLM_BudgetTable):
    """Request model for POST /project/update"""

    project_id: str
    project_alias: str | None = None
    description: str | None = None
    team_id: str | None = None
    metadata: dict | None = None
    tags: list[str] | None = None
    guardrails: list[str] | None = None
    policies: list[str] | None = None
    models: list[str] | None = None
    model_rpm_limit: dict | None = None
    model_tpm_limit: dict | None = None
    model_itpm_limit: Mapping[str, int] | None = None
    model_otpm_limit: Mapping[str, int] | None = None
    blocked: bool | None = None
    budget_id: str | None = None
    object_permission: LiteLLM_ObjectPermissionBase | None = None

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        if "tags" in values and values["tags"] is not None:
            if not isinstance(values["tags"], list):
                raise ValueError(f"tags must be a list of strings, got {type(values['tags']).__name__}")
        for field in LiteLLM_ManagementEndpoint_MetadataFields:
            if values.get(field) is not None:
                if values.get("metadata") is None:
                    values.update({"metadata": {}})
                values["metadata"][field] = values.get(field)
                values.pop(field)
        return values


class DeleteProjectRequest(LiteLLMPydanticObjectBase):
    """Request model for DELETE /project/delete"""

    project_ids: list[str]


from litellm.models.project import (  # noqa: E402
    LiteLLM_ProjectTable as LiteLLM_ProjectTable,
)


class NewProjectResponse(LiteLLM_ProjectTable):
    """Response model for POST /project/new"""

    project_id: str
    created_at: datetime
    updated_at: datetime


class LiteLLM_ProjectTableCachedObj(LiteLLM_ProjectTable):
    """Cached version for auth checks. Mirrors LiteLLM_TeamTableCachedObj pattern."""

    last_refreshed_at: float | None = None


class LiteLLM_UserTableFiltered(BaseModel):  # done to avoid exposing sensitive data
    user_id: str
    user_email: str | None = None


class LiteLLM_UserTableWithKeyCount(LiteLLM_UserTable):
    key_count: int = 0


from litellm.models.access_group import (  # noqa: E402
    LiteLLM_AccessGroupTable as LiteLLM_AccessGroupTable,
)
from litellm.models.end_user import (  # noqa: E402
    LiteLLM_EndUserTable as LiteLLM_EndUserTable,
)
from litellm.models.spend_logs import (  # noqa: E402
    LiteLLM_ErrorLogs as LiteLLM_ErrorLogs,
)
from litellm.models.spend_logs import (  # noqa: E402
    LiteLLM_SpendLogs as LiteLLM_SpendLogs,
)
from litellm.models.tag import LiteLLM_TagTable as LiteLLM_TagTable  # noqa: E402

AUDIT_ACTIONS = Literal["created", "updated", "deleted", "blocked", "unblocked", "rotated"]


class LiteLLM_AuditLogs(LiteLLMPydanticObjectBase):
    id: str
    updated_at: datetime
    changed_by: Any | None = None
    changed_by_api_key: str | None = None
    action: AUDIT_ACTIONS
    table_name: LitellmTableNames
    object_id: str
    before_value: Json | None = None
    updated_values: Json | None = None

    @model_validator(mode="before")
    @classmethod
    def cast_changed_by_to_str(cls, values):
        if values.get("changed_by") is not None:
            values["changed_by"] = str(values["changed_by"])
        return values

    @model_validator(mode="after")
    def mask_api_keys(self):
        from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker

        masker: Final = SensitiveDataMasker(sensitive_patterns={"key"})

        if self.before_value is not None:
            json_before_value: dict | None = None
            if isinstance(self.before_value, str):
                json_before_value = json.loads(self.before_value)
            elif isinstance(self.before_value, dict):
                json_before_value = self.before_value

            if json_before_value is not None:
                json_before_value = masker.mask_dict(json_before_value)
                self.before_value = json.dumps(json_before_value, default=str)

        if self.updated_values is not None:
            json_updated_values: dict | None = None
            if isinstance(self.updated_values, str):
                json_updated_values = json.loads(self.updated_values)
            elif isinstance(self.updated_values, dict):
                json_updated_values = self.updated_values

            if json_updated_values is not None:
                json_updated_values = masker.mask_dict(json_updated_values)
                self.updated_values = json.dumps(json_updated_values, default=str)

        return self


class LiteLLM_SpendLogs_ResponseObject(LiteLLMPydanticObjectBase):
    response: list[LiteLLM_SpendLogs | Any] | None = None


class TokenCountRequest(LiteLLMPydanticObjectBase):
    model: str
    prompt: str | None = None
    messages: list[dict] | None = None
    """
    Anthropic token counting endpoint uses /messages
    """

    contents: list[dict] | None = None
    """
    Google /countTokens endpoint expects contents to be a list of dicts with the following structure:
    """

    tools: list[dict] | None = None
    system: Any | None = None


class CallInfo(LiteLLMPydanticObjectBase):
    """Used for slack budget alerting"""

    spend: float
    max_budget: float | None = None
    soft_budget: float | None = None
    token: str | None = Field(default=None, description="Hashed value of that key")
    customer_id: str | None = None
    user_id: str | None = None
    team_id: str | None = None
    team_alias: str | None = None
    organization_id: str | None = None
    user_email: str | None = None
    key_alias: str | None = None
    projected_exceeded_date: str | None = None
    projected_spend: float | None = None
    event_group: Litellm_EntityType
    alert_emails: list[str] | None = Field(
        default=None,
        description="Additional email addresses to send alerts to (e.g., from team metadata)",
    )
    max_budget_alert_emails: dict[str, list[str]] | None = Field(
        default=None,
        description="Map of threshold percentage to email recipients (e.g., {'50': ['a@co.com'], '75': ['a@co.com', 'b@co.com']})",
    )


class WebhookEvent(CallInfo):
    event: Literal[
        "budget_crossed",
        "max_budget_alert",
        "soft_budget_crossed",
        "threshold_crossed",
        "projected_limit_exceeded",
        "key_created",
        "key_rotated",
        "internal_user_created",
        "spend_tracked",
    ]
    event_message: str  # human-readable description of event
    event_group: Litellm_EntityType


class SpecialModelNames(enum.Enum):
    all_team_models = "all-team-models"
    all_proxy_models = "all-proxy-models"
    no_default_models = "no-default-models"


class SpecialMCPServerNames(enum.Enum):
    no_mcp_servers = "no-mcp-servers"


class SpecialProxyStrings(enum.Enum):
    default_user_id = "default_user_id"  # global proxy admin


class InvitationNew(LiteLLMPydanticObjectBase):
    user_id: str


class InvitationUpdate(LiteLLMPydanticObjectBase):
    invitation_id: str
    is_accepted: bool


class InvitationDelete(LiteLLMPydanticObjectBase):
    invitation_id: str


class InvitationModel(LiteLLMPydanticObjectBase):
    id: str
    user_id: str
    is_accepted: bool
    accepted_at: datetime | None
    expires_at: datetime
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str


class InvitationClaim(LiteLLMPydanticObjectBase):
    invitation_link: str
    user_id: str
    password: str


class ConfigFieldInfo(LiteLLMPydanticObjectBase):
    field_name: str
    field_value: Any


class CallbackOnUI(LiteLLMPydanticObjectBase):
    litellm_callback_name: str
    litellm_callback_params: list | None
    ui_callback_name: str


class AllCallbacks(LiteLLMPydanticObjectBase):
    langfuse: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="langfuse",
        ui_callback_name="Langfuse",
        litellm_callback_params=[
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_HOST",
        ],
    )

    otel: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="otel",
        ui_callback_name="OpenTelemetry",
        litellm_callback_params=[
            "OTEL_EXPORTER",
            "OTEL_ENDPOINT",
            "OTEL_HEADERS",
        ],
    )

    s3: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="s3",
        ui_callback_name="s3 Bucket (AWS)",
        litellm_callback_params=[
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION_NAME",
        ],
    )

    azure_sentinel: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="azure_sentinel",
        ui_callback_name="Azure Sentinel",
        litellm_callback_params=[
            "AZURE_SENTINEL_DCR_IMMUTABLE_ID",
            "AZURE_SENTINEL_ENDPOINT",
            "AZURE_SENTINEL_TENANT_ID",
            "AZURE_SENTINEL_CLIENT_ID",
            "AZURE_SENTINEL_CLIENT_SECRET",
            "AZURE_SENTINEL_STREAM_NAME",
        ],
    )

    openmeter: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="openmeter",
        ui_callback_name="OpenMeter",
        litellm_callback_params=[
            "OPENMETER_API_ENDPOINT",
            "OPENMETER_API_KEY",
        ],
    )

    custom_callback_api: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="custom_callback_api",
        litellm_callback_params=["GENERIC_LOGGER_ENDPOINT", "GENERIC_LOGGER_HEADERS"],
        ui_callback_name="Custom Callback API",
    )

    generic_api: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="generic_api",
        litellm_callback_params=["GENERIC_LOGGER_ENDPOINT", "GENERIC_LOGGER_HEADERS"],
        ui_callback_name="Custom Callback API",
    )

    datadog: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="datadog",
        litellm_callback_params=["DD_API_KEY", "DD_SITE"],
        ui_callback_name="Datadog",
    )

    braintrust: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="braintrust",
        litellm_callback_params=["BRAINTRUST_API_KEY", "BRAINTRUST_API_BASE"],
        ui_callback_name="Braintrust",
    )

    langsmith: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="langsmith",
        litellm_callback_params=[
            "LANGSMITH_API_KEY",
            "LANGSMITH_PROJECT",
            "LANGSMITH_DEFAULT_RUN_NAME",
        ],
        ui_callback_name="Langsmith",
    )

    lago: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="lago",
        litellm_callback_params=[
            "LAGO_API_BASE",
            "LAGO_API_KEY",
            "LAGO_API_EVENT_CODE",
            "LAGO_API_CHARGE_BY",
        ],
        ui_callback_name="Lago Billing",
    )

    traceloop: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="traceloop",
        litellm_callback_params=[
            "TRACELOOP_API_KEY",
        ],
        ui_callback_name="Traceloop",
    )

    galileo: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="galileo",
        litellm_callback_params=[
            "GALILEO_API_KEY",
            "GALILEO_PROJECT_ID",
            "GALILEO_LOG_STREAM_ID",
            "GALILEO_BASE_URL",
            "GALILEO_USERNAME",
            "GALILEO_PASSWORD",
        ],
        ui_callback_name="Galileo",
    )

    newrelic: CallbackOnUI = CallbackOnUI(
        litellm_callback_name="newrelic",
        ui_callback_name="New Relic",
        litellm_callback_params=[
            "NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED",
        ],
    )


class SpendLogsMetadata(TypedDict):
    """
    Specific metadata k,v pairs logged to spendlogs for easier cost tracking
    """

    additional_usage_values: dict | None  # covers provider-specific usage information - e.g. prompt caching
    user_api_key: str | None
    user_api_key_alias: str | None
    user_api_key_team_id: str | None
    user_api_key_project_id: str | None
    user_api_key_project_alias: str | None
    user_api_key_org_id: str | None
    user_api_key_user_id: str | None
    user_api_key_team_alias: str | None
    spend_logs_metadata: dict | None  # special param to log k,v pairs to spendlogs for a call
    requester_ip_address: str | None
    litellm_call_id: str | None
    applied_guardrails: list[str] | None
    mcp_tool_call_metadata: StandardLoggingMCPToolCall | None
    vector_store_request_metadata: list[StandardLoggingVectorStoreRequest] | None
    routing_decision: StandardLoggingRoutingDecision | None
    internal_call_origin: InternalCallOrigin | None
    guardrail_information: list[StandardLoggingGuardrailInformation] | None
    eval_information: Any | None
    status: StandardLoggingPayloadStatus
    proxy_server_request: str | None
    batch_models: list[str] | None
    error_information: StandardLoggingPayloadErrorInformation | None
    usage_object: dict | None
    model_map_information: StandardLoggingModelInformation | None
    cold_storage_object_key: str | None  # S3/GCS object key for cold storage retrieval
    litellm_overhead_time_ms: float | None  # LiteLLM overhead time in milliseconds
    attempted_retries: int | None  # Number of retries attempted (0 = first attempt succeeded)
    max_retries: int | None  # Max retries configured for this request
    attempted_fallbacks: ReadOnly[int | None]  # Number of fallbacks attempted (0 = primary model group served)
    original_model_group: ReadOnly[str | None]  # Model group requested before any fallbacks
    cost_breakdown: CostBreakdown | None  # Detailed cost breakdown (input_cost, output_cost, margin, discount, etc.)
    compression_savings: CompressionSavingsMetadata | None
    autorouter_savings: ReadOnly[float | None]  # stamped by the logging payload; None = not auto-routed


class SpendLogsPayload(TypedDict):
    request_id: str
    call_type: str
    api_key: str
    spend: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    startTime: datetime | str
    endTime: datetime | str
    completionStartTime: datetime | str | None
    model: str
    model_id: str | None
    model_group: str | None
    mcp_namespaced_tool_name: str | None
    agent_id: str | None
    api_base: str
    user: str
    metadata: str  # json str
    cache_hit: str
    cache_key: str
    request_tags: str  # json str
    team_id: str | None
    organization_id: str | None
    end_user: str | None
    requester_ip_address: str | None
    custom_llm_provider: str | None
    messages: str | list | dict | None
    response: str | list | dict | None
    proxy_server_request: str | None
    session_id: str | None
    request_duration_ms: int | None
    status: Literal["success", "failure"]


class SpanAttributes(str, enum.Enum):
    # Note: We've taken this from opentelemetry-semantic-conventions-ai
    # I chose to not add a new dependency to litellm for this

    # Semantic Conventions for LLM requests, this needs to be removed after
    # OpenTelemetry Semantic Conventions support Gen AI.
    # Issue at https://github.com/open-telemetry/opentelemetry-python/issues/3868
    # Refer to https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/llm-spans.md

    LLM_SYSTEM = "gen_ai.system"
    LLM_REQUEST_MODEL = "gen_ai.request.model"
    LLM_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    LLM_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    LLM_REQUEST_TOP_P = "gen_ai.request.top_p"
    LLM_PROMPTS = "gen_ai.prompt"
    LLM_COMPLETIONS = "gen_ai.completion"
    LLM_RESPONSE_MODEL = "gen_ai.response.model"
    LLM_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    LLM_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"

    # OTEL 1.38 attributes
    GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
    GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    LLM_TOKEN_TYPE = "gen_ai.token.type"
    # To be added
    # LLM_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
    # LLM_RESPONSE_ID = "gen_ai.response.id"

    # LLM
    LLM_REQUEST_TYPE = "llm.request.type"
    LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
    LLM_USAGE_TOKEN_TYPE = "llm.usage.token_type"
    LLM_USER = "llm.user"
    LLM_HEADERS = "llm.headers"
    LLM_TOP_K = "llm.top_k"
    LLM_IS_STREAMING = "llm.is_streaming"
    LLM_FREQUENCY_PENALTY = "llm.frequency_penalty"
    LLM_PRESENCE_PENALTY = "llm.presence_penalty"
    LLM_CHAT_STOP_SEQUENCES = "llm.chat.stop_sequences"
    LLM_REQUEST_FUNCTIONS = "llm.request.functions"
    LLM_REQUEST_REPETITION_PENALTY = "llm.request.repetition_penalty"
    LLM_RESPONSE_FINISH_REASON = "llm.response.finish_reason"
    LLM_RESPONSE_STOP_REASON = "llm.response.stop_reason"
    LLM_CONTENT_COMPLETION_CHUNK = "llm.content.completion.chunk"

    # OpenAI
    LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT = "gen_ai.openai.system_fingerprint"
    LLM_OPENAI_API_BASE = "gen_ai.openai.api_base"
    LLM_OPENAI_API_VERSION = "gen_ai.openai.api_version"
    LLM_OPENAI_API_TYPE = "gen_ai.openai.api_type"


class ManagementEndpointLoggingPayload(LiteLLMPydanticObjectBase):
    route: str
    request_data: dict
    response: dict | None = None
    exception: Any | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


class ProxyException(Exception):
    # NOTE: DO NOT MODIFY THIS
    # This is used to map exactly to OPENAI Exceptions
    def __init__(
        self,
        message: str,
        type: str,
        param: str | None,
        code: int | str | None = None,  # maps to status code
        headers: dict[str, str] | None = None,
        openai_code: str | None = None,  # maps to 'code'  in openai
        provider_specific_fields: dict | None = None,
    ):
        self.message = str(message)
        super().__init__(self.message)
        self.type = type
        self.param = param
        self.openai_code = openai_code or code
        # If we look on official python OpenAI lib, the code should be a string:
        # https://github.com/openai/openai-python/blob/195c05a64d39c87b2dfdf1eca2d339597f1fce03/src/openai/types/shared/error_object.py#L11
        # Related LiteLLM issue: https://github.com/BerriAI/litellm/discussions/4834
        self.code = str(code)
        if headers is not None:
            for k, v in headers.items():
                if not isinstance(v, str):
                    headers[k] = str(v)
        self.headers = headers or {}
        self.provider_specific_fields = provider_specific_fields
        # rules for proxyExceptions
        # Litellm router.py returns "No healthy deployment available" when there are no deployments available
        # Should map to 429 errors https://github.com/BerriAI/litellm/issues/2487
        if "No healthy deployment available" in self.message or "No deployments available" in self.message:
            self.code = "429"
        elif RouterErrors.no_deployments_with_tag_routing.value in self.message:
            self.code = "401"

    def to_dict(self) -> dict:
        """Converts the ProxyException instance to a dictionary."""
        error_dict: Final[dict[str, str | dict | None]] = {
            "message": self.message,
            "type": self.type,
            "param": self.param,
            "code": self.code,
        }
        if self.provider_specific_fields:
            error_dict["provider_specific_fields"] = self.provider_specific_fields
        return error_dict


class CommonProxyErrors(str, enum.Enum):
    db_not_connected_error = (
        "DB not connected. This endpoint needs a database; set DATABASE_URL to a "
        "PostgreSQL connection string (postgresql://...) to enable it. "
        "See https://docs.litellm.ai/docs/proxy/virtual_keys"
    )
    no_llm_router = "No models configured on proxy"
    not_allowed_access = "Admin-only endpoint. Not allowed to access this."
    not_premium_user = "You must be a LiteLLM Enterprise user to use this feature. If you have a license please set `LITELLM_LICENSE` in your env. Get a 7 day trial key here: https://www.litellm.ai/enterprise#trial. \nPricing: https://www.litellm.ai/#pricing"
    max_parallel_request_limit_reached = "Crossed TPM / RPM / Max Parallel Request Limit"
    missing_enterprise_package = "Missing litellm-enterprise package. Please install it to use this feature. Run `pip install litellm-enterprise`"
    missing_enterprise_package_docker = "This uses the enterprise folder - only available on the Docker image."


class SpendCalculateRequest(LiteLLMPydanticObjectBase):
    model: str | None = None
    messages: list | None = None
    completion_response: dict | None = None


class ProxyErrorTypes(str, enum.Enum):
    budget_exceeded = "budget_exceeded"
    """
    Object was over budget
    """
    no_db_connection = "no_db_connection"
    """
    No database connection
    """

    token_not_found_in_db = "token_not_found_in_db"
    """
    Requested token was not found in the database
    """

    key_model_access_denied = "key_model_access_denied"
    """
    Key does not have access to the model
    """

    team_model_access_denied = "team_model_access_denied"
    """
    Team does not have access to the model
    """

    user_model_access_denied = "user_model_access_denied"
    """
    User does not have access to the model
    """

    org_model_access_denied = "org_model_access_denied"
    """
    Organization does not have access to the model
    """

    project_model_access_denied = "project_model_access_denied"
    """
    Project does not have access to the model
    """

    model_cost_map_missing = "model_cost_map_missing"

    expired_key = "expired_key"
    """
    Key has expired
    """

    auth_error = "auth_error"
    """
    General authentication error
    """

    auth_provider_unavailable = "auth_provider_unavailable"
    """
    The identity provider needed to authenticate the request (e.g. its JWKS endpoint) is unreachable
    """

    internal_server_error = "internal_server_error"
    """
    Internal server error
    """

    bad_request_error = "bad_request_error"
    """
    Bad request error
    """

    not_found_error = "not_found_error"
    """
    Not found error
    """

    validation_error = "validation_error"
    """
    Validation error
    """

    cache_ping_error = "cache_ping_error"
    """
    Cache ping error
    """

    team_member_permission_error = "team_member_permission_error"
    """
    Team member permission error
    """

    key_vector_store_access_denied = "key_vector_store_access_denied"
    """
    Key does not have access to the vector store
    """

    team_vector_store_access_denied = "team_vector_store_access_denied"
    """
    Team does not have access to the vector store
    """

    org_vector_store_access_denied = "org_vector_store_access_denied"
    """
    Organization does not have access to the vector store
    """

    team_member_already_in_team = "team_member_already_in_team"
    """
    Team member is already in team
    """

    tool_access_denied = "tool_access_denied"
    """
    Tool is not in the allowed tools list for this key/team
    """

    @classmethod
    def get_model_access_error_type_for_object(
        cls, object_type: Literal["key", "user", "team", "org", "project"]
    ) -> "ProxyErrorTypes":
        """
        Get the model access error type for object_type
        """
        if object_type == "key":
            return cls.key_model_access_denied
        elif object_type == "team":
            return cls.team_model_access_denied
        elif object_type == "user":
            return cls.user_model_access_denied
        elif object_type == "org":
            return cls.org_model_access_denied
        elif object_type == "project":
            return cls.project_model_access_denied

    @classmethod
    def get_vector_store_access_error_type_for_object(
        cls, object_type: Literal["key", "team", "org"]
    ) -> "ProxyErrorTypes":
        """
        Get the vector store access error type for object_type
        """
        if object_type == "key":
            return cls.key_vector_store_access_denied
        elif object_type == "team":
            return cls.team_vector_store_access_denied
        elif object_type == "org":
            return cls.org_vector_store_access_denied


DB_CONNECTION_ERROR_TYPES: Final = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
)

# What a NON-IDEMPOTENT write (increment upsert) may retry: only ConnectError
# proves the statements never reached the database. Post-send errors are
# ambiguous; a stalled statement can leave its transaction open on the pooled
# connection, where a retry stacks a second increment set into the same commit.
# Idempotent writes (create_many with skip_duplicates) may retry the full tuple.
DB_RETRY_SAFE_ERROR_TYPES: Final = (httpx.ConnectError,)


class SSOUserDefinedValues(TypedDict):
    models: list[str]
    user_id: str
    user_email: str | None
    user_role: str | None
    max_budget: float | None
    budget_duration: str | None


class VirtualKeyEvent(LiteLLMPydanticObjectBase):
    created_by_user_id: str
    created_by_user_role: str
    created_by_key_alias: str | None
    request_kwargs: dict


class CreatePassThroughEndpoint(LiteLLMPydanticObjectBase):
    path: str
    target: str
    headers: dict


from litellm.models.team_membership import (  # noqa: E402
    LiteLLM_TeamMembership as LiteLLM_TeamMembership,
)

#### Organization / Team Member Requests ####


class MemberAddRequest(LiteLLMPydanticObjectBase):
    member: list[Member] | Member = Field(
        description="Member object or list of member objects to add. Each member must include either user_id or user_email, and a role"
    )

    def __init__(self, **data):
        member_data: Final = data.get("member")
        if isinstance(member_data, list):
            # If member is a list of dictionaries, convert each dictionary to a Member object
            members: Final = [Member(**item) if isinstance(item, dict) else item for item in member_data]
            # Replace member_data with the list of Member objects
            data["member"] = members
        elif isinstance(member_data, dict):
            # If member is a dictionary, convert it to a single Member object
            member: Final = Member(**member_data)
            # Replace member_data with the single Member object
            data["member"] = member
        # Call the superclass __init__ method to initialize the object
        super().__init__(**data)


class OrgMemberAddRequest(LiteLLMPydanticObjectBase):
    member: list[OrgMember] | OrgMember

    def __init__(self, **data):
        member_data: Final = data.get("member")
        if isinstance(member_data, list):
            # If member is a list of dictionaries, convert each dictionary to a Member object
            if all(isinstance(item, dict) for item in member_data):
                members = [OrgMember(**item) for item in member_data]
            else:
                members = [item for item in member_data]
            # Replace member_data with the list of Member objects
            data["member"] = members
        elif isinstance(member_data, dict):
            # If member is a dictionary, convert it to a single Member object
            member: Final = OrgMember(**member_data)
            # Replace member_data with the single Member object
            data["member"] = member
        # Call the superclass __init__ method to initialize the object
        super().__init__(**data)


class TeamAddMemberResponse(LiteLLM_TeamTable):
    updated_users: list[LiteLLM_UserTable]
    updated_team_memberships: list[LiteLLM_TeamMembership]


class OrganizationAddMemberResponse(LiteLLMPydanticObjectBase):
    organization_id: str
    updated_users: list[LiteLLM_UserTable]
    updated_organization_memberships: list[LiteLLM_OrganizationMembershipTable]


class MemberDeleteRequest(LiteLLMPydanticObjectBase):
    user_id: str | None = None
    user_email: str | None = None

    @model_validator(mode="before")
    @classmethod
    def check_user_info(cls, values):
        if values.get("user_id") is None and values.get("user_email") is None:
            raise ValueError("Either user id or user email must be provided")
        return values


class MemberUpdateResponse(LiteLLMPydanticObjectBase):
    user_id: str
    user_email: str | None = None


# Team Member Requests
class TeamMemberAddRequest(MemberAddRequest):
    """
    Request body for adding members to a team.

    Example:
    ```json
    {
        "team_id": "45e3e396-ee08-4a61-a88e-16b3ce7e0849",
        "member": {
            "role": "user",
            "user_id": "user123"
        },
        "max_budget_in_team": 100.0
    }
    ```
    """

    team_id: str = Field(description="The ID of the team to add the member to")
    max_budget_in_team: float | None = Field(
        default=None,
        description="Maximum budget allocated to this user within the team. If not set, user has unlimited budget within team limits",
    )
    budget_duration: str | None = Field(
        default=None,
        description="Duration after which this team member's budget resets (e.g. '1h', '24h', '7d', '30d'). If not set, the budget never resets.",
    )
    allowed_models: list[str] | None = Field(
        default=None,
        description="List of models this team member can access. If not set, inherits the team's default_team_member_models or all team models.",
    )


class TeamMemberDeleteRequest(MemberDeleteRequest):
    team_id: str


class TeamMemberUpdateRequest(TeamMemberDeleteRequest):
    max_budget_in_team: float | None = None
    role: Literal["admin", "user"] | None = None
    tpm_limit: int | None = Field(default=None, description="Tokens per minute limit for this team member")
    rpm_limit: int | None = Field(default=None, description="Requests per minute limit for this team member")
    budget_duration: str | None = Field(
        default=None,
        description="Duration after which this team member's budget resets (e.g. '1h', '24h', '7d', '30d'). If not set, the budget never resets.",
    )
    allowed_models: list[str] | None = Field(
        default=None,
        description="List of models this team member can access. Pass an empty list to remove per-member model restrictions.",
    )


class TeamMemberUpdateResponse(MemberUpdateResponse):
    team_id: str
    max_budget_in_team: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    allowed_models: list[str] | None = None


class TeamModelAddRequest(BaseModel):
    """Request to add models to a team"""

    team_id: str
    models: list[str]


class TeamModelDeleteRequest(BaseModel):
    """Request to delete models from a team"""

    team_id: str
    models: list[str]


# Organization Member Requests
class OrganizationMemberAddRequest(OrgMemberAddRequest):
    organization_id: str
    max_budget_in_organization: float | None = None  # Users max budget within the organization


class OrganizationMemberDeleteRequest(MemberDeleteRequest):
    organization_id: str


ROLES_WITHIN_ORG: Final = [
    LitellmUserRoles.ORG_ADMIN,
    LitellmUserRoles.INTERNAL_USER,
    LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
]


class OrganizationMemberUpdateRequest(OrganizationMemberDeleteRequest):
    max_budget_in_organization: float | None = None
    role: LitellmUserRoles | None = None

    @field_validator("role")
    def validate_role(cls, value: LitellmUserRoles | None) -> LitellmUserRoles | None:
        if value is not None and value not in ROLES_WITHIN_ORG:
            raise ValueError(f"Invalid role. Must be one of: {[role.value for role in ROLES_WITHIN_ORG]}")
        return value


class OrganizationMemberUpdateResponse(MemberUpdateResponse):
    organization_id: str
    max_budget_in_organization: float


##########################################


class TeamAccessGroupModelGrant(LiteLLMPydanticObjectBase):
    access_group_id: str
    access_group_name: str
    models: tuple[str, ...]


class TeamInfoResponseObjectTeamTable(LiteLLM_TeamTable):
    team_member_budget_table: LiteLLM_BudgetTableFull | None = None
    # Resources inherited from access groups (separate from direct assignments)
    access_group_models: list[str] | None = None
    access_group_mcp_server_ids: list[str] | None = None
    access_group_agent_ids: list[str] | None = None
    access_group_details: tuple[TeamAccessGroupModelGrant, ...] | None = None


class TeamInfoResponseObject(TypedDict):
    team_id: str
    team_info: TeamInfoResponseObjectTeamTable
    keys: list
    team_memberships: list[LiteLLM_TeamMembership]


class TeamListResponseObject(LiteLLM_TeamTable):
    team_memberships: list[LiteLLM_TeamMembership]
    keys: list  # list of keys that belong to the team


class KeyListResponseObject(TypedDict, total=False):
    keys: list[str | UserAPIKeyAuth | LiteLLM_DeletedVerificationToken]
    total_count: int | None
    current_page: int | None
    total_pages: int | None


class CurrentItemRateLimit(TypedDict):
    current_requests: int
    current_tpm: int
    current_rpm: int


class LoggingCallbackStatus(TypedDict, total=False):
    callbacks: list[str]
    status: Literal["healthy", "unhealthy"]
    details: str | None


class KeyHealthResponse(TypedDict, total=False):
    key: Literal["healthy", "unhealthy"]
    logging_callbacks: LoggingCallbackStatus | None


class CreateJWTKeyMappingRequest(LiteLLMPydanticObjectBase):
    jwt_claim_name: str
    jwt_claim_value: str
    key: str
    description: str | None = None


class UpdateJWTKeyMappingRequest(LiteLLMPydanticObjectBase):
    id: str
    key: str | None = None
    description: str | None = None
    is_active: bool | None = None


class DeleteJWTKeyMappingRequest(LiteLLMPydanticObjectBase):
    id: str


class JWTKeyMappingResponse(LiteLLMPydanticObjectBase):
    id: str
    jwt_claim_name: str
    jwt_claim_value: str
    description: str | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    created_by: str | None = None
    updated_by: str | None = None


class SpecialHeaders(enum.Enum):
    """Used by user_api_key_auth.py to get litellm key"""

    openai_authorization = "Authorization"
    azure_authorization = "API-Key"
    anthropic_authorization = "x-api-key"
    google_ai_studio_authorization = "x-goog-api-key"
    azure_apim_authorization = "Ocp-Apim-Subscription-Key"
    custom_litellm_api_key = "x-litellm-api-key"
    mcp_auth = "x-mcp-auth"
    mcp_servers = "x-mcp-servers"
    mcp_access_groups = "x-mcp-access-groups"

    @classmethod
    def litellm_credential_header_names(cls) -> "frozenset[str]":
        """Lowercased header names user_api_key_auth accepts as a litellm key.

        Every header here authenticates the caller, so any code that forwards a
        request onward (e.g. the plugin reverse proxy) must strip all of them to
        avoid leaking the caller's litellm credential downstream. The static
        custom-key header (general_settings.litellm_key_header_name) is runtime
        config and must be added on top of this set by the caller.
        """
        return frozenset(
            header.value.lower()
            for header in (
                cls.openai_authorization,
                cls.azure_authorization,
                cls.anthropic_authorization,
                cls.google_ai_studio_authorization,
                cls.azure_apim_authorization,
                cls.custom_litellm_api_key,
            )
        )


class LitellmDataForBackendLLMCall(TypedDict, total=False):
    headers: dict
    organization: str
    timeout: float | None
    stream_timeout: float | None
    user: str | None
    num_retries: int | None
    # True when the effective timeout came from a caller-controlled source (the
    # `x-litellm-timeout`/`x-litellm-stream-timeout` headers, or a `timeout`/`request_timeout`/
    # `stream_timeout` field in the request body) rather than deployment config, so a
    # deliberately tiny value isn't treated as a deployment health signal (see
    # cooldown_handlers._trigger_cooldown_for_failed_deployment).
    client_side_timeout: bool
    keepalive_seconds: float | None


class LitellmMetadataFromRequestHeaders(TypedDict, total=False):
    """
    Headers a user can pass that will get added to litellm metadata for the request
    """

    spend_logs_metadata: dict | None
    agent_id: str | None
    trace_id: str | None
    session_id: str | None


class JWTKeyItem(TypedDict, total=False):
    kid: str


JWKKeyValue = list[JWTKeyItem] | JWTKeyItem


class JWKUrlResponse(TypedDict, total=False):
    keys: JWKKeyValue


class UserManagementEndpointParamDocStringEnums(str, enum.Enum):
    user_id_doc_str = "Optional[str] - Specify a user id. If not set, a unique id will be generated."
    user_alias_doc_str = "Optional[str] - A descriptive name for you to know who this user id refers to."
    teams_doc_str = "Optional[list] - specify a list of team id's a user belongs to."
    user_email_doc_str = "Optional[str] - Specify a user email."
    send_invite_email_doc_str = "Optional[bool] - Specify if an invite email should be sent."
    user_role_doc_str = """Optional[str] - Specify a user role - "proxy_admin", "proxy_admin_viewer", "internal_user", "internal_user_viewer", "team", "customer". Info about each role here: `https://github.com/BerriAI/litellm/litellm/proxy/_types.py#L20`"""
    max_budget_doc_str = """Optional[float] - Specify max budget for a given user."""
    budget_duration_doc_str = """Optional[str] - Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d"), months ("1mo")."""
    models_doc_str = (
        """Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)"""
    )
    tpm_limit_doc_str = """Optional[int] - Specify tpm limit for a given user (Tokens per minute)"""
    rpm_limit_doc_str = """Optional[int] - Specify rpm limit for a given user (Requests per minute)"""
    auto_create_key_doc_str = """bool - Default=True. Flag used for returning a key as part of the /user/new response"""
    aliases_doc_str = """Optional[dict] - Model aliases for the user - [Docs](https://litellm.vercel.app/docs/proxy/virtual_keys#model-aliases)"""
    config_doc_str = """Optional[dict] - [DEPRECATED PARAM] User-specific config."""
    allowed_cache_controls_doc_str = """Optional[list] - List of allowed cache control values. Example - ["no-cache", "no-store"]. See all values - https://docs.litellm.ai/docs/proxy/caching#turn-on--off-caching-per-request-"""
    blocked_doc_str = """Optional[bool] - [Not Implemented Yet] Whether the user is blocked."""
    guardrails_doc_str = """Optional[List[str]] - [Not Implemented Yet] List of active guardrails for the user"""
    permissions_doc_str = (
        """Optional[dict] - [Not Implemented Yet] User-specific permissions, eg. turning off pii masking."""
    )
    metadata_doc_str = """Optional[dict] - Metadata for user, store information for user. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }"""
    max_parallel_requests_doc_str = """Optional[int] - Rate limit a user based on the number of parallel requests. Raises 429 error, if user's parallel requests > x."""
    soft_budget_doc_str = """Optional[float] - Get alerts when user crosses given budget, doesn't block requests."""
    model_max_budget_doc_str = """Optional[dict] - Model-specific max budget for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-budgets-to-keys)"""
    model_rpm_limit_doc_str = """Optional[float] - Model-specific rpm limit for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-limits-to-keys)"""
    model_tpm_limit_doc_str = """Optional[float] - Model-specific tpm limit for user. [Docs](https://docs.litellm.ai/docs/proxy/users#add-model-specific-limits-to-keys)"""
    spend_doc_str = (
        """Optional[float] - Amount spent by user. Default is 0. Will be updated by proxy whenever user is used."""
    )
    team_id_doc_str = """Optional[str] - [DEPRECATED PARAM] The team id of the user. Default is None."""
    duration_doc_str = """Optional[str] - Duration for the key auto-created on `/user/new`. Default is None."""


PassThroughEndpointLoggingResultValues = (
    ModelResponse
    | TextCompletionResponse
    | ImageResponse
    | EmbeddingResponse
    | VideoObject
    | StandardPassThroughResponseObject
    | ResponsesAPIResponse
)


class PassThroughEndpointLoggingTypedDict(TypedDict):
    result: PassThroughEndpointLoggingResultValues | None
    kwargs: dict


LiteLLM_ManagementEndpoint_MetadataFields: Final = [
    "model_rpm_limit",
    "model_tpm_limit",
    "model_itpm_limit",
    "model_otpm_limit",
    "default_estimated_output_tokens",
    "default_estimated_output_tokens_per_model",
    "mcp_rpm_limit",
    "tag_rpm_limit",
    "rpm_limit_type",
    "tpm_limit_type",
    "enforced_params",
    "temp_budget_increase",
    "temp_budget_expiry",
    "allowed_vector_store_indexes",
    "enforced_batch_output_expires_after",
    "enforced_file_expires_after",
    "throttle_on_budget_exceeded",
    "enable_prompt_caching",
]

LiteLLM_ManagementEndpoint_MetadataFields_Premium: Final = [
    "disable_global_guardrails",
    "guardrails",
    "policies",
    "tags",
    "team_member_key_duration",
    "prompts",
    "logging",
    "secret_manager_settings",
    "allowed_passthrough_routes",
]

# Metadata keys that are immutable once set: preserved when an update omits them,
# and rejected (400) when an update tries to change them.
LiteLLM_Reserved_Metadata_Fields: Final = [
    "service_account_id",
]


class ProviderBudgetResponseObject(LiteLLMPydanticObjectBase):
    """
    Configuration for a single provider's budget settings
    """

    budget_limit: float | None  # Budget limit in USD for the time period
    time_period: str | None  # Time period for budget (e.g., '1d', '30d', '1mo')
    spend: float | None = 0.0  # Current spend for this provider
    budget_reset_at: str | None = None  # When the current budget period resets


class ProviderBudgetResponse(LiteLLMPydanticObjectBase):
    """
    Complete provider budget configuration and status.
    Maps provider names to their budget configs.
    """

    providers: dict[
        str, ProviderBudgetResponseObject
    ] = {}  # Dictionary mapping provider names to their budget configurations


class ProxyStateVariables(TypedDict):
    """
    TypedDict for Proxy state variables.
    """

    spend_logs_row_count: int


UI_TEAM_ID = "litellm-dashboard"


class JWTAuthBuilderResult(TypedDict):
    is_proxy_admin: bool
    team_object: LiteLLM_TeamTable | None
    user_object: LiteLLM_UserTable | None
    end_user_object: LiteLLM_EndUserTable | None
    org_object: LiteLLM_OrganizationTable | None
    token: str
    team_id: str | None
    user_id: str | None
    user_email: str | None
    end_user_id: str | None
    org_id: str | None
    team_membership: LiteLLM_TeamMembership | None
    jwt_claims: dict  # Decoded JWT token claims (avoids re-decoding)


class ClientSideFallbackModel(TypedDict, total=False):
    """
    Dictionary passed when client configuring input
    """

    model: Required[str]
    messages: list[AllMessageValues]


ALL_FALLBACK_MODEL_VALUES = str | ClientSideFallbackModel


RBAC_ROLES = Literal[
    LitellmUserRoles.PROXY_ADMIN,
    LitellmUserRoles.TEAM,
    LitellmUserRoles.INTERNAL_USER,
]


class OIDCPermissions(LiteLLMPydanticObjectBase):
    models: list[str] | None = None
    routes: list[str] | None = None


class RoleBasedPermissions(OIDCPermissions):
    role: RBAC_ROLES

    model_config = {
        "extra": "forbid",
    }


class RoleMapping(BaseModel):
    role: str
    internal_role: RBAC_ROLES


class JWTLiteLLMRoleMap(BaseModel):
    jwt_role: str
    litellm_role: LitellmUserRoles


class ScopeMapping(OIDCPermissions):
    scope: str

    model_config = {
        "extra": "forbid",
    }


class JWTRoutingOverride(BaseModel):
    """
    Override default auth routing for JWT-shaped bearer tokens.

    A rule matches when all provided selectors match token claims.
    If matched, request is routed to the configured auth path.

    Wildcard selectors use shell-style patterns (* and ?) and are matched with
    case-sensitive semantics; use the same casing your IdP emits in JWT claims.
    Space-delimited tokenization applies only to the ``scope`` claim (OAuth/OIDC
    scope strings), not to ``iss``, ``aud``, or ``client_id``.
    """

    iss: str | list[str]
    client_id: str | list[str] | None = None
    scope: str | list[str] | None = None
    aud: str | list[str] | None = None
    path: Literal["oauth2"] = "oauth2"

    model_config = {
        "extra": "forbid",
    }


class UnregisteredJWTClientBehavior(str, enum.Enum):
    """
    Controls what happens when `virtual_key_claim_field` is configured but the
    JWT claim value has no registered mapping in `litellm_jwtkeymapping`.

    - fallback_team_mapping: Fall through to standard team-based JWT auth (default,
      backward-compatible).
    - reject: Immediately return HTTP 403. Use this when every valid JWT client
      must have a pre-registered virtual key — unknown callers are denied.
    - auto_register: Automatically create a new virtual key and mapping on first
      encounter. The new key has no budget/model restrictions; admins can tighten
      it later via /jwt_client/update.
    """

    FALLBACK_TEAM_MAPPING = "fallback_team_mapping"
    REJECT = "reject"
    AUTO_REGISTER = "auto_register"


class JWTIssuerConfig(BaseModel):
    """
    Issuer-bound JWT validation configuration.

    When a token's unverified `iss` claim matches an entry in
    ``LiteLLM_JWTAuth.issuers``, LiteLLM validates it only against that
    issuer's JWKS and audience. Tokens whose `iss` does not match any
    configured issuer fall back to the global JWT_AUDIENCE/JWT_ISSUER
    validation path; `issuers` is additive routing, not an allow-list.
    """

    issuer: str = Field(description="Exact expected JWT issuer (`iss`) value.")
    jwks_url: str | None = Field(
        default=None,
        description="Issuer JWKS URL. If omitted, LiteLLM uses the issuer's OIDC discovery document.",
    )
    audience: str | list[str] | None = Field(
        default=None,
        description="Expected token audience for this issuer.",
    )
    disable_audience_validation: bool = Field(
        default=False,
        description="Explicitly disable audience validation for this issuer. Use only when the issuer cannot provide an audience suitable for LiteLLM.",
    )
    user_id_jwt_field: str | None = Field(
        default=None,
        description="Issuer-specific claim path to normalize into LiteLLM's user id.",
    )
    user_email_jwt_field: str | None = Field(
        default=None,
        description="Issuer-specific claim path to normalize into LiteLLM's user email.",
    )
    team_id_jwt_field: str | None = Field(
        default=None,
        description="Issuer-specific claim path to normalize into LiteLLM's team id.",
    )
    team_ids_jwt_field: str | None = Field(
        default=None,
        description="Issuer-specific claim path to normalize into LiteLLM's team ids.",
    )
    org_id_jwt_field: str | None = Field(
        default=None,
        description="Issuer-specific claim path to normalize into LiteLLM's organization id.",
    )
    end_user_id_jwt_field: str | None = Field(
        default=None,
        description="Issuer-specific claim path to normalize into LiteLLM's end-user id.",
    )

    model_config = {
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def validate_audience_configured(self) -> "JWTIssuerConfig":
        if self.audience is None and not self.disable_audience_validation:
            raise ValueError(
                f"JWT issuer {self.issuer} must configure audience or set disable_audience_validation=True"
            )
        if self.audience is not None and self.disable_audience_validation:
            raise ValueError(
                f"JWT issuer {self.issuer} cannot set audience and disable_audience_validation=True together"
            )
        return self


DEFAULT_JWKS_STALE_TTL: Final = 3600


class LiteLLM_JWTAuth(LiteLLMPydanticObjectBase):
    """
    A class to define the roles and permissions for a LiteLLM Proxy w/ JWT Auth.

    Attributes:
    - admin_jwt_scope: The JWT scope required for proxy admin roles.
    - admin_allowed_routes: list of allowed routes for proxy admin roles.
    - team_jwt_scope: The JWT scope required for proxy team roles.
    - team_id_jwt_field: The field in the JWT token that stores the team ID. Default - `client_id`.
    - team_allowed_routes: list of allowed routes for proxy team roles.
    - user_id_jwt_field: The field in the JWT token that stores the user id (maps to `LiteLLMUserTable`). Use this for internal employees.
    - user_email_jwt_field: The field in the JWT token that stores the user email (maps to `LiteLLMUserTable`). Use this for internal employees.
    - user_allowed_email_subdomain: If specified, only emails from specified subdomain will be allowed to access proxy.
    - end_user_id_jwt_field: The field in the JWT token that stores the end-user ID (maps to `LiteLLMEndUserTable`). Turn this off by setting to `None`. Enables end-user cost tracking. Use this for external customers.
    - public_key_ttl: Default - 600s. TTL for caching public JWT keys.
    - public_key_stale_ttl: Default - 3600s. Extra time past `public_key_ttl` that the last-known-good JWKS response
        stays usable while the identity provider is unreachable. Set to 0 to fail closed instead.
    - public_allowed_routes: list of allowed routes for authenticated but unknown litellm role jwt tokens.
    - enforce_rbac: If true, enforce RBAC for all routes.
    - custom_validate: A custom function to validates the JWT token.
    - oidc_userinfo_endpoint: OIDC UserInfo endpoint URL. When set along with oidc_userinfo_enabled, LiteLLM will call this endpoint with the access token to retrieve user identity information.
    - oidc_userinfo_enabled: Enable fetching user info from OIDC UserInfo endpoint instead of just decoding JWT token. Default: False.
    - oidc_userinfo_cache_ttl: TTL (in seconds) for caching UserInfo responses. Default: 300s (5 minutes).

    See `auth_checks.py` for the specific routes
    """

    admin_jwt_scope: str = "litellm_proxy_admin"
    admin_allowed_routes: list[str] = [
        "management_routes",
        "spend_tracking_routes",
        "global_spend_tracking_routes",
        "info_routes",
    ]
    team_id_jwt_field: str | None = None
    team_id_upsert: bool = False
    team_ids_jwt_field: str | None = None
    upsert_sso_user_to_team: bool = False
    team_allowed_routes: list[str] = [
        "openai_routes",
        "info_routes",
        "mcp_routes",
        "/v1/messages",
        "/v1/messages/count_tokens",
    ]
    team_id_default: str | None = Field(
        default=None,
        description="If no team_id given, default permissions/spend-tracking to this team.s",
    )
    team_alias_jwt_field: str | None = Field(
        default=None,
        description="The field in the JWT token that stores the team name/alias. Will be resolved to team_id via database lookup.",
    )

    org_id_jwt_field: str | None = None
    org_alias_jwt_field: str | None = Field(
        default=None,
        description="The field in the JWT token that stores the organization name/alias. Will be resolved to org_id via database lookup.",
    )
    user_id_jwt_field: str | None = None
    user_email_jwt_field: str | None = None
    user_allowed_email_domain: str | None = None
    user_roles_jwt_field: str | None = None
    user_allowed_roles: list[str] | None = None
    user_id_upsert: bool = Field(default=False, description="If user doesn't exist, upsert them into the db.")
    end_user_id_jwt_field: str | None = None
    public_key_ttl: float = 600
    public_key_stale_ttl: float = Field(
        default=DEFAULT_JWKS_STALE_TTL,
        ge=0,
        description=(
            "Seconds beyond `public_key_ttl` that the last-known-good JWKS response stays usable while the identity "
            "provider is unreachable. Bounds how long a signing key the provider has since removed can still be "
            "trusted. Set to 0 to fail closed and reject requests as soon as the cached keys expire."
        ),
    )
    public_allowed_routes: list[str] = ["public_routes"]
    enforce_rbac: bool = False
    roles_jwt_field: str | None = None  # v2 on role mappings
    role_mappings: list[RoleMapping] | None = None
    object_id_jwt_field: str | None = None  # can be either user / team, inferred from the role mapping
    scope_mappings: list[ScopeMapping] | None = None
    enforce_scope_based_access: bool = False
    enforce_team_based_model_access: bool = False
    custom_validate: Callable[..., Literal[True]] | None = None
    #########################################################
    # Fields for syncing user team membership and roles with IDP provider
    jwt_litellm_role_map: list[JWTLiteLLMRoleMap] | None = None
    sync_user_role_and_teams: bool = False
    #########################################################
    #########################################################
    # OIDC UserInfo Endpoint Configuration
    oidc_userinfo_endpoint: str | None = Field(
        default=None,
        description="OIDC UserInfo endpoint URL. If set, LiteLLM will call this endpoint with the access token to retrieve user identity information.",
    )
    oidc_userinfo_enabled: bool = Field(
        default=False,
        description="Enable fetching user info from OIDC UserInfo endpoint instead of just decoding JWT token.",
    )
    oidc_userinfo_cache_ttl: float = Field(
        default=300,
        description="TTL (in seconds) for caching UserInfo responses. Default: 300s (5 minutes).",
    )
    # JWT-to-Virtual-Key Mapping
    virtual_key_claim_field: str | None = Field(
        default=None,
        description="JWT claim field for virtual key mapping lookup (e.g. 'sub', 'email'). Supports dot notation.",
    )
    virtual_key_mapping_cache_ttl: float = Field(
        default=300,
        description="TTL (seconds) for caching JWT-to-virtual-key mapping lookups.",
    )
    unregistered_jwt_client_behavior: UnregisteredJWTClientBehavior = Field(
        default=UnregisteredJWTClientBehavior.FALLBACK_TEAM_MAPPING,
        description=(
            "What to do when virtual_key_claim_field is set but the JWT claim value "
            "has no registered mapping. 'fallback_team_mapping' (default): fall through "
            "to team-based JWT auth. 'reject': return HTTP 403. "
            "'auto_register': auto-create a virtual key and mapping on first encounter."
        ),
    )
    routing_overrides: list[JWTRoutingOverride] | None = Field(
        default=None,
        description="Optional claim-based routing overrides for JWT-shaped tokens. Matching rules route requests to oauth2 before default JWT flow.",
    )
    team_claim_fallback: bool = Field(
        default=False,
        description=(
            "If True, when a configured team_id_jwt_field / team_ids_jwt_field "
            "claim is present but does not resolve to any known team, defer to "
            "the single-team DB fallback (caller's only team membership) "
            "instead of raising. Default False preserves strict claim-based "
            "authorization."
        ),
    )
    fallback_to_db_teams: bool = Field(
        default=False,
        description=(
            "When True, users whose JWT contains no team claims are authenticated "
            "using their database team memberships instead of receiving HTTP 403. "
            "Usage is attributed to the user's first resolvable DB team, or to the "
            "team specified via the x-litellm-team-id request header (validated "
            "against DB membership). Requires user_id_upsert=True so that user "
            "records exist before the fallback runs."
        ),
    )
    issuers: list[JWTIssuerConfig] | None = Field(
        default=None,
        description="Optional issuer-bound JWT validation rules. When a token's `iss` matches a configured issuer, validation uses that issuer's JWKS, audience, and claim mappings. Tokens with an unlisted `iss` fall back to the global JWT_AUDIENCE/JWT_ISSUER validation path — this is additive routing, not an allow-list.",
    )
    #########################################################

    def __init__(self, **kwargs: Any) -> None:
        # ``config_file_path`` is a non-field kwarg threaded by the
        # startup-load path so an operator-configured
        # ``custom_validate: s3://bucket/module.fn`` resolves through
        # the documented config-file flow. Pop before the invalid-keys
        # check; the runtime gate in ``get_instance_fn`` refuses
        # ``s3://`` / ``gcs://`` when this is None.
        config_file_path: Final = kwargs.pop("config_file_path", None)

        # Backward-compat: jwt_client_id_field was renamed to virtual_key_claim_field
        if "jwt_client_id_field" in kwargs:
            if "virtual_key_claim_field" not in kwargs:
                kwargs["virtual_key_claim_field"] = kwargs.pop("jwt_client_id_field")
            else:
                kwargs.pop("jwt_client_id_field")

        # get the attribute names for this Pydantic model
        allowed_keys: Final = LiteLLM_JWTAuth.__annotations__.keys()

        invalid_keys: Final = set(kwargs.keys()) - allowed_keys
        user_roles_jwt_field: Final = kwargs.get("user_roles_jwt_field")
        user_allowed_roles: Final = kwargs.get("user_allowed_roles")
        object_id_jwt_field: Final = kwargs.get("object_id_jwt_field")
        role_mappings: Final = kwargs.get("role_mappings")
        scope_mappings: Final = kwargs.get("scope_mappings")
        enforce_scope_based_access: Final = kwargs.get("enforce_scope_based_access")
        custom_validate: Final = kwargs.get("custom_validate")

        if custom_validate is not None:
            fn: Final = get_instance_fn(custom_validate, config_file_path=config_file_path)
            validate_custom_validate_return_type(fn)
            kwargs["custom_validate"] = fn

        if invalid_keys:
            raise ValueError(
                f"Invalid arguments provided: {', '.join(invalid_keys)}. Allowed arguments are: {', '.join(allowed_keys)}."
            )
        if (user_roles_jwt_field is not None and user_allowed_roles is None) or (
            user_roles_jwt_field is None and user_allowed_roles is not None
        ):
            raise ValueError("user_allowed_roles must be provided if user_roles_jwt_field is set.")

        if object_id_jwt_field is not None and role_mappings is None:
            raise ValueError(
                "if object_id_jwt_field is set, role_mappings must also be set. Needed to infer if the caller is a user or team."
            )

        if scope_mappings is not None and not enforce_scope_based_access:
            raise ValueError("scope_mappings must be set if enforce_scope_based_access is true.")

        super().__init__(**kwargs)


class PrismaCompatibleUpdateDBModel(TypedDict, total=False):
    model_name: str
    litellm_params: str
    model_info: str
    blocked: bool
    updated_at: str
    updated_by: str


class SpecialManagementEndpointEnums(enum.Enum):
    DEFAULT_ORGANIZATION = "default_organization"


class TransformRequestBody(BaseModel):
    call_type: CallTypes
    request_body: dict


class DefaultInternalUserParams(LiteLLMPydanticObjectBase):
    """
    Default parameters to apply when a new user signs in via SSO or is created on the /user/new API endpoint
    """

    user_role: (
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
        | None
    ) = Field(
        default=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        description="Default role assigned to new users created",
    )
    max_budget: float | None = Field(
        default=None,
        description="Default maximum budget (in USD) for new users created",
    )
    budget_duration: str | None = Field(
        default=None,
        description="Default budget duration for new users (e.g. 'daily', 'weekly', 'monthly')",
    )
    models: list[str] | None = Field(default=None, description="Default list of models that new users can access")

    teams: list[str] | list[NewUserRequestTeam] | None = Field(
        default=None,
        description="Default teams for new users created",
    )


class BaseDailySpendTransaction(TypedDict):
    date: str
    api_key: str
    model: str | None
    model_group: str | None
    mcp_namespaced_tool_name: str | None
    custom_llm_provider: str | None
    endpoint: str | None

    # token count metrics
    prompt_tokens: int
    completion_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    compression_saved_tokens: int

    # cost-savings metrics (dollars, priced per request before aggregation)
    compression_savings_spend: float
    prompt_caching_savings_spend: float
    # Not required: rows queued by a pod running the previous release, or replayed from
    # the Redis buffer across an upgrade, carry no such key. Every reader coalesces a
    # missing value to zero, so requiring it here would describe a shape the aggregation
    # is explicitly tested against.
    autorouter_savings_spend: NotRequired[float]

    # request level metrics
    spend: float
    api_requests: int
    successful_requests: int
    failed_requests: int


class DailyTeamSpendTransaction(BaseDailySpendTransaction):
    team_id: str


class DailyOrganizationSpendTransaction(BaseDailySpendTransaction):
    organization_id: str


class DailyUserSpendTransaction(BaseDailySpendTransaction):
    user_id: str


class DailyEndUserSpendTransaction(BaseDailySpendTransaction):
    end_user_id: str


class DailyTagSpendTransaction(BaseDailySpendTransaction):
    request_id: str | None
    tag: str


class DailyAgentSpendTransaction(BaseDailySpendTransaction):
    agent_id: str


class DBSpendUpdateTransactions(TypedDict):
    """
    Internal Data Structure for buffering spend updates in Redis or in memory before committing them to the database
    """

    user_list_transactions: dict[str, float] | None
    end_user_list_transactions: dict[str, float] | None
    key_list_transactions: dict[str, float] | None
    team_list_transactions: dict[str, float] | None
    team_member_list_transactions: dict[str, float] | None
    org_list_transactions: dict[str, float] | None
    tag_list_transactions: dict[str, float] | None
    agent_list_transactions: dict[str, float] | None


class SpendUpdateQueueItem(TypedDict, total=False):
    entity_type: Litellm_EntityType
    entity_id: str
    response_cost: float | None


class ToolDiscoveryQueueItem(TypedDict, total=False):
    tool_name: str
    origin: str | None  # MCP server name or "user_defined"
    created_by: str | None
    key_hash: str | None  # hash of virtual key that triggered discovery
    team_id: str | None  # team that triggered discovery
    key_alias: str | None  # human-readable key alias
    user_agent: str | None  # HTTP User-Agent of the caller


from litellm.models.managed_files import (  # noqa: E402
    LiteLLM_ManagedFileTable as LiteLLM_ManagedFileTable,
)
from litellm.models.managed_files import (  # noqa: E402
    LiteLLM_ManagedObjectTable as LiteLLM_ManagedObjectTable,
)
from litellm.models.managed_files import (  # noqa: E402
    LiteLLM_ManagedVectorStoresTable as LiteLLM_ManagedVectorStoresTable,
)
from litellm.models.managed_files import (  # noqa: E402
    LiteLLM_ManagedVectorStoreTable as LiteLLM_ManagedVectorStoreTable,
)


class EnterpriseLicenseData(TypedDict, total=False):
    expiration_date: str
    user_id: str
    allowed_features: list[str]
    max_users: int
    max_teams: int


class ResponseLiteLLM_ManagedVectorStore(TypedDict, total=False):
    vector_store: LiteLLM_ManagedVectorStoresTable


class CostEstimateRequest(LiteLLMPydanticObjectBase):
    """Request body for /cost/estimate endpoint."""

    model: str = Field(description="Model name (from /model_group/info)")
    input_tokens: int = Field(description="Expected input tokens per request", ge=0)
    output_tokens: int = Field(description="Expected output tokens per request", ge=0)
    num_requests_per_day: int | None = Field(default=None, description="Number of requests per day", ge=0)
    num_requests_per_month: int | None = Field(default=None, description="Number of requests per month", ge=0)


class CostEstimateResponse(LiteLLMPydanticObjectBase):
    """Response body for /cost/estimate endpoint."""

    model: str
    input_tokens: int
    output_tokens: int
    num_requests_per_day: int | None = None
    num_requests_per_month: int | None = None
    # Per-request costs
    cost_per_request: float = Field(description="Total cost per request (includes margin)")
    input_cost_per_request: float = Field(description="Input token cost per request (before margin)")
    output_cost_per_request: float = Field(description="Output token cost per request (before margin)")
    margin_cost_per_request: float = Field(default=0.0, description="Margin/fee added per request")
    # Daily costs (if num_requests_per_day provided)
    daily_cost: float | None = Field(default=None, description="Total daily cost (includes margin)")
    daily_input_cost: float | None = Field(default=None, description="Daily input token cost")
    daily_output_cost: float | None = Field(default=None, description="Daily output token cost")
    daily_margin_cost: float | None = Field(default=None, description="Daily margin/fee")
    # Monthly costs (if num_requests_per_month provided)
    monthly_cost: float | None = Field(default=None, description="Total monthly cost (includes margin)")
    monthly_input_cost: float | None = Field(default=None, description="Monthly input token cost")
    monthly_output_cost: float | None = Field(default=None, description="Monthly output token cost")
    monthly_margin_cost: float | None = Field(default=None, description="Monthly margin/fee")
    # Pricing info
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    provider: str | None = None
