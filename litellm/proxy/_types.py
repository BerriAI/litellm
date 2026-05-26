import enum
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Json,
    field_validator,
    model_validator,
)
from typing_extensions import Required, TypedDict

from litellm._uuid import uuid
from litellm.constants import MCP_STDIO_ALLOWED_COMMANDS
from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    validate_no_callback_env_reference,
)
from litellm.types.integrations.slack_alerting import AlertType
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIFileObject,
    ResponsesAPIResponse,
)
from litellm.types.mcp import (
    MCPAuthType,
    MCPCredentials,
    MCPTransport,
    MCPTransportType,
)
from litellm.types.mcp_server.mcp_server_manager import MCPInfo
from litellm.types.router import RouterErrors, UpdateRouterConfig
from litellm.types.secret_managers.main import KeyManagementSystem
from litellm.types.utils import (
    CallTypes,
    CostBreakdown,
    EmbeddingResponse,
    GenericBudgetConfigType,
    ImageResponse,
    LiteLLMBatch,
    LiteLLMFineTuningJob,
    LiteLLMPydanticObjectBase,
    ModelResponse,
    ProviderField,
    StandardCallbackDynamicParams,
    StandardLoggingGuardrailInformation,
    StandardLoggingMCPToolCall,
    StandardLoggingModelInformation,
    StandardLoggingPayloadErrorInformation,
    StandardLoggingPayloadStatus,
    StandardLoggingVectorStoreRequest,
    StandardPassThroughResponseObject,
    TextCompletionResponse,
)
from litellm.types.videos.main import VideoObject

from .types_utils.utils import get_instance_fn, validate_custom_validate_return_type

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


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

    def values(self) -> List[str]:
        return list(self.__annotations__.keys())

    @property
    def description(self):
        """
        Descriptions for the enum values
        """
        descriptions = {
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
        ui_labels = {
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
    hashed_token = hashlib.sha256(token.encode()).hexdigest()

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

    # info and health routes
    KEY_INFO = "/key/info"
    KEY_HEALTH = "/key/health"

    # list routes
    KEY_LIST = "/key/list"
    KEY_ALIASES = "/key/aliases"

    # team usage routes
    TEAM_DAILY_ACTIVITY = "/team/daily/activity"

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
        # responses API
        "/responses",
        "/v1/responses",
        "/responses/{response_id}",
        "/v1/responses/{response_id}",
        "/responses/{response_id}/input_items",
        "/v1/responses/{response_id}/input_items",
        "/responses/{response_id}/cancel",
        "/v1/responses/{response_id}/cancel",
        # vector stores
        "/vector_stores",
        "/v1/vector_stores",
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
    ]

    # MCP server CRUD routes — control-plane. Gated by DISABLE_ADMIN_ENDPOINTS.
    mcp_management_routes = [
        "/v1/mcp/server",
        "/v1/mcp/server/{path:path}",
    ]

    # Backwards-compat union — virtual keys may be configured with
    # allowed_routes=["mcp_routes"], which should cover both halves.
    mcp_routes = mcp_inference_routes + mcp_management_routes

    agent_routes = [
        "/v1/agents",
        "/v1/agents/{agent_id}",
        "/agents",
        "/a2a/{agent_id}",
        "/a2a/{agent_id}/message/send",
        "/a2a/{agent_id}/message/stream",
        "/a2a/{agent_id}/.well-known/agent-card.json",
    ]

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

    llm_api_routes = (
        openai_routes
        + anthropic_routes
        + google_routes
        + mapped_pass_through_routes
        + passthrough_routes_wildcard
        + apply_guardrail_routes
        + mcp_inference_routes
        + litellm_native_routes
        + agent_routes
    )
    info_routes = [
        "/key/info",
        "/key/health",
        "/team/info",
        "/team/list",
        "/v2/team/list",
        "/organization/list",
        "/team/available",
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
        KeyManagementRoutes.SPEND_LOGS.value,
        KeyManagementRoutes.SPEND_LOGS_V2.value,
        KeyManagementRoutes.KEY_RESET_SPEND.value,
        KeyManagementRoutes.KEY_ALIASES.value,
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
            "/team/delete",
            "/team/list",
            "/v2/team/list",
            "/team/info",
            "/team/block",
            "/team/unblock",
            "/team/available",
            "/team/permissions_list",
            "/team/permissions_update",
            "/team/permissions_bulk_update",
            "/team/daily/activity",
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

    public_routes = set(
        [
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
        ]
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
        "/team/permissions_list",
        "/team/permissions_update",
        "/team/daily/activity",
        "/team/{team_id}/members/me",
        "/model/new",
        "/model/update",
        "/model/delete",
        "/user/daily/activity",
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
            # UI Logs page detail drawer (single + session). The list endpoint
            # `/spend/logs/ui` is covered via spend_tracking_routes below.
            "/spend/logs/ui/{logId}",
            "/spend/logs/session/ui",
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
    org_admin_allowed_routes = (
        org_admin_only_routes
        + management_routes
        + self_managed_routes
        + admin_viewer_routes
    )


class LiteLLMPromptInjectionParams(LiteLLMPydanticObjectBase):
    heuristics_check: bool = False
    vector_db_check: bool = False
    llm_api_check: bool = False
    llm_api_name: Optional[str] = None
    llm_api_system_prompt: Optional[str] = None
    llm_api_fail_call_string: Optional[str] = None
    reject_as_response: Optional[bool] = Field(
        default=False,
        description="Return rejected request error message as a string to the user. Default behaviour is to raise an exception.",
    )

    @model_validator(mode="before")
    @classmethod
    def check_llm_api_params(cls, values):
        llm_api_check = values.get("llm_api_check")
        if llm_api_check is True:
            if "llm_api_name" not in values or not values["llm_api_name"]:
                raise ValueError(
                    "If llm_api_check is set to True, llm_api_name must be provided"
                )
            if (
                "llm_api_system_prompt" not in values
                or not values["llm_api_system_prompt"]
            ):
                raise ValueError(
                    "If llm_api_check is set to True, llm_api_system_prompt must be provided"
                )
            if (
                "llm_api_fail_call_string" not in values
                or not values["llm_api_fail_call_string"]
            ):
                raise ValueError(
                    "If llm_api_check is set to True, llm_api_fail_call_string must be provided"
                )
        return values


######### Request Class Definition ######
class ProxyChatCompletionRequest(LiteLLMPydanticObjectBase):
    """
    Pydantic model for chat completion requests that includes both OpenAI standard fields
    and LiteLLM-specific parameters. This replaces the previous TypedDict version.
    """

    # Required fields (from ChatCompletionRequest)
    model: str
    messages: List[AllMessageValues]

    # Standard OpenAI completion parameters (all optional)
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    service_tier: Optional[str] = None
    stop: Optional[Union[str, List[str]]] = None
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    user: Optional[str] = None
    stream: Optional[bool] = None

    # LiteLLM-specific metadata param (from original ChatCompletionRequest)
    metadata: Optional[Dict[str, Any]] = None

    # Optional LiteLLM params
    guardrails: Optional[List[str]] = None
    caching: Optional[bool] = None
    num_retries: Optional[int] = None
    context_window_fallback_dict: Optional[Dict[str, str]] = None
    fallbacks: Optional[List[str]] = None


class ModelInfoDelete(LiteLLMPydanticObjectBase):
    id: str


class ModelInfo(LiteLLMPydanticObjectBase):
    id: Optional[str]
    mode: Optional[Literal["embedding", "chat", "completion"]]
    input_cost_per_token: Optional[float] = 0.0
    output_cost_per_token: Optional[float] = 0.0
    max_tokens: Optional[int] = 2048  # assume 2048 if not set

    # for azure models we need users to specify the base model, one azure you can call deployments - azure/my-random-model
    # we look up the base model in model_prices_and_context_window.json
    base_model: Optional[
        Literal[
            "gpt-4-1106-preview",
            "gpt-4-32k",
            "gpt-4",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo",
            "text-embedding-ada-002",
        ]
    ]

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
    fields: List[ProviderField]


class BlockUsers(LiteLLMPydanticObjectBase):
    user_ids: List[str]  # required


class ModelParams(LiteLLMPydanticObjectBase):
    model_name: str
    litellm_params: dict
    model_info: ModelInfo

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        if values.get("model_info") is None:
            values.update(
                {"model_info": ModelInfo(id=None, mode="chat", base_model=None)}
            )
        return values


class LiteLLM_ObjectPermissionBase(LiteLLMPydanticObjectBase):
    mcp_servers: Optional[List[str]] = None
    mcp_access_groups: Optional[List[str]] = None
    mcp_tool_permissions: Optional[Dict[str, List[str]]] = None
    mcp_toolsets: Optional[List[str]] = None
    blocked_tools: Optional[List[str]] = None
    vector_stores: Optional[List[str]] = None
    agents: Optional[List[str]] = None
    agent_access_groups: Optional[List[str]] = None
    models: Optional[List[str]] = None
    search_tools: Optional[List[str]] = None


class BudgetLimitEntry(LiteLLMPydanticObjectBase):
    """A single budget window with its own limit and independent reset schedule."""

    budget_duration: str  # e.g. "24h", "7d", "30d"
    max_budget: float  # max spend in USD for this window
    reset_at: Optional[datetime] = None  # populated at creation/reset time


class GenerateRequestBase(LiteLLMPydanticObjectBase):
    """
    Overlapping schema between key and user generate/update requests
    """

    key_alias: Optional[str] = None
    duration: Optional[str] = None
    models: Optional[list] = []
    spend: Optional[float] = 0
    max_budget: Optional[float] = None
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    agent_id: Optional[str] = None
    max_parallel_requests: Optional[int] = None
    metadata: Optional[dict] = {}
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None

    budget_duration: Optional[str] = None
    budget_limits: Optional[List[BudgetLimitEntry]] = (
        None  # multiple concurrent budget windows
    )
    allowed_cache_controls: Optional[list] = []
    config: Optional[dict] = {}
    permissions: Optional[dict] = {}
    model_max_budget: Optional[dict] = (
        {}
    )  # {"gpt-4": 5.0, "gpt-3.5-turbo": 5.0}, defaults to {}

    model_config = ConfigDict(protected_namespaces=())
    model_rpm_limit: Optional[dict] = None
    model_tpm_limit: Optional[dict] = None
    guardrails: Optional[List[str]] = None
    policies: Optional[List[str]] = None
    prompts: Optional[List[str]] = None
    blocked: Optional[bool] = None
    aliases: Optional[dict] = {}
    object_permission: Optional[LiteLLM_ObjectPermissionBase] = None

    @field_validator("max_budget", mode="before")
    @classmethod
    def check_max_budget(cls, v):
        if v == "":
            return None
        return v


class AllowedVectorStoreIndexItem(LiteLLMPydanticObjectBase):
    index_name: str
    index_permissions: List[Literal["read", "write"]]


class KeyRequestBase(GenerateRequestBase):
    key: Optional[str] = None
    budget_id: Optional[str] = None
    tags: Optional[List[str]] = None
    enforced_params: Optional[List[str]] = None
    allowed_routes: Optional[list] = []
    allowed_passthrough_routes: Optional[list] = None
    allowed_vector_store_indexes: Optional[List[AllowedVectorStoreIndexItem]] = None
    rpm_limit_type: Optional[
        Literal["guaranteed_throughput", "best_effort_throughput", "dynamic"]
    ] = None  # raise an error if 'guaranteed_throughput' is set and we're overallocating rpm
    tpm_limit_type: Optional[
        Literal["guaranteed_throughput", "best_effort_throughput", "dynamic"]
    ] = None  # raise an error if 'guaranteed_throughput' is set and we're overallocating tpm
    router_settings: Optional[UpdateRouterConfig] = None
    access_group_ids: Optional[List[str]] = None


class LiteLLMKeyType(str, enum.Enum):
    """
    Enum for key types that determine what routes a key can access
    """

    LLM_API = "llm_api"  # Can call LLM API routes (chat/completions, embeddings, etc.)
    MANAGEMENT = "management"  # Can call management routes (user/team/key management)
    READ_ONLY = "read_only"  # Can only call info/read routes
    DEFAULT = "default"  # Uses default allowed routes


class GenerateKeyRequest(KeyRequestBase):
    soft_budget: Optional[float] = None
    send_invite_email: Optional[bool] = None
    key_type: Optional[LiteLLMKeyType] = Field(
        default=LiteLLMKeyType.DEFAULT,
        description="Type of key that determines default allowed routes.",
    )
    auto_rotate: Optional[bool] = Field(
        default=False, description="Whether this key should be automatically rotated"
    )
    rotation_interval: Optional[str] = Field(
        default=None,
        description="How often to rotate this key (e.g., '30d', '90d'). Required if auto_rotate=True",
    )
    organization_id: Optional[str] = None
    project_id: Optional[str] = None


class GenerateKeyResponse(KeyRequestBase):
    key: str  # type: ignore
    key_name: Optional[str] = None
    expires: Optional[datetime] = None
    user_id: Optional[str] = None
    token_id: Optional[str] = None
    organization_id: Optional[str] = None
    project_id: Optional[str] = None
    litellm_budget_table: Optional[Any] = None
    token: Optional[str] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        if values.get("token") is not None:
            values.update({"key": values.get("token")})
        dict_fields = [
            "metadata",
            "aliases",
            "config",
            "permissions",
            "model_max_budget",
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
    key: str  # type: ignore
    duration: Optional[str] = None
    spend: Optional[float] = None
    metadata: Optional[dict] = None
    temp_budget_increase: Optional[float] = None
    temp_budget_expiry: Optional[datetime] = None
    auto_rotate: Optional[bool] = None
    rotation_interval: Optional[str] = None
    organization_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_temp_budget(self) -> "UpdateKeyRequest":
        if self.temp_budget_increase is not None or self.temp_budget_expiry is not None:
            if self.temp_budget_increase is None or self.temp_budget_expiry is None:
                raise ValueError(
                    "temp_budget_increase and temp_budget_expiry must be set together"
                )
        return self


class RegenerateKeyRequest(GenerateKeyRequest):
    # This needs to be different from UpdateKeyRequest, because "key" is optional for this
    key: Optional[str] = None
    new_key: Optional[str] = None
    duration: Optional[str] = None
    spend: Optional[float] = None
    metadata: Optional[dict] = None
    new_master_key: Optional[str] = None
    grace_period: Optional[str] = (
        None  # Duration to keep old key valid (e.g. "24h", "2d"); None = immediate revoke
    )


class ResetSpendRequest(LiteLLMPydanticObjectBase):
    reset_to: float


class KeyRequest(LiteLLMPydanticObjectBase):
    keys: Optional[List[str]] = None
    key_aliases: Optional[List[str]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_at_least_one(cls, values):
        if not values.get("keys") and not values.get("key_aliases"):
            raise ValueError(
                "At least one of 'keys' or 'key_aliases' must be provided."
            )
        return values


class LiteLLM_ModelTable(LiteLLMPydanticObjectBase):
    id: Optional[int] = None
    model_aliases: Optional[Union[str, dict]] = None  # json dump the dict
    created_by: str
    updated_by: str
    team: Optional["LiteLLM_TeamTable"] = None

    model_config = ConfigDict(protected_namespaces=())


class LiteLLM_ProxyModelTable(LiteLLMPydanticObjectBase):
    model_id: str
    model_name: str
    litellm_params: dict
    model_info: dict
    created_at: Optional[datetime] = None
    created_by: str
    updated_at: Optional[datetime] = None
    updated_by: str

    @model_validator(mode="before")
    @classmethod
    def check_potential_json_str(cls, values):
        if isinstance(values.get("litellm_params"), str):
            try:
                values["litellm_params"] = json.loads(values["litellm_params"])
            except json.JSONDecodeError:
                pass
        if isinstance(values.get("model_info"), str):
            try:
                values["model_info"] = json.loads(values["model_info"])
            except json.JSONDecodeError:
                pass
        return values


# MCP Types
class SpecialMCPServerName(str, enum.Enum):
    all_team_servers = "all-team-mcpservers"
    all_proxy_servers = "all-proxy-mcpservers"


class MCPApprovalStatus(str, enum.Enum):
    pending_review = "pending_review"
    active = "active"
    rejected = "rejected"


# MCP Proxy Request Types
class NewMCPServerRequest(LiteLLMPydanticObjectBase):
    server_id: Optional[str] = None
    server_name: Optional[str] = None
    alias: Optional[str] = None
    description: Optional[str] = None
    transport: MCPTransportType = MCPTransport.sse
    auth_type: Optional[MCPAuthType] = None
    credentials: Optional[MCPCredentials] = None
    url: Optional[str] = None
    spec_path: Optional[str] = None
    mcp_info: Optional[MCPInfo] = None
    mcp_access_groups: List[str] = Field(default_factory=list)
    allowed_tools: Optional[List[str]] = None
    tool_name_to_display_name: Optional[Dict[str, str]] = None
    tool_name_to_description: Optional[Dict[str, str]] = None
    extra_headers: Optional[List[str]] = None
    static_headers: Optional[Dict[str, str]] = None
    instructions: Optional[str] = None
    # Stdio-specific fields
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    registration_url: Optional[str] = None
    oauth2_flow: Optional[Literal["client_credentials", "authorization_code"]] = None
    allow_all_keys: bool = False
    available_on_public_internet: bool = True
    delegate_auth_to_upstream: bool = False
    is_byok: bool = False
    byok_description: List[str] = Field(default_factory=list)
    byok_api_key_help_url: Optional[str] = None
    source_url: Optional[str] = None
    # BYOM submission fields — set by the endpoint, not by the caller.
    # Any caller-provided values are silently overridden before persistence.
    approval_status: Optional[str] = Field(
        None,
        description="Server-managed: set by the endpoint; caller values are overridden.",
    )
    submitted_by: Optional[str] = Field(
        None,
        description="Server-managed: set by the endpoint; caller values are overridden.",
    )
    submitted_at: Optional[datetime] = Field(
        None,
        description="Server-managed: set by the endpoint; caller values are overridden.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_transport_fields(cls, values):
        if isinstance(values, dict):
            transport = values.get("transport")
            if transport == MCPTransport.stdio:
                if not values.get("command"):
                    raise ValueError("command is required for stdio transport")
                if not values.get("args"):
                    raise ValueError("args is required for stdio transport")
                # Validate command against allowlist to prevent arbitrary execution
                base_command = os.path.basename(values["command"])
                if base_command not in MCP_STDIO_ALLOWED_COMMANDS:
                    raise ValueError(
                        f"Command '{values['command']}' is not in the allowed commands list "
                        f"for stdio transport. Allowed commands: {sorted(MCP_STDIO_ALLOWED_COMMANDS)}"
                    )
            elif transport in [MCPTransport.http, MCPTransport.sse]:
                if not values.get("url") and not values.get("spec_path"):
                    raise ValueError(
                        "url or spec_path is required for HTTP/SSE transport"
                    )
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


class UpdateMCPServerRequest(LiteLLMPydanticObjectBase):
    server_id: str
    server_name: Optional[str] = None
    alias: Optional[str] = None
    description: Optional[str] = None
    transport: MCPTransportType = MCPTransport.sse
    auth_type: Optional[MCPAuthType] = None
    credentials: Optional[MCPCredentials] = None
    url: Optional[str] = None
    spec_path: Optional[str] = None
    mcp_info: Optional[MCPInfo] = None
    mcp_access_groups: List[str] = Field(default_factory=list)
    allowed_tools: Optional[List[str]] = None
    tool_name_to_display_name: Optional[Dict[str, str]] = None
    tool_name_to_description: Optional[Dict[str, str]] = None
    extra_headers: Optional[List[str]] = None
    static_headers: Optional[Dict[str, str]] = None
    instructions: Optional[str] = None
    # Stdio-specific fields
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    registration_url: Optional[str] = None
    allow_all_keys: bool = False
    available_on_public_internet: bool = True
    delegate_auth_to_upstream: bool = False
    is_byok: bool = False
    byok_description: List[str] = Field(default_factory=list)
    byok_api_key_help_url: Optional[str] = None
    source_url: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_transport_fields(cls, values):
        if isinstance(values, dict):
            transport = values.get("transport")
            if transport == MCPTransport.stdio:
                if not values.get("command"):
                    raise ValueError("command is required for stdio transport")
                if not values.get("args"):
                    raise ValueError("args is required for stdio transport")
                # Validate command against allowlist to prevent arbitrary execution
                base_command = os.path.basename(values["command"])
                if base_command not in MCP_STDIO_ALLOWED_COMMANDS:
                    raise ValueError(
                        f"Command '{values['command']}' is not in the allowed commands list "
                        f"for stdio transport. Allowed commands: {sorted(MCP_STDIO_ALLOWED_COMMANDS)}"
                    )
            elif transport in [MCPTransport.http, MCPTransport.sse]:
                if not values.get("url") and not values.get("spec_path"):
                    raise ValueError(
                        "url or spec_path is required for HTTP/SSE transport"
                    )
        return values


class LiteLLM_MCPServerTable(LiteLLMPydanticObjectBase):
    """Represents a LiteLLM_MCPServerTable record"""

    server_id: str
    server_name: Optional[str] = None
    alias: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    spec_path: Optional[str] = None
    transport: MCPTransportType
    auth_type: Optional[MCPAuthType] = None
    credentials: Optional[MCPCredentials] = None
    instructions: Optional[str] = None
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None
    teams: List[Dict[str, Optional[str]]] = Field(default_factory=list)
    mcp_access_groups: List[str] = Field(default_factory=list)
    allowed_tools: List[str] = Field(default_factory=list)
    tool_name_to_display_name: Optional[Dict[str, str]] = None
    tool_name_to_description: Optional[Dict[str, str]] = None
    extra_headers: List[str] = Field(default_factory=list)
    mcp_info: Optional[MCPInfo] = None
    static_headers: Optional[Dict[str, str]] = None
    # Health check status
    status: Optional[Literal["healthy", "unhealthy", "unknown"]] = Field(
        default="unknown",
        description="Health status: 'healthy', 'unhealthy', 'unknown'",
    )
    last_health_check: Optional[datetime] = None
    health_check_error: Optional[str] = None
    # Stdio-specific fields
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    registration_url: Optional[str] = None
    allow_all_keys: bool = False
    available_on_public_internet: bool = True
    delegate_auth_to_upstream: bool = False
    is_byok: bool = False
    byok_description: List[str] = Field(default_factory=list)
    byok_api_key_help_url: Optional[str] = None
    has_user_credential: Optional[bool] = None
    source_url: Optional[str] = None
    # BYOM submission fields
    approval_status: Optional[str] = Field(
        default="active",
        description="Approval status: 'pending_review', 'active', 'rejected'",
    )
    submitted_by: Optional[str] = None
    submitted_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None


class MakeMCPServersPublicRequest(LiteLLMPydanticObjectBase):
    mcp_server_ids: List[str]


class MCPUserCredentialRequest(LiteLLMPydanticObjectBase):
    credential: str
    save: bool = True


class MCPUserCredentialResponse(LiteLLMPydanticObjectBase):
    server_id: str
    has_credential: bool


class MCPOAuthUserCredentialRequest(LiteLLMPydanticObjectBase):
    """Stores a user's OAuth2 token for an OpenAPI MCP server."""

    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None  # seconds until expiry
    scopes: Optional[List[str]] = None


class MCPOAuthUserCredentialStatus(LiteLLMPydanticObjectBase):
    """Describes whether the calling user has a stored OAuth credential."""

    server_id: str
    has_credential: bool
    expires_at: Optional[str] = None  # ISO-8601
    is_expired: bool = False
    connected_at: Optional[str] = None  # ISO-8601


class MCPUserCredentialListItem(LiteLLMPydanticObjectBase):
    """One entry in the /user-credentials list."""

    server_id: str
    server_name: Optional[str] = None
    alias: Optional[str] = None
    credential_type: str  # "oauth2" or "byok"
    has_credential: bool
    expires_at: Optional[str] = None  # ISO-8601; None means non-expiring
    connected_at: Optional[str] = None  # ISO-8601


class RejectMCPServerRequest(LiteLLMPydanticObjectBase):
    review_notes: Optional[str] = None


class MCPSubmissionsSummary(LiteLLMPydanticObjectBase):
    total: int
    pending_review: int
    active: int
    rejected: int
    items: List["LiteLLM_MCPServerTable"]


######## Skills API Types ########


class NewSkillRequest(LiteLLMPydanticObjectBase):
    """Request to create a new skill in LiteLLM database"""

    display_title: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    file_content: Optional[bytes] = None  # Binary content of skill files (zip)
    file_name: Optional[str] = None  # Original filename
    file_type: Optional[str] = None  # MIME type (e.g., "application/zip")
    metadata: Optional[Dict[str, Any]] = None
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    registration_url: Optional[str] = None


class UpdateSkillRequest(LiteLLMPydanticObjectBase):
    """Request to update an existing skill"""

    skill_id: str
    display_title: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    file_content: Optional[bytes] = None  # Binary content of skill files (zip)
    file_name: Optional[str] = None  # Original filename
    file_type: Optional[str] = None  # MIME type
    metadata: Optional[Dict[str, Any]] = None


class LiteLLM_SkillsTable(LiteLLMPydanticObjectBase):
    """Represents a LiteLLM_SkillsTable record"""

    skill_id: str
    display_title: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    source: str = "custom"
    latest_version: Optional[str] = None
    file_content: Optional[bytes] = None  # Binary content of skill files (zip)
    file_name: Optional[str] = None  # Original filename
    file_type: Optional[str] = None  # MIME type
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class ListSkillsRequest(LiteLLMPydanticObjectBase):
    """Request to list skills from LiteLLM database"""

    limit: Optional[int] = 20
    offset: Optional[int] = 0


class NewUserRequestTeam(LiteLLMPydanticObjectBase):
    team_id: str
    max_budget_in_team: Optional[float] = None
    user_role: Literal["user", "admin"] = "user"


class NewUserRequest(GenerateRequestBase):
    max_budget: Optional[float] = None
    user_email: Optional[str] = None
    user_alias: Optional[str] = None
    user_role: Optional[
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
    ] = None
    teams: Optional[Union[List[str], List[NewUserRequestTeam]]] = None
    auto_create_key: bool = (
        True  # flag used for returning a key as part of the /user/new response
    )
    send_invite_email: Optional[bool] = None
    sso_user_id: Optional[str] = None
    organizations: Optional[List[str]] = None


class NewUserResponse(GenerateKeyResponse):
    max_budget: Optional[float] = None
    user_email: Optional[str] = None
    user_role: Optional[
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
    ] = None
    teams: Optional[list] = None
    user_alias: Optional[str] = None
    model_max_budget: Optional[dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UpdateUserRequestNoUserIDorEmail(
    GenerateRequestBase
):  # shared with BulkUpdateUserRequest
    password: Optional[str] = None
    spend: Optional[float] = None
    metadata: Optional[dict] = None
    user_alias: Optional[str] = None
    user_role: Optional[
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
    ] = None
    max_budget: Optional[float] = None


class UpdateUserRequest(UpdateUserRequestNoUserIDorEmail):
    # Note: the defaults of all Params here MUST BE NONE
    # else they will get overwritten
    user_id: Optional[str] = None
    user_email: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def check_user_info(cls, values):
        if values.get("user_id") is None and values.get("user_email") is None:
            raise ValueError("Either user id or user email must be provided")
        return values


class DeleteUserRequest(LiteLLMPydanticObjectBase):
    user_ids: List[str]  # required


AllowedModelRegion = Literal["eu", "us"]


class BudgetNewRequest(LiteLLMPydanticObjectBase):
    budget_id: Optional[str] = Field(default=None, description="The unique budget id.")
    max_budget: Optional[float] = Field(
        default=None,
        description="Requests will fail if this budget (in USD) is exceeded.",
    )
    soft_budget: Optional[float] = Field(
        default=None,
        description="Requests will NOT fail if this is exceeded. Will fire alerting though.",
    )
    max_parallel_requests: Optional[int] = Field(
        default=None, description="Max concurrent requests allowed for this budget id."
    )
    tpm_limit: Optional[int] = Field(
        default=None, description="Max tokens per minute, allowed for this budget id."
    )
    rpm_limit: Optional[int] = Field(
        default=None, description="Max requests per minute, allowed for this budget id."
    )
    budget_duration: Optional[str] = Field(
        default=None,
        description="Max duration budget should be set for (e.g. '1hr', '1d', '28d')",
    )
    model_max_budget: Optional[GenericBudgetConfigType] = Field(
        default=None,
        description="Max budget for each model (e.g. {'gpt-4o': {'max_budget': '0.0000001', 'budget_duration': '1d', 'tpm_limit': 1000, 'rpm_limit': 1000}})",
    )
    budget_reset_at: Optional[datetime] = Field(
        default=None,
        description="Datetime when the budget is reset",
    )


class BudgetRequest(LiteLLMPydanticObjectBase):
    budgets: List[str]


class BudgetDeleteRequest(LiteLLMPydanticObjectBase):
    id: str


class CustomerBase(LiteLLMPydanticObjectBase):
    user_id: str
    alias: Optional[str] = None
    spend: float = 0.0
    allowed_model_region: Optional[AllowedModelRegion] = None
    default_model: Optional[str] = None
    budget_id: Optional[str] = None
    litellm_budget_table: Optional[BudgetNewRequest] = None
    blocked: bool = False


class NewCustomerRequest(BudgetNewRequest):
    """
    Create a new customer, allocate a budget to them
    """

    user_id: str
    alias: Optional[str] = None  # human-friendly alias
    blocked: bool = False  # allow/disallow requests for this end-user
    budget_id: Optional[str] = None  # give either a budget_id or max_budget
    spend: Optional[float] = None
    allowed_model_region: Optional[AllowedModelRegion] = (
        None  # require all user requests to use models in this specific region
    )
    default_model: Optional[str] = (
        None  # if no equivalent model in allowed region - default all requests to this model
    )
    object_permission: Optional[LiteLLM_ObjectPermissionBase] = None

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
    alias: Optional[str] = None  # human-friendly alias
    blocked: bool = False  # allow/disallow requests for this end-user
    max_budget: Optional[float] = None
    budget_id: Optional[str] = None  # give either a budget_id or max_budget
    allowed_model_region: Optional[AllowedModelRegion] = (
        None  # require all user requests to use models in this specific region
    )
    default_model: Optional[str] = (
        None  # if no equivalent model in allowed region - default all requests to this model
    )
    object_permission: Optional[LiteLLM_ObjectPermissionBase] = None


class DeleteCustomerRequest(LiteLLMPydanticObjectBase):
    """
    Delete multiple Customers
    """

    user_ids: List[str]


class MemberBase(LiteLLMPydanticObjectBase):
    user_id: Optional[str] = Field(
        default=None,
        description="The unique ID of the user to add. Either user_id or user_email must be provided",
    )
    user_email: Optional[str] = Field(
        default=None,
        description="The email address of the user to add. Either user_id or user_email must be provided",
    )

    @model_validator(mode="before")
    @classmethod
    def check_user_info(cls, values):
        if not isinstance(values, dict):
            raise ValueError("input needs to be a dictionary")
        if values.get("user_id") is None and values.get("user_email") is None:
            raise ValueError("Either user id or user email must be provided")
        return values


class Member(MemberBase):
    role: Literal[
        "admin",
        "user",
    ] = Field(
        description="The role of the user within the team. 'admin' users can manage team settings and members, 'user' is a regular team member"
    )


class OrgMember(MemberBase):
    role: Literal[
        LitellmUserRoles.ORG_ADMIN,
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
    ]


class TeamBase(LiteLLMPydanticObjectBase):
    team_alias: Optional[str] = None
    team_id: Optional[str] = None
    organization_id: Optional[str] = None
    admins: list = []
    members: list = []
    members_with_roles: List[Member] = []
    team_member_permissions: Optional[List[str]] = None
    metadata: Optional[dict] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None

    # Budget fields
    max_budget: Optional[float] = None
    soft_budget: Optional[float] = None
    budget_duration: Optional[str] = None
    budget_limits: Optional[List[BudgetLimitEntry]] = (
        None  # multiple concurrent budget windows
    )

    models: list = []
    blocked: bool = False
    router_settings: Optional[dict] = None
    access_group_ids: Optional[List[str]] = None
    default_team_member_models: Optional[List[str]] = (
        None  # default allowed_models seeded onto new team members
    )


class NewTeamRequest(TeamBase):
    model_aliases: Optional[dict] = None
    tags: Optional[list] = None
    guardrails: Optional[List[str]] = None
    policies: Optional[List[str]] = None
    prompts: Optional[List[str]] = None
    object_permission: Optional[LiteLLM_ObjectPermissionBase] = None
    allowed_passthrough_routes: Optional[list] = None
    secret_manager_settings: Optional[dict] = None
    model_rpm_limit: Optional[Dict[str, int]] = None
    rpm_limit_type: Optional[
        Literal["guaranteed_throughput", "best_effort_throughput"]
    ] = None  # raise an error if 'guaranteed_throughput' is set and we're overallocating rpm
    tpm_limit_type: Optional[
        Literal["guaranteed_throughput", "best_effort_throughput"]
    ] = None  # raise an error if 'guaranteed_throughput' is set and we're overallocating tpm

    model_tpm_limit: Optional[Dict[str, int]] = None
    team_member_budget: Optional[float] = (
        None  # allow user to set a budget for all team members
    )
    team_member_rpm_limit: Optional[int] = (
        None  # allow user to set RPM limit for all team members
    )
    team_member_tpm_limit: Optional[int] = (
        None  # allow user to set TPM limit for all team members
    )
    team_member_key_duration: Optional[str] = None  # e.g. "1d", "1w", "1m"
    team_member_budget_duration: Optional[str] = None  # e.g. "30d", "1mo"
    allowed_vector_store_indexes: Optional[List[AllowedVectorStoreIndexItem]] = None
    enforced_batch_output_expires_after: Optional[dict] = None
    enforced_file_expires_after: Optional[dict] = None

    model_config = ConfigDict(protected_namespaces=())


class GlobalEndUsersSpend(LiteLLMPydanticObjectBase):
    api_key: Optional[str] = None
    startTime: Optional[datetime] = None
    endTime: Optional[datetime] = None


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
    team_alias: Optional[str] = None
    organization_id: Optional[str] = None
    metadata: Optional[dict] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    max_budget: Optional[float] = None
    soft_budget: Optional[float] = None
    models: Optional[list] = None
    blocked: Optional[bool] = None
    budget_duration: Optional[str] = None
    tags: Optional[list] = None
    model_aliases: Optional[dict] = None
    guardrails: Optional[List[str]] = None
    policies: Optional[List[str]] = None
    object_permission: Optional[LiteLLM_ObjectPermissionBase] = None
    team_member_budget: Optional[float] = None
    team_member_budget_duration: Optional[str] = None
    team_member_rpm_limit: Optional[int] = None
    team_member_tpm_limit: Optional[int] = None
    team_member_key_duration: Optional[str] = None
    allowed_passthrough_routes: Optional[list] = None
    secret_manager_settings: Optional[dict] = None
    prompts: Optional[List[str]] = None
    model_rpm_limit: Optional[Dict[str, int]] = None
    model_tpm_limit: Optional[Dict[str, int]] = None
    allowed_vector_store_indexes: Optional[List[AllowedVectorStoreIndexItem]] = None
    enforced_batch_output_expires_after: Optional[dict] = None
    enforced_file_expires_after: Optional[dict] = None
    router_settings: Optional[dict] = None
    access_group_ids: Optional[List[str]] = None
    budget_limits: Optional[List[BudgetLimitEntry]] = (
        None  # multiple concurrent budget windows
    )
    default_team_member_models: Optional[List[str]] = (
        None  # default allowed_models seeded onto new team members
    )


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
    team_ids: List[str]  # required


class BlockTeamRequest(LiteLLMPydanticObjectBase):
    team_id: str  # required


class BlockKeyRequest(LiteLLMPydanticObjectBase):
    key: str  # required


class AddTeamCallback(LiteLLMPydanticObjectBase):
    callback_name: str
    callback_type: Optional[Literal["success", "failure", "success_and_failure"]] = (
        "success_and_failure"
    )
    callback_vars: Dict[str, str]

    @model_validator(mode="before")
    @classmethod
    def validate_callback_vars(cls, values):
        callback_vars = values.get("callback_vars", {})
        valid_keys = set(StandardCallbackDynamicParams.__annotations__.keys())
        for key, value in callback_vars.items():
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid callback variable: {key}. Must be one of {valid_keys}"
                )
            callback_vars[key] = str(value)
            validate_no_callback_env_reference(
                key, callback_vars[key], source="key/team callback metadata"
            )
        return values


class TeamCallbackMetadata(LiteLLMPydanticObjectBase):
    success_callback: Optional[List[str]] = []
    failure_callback: Optional[List[str]] = []
    callbacks: Optional[List[str]] = []
    # for now - only supported for langfuse
    callback_vars: Optional[Dict[str, str]] = {}

    @model_validator(mode="before")
    @classmethod
    def validate_callback_vars(cls, values):
        success_callback = values.get("success_callback", [])
        if success_callback is None:
            values.pop("success_callback", None)
        failure_callback = values.get("failure_callback", [])
        if failure_callback is None:
            values.pop("failure_callback", None)
        callbacks = values.get("callbacks", [])
        if callbacks is None:
            values.pop("callbacks", None)

        callback_vars = values.get("callback_vars", {})
        if callback_vars is None:
            values.pop("callback_vars", None)
        if all(val is None for val in values.values()):
            return {
                "success_callback": [],
                "failure_callback": [],
                "callbacks": [],
                "callback_vars": {},
            }
        valid_keys = set(StandardCallbackDynamicParams.__annotations__.keys())
        if callback_vars is not None:
            for key in callback_vars:
                if key not in valid_keys:
                    raise ValueError(
                        f"Invalid callback variable: {key}. Must be one of {valid_keys}"
                    )
        return values


class LiteLLM_ObjectPermissionTable(LiteLLMPydanticObjectBase):
    """Represents a LiteLLM_ObjectPermissionTable record"""

    object_permission_id: str
    mcp_servers: Optional[List[str]] = []
    mcp_access_groups: Optional[List[str]] = []
    mcp_tool_permissions: Optional[Dict[str, List[str]]] = None
    """
    Mapping - server_id -> list of tools

    Enforces allowed tools for a specific key/team/organization
    """

    vector_stores: Optional[List[str]] = []
    agents: Optional[List[str]] = []
    agent_access_groups: Optional[List[str]] = []
    mcp_toolsets: Optional[List[str]] = None
    blocked_tools: Optional[List[str]] = []
    search_tools: Optional[List[str]] = []
