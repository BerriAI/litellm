"""
MCP server table model.

Canonical definition for ``litellm_mcpservertable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

import enum
from datetime import datetime
from typing import Literal

from pydantic import Field

from litellm.types.llms.base import LiteLLMPydanticObjectBase
from litellm.types.mcp import MCPAuthType, MCPCredentials, MCPTransportType
from litellm.types.mcp_server.mcp_server_manager import MCPInfo


class MCPEnvVarScope(str, enum.Enum):
    """Scope for an MCP server environment variable.

    - ``global``: value is provided by the admin and used for all users.
    - ``user``: each user must provide their own value via the per-user
      env-var endpoint. The admin-supplied ``value`` is treated as a
      placeholder/hint and is not used at request time.
    """

    global_ = "global"
    user = "user"


class MCPEnvVar(LiteLLMPydanticObjectBase):
    """One environment variable for an MCP server.

    Variables can be interpolated into ``static_headers`` using ``${NAME}``
    syntax. ``scope=global`` values are stored on the server. ``scope=user``
    values are stored per-user in ``LiteLLM_MCPUserEnvVars`` and supplied by
    each user.
    """

    name: str
    value: str = ""
    scope: MCPEnvVarScope = MCPEnvVarScope.global_
    description: str | None = None


class LiteLLM_MCPServerTable(LiteLLMPydanticObjectBase):
    """Represents a LiteLLM_MCPServerTable record"""

    server_id: str
    server_name: str | None = None
    alias: str | None = None
    description: str | None = None
    url: str | None = None
    spec_path: str | None = None
    transport: MCPTransportType
    auth_type: MCPAuthType | None = None
    credentials: MCPCredentials | None = None
    instructions: str | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None
    teams: list[dict[str, str | None]] = Field(default_factory=list)
    mcp_access_groups: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    tool_name_to_display_name: dict[str, str] | None = None
    tool_name_to_description: dict[str, str] | None = None
    extra_headers: list[str] = Field(default_factory=list)
    mcp_info: MCPInfo | None = None
    static_headers: dict[str, str] | None = None
    env_vars: list[MCPEnvVar] | None = None
    status: Literal["healthy", "unhealthy", "unknown"] | None = Field(
        default="unknown",
        description="Health status: 'healthy', 'unhealthy', 'unknown'",
    )
    last_health_check: datetime | None = None
    health_check_error: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    issuer: str | None = None
    authorization_url: str | None = None
    token_url: str | None = None
    registration_url: str | None = None
    oauth2_flow: Literal["client_credentials", "authorization_code"] | None = None
    # Token Exchange (OBO) fields — RFC 8693. ``audience`` is named for the RFC's
    # request parameter (token-exchange only); RFC 8707 resource indicators are a
    # separate concept named ``resource`` in the v2 egress types. A null
    # ``subject_token_type`` means DEFAULT_SUBJECT_TOKEN_TYPE (litellm.types.mcp),
    # applied at the egress build sites.
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
    has_user_credential: bool | None = None
    connected_app_reachable: bool | None = None
    source_url: str | None = None
    timeout: float | None = None
    max_concurrent_requests: int | None = None
    approval_status: str | None = Field(
        default="active",
        description="Approval status: 'pending_review', 'active', 'rejected'",
    )
    submitted_by: str | None = None
    submitted_at: datetime | None = None
    reviewed_at: datetime | None = None
    review_notes: str | None = None
