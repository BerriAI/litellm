"""
MCP server table model.

Canonical definition for ``litellm_mcpservertable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

import enum
from datetime import datetime
from typing import Dict, List, Literal, Optional

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
    description: Optional[str] = None


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
    allowed_response_headers: list[str] = Field(default_factory=list)
    mcp_info: Optional[MCPInfo] = None
    static_headers: Optional[Dict[str, str]] = None
    env_vars: Optional[List[MCPEnvVar]] = None
    status: Optional[Literal["healthy", "unhealthy", "unknown"]] = Field(
        default="unknown",
        description="Health status: 'healthy', 'unhealthy', 'unknown'",
    )
    last_health_check: Optional[datetime] = None
    health_check_error: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    issuer: Optional[str] = None
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    registration_url: Optional[str] = None
    oauth2_flow: Optional[Literal["client_credentials", "authorization_code"]] = None
    # Token Exchange (OBO) fields — RFC 8693. ``audience`` is named for the RFC's
    # request parameter (token-exchange only); RFC 8707 resource indicators are a
    # separate concept named ``resource`` in the v2 egress types. A null
    # ``subject_token_type`` means DEFAULT_SUBJECT_TOKEN_TYPE (litellm.types.mcp),
    # applied at the egress build sites.
    token_exchange_endpoint: Optional[str] = None
    audience: Optional[str] = None
    subject_token_type: Optional[str] = None
    token_exchange_profile: Optional[str] = None
    allow_all_keys: bool = False
    available_on_public_internet: bool = True
    delegate_auth_to_upstream: bool = False
    oauth_passthrough: bool = False
    dcr_bridge: Optional[bool] = None
    is_byok: bool = False
    byok_description: List[str] = Field(default_factory=list)
    byok_api_key_help_url: Optional[str] = None
    has_user_credential: Optional[bool] = None
    source_url: Optional[str] = None
    timeout: Optional[float] = None
    max_concurrent_requests: Optional[int] = None
    approval_status: Optional[str] = Field(
        default="active",
        description="Approval status: 'pending_review', 'active', 'rejected'",
    )
    submitted_by: Optional[str] = None
    submitted_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None
