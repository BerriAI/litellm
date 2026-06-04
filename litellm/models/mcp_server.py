"""
MCP server table model.

Canonical definition for ``litellm_mcpservertable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import Field

from litellm.types.llms.base import LiteLLMPydanticObjectBase
from litellm.types.mcp import MCPAuthType, MCPCredentials, MCPTransportType
from litellm.types.mcp_server.mcp_server_manager import MCPInfo


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
    status: Optional[Literal["healthy", "unhealthy", "unknown"]] = Field(
        default="unknown",
        description="Health status: 'healthy', 'unhealthy', 'unknown'",
    )
    last_health_check: Optional[datetime] = None
    health_check_error: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    registration_url: Optional[str] = None
    allow_all_keys: bool = False
    available_on_public_internet: bool = True
    delegate_auth_to_upstream: bool = False
    oauth_passthrough: bool = False
    is_byok: bool = False
    byok_description: List[str] = Field(default_factory=list)
    byok_api_key_help_url: Optional[str] = None
    has_user_credential: Optional[bool] = None
    source_url: Optional[str] = None
    approval_status: Optional[str] = Field(
        default="active",
        description="Approval status: 'pending_review', 'active', 'rejected'",
    )
    submitted_by: Optional[str] = None
    submitted_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None
