from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from litellm.proxy._types import MCPAuthType, MCPTransportType
from litellm.types.mcp import MCPAuth

# MCPInfo now allows arbitrary additional fields for custom metadata
MCPInfo = Dict[str, Any]


class MCPOAuthMetadata(BaseModel):
    scopes: Optional[List[str]] = None
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    registration_url: Optional[str] = None


class MCPServer(BaseModel):
    server_id: str
    name: str
    alias: Optional[str] = None
    server_name: Optional[str] = None
    url: Optional[str] = None
    transport: MCPTransportType
    spec_path: Optional[str] = None
    auth_type: Optional[MCPAuthType] = None
    authentication_token: Optional[str] = None
    instructions: Optional[str] = None
    mcp_info: Optional[MCPInfo] = None
    extra_headers: Optional[List[str]] = (
        None  # allow admin to specify which headers to forward from client to the MCP server
    )
    allowed_tools: Optional[List[str]] = None
    disallowed_tools: Optional[List[str]] = None
    tool_name_to_display_name: Optional[Dict[str, str]] = None
    tool_name_to_description: Optional[Dict[str, str]] = None
    allowed_params: Optional[Dict[str, List[str]]] = (
        None  # map of tool names to allowed parameter lists
    )
    static_headers: Optional[Dict[str, str]] = (
        None  # static headers to forward to the MCP server
    )
    # OAuth-specific fields
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    scopes: Optional[List[str]] = None
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    registration_url: Optional[str] = None
    # AWS SigV4 fields
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    aws_region_name: Optional[str] = None
    aws_service_name: Optional[str] = None  # defaults to "bedrock-agentcore"
    aws_role_name: Optional[str] = None  # IAM role ARN for STS AssumeRole
    aws_session_name: Optional[str] = None  # session name for CloudTrail auditing
    # Token Exchange (OBO) fields — RFC 8693
    token_exchange_endpoint: Optional[str] = None
    audience: Optional[str] = None
    subject_token_type: str = "urn:ietf:params:oauth:token-type:access_token"
    # Stdio-specific fields
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    access_groups: Optional[List[str]] = None
    allow_all_keys: bool = False
    available_on_public_internet: bool = True
    # When True AND auth_type == oauth2, MCP requests targeting this server
    # bypass LiteLLM API-key/SSO auth (and the pre-emptive 401) so the client
    # completes PKCE directly with the upstream MCP server. Honored only for
    # auth_type=oauth2; ignored for any other auth_type. See
    # MCPRequestHandler._target_servers_delegate_auth_to_upstream.
    delegate_auth_to_upstream: bool = False
    is_byok: bool = False
    byok_description: List[str] = []
    byok_api_key_help_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # OAuth2 flow type.  Defaults to None (interactive / authorization_code).
    # Set to "client_credentials" to enable M2M token fetching.
    oauth2_flow: Optional[Literal["client_credentials", "authorization_code"]] = None
    # Per-user OAuth server-side storage config.
    # token_validation: key-value pairs that must match fields in the OAuth token
    # response (supports dot-notation for nested fields, e.g. "team.enterprise_id").
    # Tokens that fail validation are rejected before storage.
    token_validation: Optional[Dict[str, Any]] = None
    # Optional TTL override (seconds) for the Redis per-user token cache.
    # Defaults to the token's expires_in minus the expiry buffer, or
    # MCP_PER_USER_TOKEN_DEFAULT_TTL when expires_in is absent.
    token_storage_ttl_seconds: Optional[int] = None
    # Resolved short-ID tool prefix when LITELLM_USE_SHORT_MCP_TOOL_PREFIX is
    # enabled.  Set by ``MCPServerManager._assign_unique_short_prefix`` at
    # registration time so that natural-hash collisions between two
    # different ``server_id`` values are bumped deterministically.  Left
    # ``None`` in default-prefix mode.
    short_prefix: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def has_client_credentials(self) -> bool:
        """True if this server should use the OAuth2 client_credentials (M2M) flow.

        M2M flow must be opted into explicitly via ``oauth2_flow: client_credentials``.
        Having client_id / client_secret / token_url present is NOT sufficient —
        those fields are also used for interactive (authorization_code) OAuth,
        e.g. GitHub Enterprise.  Auto-detecting M2M from field presence was a
        breaking regression introduced with the M2M feature.
        """
        return self.oauth2_flow == "client_credentials"

    @property
    def needs_user_oauth_token(self) -> bool:
        """True if this is an OAuth2 server that relies on per-user tokens (no client_credentials)."""
        return self.auth_type == MCPAuth.oauth2 and not self.has_client_credentials

    @property
    def requires_per_user_auth(self) -> bool:
        """
        True if this server requires per-user/per-request authentication.
        This includes:
        - OAuth2 servers without client credentials
        - Servers with auth_type=none but extra_headers configured for auth passthrough

        Health checks should be skipped for these servers since they cannot
        authenticate without user-provided credentials.
        """
        # OAuth2 without client credentials
        if self.needs_user_oauth_token:
            return True

        # PAT passthrough: auth_type is none but extra_headers includes auth headers
        if self.auth_type == MCPAuth.none and self.extra_headers:
            auth_header_names = {"authorization", "x-api-key", "api-key", "apikey"}
            return any(h.lower() in auth_header_names for h in self.extra_headers)

        return False

    @property
    def has_token_exchange_config(self) -> bool:
        """True if this server is configured for OAuth2 token exchange (OBO / RFC 8693)."""
        return (
            self.auth_type == MCPAuth.oauth2_token_exchange
            and bool(self.client_id and self.client_secret)
            and bool(self.token_exchange_endpoint or self.token_url)
        )
