from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from litellm.types.mcp import (
    DEFAULT_SUBJECT_TOKEN_TYPE,
    MCPAuth,
    MCPAuthType,
    MCPTokenEndpointAuthMethod,
    MCPTransportType,
)

# MCPInfo now allows arbitrary additional fields for custom metadata
MCPInfo = Dict[str, Any]


class MCPOAuthMetadata(BaseModel):
    scopes: Optional[List[str]] = None
    """Resource-driven scopes for the authorization request: the RFC 9728 protected-resource
    ``scopes_supported``, or the ``scope`` from the WWW-Authenticate 401 challenge when the resource
    supplied one, else the authorization server's ``scopes_supported``. This is the scope value a
    client requests per the MCP authorization spec Scope Selection Strategy; scope minimization and
    inflation control are the authorization server's and user's job at consent (RFC 6749 §3.3), not
    the client's."""
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    registration_url: Optional[str] = None
    discovered_issuer: Optional[str] = None
    """The ``issuer`` the authorization-server metadata document self-attests (RFC 8414). Persisted
    trust-on-first-use as the server's ``issuer`` when none is configured, so that later rebuilds
    anchor discovery on it (RFC 8414 §3.3) and a subsequently compromised resource cannot re-point
    it. Never overwrites an admin-configured issuer."""
    from_origin_fallback: bool = False
    """True when the metadata came from guessing the resource origin as its authorization
    server rather than from an RFC 9728/8414-advertised document. Guessed endpoints are
    usable in memory but must never be persisted as configuration."""
    grant_profiles: Optional[List[str]] = None
    """The authorization server's ``authorization_grant_profiles_supported`` (draft OAuth
    identity-assertion-authz-grant); the enterprise-managed-authorization gate requires the
    id-jag profile in it before an autofilled ID-JAG endpoint is trusted. ``None`` means the
    document did not carry the field, distinct from an empty advertisement."""


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
    allowed_params: Optional[Dict[str, List[str]]] = None  # map of tool names to allowed parameter lists
    static_headers: Optional[Dict[str, str]] = None  # static headers to forward to the MCP server
    # Admin-configured env vars. Each entry is {name, value, scope, description}.
    # scope=="global" values are interpolated into static_headers using ${NAME}.
    # scope=="user" values must be supplied per-user.
    env_vars: Optional[List[Dict[str, Any]]] = None
    # OAuth-specific fields
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    issuer: Optional[str] = None
    issuer_is_anchored: bool = False
    scopes: Optional[List[str]] = None
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    registration_url: Optional[str] = None
    # How the gateway authenticates to the upstream token endpoint. When
    # "client_secret_basic" the credentials go in an HTTP Basic Authorization
    # header (omitted from the body); None defaults to "client_secret_post".
    token_endpoint_auth_method: Optional[MCPTokenEndpointAuthMethod] = None
    # AWS SigV4 fields
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    aws_region_name: Optional[str] = None
    aws_service_name: Optional[str] = None  # defaults to "bedrock-agentcore"
    aws_role_name: Optional[str] = None  # IAM role ARN for STS AssumeRole
    aws_session_name: Optional[str] = None  # session name for CloudTrail auditing
    # Token Exchange (OBO) fields
    token_exchange_endpoint: Optional[str] = None
    audience: Optional[str] = None
    subject_token_type: str = DEFAULT_SUBJECT_TOKEN_TYPE
    # ID-JAG fields (draft-ietf-oauth-identity-assertion-authz-grant).
    # Leg 1 reuses token_exchange_endpoint (IdP org-AS), audience (resource-AS
    # identifier), scopes, subject_token_type, client_id/client_secret. Leg 2
    # posts the ID-JAG assertion to id_jag_resource_token_endpoint.
    id_jag_resource_token_endpoint: Optional[str] = None
    id_jag_resource: Optional[str] = None
    client_private_key: Optional[str] = None
    client_private_key_id: Optional[str] = None
    client_assertion_signing_alg: str = "RS256"
    # Wire dialect: "rfc8693" (standard token-exchange grant) or "entra_obo" (Microsoft Entra
    # On-Behalf-Of, the RFC 7523 jwt-bearer grant + requested_token_use extension)
    token_exchange_profile: str = "rfc8693"
    # Stdio-specific fields
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    access_groups: Optional[List[str]] = None
    allow_all_keys: bool = False
    available_on_public_internet: bool = True
    # Explicit opt-in to upstream-delegated authentication for ``oauth2``
    # servers. When ``auth_type == oauth2`` and this is ``True``, MCP requests
    # bypass LiteLLM API-key/SSO auth (and the pre-emptive 401) so the client
    # completes PKCE directly with the upstream MCP server. See
    # ``MCPRequestHandler._target_servers_delegate_auth_to_upstream``.
    #
    # Honored only for ``auth_type == oauth2``; ignored for any other
    # ``auth_type``. OAuth pass-through for non-oauth2 servers
    # (``auth_type in (None, MCPAuth.none)``) is a separate, explicit opt-in —
    # see ``oauth_passthrough`` / ``is_oauth_passthrough``.
    delegate_auth_to_upstream: bool = False
    # Explicit opt-in to OAuth pass-through for non-oauth2 servers. When this
    # is ``True`` AND ``auth_type in (None, MCPAuth.none)`` AND ``extra_headers``
    # contains ``Authorization``, the gateway proxies upstream
    # ``/.well-known/oauth-protected-resource`` metadata, emits spec-compliant
    # 401 challenges when no bearer is supplied, and propagates upstream
    # 401/403 responses instead of swallowing them. See ``is_oauth_passthrough``.
    #
    # Intentionally distinct from ``delegate_auth_to_upstream`` (oauth2-only):
    # reusing that flag would silently change behavior for servers that forward
    # ``Authorization`` for non-OAuth reasons (e.g. static bearer tokens). Must
    # be set explicitly to avoid regressing servers that did not opt in.
    oauth_passthrough: bool = False
    dcr_bridge: Optional[bool] = None
    is_byok: bool = False
    byok_description: List[str] = []
    byok_api_key_help_url: Optional[str] = None
    source_url: Optional[str] = None
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
    # Optional TTL override (seconds) for the Redis per-user token cache, capped
    # at the token's expires_in minus the expiry buffer so a cached entry never
    # outlives the token. Defaults to the token's expires_in minus the expiry
    # buffer, or MCP_PER_USER_TOKEN_DEFAULT_TTL when expires_in is absent.
    token_storage_ttl_seconds: Optional[int] = None
    timeout: Optional[float] = None
    # Max concurrent outbound tool calls to this server; excess calls queue.
    # None or a value <= 0 means unlimited.
    max_concurrent_requests: Optional[int] = None
    # Resolved short-ID tool prefix when LITELLM_USE_SHORT_MCP_TOOL_PREFIX is
    # enabled.  Set by ``MCPServerManager._assign_unique_short_prefix`` at
    # registration time so that natural-hash collisions between two
    # different ``server_id`` values are bumped deterministically.  Left
    # ``None`` in default-prefix mode.
    short_prefix: Optional[str] = None
    allow_sampling: bool = False
    allow_elicitation: bool = False
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
    def is_true_passthrough(self) -> bool:
        """True for the transparent-proxy mode: LiteLLM performs no admission auth and forwards the
        client's ``Authorization`` to the upstream unchanged."""
        return self.auth_type == MCPAuth.true_passthrough

    @property
    def is_oauth_delegate(self) -> bool:
        """True for the delegated-upstream-OAuth mode: LiteLLM still admits the caller (API key / SSO /
        JWT) but forwards the caller's separate upstream ``Authorization`` unchanged, minting nothing."""
        return self.auth_type == MCPAuth.oauth_delegate

    @property
    def is_dcr_bridge(self) -> bool:
        """True when this client-forwarded-token server serves the gateway-hosted DCR front door
        (gateway-self protected-resource and authorization-server metadata plus the register,
        authorize, and token relays) instead of relaying the upstream's own OAuth discovery
        verbatim. ``dcr_bridge`` is rejected on every other auth type at create, update, and
        config load, so the mode gate here only defends rows edited outside those paths."""
        return bool(self.dcr_bridge) and (self.is_true_passthrough or self.is_oauth_delegate)

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

        if self.is_true_passthrough or self.is_oauth_delegate:
            return True

        # PAT passthrough: auth_type is none but extra_headers includes auth headers
        if self.auth_type == MCPAuth.none and self.extra_headers:
            auth_header_names = {"authorization", "x-api-key", "api-key", "apikey"}
            return any(h.lower() in auth_header_names for h in self.extra_headers)

        return False

    @property
    def is_oauth_passthrough(self) -> bool:
        """True iff the gateway should transparently forward upstream OAuth
        (discovery + 401s) rather than participating as an authorization
        server itself.

        A server is pass-through for OAuth purposes when ALL three conditions
        hold:
        1. ``auth_type`` is ``None`` or ``MCPAuth.none`` (the gateway does
           not manage OAuth for this server).
        2. ``extra_headers`` includes ``Authorization`` — the admin has
           opted this server into forwarding the client's bearer token
           straight to the upstream MCP server.
        3. ``oauth_passthrough`` is ``True`` — the admin has
           explicitly opted into upstream-delegated OAuth semantics for
           this server. This is the explicit detection flag: without it,
           a server that merely forwards ``Authorization`` (e.g. for
           static bearer tokens or custom auth schemes) keeps the
           pre-PR behavior and is not treated as OAuth pass-through.
           This is deliberately a separate flag from
           ``delegate_auth_to_upstream`` (which is oauth2-only) so enabling
           pass-through here never changes behavior for oauth2 servers.

        This is intentionally narrower than ``requires_per_user_auth``,
        which also covers PATs (``x-api-key``, ``api-key``, ``apikey``).
        Those are static credentials, not OAuth bearer tokens, so they
        must not trigger upstream OAuth discovery or 401 propagation.
        """
        if self.auth_type not in (None, MCPAuth.none):
            return False
        if not self.extra_headers:
            return False
        if self.oauth_passthrough is not True:
            return False
        return any(h.lower() == "authorization" for h in self.extra_headers)

    @property
    def has_token_exchange_config(self) -> bool:
        """True if this server is configured for OAuth2 token exchange (OBO / RFC 8693)."""
        return (
            self.auth_type == MCPAuth.oauth2_token_exchange
            and bool(self.client_id and self.client_secret)
            and bool(self.token_exchange_endpoint or self.token_url)
        )
