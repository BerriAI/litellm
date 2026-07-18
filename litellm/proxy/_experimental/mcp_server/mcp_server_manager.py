"""
MCP Client Manager

This class is responsible for managing MCP clients with support for both SSE and HTTP streamable transports.

This is a Proxy
"""

import asyncio
import datetime
import hashlib
import json
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Literal, Optional, Union, cast
from urllib.parse import urlparse

import anyio
import httpx
from fastapi import HTTPException
from httpx import HTTPStatusError
from mcp import ReadResourceResult, Resource
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import (
    CallToolResult,
    GetPromptRequestParams,
    GetPromptResult,
    Prompt,
    ResourceTemplate,
)
from mcp.types import Tool as MCPTool
from pydantic import AnyUrl

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    MCP_CLIENT_TIMEOUT,
    MCP_HEALTH_CHECK_TIMEOUT,
    MCP_METADATA_TIMEOUT,
    MCP_NPM_CACHE_DIR,
    MCP_STDIO_ALLOWED_COMMANDS,
    MCP_TOOL_LISTING_TIMEOUT,
)
from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
from litellm.experimental_mcp_client.client import MCPClient, MCPSigV4Auth
from litellm.litellm_core_utils.url_utils import SSRFError, async_safe_get
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.exceptions import MCPUpstreamAuthError
from litellm.proxy._experimental.mcp_server.elicitation_handler import (
    MCP_ELICITATION_AVAILABLE,
)
from litellm.proxy._experimental.mcp_server.sampling_handler import (
    MCP_SAMPLING_AVAILABLE,
)
from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
    MCPPerUserTokenCache,
    mcp_per_user_token_cache,
    resolve_mcp_auth,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials import (
    Error,
    Ok,
    UpstreamCredentialProvider,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (
    raise_public,
    raise_token_exchange_challenge,
    raise_user_oauth_challenge,
    to_server_spec,
    to_subject,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.idp_subject_provider import (
    build_idp_subject_source,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    InvalidatableOAuthTokenStore,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.per_user_oauth_store import (
    LazyPerUserOAuthTokenStore,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchange_provider import (
    build_token_exchanger,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    AuthorizationCodeConfig,
    PassthroughConfig,
    ServerSpec,
    TokenExchangeConfig,
)
from litellm.proxy._experimental.mcp_server.utils import (
    MCP_TOOL_PREFIX_SEPARATOR,
    MCPMissingUserEnvVarsError,
    add_server_prefix_to_name,
    build_env_var_setup_url,
    collect_env_var_references,
    compute_short_server_prefix,
    get_server_prefix,
    interpolate_headers,
    is_short_mcp_tool_prefix_enabled,
    is_tool_name_prefixed,
    iter_known_server_prefixes,
    merge_mcp_headers,
    normalize_server_name,
    parse_admin_env_vars,
    split_server_prefix_from_name,
    strip_known_server_prefix,
    validate_mcp_server_name,
)
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPAuthType,
    MCPEnvVar,
    MCPTransport,
    MCPTransportType,
    SpecialMCPServerNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.proxy.common_utils.user_api_key_cache import get_management_object_ttl
from litellm.proxy.utils import ProxyLogging, get_server_root_path
from litellm.repositories.table_repositories import MCPServerRepository
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.mcp import DEFAULT_SUBJECT_TOKEN_TYPE, MCPAuth, MCPStdioConfig
from litellm.types.mcp_server.mcp_server_manager import (
    MCPInfo,
    MCPOAuthMetadata,
    MCPServer,
)
from litellm.types.utils import CallTypes

try:
    from mcp.shared.tool_name_validation import (
        validate_tool_name,  # pyright: ignore[reportAssignmentType]
    )
    from mcp.shared.tool_name_validation import (
        SEP_986_URL,
    )
except ImportError:
    from pydantic import BaseModel

    SEP_986_URL = "https://github.com/modelcontextprotocol/protocol/blob/main/proposals/0001-tool-name-validation.md"

    class _ToolNameValidationResult(BaseModel):
        is_valid: bool = True
        warnings: list = []

    def validate_tool_name(name: str) -> _ToolNameValidationResult:  # type: ignore[misc]
        return _ToolNameValidationResult()


# Probe includes characters on both sides of the separator to mimic real prefixed tool names.
_separator_probe_tool_name = f"litellm{MCP_TOOL_PREFIX_SEPARATOR}probe"
_separator_probe = validate_tool_name(_separator_probe_tool_name)
if not _separator_probe.is_valid:
    verbose_logger.warning(
        "MCP tool prefix separator '%s' violates SEP-986. See %s",
        MCP_TOOL_PREFIX_SEPARATOR,
        SEP_986_URL,
    )

_AZURE_ENTRA_HOSTS = {
    "login.microsoftonline.com",  # Global
    "login.microsoftonline.us",  # US Government
    "login.chinacloudapi.cn",  # China
}

# Short-lived in-memory cache for per-user MCP env var values, mirroring the
# BYOK credential cache. Keyed by (user_id, server_id); value is
# (values_dict, monotonic_timestamp). Keeps the tool-call and tool-listing
# paths off the DB on every request within the TTL window.
_user_env_vars_cache: dict[tuple[str, str], tuple[dict[str, str], float]] = {}
_USER_ENV_VARS_CACHE_TTL = 60  # seconds
_USER_ENV_VARS_CACHE_MAX_SIZE = 4096  # cap to prevent unbounded growth

# Auth types whose upstream OAuth endpoints (protected-resource + authorization-server metadata) the
# gateway discovers from the upstream itself: interactive oauth2 and the two client-forwarded modes.
# OBO/M2M endpoint discovery is decided separately via _obo_needs_endpoint_discovery. Shared by the
# config-YAML and DB server loaders so the two paths cannot drift on which modes trigger discovery.
_UPSTREAM_OAUTH_DISCOVERY_AUTH_TYPES: tuple[MCPAuth, ...] = (
    MCPAuth.oauth2,
    MCPAuth.true_passthrough,
    MCPAuth.oauth_delegate,
)


def _blank_to_none(value: str | None) -> str | None:
    """Collapse an absent, empty, or whitespace-only string to ``None``.

    OAuth endpoint fields are consumed by truthiness-based merges (``row or discovered``) and by the
    corroboration gate. A whitespace-only value is truthy to ``or`` but is not a usable endpoint, so
    without this the merge would keep the blank value for redirects while the gate treats it as
    unpinned and backfills the other fields, yielding a broken half-discovered config. Normalizing
    the pinned fields once, at each build entry point, gives every downstream consumer a single
    notion of "blank" so those code paths cannot disagree.
    """
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _uses_issuer_anchor(manual_issuer: str | None, is_discovery_auth_type: bool) -> bool:
    """Whether the endpoints are authoritatively anchored to an admin-pinned issuer (RFC 8414 §3.3).

    This is the trust/provenance property, distinct from whether the ``issuer`` field is merely
    populated: a trust-on-first-use discovered issuer sets ``issuer`` for token identity but is NOT
    anchored, so its endpoints stay resource-rooted. Anchoring holds only when the issuer was pinned
    (present on the row/config) on a discovery auth type. Every consumer of "is this anchored" reads
    this one definition, so the answer cannot diverge across build paths.
    """
    return _blank_to_none(manual_issuer) is not None and is_discovery_auth_type


def _endpoints_yield_to_issuer(
    issuer: str | None,
    is_discovery_auth_type: bool,
    authorization_url: str | None,
    token_url: str | None,
    registration_url: str | None,
) -> tuple[str | None, str | None, str | None]:
    """The single rule that makes an admin-configured ``issuer`` the sole authoritative endpoint
    source (RFC 8414 §3.3): when it is set for a discovery auth type, the stored/manual
    ``authorization_url``/``token_url``/``registration_url`` do not apply. They neither anchor nor
    short-circuit discovery, never override the issuer document in the merge, and never substitute for
    it when the issuer fetch fails (fail-closed). Returns the endpoint values that remain in force,
    i.e. all ``None`` when issuer-anchored, else the inputs unchanged. Called at every resolution site
    so the invariant holds in one place instead of being re-derived per merge.
    """
    if issuer is not None and is_discovery_auth_type:
        return None, None, None
    return authorization_url, token_url, registration_url


def _normalized_authorize_endpoint(url: str) -> str:
    """Compare authorize endpoints on scheme, host, and path only. The default port is elided and
    the host is lowercased so ``https://IDP.example.com:443/authorize/`` and
    ``https://idp.example.com/authorize`` are the same identity; query and trailing slash are not."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    default_port = {"https": 443, "http": 80}.get(scheme)
    try:
        port = parsed.port
    except ValueError:
        port = None
    authority = host if port is None or port == default_port else f"{host}:{port}"
    return f"{scheme}://{authority}{parsed.path.rstrip('/')}"


def _issuer_matches(claimed_issuer: object, configured_issuer: str) -> bool:
    """RFC 8414 §3.3 issuer equality between the metadata document's self-attested ``issuer`` and the
    admin-configured issuer, tolerant only of URL-insignificant differences (scheme/host case, the
    default port, a trailing slash). A non-string or empty claimed issuer never matches, so a
    document that omits ``issuer`` fails closed under issuer-anchored discovery.
    """
    if not isinstance(claimed_issuer, str) or not claimed_issuer:
        return False
    return _normalized_authorize_endpoint(claimed_issuer) == _normalized_authorize_endpoint(configured_issuer)


def _endpoints_corroborate_authorization_url(
    source_authorization_url: str | None,
    trusted_authorization_url: str | None,
) -> bool:
    """Whether a source's ``token_url``/``registration_url`` may be paired with a trusted authorize
    endpoint. This is the single trust rule for adopting OAuth endpoints from any non-manual source.

    Discovery is rooted at the MCP resource (RFC 9728), so a compromised upstream can advertise an
    attacker-run authorization server. When ``authorization_url`` is admin-pinned, pairing it with a
    ``token_url`` from a different source is the RFC 9700 authorization-server mix-up: the user signs
    in at the trusted authorize endpoint while the gateway redeems the code, with the stored client
    secret and PKCE verifier, at the attacker's token endpoint. Endpoints are trustworthy together
    only when they share an authorization server, so a source's endpoints are adopted only when the
    same source advertised an ``authorization_endpoint`` matching the pinned value. With no pinned
    value (``trusted_authorization_url is None``) there is nothing to protect: the authorize endpoint
    comes from the same source as the token endpoint, so they corroborate each other by construction.
    """
    if not (trusted_authorization_url and trusted_authorization_url.strip()):
        return True
    return bool(source_authorization_url) and _normalized_authorize_endpoint(
        source_authorization_url
    ) == _normalized_authorize_endpoint(trusted_authorization_url)


def _carry_forward_resolved_oauth_endpoints(new_server: MCPServer, previous_server: MCPServer | None) -> None:
    """Keep the last known good OAuth endpoints when a rebuild's re-discovery comes back empty.

    A rebuild wholesale-replaces the registry entry, so without this a transient upstream outage
    during re-discovery downgrades a working server (``authorization_url`` set) to a broken one
    (``None``, /authorize 400s) with no configuration change. Mirrors the ``short_prefix``
    carry-forward. Skipped when the server's ``url`` or ``auth_type`` changed, since the previous
    endpoints may then belong to a different upstream. ``registration_url`` IS carried even though
    ``_persist_discovered_oauth_endpoints`` refuses to write it to the row: carrying only restores
    the same in-memory value the previous build already ran with, while persisting it would flip
    ``_dcr_bridge_relays_client_registration`` (which keys off the stored column) for dcr_bridge
    servers that never had one configured.

    Carry-forward is a non-manual endpoint source, so the same trust rule as discovery applies: the
    previous ``token_url``/``registration_url``/``scopes`` are carried only when the previous
    ``authorization_url`` corroborates the authorize endpoint this build will use, i.e. when the
    incoming build has no pinned authorize endpoint (``None`` -> we adopt the previous one too, a
    consistent group) or pins the same one. An admin re-pointing ``authorization_url`` to a different
    server must not keep serving the old server's token endpoint or granted scopes.

    When the server is issuer-anchored (``issuer_is_anchored`` -- a pinned issuer on a discovery auth
    type), the endpoints come solely from the §3.3-validated issuer document, so carry-forward is
    skipped entirely for its endpoints: a failed issuer fetch leaves them ``None`` and must stay
    ``None`` (fail-closed), never resurrected from the previous registry entry. A merely discovered
    (trust-on-first-use) issuer is NOT anchored -- ``issuer`` is set for token identity but the
    endpoints are resource-rooted, so they still carry forward as last-known-good, gated by the
    corroboration check below like any other resource-rooted server. Scopes stay resource-driven and
    can carry either way.
    """
    if previous_server is None:
        return
    if previous_server.url != new_server.url or previous_server.auth_type != new_server.auth_type:
        return
    if new_server.issuer_is_anchored:
        # Endpoints come solely from the §3.3-validated issuer document; a failed fetch stays
        # fail-closed and must not be resurrected from the previous entry. Only the resource-driven
        # scopes carry as last-known-good.
        if not new_server.scopes and previous_server.scopes:
            new_server.scopes = previous_server.scopes
        return
    may_carry = _endpoints_corroborate_authorization_url(
        previous_server.authorization_url, new_server.authorization_url
    )
    if new_server.authorization_url is None and previous_server.authorization_url:
        new_server.authorization_url = previous_server.authorization_url
    if may_carry and new_server.token_url is None and previous_server.token_url:
        new_server.token_url = previous_server.token_url
    if may_carry and new_server.registration_url is None and previous_server.registration_url:
        new_server.registration_url = previous_server.registration_url
    if may_carry and not new_server.scopes and previous_server.scopes:
        new_server.scopes = previous_server.scopes


def _restrict_discovery_to_corroborated_authorization_server(
    metadata: MCPOAuthMetadata | None,
    manual_authorization_url: str | None,
    server_identifier: str,
    is_dcr_bridge: bool,
) -> MCPOAuthMetadata | None:
    """Reject discovered token/registration endpoints a manually pinned authorize endpoint cannot
    vouch for (the RFC 9700 authorization-server mix-up).

    Discovery is rooted at the MCP resource, so a compromised upstream can advertise an attacker
    ``token_endpoint``: with ``authorization_url`` admin-pinned but ``token_url`` blank, the merge
    would pair the trusted authorize endpoint with that attacker token endpoint, and the gateway would
    post the authorization code and client secret there. So the discovered ``token_url`` and
    ``registration_url`` are kept only if the document corroborates the pin (its
    ``authorization_endpoint`` matches). ``scopes`` are deliberately NOT gated here: per the MCP
    authorization spec Scope Selection Strategy and RFC 9700 §2.3, the scopes a client requests are
    resource-driven (the WWW-Authenticate challenge or the RFC 9728 protected-resource
    ``scopes_supported``), and scope inflation by a compromised resource is bounded by the
    authorization server and user consent (RFC 6749 §3.3), not by the client second-guessing the
    request. With no pin there is no trust anchor to protect, so discovery is returned as-is.
    """
    if metadata is None or not (manual_authorization_url and manual_authorization_url.strip()):
        return metadata
    if _endpoints_corroborate_authorization_url(metadata.authorization_url, manual_authorization_url):
        return metadata
    if not metadata.token_url and not metadata.registration_url:
        return metadata
    bridge_note = (
        " The discovered registration_url is rejected with it, so this dcr_bridge server stays on the"
        " short-circuit registration arm."
        if is_dcr_bridge and metadata.registration_url
        else ""
    )
    verbose_logger.warning(
        "MCP OAuth discovery for server %s advertised authorization_endpoint %s, which does not match the "
        "manually configured authorization_url %s; rejecting the discovered token_url/registration_url so "
        "authorization codes and client credentials only follow the configured authorization server. "
        "Configure Token URL manually if the mismatch is intentional.%s",
        server_identifier,
        _normalized_authorize_endpoint(metadata.authorization_url) if metadata.authorization_url else "<absent>",
        _normalized_authorize_endpoint(manual_authorization_url),
        bridge_note,
    )
    return metadata.model_copy(update={"token_url": None, "registration_url": None})


def invalidate_user_env_vars_cache(user_id: str, server_id: str) -> None:
    """Drop a cached entry after the user stores or clears their env var values
    so the next request reads the fresh value instead of a stale one."""
    _user_env_vars_cache.pop((user_id, server_id), None)


def _write_user_env_vars_cache(user_id: str, server_id: str, values: dict[str, str]) -> None:
    cache_key = (user_id, server_id)
    # Re-insert at the tail so eviction drops the oldest-written entry, not a
    # freshly refreshed one, and only sheds a single entry instead of wiping the
    # whole cache (which would stampede the DB).
    _user_env_vars_cache.pop(cache_key, None)
    if len(_user_env_vars_cache) >= _USER_ENV_VARS_CACHE_MAX_SIZE:
        _user_env_vars_cache.pop(next(iter(_user_env_vars_cache)), None)
    _user_env_vars_cache[cache_key] = (values, time.monotonic())


def _should_strip_caller_authorization(
    mcp_server: MCPServer,
    raw_headers: Optional[dict[str, str]],
    user_api_key_auth: Optional[UserAPIKeyAuth],
) -> bool:
    """Decide whether the caller's ``Authorization`` header must NOT be
    forwarded upstream when populating ``extra_headers`` for an MCP server.

    Centralized so ``_call_regular_mcp_tool`` (this module) and
    ``_prepare_mcp_server_headers`` (``server.py``) cannot drift apart on
    this security-sensitive decision.

    Strip rules:
    - **M2M (client_credentials) servers**: never forward the caller's
      ``Authorization`` — the proxy fetches its own upstream token.
    - **Migrated per-user OAuth (authorization_code) servers**: never forward
      the caller's ``Authorization`` — the v2 resolver injects the stored
      per-user token, so a caller-supplied bearer cannot override another
      user's stored credential. Delegate / pass-through keep forwarding it.
    - **OAuth pass-through servers**: strip when the ``Authorization``
      header is actually the LiteLLM API key — either because admission
      validated it (``user_api_key_auth.api_key`` is set) and the caller
      did NOT also supply ``x-litellm-api-key`` to disambiguate, or
      because the legacy ``user_api_key_auth is None`` call sites did
      not supply an explicit admission header.  In the anonymous /
      pass-through cold-start case (RFC 9728) the bearer in
      ``Authorization`` is the upstream OAuth token and must be
      forwarded, so we keep it.
    - **oauth_delegate servers**: admission always runs and there is no
      anonymous path, so the caller's separate ``Authorization`` is
      forwarded only when a distinct ``x-litellm-api-key`` carried
      admission. Without that header the ``Authorization`` *was* the
      admission credential — a virtual key, an IdP JWT, or an SSO / OIDC /
      session token whose ``api_key`` is ``None`` — and must never reach
      the upstream, so it is stripped regardless of the ``api_key`` value.
    """
    if mcp_server.auth_type == MCPAuth.oauth2_token_exchange:
        # OBO: the inbound Authorization is the subject token. It is exchanged at the IdP and only the
        # exchanged token is sent upstream, so the raw caller bearer must never be forwarded.
        return True
    if mcp_server.has_client_credentials:
        return True
    if mcp_server.auth_type == MCPAuth.oauth2 and to_server_spec(mcp_server) is not None:
        # Migrated per-user OAuth (authorization_code): the v2 resolver injects the
        # stored token, so a caller-forwarded Authorization must not be forwarded
        # upstream — it would override another user's stored credential. Delegate and
        # pass-through return None from to_server_spec and keep forwarding the bearer.
        return True
    if not (mcp_server.is_oauth_passthrough or mcp_server.is_oauth_delegate):
        return False

    normalized_raw_headers = {str(k).lower(): v for k, v in (raw_headers or {}).items() if isinstance(k, str)}
    has_explicit_litellm_admission_header = normalized_raw_headers.get("x-litellm-api-key") is not None
    if mcp_server.is_oauth_delegate:
        return not has_explicit_litellm_admission_header
    admission_consumed_authorization_as_litellm_key = (
        user_api_key_auth is not None
        and bool(getattr(user_api_key_auth, "api_key", None))
        and not has_explicit_litellm_admission_header
    )
    return admission_consumed_authorization_as_litellm_key or (
        user_api_key_auth is None and not has_explicit_litellm_admission_header
    )


def _without_authorization(
    headers: Optional[dict[str, str]],
) -> Optional[dict[str, str]]:
    """A copy of ``headers`` with any ``Authorization`` key removed (case-insensitive), or
    None if nothing remains. Drops only the credential, keeping other forwarded headers.
    """
    if not headers:
        return None
    filtered = {k: v for k, v in headers.items() if k.lower() != "authorization"}
    return filtered or None


def _format_byok_openapi_auth_header(mcp_server: MCPServer, mcp_auth_header: str) -> str:
    """Format a raw BYOK credential for OpenAPI tool ``Authorization`` injection."""
    if mcp_server.auth_type == MCPAuth.api_key:
        return f"ApiKey {mcp_auth_header}"
    if mcp_server.auth_type == MCPAuth.basic:
        return f"Basic {mcp_auth_header}"
    return f"Bearer {mcp_auth_header}"


def _openapi_forwarded_extra_headers(
    mcp_server: MCPServer,
    raw_headers: Optional[dict[str, str]],
    user_api_key_auth: Optional[UserAPIKeyAuth],
) -> Optional[dict[str, str]]:
    if not mcp_server.extra_headers or not raw_headers:
        return None
    normalized_raw = {str(k).lower(): v for k, v in raw_headers.items() if isinstance(k, str)}
    skip_caller_authorization = _should_strip_caller_authorization(
        mcp_server=mcp_server,
        raw_headers=raw_headers,
        user_api_key_auth=user_api_key_auth,
    )
    forwarded: dict[str, str] = {}
    for header_name in mcp_server.extra_headers:
        if not isinstance(header_name, str):
            continue
        if skip_caller_authorization and header_name.lower() == "authorization":
            continue
        value = normalized_raw.get(header_name.lower())
        if value is not None:
            forwarded[header_name] = value
    return forwarded or None


async def _resolve_byok_mcp_auth_header(
    mcp_server: MCPServer,
    user_api_key_auth: Optional[UserAPIKeyAuth],
    mcp_auth_header: Optional[str],
) -> Optional[str]:
    """Resolve BYOK credential for tool calls that bypass ``execute_mcp_tool``."""
    if not mcp_server.is_byok:
        return mcp_auth_header

    from litellm.proxy._experimental.mcp_server.server import (
        _check_byok_credential,
        _get_byok_credential,
    )

    if not mcp_auth_header:
        byok_cred = await _get_byok_credential(mcp_server, user_api_key_auth)
        if byok_cred is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "byok_auth_required",
                    "server_id": mcp_server.server_id,
                    "server_name": mcp_server.server_name or mcp_server.name,
                    "message": (
                        "No stored credential found for this BYOK server. "
                        "Complete the OAuth authorization flow to provide your API key."
                    ),
                },
                headers={"WWW-Authenticate": 'Bearer resource_metadata="/.well-known/oauth-protected-resource"'},
            )
        return byok_cred

    await _check_byok_credential(mcp_server, user_api_key_auth)
    return mcp_auth_header


def _client_forwarded_authorization_headers(
    mcp_server: MCPServer,
    oauth2_headers: Optional[dict[str, str]],
    raw_headers: Optional[dict[str, str]],
    user_api_key_auth: Optional[UserAPIKeyAuth],
) -> Optional[dict[str, str]]:
    """Egress headers for the client-forwarded-token modes (``true_passthrough`` / ``oauth_delegate``).

    Forwards the caller's ``Authorization`` to the upstream, stripped when
    ``_should_strip_caller_authorization`` says it was consumed as the LiteLLM admission key. Shared by
    ``_call_regular_mcp_tool`` and ``server.py``'s ``_prepare_mcp_server_headers`` so the two egress
    paths cannot drift, mirroring the ``_should_strip_caller_authorization`` split.
    """
    extra_headers = oauth2_headers.copy() if oauth2_headers else None
    if extra_headers and _should_strip_caller_authorization(
        mcp_server=mcp_server,
        raw_headers=raw_headers,
        user_api_key_auth=user_api_key_auth,
    ):
        return _without_authorization(extra_headers)
    return extra_headers


def _take_forwarded_authorization(
    headers: Optional[dict[str, str]],
) -> tuple[Optional[str], Optional[dict[str, str]]]:
    """Pop the ``Authorization`` value out of ``headers`` (case-insensitive), returning it with the
    remaining headers, so the passthrough resolver arm is the single Authorization source rather than
    the header also riding in ``extra_headers`` (which the resolved auth would then defer to)."""
    if not headers:
        return None, headers
    value = next((v for k, v in headers.items() if k.lower() == "authorization"), None)
    return value, _without_authorization(headers)


def _passthrough_token_from_mcp_auth_header(
    mcp_auth_header: Optional[Union[str, dict[str, str]]],
) -> Optional[str]:
    """The caller's per-server upstream credential for a passthrough-mode server, or None.

    Sourced from ``x-mcp-{alias}-authorization`` (string or per-header dict form) or the deprecated
    global ``x-mcp-auth`` fallback. Per-server headers are the multi-server shape: they bind one
    token to one server, so an aggregate scope with several passthrough-mode servers never replays
    a single credential across upstreams. The value is forwarded verbatim, so it must be the full
    header value (e.g. ``Bearer <upstream-token>``)."""
    if isinstance(mcp_auth_header, str):
        return mcp_auth_header or None
    if isinstance(mcp_auth_header, dict):
        return next((v for k, v in mcp_auth_header.items() if k.lower() == "authorization"), None)
    return None


def _consumes_caller_authorization(server: MCPServer) -> bool:
    """True when this server's egress forwards the caller's request-wide ``Authorization`` upstream:
    the client-forwarded token modes, legacy OAuth pass-through, and legacy upstream-delegated
    interactive oauth2. An unstamped oauth2 row (flow column not yet backfilled) reads as a consumer,
    which errs toward suppression — the fail-safe direction."""
    if server.is_true_passthrough or server.is_oauth_delegate or server.is_oauth_passthrough:
        return True
    return (
        server.auth_type == MCPAuth.oauth2
        and getattr(server, "delegate_auth_to_upstream", False) is True
        and not server.has_client_credentials
    )


def _caller_authorization_fans_out(
    server: MCPServer,
    scope_servers: Optional[list[MCPServer]],
) -> bool:
    """True when forwarding the caller's request-wide ``Authorization`` to ``server`` inside a
    listing fan-out would replay one credential against multiple upstreams: another server in the
    scope also consumes it (RFC 9700 cross-resource replay). ``scope_servers`` is None for
    explicitly-addressed operations (tool call, get_prompt, read_resource, single-server routes),
    where the client named the one target and the gateway is not choosing recipients."""
    if scope_servers is None:
        return False
    return any(
        other is not None and other.server_id != server.server_id and _consumes_caller_authorization(other)
        for other in scope_servers
    )


def _extract_upstream_auth_failure(
    exc: BaseException,
) -> Optional[tuple[int, Optional[str]]]:
    """Walk the exception tree looking for an HTTP 401/403 response from the
    upstream MCP server.

    The MCP SDK wraps transport errors in anyio ``ExceptionGroup`` objects and
    may chain through ``__cause__`` / ``__context__``. We inspect all of those
    layers for an ``httpx.Response``-bearing exception (typically
    ``httpx.HTTPStatusError``) and extract the status code and any upstream
    ``WWW-Authenticate`` header.

    Returns ``(status_code, www_authenticate)`` on match, else ``None``.
    """
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        response = getattr(current, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
            if isinstance(status_code, int) and status_code in (401, 403):
                www_authenticate: Optional[str] = None
                headers = getattr(response, "headers", None)
                if headers is not None:
                    try:
                        www_authenticate = headers.get("www-authenticate")
                    except Exception:
                        www_authenticate = None
                return status_code, www_authenticate

        # anyio / PEP 654 ExceptionGroup
        sub_exceptions = getattr(current, "exceptions", None)
        if sub_exceptions:
            stack.extend(sub_exceptions)

        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None and current.__context__ is not current.__cause__:
            stack.append(current.__context__)

    return None


def _warn_on_server_name_fields(
    *,
    server_id: str,
    alias: Optional[str],
    server_name: Optional[str],
):
    def _warn(field_name: str, value: Optional[str]) -> None:
        if not value:
            return
        result = validate_tool_name(value)
        if result.is_valid:
            return

        warning_text = "; ".join(result.warnings) if result.warnings else "Validation failed"
        verbose_logger.warning(
            "MCP server '%s' has invalid %s '%s': %s",
            server_id,
            field_name,
            value,
            warning_text,
        )

    _warn("alias", alias)
    _warn("server_name", server_name)


def _warn_internal_delegate_pkce_if_applicable(server: MCPServer, *, source: str) -> None:
    """Surface internal + upstream PKCE delegate in logs for operators."""
    if server.auth_type != MCPAuth.oauth2:
        return
    if getattr(server, "delegate_auth_to_upstream", False) is not True:
        return
    if getattr(server, "available_on_public_internet", True):
        return
    if server.has_client_credentials:
        return
    label = get_server_prefix(server)
    verbose_logger.warning(
        "MCP server %r (id=%s, source=%s): internal-only (available_on_public_internet=false) "
        "with delegate_auth_to_upstream=true. Anonymous callers can reach the upstream OAuth2 "
        "/authorize flow and complete PKCE without a LiteLLM API key session; ensure the "
        "upstream IdP and network enforce your access policy.",
        label,
        server.server_id,
        source,
    )


def _deserialize_json_dict(data: Any) -> Optional[dict[str, str]]:
    """
    Deserialize optional JSON mappings stored in the database.

    Accepts values kept as JSON strings or materialized dictionaries and
    returns None when the input is empty or cannot be decoded.
    """
    if not data:
        return None

    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            # If it's not valid JSON, return as-is (shouldn't happen but safety)
            return None
    else:
        # Already a dictionary
        return data


def _deserialize_json_list(data: Any) -> Optional[list[dict[str, Any]]]:
    """Deserialize a JSON array stored in the DB (``env_vars`` and friends).

    Returns ``None`` for empty / null / unparseable input. Accepts strings
    (raw JSON), already-materialized lists of dicts, and lists of Pydantic
    models (Prisma may hydrate a JSON column such as ``env_vars`` into
    ``MCPEnvVar`` objects); model entries are normalized to plain dicts so
    downstream consumers expecting ``List[Dict[str, Any]]`` validate.
    """
    if data is None or data == "" or data == []:
        return None
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
        data = parsed
    if not isinstance(data, list):
        return None
    return [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in data]


def _normalize_mcp_server_cost_info(mcp_info: MCPInfo) -> None:
    """Coerce ``mcp_server_cost_info`` numeric fields to ``float`` at ingest.

    YAML 1.1 parses scientific notation without a decimal point (e.g.
    ``7e-05``) as a string, and ``MCPServerCostInfo`` is a TypedDict with no
    runtime validation, so string-typed costs flow through to the UI and
    crash its ``.toFixed`` formatting. Values that cannot be coerced are
    dropped with a warning instead of failing the server load.
    """
    cost_info = mcp_info.get("mcp_server_cost_info")
    if not isinstance(cost_info, dict):
        return

    server_name = mcp_info.get("server_name")
    normalized = dict(cost_info)

    default_cost = normalized.get("default_cost_per_query")
    if default_cost is not None:
        try:
            normalized["default_cost_per_query"] = float(default_cost)
        except (TypeError, ValueError):
            verbose_logger.warning(
                "MCP server '%s' has non-numeric default_cost_per_query %r; ignoring it",
                server_name,
                default_cost,
            )
            del normalized["default_cost_per_query"]

    tool_costs = normalized.get("tool_name_to_cost_per_query")
    if isinstance(tool_costs, dict):
        normalized_tool_costs = {}
        for tool_name, cost in tool_costs.items():
            try:
                normalized_tool_costs[tool_name] = float(cost)
            except (TypeError, ValueError):
                verbose_logger.warning(
                    "MCP server '%s' has non-numeric cost %r for tool '%s'; ignoring it",
                    server_name,
                    cost,
                    tool_name,
                )
        normalized["tool_name_to_cost_per_query"] = normalized_tool_costs

    mcp_info["mcp_server_cost_info"] = normalized


def _create_sampling_callback(user_api_key_auth: Optional[Any] = None):
    """
    Create a sampling callback for MCP ClientSession.
    Returns a callable that handles sampling/createMessage requests from
    upstream MCP servers by routing them through litellm.acompletion().
    """
    if not MCP_SAMPLING_AVAILABLE:
        return None

    async def _sampling_callback(context, params):
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )
        import litellm
        from litellm.proxy._experimental.mcp_server.server import (
            get_active_auth_context,
        )

        auth_context = get_active_auth_context()
        resolved_auth = user_api_key_auth or (auth_context.user_api_key_auth if auth_context else None)
        # Forward original HTTP headers and client IP so that
        # header-dependent guardrails, tag-based routing, trace
        # correlation, and forward_llm_provider_auth_headers work
        # correctly for sampling sub-calls.
        _raw_headers = getattr(auth_context, "raw_headers", None)
        _client_ip = getattr(auth_context, "client_ip", None)

        return await handle_sampling_create_message(
            context=context,
            params=params,
            default_model=getattr(litellm, "default_mcp_sampling_model", None),
            user_api_key_auth=resolved_auth,
            raw_headers=_raw_headers,
            client_ip=_client_ip,
        )

    return _sampling_callback


def _create_elicitation_callback():
    """
    Create an elicitation callback for MCP ClientSession.
    Returns a callable that handles elicitation/create requests from
    upstream MCP servers. In gateway mode, this relays to the downstream
    client; in tool bridge mode, it returns a decline response.
    """
    if not MCP_ELICITATION_AVAILABLE:
        return None

    async def _elicitation_callback(context, params):
        from litellm.proxy._experimental.mcp_server.elicitation_handler import (
            handle_elicitation_request,
        )
        from litellm.proxy._experimental.mcp_server.server import get_active_mcp_session

        # In Gateway mode, we relay the elicitation request to the downstream client
        # that triggered the current operation.
        downstream_session = get_active_mcp_session()
        downstream_capabilities = getattr(downstream_session, "capabilities", None) if downstream_session else None

        return await handle_elicitation_request(
            context=context,
            params=params,
            downstream_session=downstream_session,
            downstream_capabilities=downstream_capabilities,
        )

    return _elicitation_callback


class MCPServerManager:
    _STDIO_ENV_TEMPLATE_PATTERN = re.compile(r"^\$\{(X-[^}]+)\}$")

    @staticmethod
    def _explicit_oauth2_flow(
        oauth2_flow: Optional[str],
    ) -> Optional[Literal["client_credentials", "authorization_code"]]:
        """DB rows persist their flow (write-time stamps plus the startup backfill) and
        config servers must declare it (validated at load), so both builds read the
        value verbatim: unknown or null resolves to None, which
        ``needs_user_oauth_token`` already treats as interactive. Field-shape inference
        survives only in the request-time security helpers (``effective_oauth2_flow`` /
        ``resolve_oauth2_flow_for_request``).
        """
        if oauth2_flow in ("client_credentials", "authorization_code"):
            return cast(Literal["client_credentials", "authorization_code"], oauth2_flow)
        return None

    @staticmethod
    def _resolve_oauth2_flow(
        *,
        auth_type: Optional[MCPAuthType],
        oauth2_flow: Optional[str],
        token_url: Optional[str],
        authorization_url: Optional[str],
        client_id: Optional[str],
        client_secret: Optional[str],
    ) -> Optional[Literal["client_credentials", "authorization_code"]]:
        """Infer oauth2_flow from field shape when the value is omitted.

        SECURITY-SENSITIVE: this is the shape-inference engine both request-time security
        helpers delegate to, so it is what decides M2M-vs-interactive for an unstamped row.
        Always access it through ``effective_oauth2_flow`` (boolean/enum decisions) or
        ``resolve_oauth2_flow_for_request`` (the egress object backstop), which are the single
        choke points for request-time resolution; do not call it directly from security sites
        and do not weaken its M2M-shape branch without accounting for those callers. DB rows
        are stamped at write time and by the startup backfill, config servers must declare
        oauth2_flow (validated at load), and both builds read the value verbatim via
        ``_explicit_oauth2_flow``. Delete this whole request-time layer only once the backstop
        warning stays silent in production.
        """
        if oauth2_flow in ("client_credentials", "authorization_code"):
            return cast(Literal["client_credentials", "authorization_code"], oauth2_flow)
        if oauth2_flow:
            # Ignore unknown/untyped values and continue legacy inference.
            return None
        if auth_type != MCPAuth.oauth2:
            return None
        if authorization_url:
            return None
        if token_url and client_id and client_secret:
            return "client_credentials"
        return None

    @staticmethod
    def effective_oauth2_flow(server: "MCPServer") -> Optional[Literal["client_credentials", "authorization_code"]]:
        """The oauth2_flow a security decision must use for ``server`` this request.

        Column-first, shape-fallback: a stamped row returns its explicit value; an
        unstamped (null) row whose fields carry the M2M shape resolves to
        ``client_credentials`` so it is treated as M2M and fails closed. Every
        security-sensitive reader (anonymous-delegate allowlist and gate, egress flow
        resolution) goes through this one helper rather than reading the bare
        ``has_client_credentials`` column, which is unreliable for null rows.
        """
        return MCPServerManager._resolve_oauth2_flow(
            auth_type=server.auth_type,
            oauth2_flow=server.oauth2_flow,
            token_url=server.token_url,
            authorization_url=server.authorization_url,
            client_id=server.client_id,
            client_secret=server.client_secret,
        )

    @staticmethod
    def resolve_oauth2_flow_for_request(server: "MCPServer") -> "MCPServer":
        """Return ``server`` with its effective oauth2_flow applied, for egress paths.

        A stamped row is returned unchanged (its effective flow equals the stored value).
        An unstamped M2M-shape row is returned as a per-request copy carrying
        ``oauth2_flow=client_credentials`` so downstream ``has_client_credentials`` /
        ``needs_user_oauth_token`` compute correctly and the stored client credentials are
        used instead of forwarding the caller's Authorization. Use this at every point that
        resolves an allowed server id into an ``MCPServer`` for a tool call or listing.
        """
        effective = MCPServerManager.effective_oauth2_flow(server)
        if effective is None or effective == server.oauth2_flow:
            return server
        verbose_logger.warning(
            "MCP server %s has no persisted oauth2_flow but matches the %s shape; using the "
            "inferred flow for this request. The startup backfill leaves this ambiguous M2M "
            "shape unstamped on purpose, so it will NOT self-heal: set oauth2_flow explicitly "
            "in the dashboard or via PUT /v1/mcp/server (client_credentials for M2M, or "
            "authorization_code after an interactive sign-in).",
            server.server_id,
            effective,
        )
        return server.model_copy(update={"oauth2_flow": effective})

    @staticmethod
    def _obo_needs_endpoint_discovery(
        auth_type: Optional[MCPAuthType],
        token_exchange_endpoint: Optional[str],
        token_url: Optional[str],
    ) -> bool:
        """An ``oauth2_token_exchange`` server with no configured token endpoint can have it
        discovered (RFC 9728 -> RFC 8414) the same way the ``oauth2`` flow already does; an explicitly
        configured ``token_exchange_endpoint``/``token_url`` wins and skips the discovery round-trip.
        """
        return auth_type == MCPAuth.oauth2_token_exchange and not (token_exchange_endpoint or token_url)

    def __init__(
        self,
        cred_provider: Optional[UpstreamCredentialProvider] = None,
        per_user_oauth_token_store: Optional[InvalidatableOAuthTokenStore] = None,
        per_user_token_cache: Optional[MCPPerUserTokenCache] = None,
    ):
        self._per_user_oauth_token_store = per_user_oauth_token_store or LazyPerUserOAuthTokenStore(
            self.get_mcp_server_by_id
        )
        self._per_user_token_cache = per_user_token_cache or mcp_per_user_token_cache
        self._cred_provider = cred_provider or UpstreamCredentialProvider(
            oauth_token_store=self._per_user_oauth_token_store,
            token_exchanger=build_token_exchanger(),
            idp_subject_source=build_idp_subject_source(),
        )
        self.registry: dict[str, MCPServer] = {}
        self.config_mcp_servers: dict[str, MCPServer] = {}
        """
        eg.
        [
            "server-1": {
                "name": "zapier_mcp_server",
                "url": "https://actions.zapier.com/mcp/<your-api-key>/sse"
                "transport": "sse",
                "auth_type": "api_key"
            },
            "uuid-2": {
                "name": "google_drive_mcp_server",
                "url": "https://actions.zapier.com/mcp/<your-api-key>/sse"
            }
        ]
        """

        # Per-server outbound tool-call concurrency limiters, lazily created from
        # each server's max_concurrent_requests. Keyed by server_id so the cap
        # survives the registry atomic-swap on config reload; a missing key means
        # the server has no configured limit. The limit is cached alongside the
        # semaphore so an edited limit rebuilds it instead of keeping the old cap
        # until restart.
        self._server_call_semaphores: dict[str, tuple[int, asyncio.Semaphore]] = {}
        self.tool_name_to_mcp_server_name_mapping: dict[str, str] = {}
        """
        {
            "gmail_send_email": "zapier_mcp_server",
        }
        """
        self._upstream_initialize_instructions_by_server_id: dict[str, str] = {}
        # Per-server monotonic timestamp of last upstream prefetch attempt (success,
        # empty result, or failure). Used to throttle re-probes for servers that do
        # not return instructions, and to apply a short cooldown after failures.
        self._upstream_initialize_instructions_probed_at: dict[str, float] = {}

    def _remember_upstream_initialize_instructions(self, server: MCPServer, client: MCPClient) -> None:
        raw = getattr(client, "_last_initialize_instructions", None)
        if raw and str(raw).strip():
            self._upstream_initialize_instructions_by_server_id[server.server_id] = str(raw).strip()

    async def _ensure_upstream_initialize_instructions_cached(self, server: MCPServer) -> None:
        """
        Open one upstream session and cache InitializeResult.instructions if missing.

        No-op when:
          - YAML/DB instructions are set on the server record,
          - server is OpenAPI (spec_path),
          - non-empty upstream instructions are already cached,
          - auth preconditions match health_check_server's skip rules
            (per-user auth / missing static auth token / static headers that
            reference a per-user env var),
          - a prior probe attempt for this server is within
            MCP_HEALTH_CHECK_TIMEOUT seconds (the probe is a health-check-shaped
            op and already uses this knob for its inner call timeout; reusing it
            as the cooldown avoids reconnecting on every gateway initialize when
            upstream returns empty or fails).
        """
        if server.spec_path:
            return
        if server.instructions and server.instructions.strip():
            return
        if self._upstream_initialize_instructions_by_server_id.get(server.server_id):
            return
        if server.requires_per_user_auth:
            return
        if self._references_per_user_env_var(server):
            return
        if (
            server.auth_type
            and server.auth_type != MCPAuth.none
            and server.auth_type != MCPAuth.aws_sigv4
            and not server.authentication_token
        ):
            return

        last_probed_at = self._upstream_initialize_instructions_probed_at.get(server.server_id)
        if last_probed_at is not None and (time.monotonic() - last_probed_at) < MCP_HEALTH_CHECK_TIMEOUT:
            return

        # Record the attempt up-front so that a failure / empty response does not
        # cause every subsequent initialize request to re-open the upstream session.
        self._upstream_initialize_instructions_probed_at[server.server_id] = time.monotonic()

        try:
            resolved_static_headers = await self._resolve_static_headers_with_env_vars(
                server=server,
                user_api_key_auth=None,
                raise_on_missing=False,
            )
            extra_headers: Optional[dict[str, str]] = dict(resolved_static_headers) if resolved_static_headers else None
            client = await self._create_mcp_client(
                server=server,
                mcp_auth_header=None,
                extra_headers=extra_headers,
                stdio_env=None,
            )

            async def _noop(_session):
                return "ok"

            await asyncio.wait_for(client.run_with_session(_noop), timeout=MCP_HEALTH_CHECK_TIMEOUT)
            self._remember_upstream_initialize_instructions(server, client)
        except Exception as e:
            verbose_logger.debug(
                "Upstream initialize instructions prefetch failed for %s: %s",
                server.name,
                e,
            )

    def get_registry(self) -> dict[str, MCPServer]:
        """
        Get the registered MCP Servers from the registry and union with the config MCP Servers
        """
        return self.config_mcp_servers | self.registry

    async def load_servers_from_config(
        self,
        mcp_servers_config: dict[str, Any],
        mcp_aliases: Optional[dict[str, str]] = None,
    ):
        """
        Load the MCP Servers from the config

        Args:
            mcp_servers_config: Dictionary of MCP server configurations
            mcp_aliases: Optional dictionary mapping aliases to server names from litellm_settings
        """
        verbose_logger.debug("Loading MCP Servers from config-----")
        self._upstream_initialize_instructions_by_server_id.clear()
        self._upstream_initialize_instructions_probed_at.clear()

        # Track which aliases have been used to ensure only first occurrence is used
        used_aliases = set()

        for server_name, server_config in mcp_servers_config.items():
            validate_mcp_server_name(server_name)
            _mcp_info: dict[str, Any] = server_config.get("mcp_info", None) or {}
            # Preserve all custom fields from config while setting defaults for core fields
            mcp_info: MCPInfo = _mcp_info.copy()
            # Set default values for core fields if not present
            if "server_name" not in mcp_info:
                mcp_info["server_name"] = server_name
            if "description" not in mcp_info and server_config.get("description"):
                mcp_info["description"] = server_config.get("description")
            _normalize_mcp_server_cost_info(mcp_info)

            # Use alias for name if present, else server_name
            alias = server_config.get("alias", None)

            # Apply mcp_aliases mapping if provided
            if mcp_aliases and alias is None:
                # Check if this server_name has an alias in mcp_aliases
                for alias_name, target_server_name in mcp_aliases.items():
                    if target_server_name == server_name and alias_name not in used_aliases:
                        alias = alias_name
                        used_aliases.add(alias_name)
                        verbose_logger.debug(f"Mapped alias '{alias_name}' to server '{server_name}'")
                        break

            # Create a temporary server object to use with get_server_prefix utility
            temp_server = type(
                "TempServer",
                (),
                {"alias": alias, "server_name": server_name, "server_id": None},
            )()
            name_for_prefix = get_server_prefix(temp_server)

            server_url = server_config.get("url", None) or ""
            # Generate stable server ID based on parameters
            server_id = self._generate_stable_server_id(
                server_name=server_name,
                url=server_url,
                transport=server_config.get("transport", MCPTransport.http),
                auth_type=server_config.get("auth_type", None),
                alias=alias,
            )

            _warn_on_server_name_fields(
                server_id=server_id,
                alias=alias,
                server_name=server_name,
            )

            auth_type = server_config.get("auth_type", None)
            manual_issuer = _blank_to_none(server_config.get("issuer"))
            manual_authorization_url = _blank_to_none(server_config.get("authorization_url"))
            manual_token_url = _blank_to_none(server_config.get("token_url"))
            manual_registration_url = _blank_to_none(server_config.get("registration_url"))
            is_discovery_auth_type = auth_type in _UPSTREAM_OAUTH_DISCOVERY_AUTH_TYPES
            use_issuer_anchor = _uses_issuer_anchor(manual_issuer, is_discovery_auth_type)
            manual_authorization_url, manual_token_url, manual_registration_url = _endpoints_yield_to_issuer(
                manual_issuer,
                is_discovery_auth_type,
                manual_authorization_url,
                manual_token_url,
                manual_registration_url,
            )
            should_discover = bool(server_url) and (
                is_discovery_auth_type
                or self._obo_needs_endpoint_discovery(
                    auth_type,
                    server_config.get("token_exchange_endpoint"),
                    manual_token_url,
                )
            )
            if not should_discover:
                mcp_oauth_metadata = None
            elif manual_issuer is not None and is_discovery_auth_type:
                mcp_oauth_metadata = await self._fetch_issuer_anchored_oauth_metadata(manual_issuer, server_url)
            else:
                mcp_oauth_metadata = await self._descovery_metadata(
                    server_url=server_url,
                    allow_origin_fallback=is_discovery_auth_type,
                )

            if use_issuer_anchor:
                gated_oauth_metadata = mcp_oauth_metadata
            elif is_discovery_auth_type:
                gated_oauth_metadata = _restrict_discovery_to_corroborated_authorization_server(
                    mcp_oauth_metadata,
                    manual_authorization_url,
                    server_name or server_id,
                    bool(server_config.get("dcr_bridge")),
                )
            else:
                gated_oauth_metadata = mcp_oauth_metadata

            # Filter blank scopes (e.g. YAML ``scopes: [""]``) the same way the DB-build path does, so
            # an all-blank list normalizes to None rather than a ``("",)`` tuple that skips the
            # entra_obo fail-closed scope precondition and POSTs an empty scope to the IdP.
            resolved_scopes = self._extract_scopes(server_config.get("scopes")) or (
                gated_oauth_metadata.scopes if gated_oauth_metadata else None
            )
            resolved_authorization_url = manual_authorization_url or (
                gated_oauth_metadata.authorization_url if gated_oauth_metadata else None
            )
            resolved_token_url = manual_token_url or (gated_oauth_metadata.token_url if gated_oauth_metadata else None)
            resolved_registration_url = manual_registration_url or (
                gated_oauth_metadata.registration_url if gated_oauth_metadata else None
            )
            discovered_issuer = (
                gated_oauth_metadata.discovered_issuer
                if gated_oauth_metadata and not gated_oauth_metadata.from_origin_fallback
                else None
            )
            effective_issuer = manual_issuer or discovered_issuer

            config_oauth2_flow = server_config.get("oauth2_flow", None)
            if auth_type == MCPAuth.oauth2 and config_oauth2_flow not in (
                "client_credentials",
                "authorization_code",
            ):
                raise ValueError(
                    f"Invalid config for MCP server '{server_name or server_id}': auth_type oauth2 "
                    f"requires an explicit oauth2_flow (got {config_oauth2_flow!r}). Set "
                    "oauth2_flow: client_credentials for machine-to-machine servers (the proxy mints "
                    "a shared token at token_url using client_id/client_secret, no user interaction) "
                    "or oauth2_flow: authorization_code for interactive servers (per-user tokens via "
                    "browser sign-in, including delegate_auth_to_upstream)."
                )

            config_dcr_bridge = server_config.get("dcr_bridge", None)
            if config_dcr_bridge is not None and not isinstance(config_dcr_bridge, bool):
                raise ValueError(
                    f"Invalid config for MCP server '{server_name or server_id}': dcr_bridge "
                    f"must be a boolean (got {config_dcr_bridge!r})."
                )
            if config_dcr_bridge and auth_type not in (
                MCPAuth.true_passthrough,
                MCPAuth.oauth_delegate,
            ):
                raise ValueError(
                    f"Invalid config for MCP server '{server_name or server_id}': dcr_bridge is only "
                    f"supported for auth_type true_passthrough or oauth_delegate (got {auth_type!r}). "
                    "The DCR bridge serves gateway-hosted OAuth discovery for the client-forwarded "
                    "token modes; interactive oauth2 servers already run the gateway "
                    "authorization-code flow."
                )

            new_server = MCPServer(
                server_id=server_id,
                name=name_for_prefix,
                alias=alias,
                server_name=server_name,
                spec_path=server_config.get("spec_path", None),
                url=server_url,
                command=server_config.get("command", None) or "",
                args=server_config.get("args", None) or [],
                env=server_config.get("env", None) or {},
                # oauth specific fields
                client_id=server_config.get("client_id", None),
                client_secret=server_config.get("client_secret", None),
                oauth2_flow=self._explicit_oauth2_flow(config_oauth2_flow),
                scopes=resolved_scopes,
                issuer=effective_issuer,
                issuer_is_anchored=use_issuer_anchor,
                authorization_url=resolved_authorization_url,
                token_url=resolved_token_url,
                registration_url=resolved_registration_url,
                token_endpoint_auth_method=server_config.get("token_endpoint_auth_method", None),
                # TODO: utility fn the default values
                transport=server_config.get("transport", MCPTransport.http),
                auth_type=auth_type,
                authentication_token=server_config.get("authentication_token", server_config.get("auth_value", None)),
                mcp_info=mcp_info,
                extra_headers=server_config.get("extra_headers", None),
                allowed_tools=server_config.get("allowed_tools", None),
                disallowed_tools=server_config.get("disallowed_tools", None),
                allowed_params=server_config.get("allowed_params", None),
                access_groups=server_config.get("access_groups", None),
                static_headers=server_config.get("static_headers", None),
                env_vars=server_config.get("env_vars", None),
                allow_all_keys=bool(server_config.get("allow_all_keys", False)),
                available_on_public_internet=bool(server_config.get("available_on_public_internet", True)),
                delegate_auth_to_upstream=bool(server_config.get("delegate_auth_to_upstream", False)),
                oauth_passthrough=bool(server_config.get("oauth_passthrough", False)),
                dcr_bridge=config_dcr_bridge,
                # AWS SigV4 fields
                aws_access_key_id=server_config.get("aws_access_key_id", None),
                aws_secret_access_key=server_config.get("aws_secret_access_key", None),
                aws_session_token=server_config.get("aws_session_token", None),
                aws_region_name=server_config.get("aws_region_name", None),
                aws_service_name=server_config.get("aws_service_name", None),
                aws_role_name=server_config.get("aws_role_name", None),
                aws_session_name=server_config.get("aws_session_name", None),
                instructions=server_config.get("instructions", None),
                # Token Exchange (OBO) fields
                token_exchange_endpoint=server_config.get("token_exchange_endpoint", None),
                audience=server_config.get("audience", None),
                subject_token_type=server_config.get(
                    "subject_token_type",
                    DEFAULT_SUBJECT_TOKEN_TYPE,
                ),
                token_exchange_profile=server_config.get("token_exchange_profile", "rfc8693"),
                allow_sampling=bool(server_config.get("allow_sampling", False)),
                allow_elicitation=bool(server_config.get("allow_elicitation", False)),
                timeout=server_config.get("timeout", None),
                max_concurrent_requests=server_config.get("max_concurrent_requests", None),
            )
            self._assign_unique_short_prefix(new_server)
            _warn_internal_delegate_pkce_if_applicable(new_server, source="config")
            self.config_mcp_servers[server_id] = new_server

            # Check if this is an OpenAPI-based server
            spec_path = server_config.get("spec_path", None)
            if spec_path:
                verbose_logger.info(f"Loading OpenAPI spec from {spec_path} for server {server_name}")
                await self._register_openapi_tools(
                    spec_path=spec_path,
                    server=new_server,
                    base_url=server_config.get("url", ""),
                )

        verbose_logger.debug(f"Loaded MCP Servers: {json.dumps(self.config_mcp_servers, indent=4, default=str)}")

        self.initialize_tool_name_to_mcp_server_name_mapping()

    async def _register_openapi_tools(self, spec_path: str, server: MCPServer, base_url: str):
        """
        Register tools from an OpenAPI specification for a given server.

        This creates "virtual" MCP tools from OpenAPI endpoints that are:
        1. Registered in the global tool registry with server prefix
        2. Mapped to the server for routing
        3. Executed via the local tool handler

        Args:
            spec_path: Path to the OpenAPI specification file
            server: The MCPServer instance to register tools for
            base_url: Base URL for API calls
        """
        from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
            build_input_schema,
            create_tool_function,
        )
        from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
            get_base_url as get_openapi_base_url,
        )
        from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
            load_openapi_spec_async,
            resolve_operation_params,
        )
        from litellm.proxy._experimental.mcp_server.tool_registry import (
            global_mcp_tool_registry,
        )

        try:
            # Load OpenAPI spec (async to avoid "called from within a running event loop")
            spec = await load_openapi_spec_async(spec_path)

            # Use base_url from config if provided, otherwise extract from spec
            if not base_url:
                base_url = get_openapi_base_url(spec, spec_path)
            verbose_logger.info(f"Registering OpenAPI tools for server {server.name} with base URL: {base_url}")

            # Get server prefix for tool naming
            server_prefix = get_server_prefix(server)

            # Build headers from server configuration
            headers: dict[str, str] = {}

            # Add authentication headers if configured
            if server.authentication_token:
                from litellm.types.mcp import MCPAuth

                if server.auth_type == MCPAuth.bearer_token:
                    headers["Authorization"] = f"Bearer {server.authentication_token}"
                elif server.auth_type == MCPAuth.api_key:
                    headers["Authorization"] = f"ApiKey {server.authentication_token}"
                elif server.auth_type == MCPAuth.basic:
                    headers["Authorization"] = f"Basic {server.authentication_token}"
                elif server.auth_type == MCPAuth.token:
                    headers["Authorization"] = f"token {server.authentication_token}"

            # Add any static headers from server config.
            #
            # Note: `extra_headers` on MCPServer is a List[str] of header names to forward
            # from each client MCP request; values are applied at call time via
            # `_request_extra_headers` in server.py (not baked in here).
            # `static_headers` is a dict of concrete headers to always send.
            headers = (
                merge_mcp_headers(
                    extra_headers=headers,
                    static_headers=server.static_headers,
                )
                or {}
            )

            verbose_logger.debug(
                f"Using headers for OpenAPI tools (excluding sensitive values): {list(headers.keys())}"
            )

            # Extract and register tools from OpenAPI paths
            paths = spec.get("paths", {})
            components = spec.get("components", {})
            registered_count = 0

            verbose_logger.debug(f"Processing {len(paths)} paths from OpenAPI spec")

            for path, path_item in paths.items():
                for method in ["get", "post", "put", "delete", "patch"]:
                    if method not in path_item:
                        continue

                    operation = path_item[method]

                    # Resolve $ref params and merge path-level params into the operation.
                    resolved_operation = resolve_operation_params(operation, path_item, components)

                    # Generate tool name (without prefix initially)
                    operation_id = operation.get("operationId", f"{method}_{path.replace('/', '_')}")
                    base_tool_name = operation_id.replace(" ", "_").lower()

                    # Add server prefix to tool name
                    prefixed_tool_name = add_server_prefix_to_name(base_tool_name, server_prefix)

                    # Get description
                    description = operation.get(
                        "summary",
                        operation.get("description", f"{method.upper()} {path}"),
                    )

                    # Build input schema using imported function
                    input_schema = build_input_schema(resolved_operation)

                    # Create tool function with headers using imported function
                    tool_func = create_tool_function(path, method, resolved_operation, base_url, headers=headers)
                    tool_func.__name__ = prefixed_tool_name
                    tool_func.__doc__ = description

                    # Register tool with prefixed name in global registry
                    global_mcp_tool_registry.register_tool(
                        name=prefixed_tool_name,
                        description=description,
                        input_schema=input_schema,
                        handler=tool_func,
                    )

                    # Update tool name to server name mapping (for both prefixed and base names)
                    self.tool_name_to_mcp_server_name_mapping[base_tool_name] = server_prefix
                    self.tool_name_to_mcp_server_name_mapping[prefixed_tool_name] = server_prefix

                    registered_count += 1
                    verbose_logger.debug(f"Registered OpenAPI tool: {prefixed_tool_name} for server {server.name}")

            verbose_logger.info(f"Successfully registered {registered_count} OpenAPI tools for server {server.name}")

        except Exception as e:
            verbose_logger.error(f"Failed to register OpenAPI tools for server {server.name}: {str(e)}")
            raise e

    def _cleanup_server_tool_routing_artifacts(self, server: MCPServer) -> None:
        """Drop OpenAPI global tools and name-mapping rows owned by ``server``.

        When a server leaves ``self.registry`` (eviction, ``remove_server``, etc.),
        OpenAPI tools remain in ``global_mcp_tool_registry`` and
        ``tool_name_to_mcp_server_name_mapping`` unless removed here. Stale
        mappings make ``_get_mcp_server_from_tool_name`` resolve to a prefix that
        no longer exists in the live registry.
        """
        from litellm.proxy._experimental.mcp_server.tool_registry import (
            global_mcp_tool_registry,
        )

        prefix_root = normalize_server_name(get_server_prefix(server))
        if server.spec_path and prefix_root:
            openapi_key_prefix = prefix_root + MCP_TOOL_PREFIX_SEPARATOR
            global_mcp_tool_registry.unregister_tools_with_prefix(openapi_key_prefix)

        owned_raw: set[str] = set()
        for p in iter_known_server_prefixes(server):
            if p:
                owned_raw.add(p)
        if server.name:
            owned_raw.add(server.name)

        owned_normalized = {normalize_server_name(x) for x in owned_raw}

        stale_mapping_keys: list[str] = []
        for tool_name, mapped_server in list(self.tool_name_to_mcp_server_name_mapping.items()):
            if mapped_server in owned_raw:
                stale_mapping_keys.append(tool_name)
            elif normalize_server_name(str(mapped_server)) in owned_normalized:
                stale_mapping_keys.append(tool_name)

        for key in stale_mapping_keys:
            del self.tool_name_to_mcp_server_name_mapping[key]

    def remove_server(self, mcp_server: LiteLLM_MCPServerTable):
        """
        Remove a server from the registry
        """
        evicted: Optional[MCPServer] = self.registry.pop(mcp_server.server_id, None)
        if evicted is None and mcp_server.server_name:
            evicted = self.registry.pop(mcp_server.server_name, None)
        if evicted is not None:
            verbose_logger.debug("Removed MCP Server: %s", mcp_server.server_id or mcp_server.server_name)
            self._cleanup_server_tool_routing_artifacts(evicted)
        else:
            verbose_logger.warning(f"Server ID {mcp_server.server_id} not found in registry")

    def _resolve_env_vars_list(
        self,
        mcp_server: LiteLLM_MCPServerTable,
        *,
        env_vars_are_encrypted: bool,
    ) -> Optional[list[dict[str, Any]]]:
        env_vars_list = _deserialize_json_list(getattr(mcp_server, "env_vars", None))
        if env_vars_are_encrypted:
            from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415
                decrypt_global_env_var_values,
            )

            decrypt_global_env_var_values(env_vars_list)
        return env_vars_list

    async def _resolve_table_oauth_metadata(
        self,
        *,
        mcp_server: LiteLLM_MCPServerTable,
        auth_type: MCPAuthType,
        server_url: Optional[str],
        manual_issuer: Optional[str],
        manual_authorization_url: Optional[str],
        manual_token_url: Optional[str],
        is_discovery_auth_type: bool,
        use_issuer_anchor: bool,
        scopes: Optional[list[str]],
        token_exchange_endpoint: Optional[str],
    ) -> Optional[MCPOAuthMetadata]:
        has_all_upstream_oauth_fields = bool(manual_authorization_url and manual_token_url and scopes)
        needs_discovery = bool(server_url) and (
            (is_discovery_auth_type and not has_all_upstream_oauth_fields)
            or self._obo_needs_endpoint_discovery(auth_type, token_exchange_endpoint, manual_token_url)
        )
        if not needs_discovery:
            mcp_oauth_metadata: Optional[MCPOAuthMetadata] = None
        elif use_issuer_anchor and manual_issuer is not None:
            mcp_oauth_metadata = await self._fetch_issuer_anchored_oauth_metadata(manual_issuer, server_url)
        else:
            mcp_oauth_metadata = await self._descovery_metadata(
                server_url=server_url,  # type: ignore[arg-type]
                allow_origin_fallback=is_discovery_auth_type,
            )
        if needs_discovery and not use_issuer_anchor and mcp_oauth_metadata is None:
            verbose_logger.warning(
                "MCP OAuth discovery yielded no metadata for server %s (%s); "
                "OAuth endpoints/scopes stay unresolved until a rebuild succeeds",
                mcp_server.server_id,
                server_url,
            )
        if use_issuer_anchor:
            return mcp_oauth_metadata
        if is_discovery_auth_type:
            return _restrict_discovery_to_corroborated_authorization_server(
                mcp_oauth_metadata,
                manual_authorization_url,
                mcp_server.server_id,
                bool(getattr(mcp_server, "dcr_bridge", None)),
            )
        return mcp_oauth_metadata

    async def build_mcp_server_from_table(
        self,
        mcp_server: LiteLLM_MCPServerTable,
        *,
        credentials_are_encrypted: bool = True,
        env_vars_are_encrypted: Optional[bool] = None,
        persist_discovered_endpoints: bool = True,
    ) -> MCPServer:
        _mcp_info: MCPInfo = mcp_server.mcp_info or {}
        env_dict = _deserialize_json_dict(getattr(mcp_server, "env", None))
        static_headers_dict = _deserialize_json_dict(getattr(mcp_server, "static_headers", None))
        env_vars_list = self._resolve_env_vars_list(
            mcp_server,
            env_vars_are_encrypted=(
                credentials_are_encrypted if env_vars_are_encrypted is None else env_vars_are_encrypted
            ),
        )
        credentials_dict = _deserialize_json_dict(getattr(mcp_server, "credentials", None))

        encrypted_auth_value: Optional[str] = None
        encrypted_client_id: Optional[str] = None
        encrypted_client_secret: Optional[str] = None
        if credentials_dict:
            encrypted_auth_value = credentials_dict.get("auth_value")
            encrypted_client_id = credentials_dict.get("client_id")
            encrypted_client_secret = credentials_dict.get("client_secret")

        auth_value: Optional[str] = None
        if encrypted_auth_value:
            if credentials_are_encrypted:
                auth_value = decrypt_value_helper(
                    value=encrypted_auth_value,
                    key="auth_value",
                    exception_type="debug",
                    return_original_value=True,
                )
            else:
                auth_value = encrypted_auth_value

        client_id_value: Optional[str] = None
        if encrypted_client_id:
            if credentials_are_encrypted:
                client_id_value = decrypt_value_helper(
                    value=encrypted_client_id,
                    key="client_id",
                    exception_type="debug",
                    return_original_value=True,
                )
            else:
                client_id_value = encrypted_client_id

        client_secret_value: Optional[str] = None
        if encrypted_client_secret:
            if credentials_are_encrypted:
                client_secret_value = decrypt_value_helper(
                    value=encrypted_client_secret,
                    key="client_secret",
                    exception_type="debug",
                    return_original_value=True,
                )
            else:
                client_secret_value = encrypted_client_secret

        # AWS SigV4 credential fields
        aws_creds = self._extract_aws_credentials(credentials_dict, credentials_are_encrypted)

        scopes: Optional[list[str]] = None
        if credentials_dict:
            scopes_value = credentials_dict.get("scopes")
            if scopes_value is not None:
                scopes = self._extract_scopes(scopes_value)

        name_for_prefix = mcp_server.alias or mcp_server.server_name or mcp_server.server_id

        mcp_info: MCPInfo = _mcp_info.copy()
        if "server_name" not in mcp_info:
            mcp_info["server_name"] = mcp_server.server_name or mcp_server.server_id
        if "description" not in mcp_info and mcp_server.description:
            mcp_info["description"] = mcp_server.description
        _normalize_mcp_server_cost_info(mcp_info)

        auth_type = cast(MCPAuthType, mcp_server.auth_type)
        server_url = mcp_server.url
        manual_issuer = _blank_to_none(mcp_server.issuer)
        manual_authorization_url = _blank_to_none(mcp_server.authorization_url)
        manual_token_url = _blank_to_none(mcp_server.token_url)
        manual_registration_url = _blank_to_none(mcp_server.registration_url)
        is_discovery_auth_type = auth_type in _UPSTREAM_OAUTH_DISCOVERY_AUTH_TYPES
        use_issuer_anchor = _uses_issuer_anchor(manual_issuer, is_discovery_auth_type)
        manual_authorization_url, manual_token_url, manual_registration_url = _endpoints_yield_to_issuer(
            manual_issuer, is_discovery_auth_type, manual_authorization_url, manual_token_url, manual_registration_url
        )
        token_exchange_endpoint = mcp_server.token_exchange_endpoint or (
            credentials_dict.get("token_exchange_endpoint") if credentials_dict else None
        )
        gated_oauth_metadata = await self._resolve_table_oauth_metadata(
            mcp_server=mcp_server,
            auth_type=auth_type,
            server_url=server_url,
            manual_issuer=manual_issuer,
            manual_authorization_url=manual_authorization_url,
            manual_token_url=manual_token_url,
            is_discovery_auth_type=is_discovery_auth_type,
            use_issuer_anchor=use_issuer_anchor,
            scopes=scopes,
            token_exchange_endpoint=token_exchange_endpoint,
        )

        resolved_scopes = scopes or (gated_oauth_metadata.scopes if gated_oauth_metadata else None)
        discovered_issuer = (
            gated_oauth_metadata.discovered_issuer
            if gated_oauth_metadata and not gated_oauth_metadata.from_origin_fallback
            else None
        )
        effective_issuer = manual_issuer or discovered_issuer

        new_server = MCPServer(
            server_id=mcp_server.server_id,
            name=name_for_prefix,
            alias=getattr(mcp_server, "alias", None),
            server_name=getattr(mcp_server, "server_name", None),
            url=mcp_server.url,
            spec_path=getattr(mcp_server, "spec_path", None),
            transport=cast(MCPTransportType, mcp_server.transport),
            auth_type=auth_type,
            authentication_token=auth_value,
            mcp_info=mcp_info,
            extra_headers=getattr(mcp_server, "extra_headers", None),
            static_headers=static_headers_dict,
            env_vars=env_vars_list,
            client_id=client_id_value or getattr(mcp_server, "client_id", None),
            client_secret=client_secret_value or getattr(mcp_server, "client_secret", None),
            oauth2_flow=self._explicit_oauth2_flow(getattr(mcp_server, "oauth2_flow", None)),
            scopes=resolved_scopes,
            issuer=effective_issuer,
            issuer_is_anchored=use_issuer_anchor,
            authorization_url=manual_authorization_url or getattr(gated_oauth_metadata, "authorization_url", None),
            token_url=manual_token_url or getattr(gated_oauth_metadata, "token_url", None),
            registration_url=manual_registration_url or getattr(gated_oauth_metadata, "registration_url", None),
            token_endpoint_auth_method=(
                credentials_dict.get("token_endpoint_auth_method") if credentials_dict else None
            ),
            command=getattr(mcp_server, "command", None),
            args=getattr(mcp_server, "args", None) or [],
            env=env_dict,
            access_groups=getattr(mcp_server, "mcp_access_groups", None),
            allowed_tools=getattr(mcp_server, "allowed_tools", None),
            disallowed_tools=getattr(mcp_server, "disallowed_tools", None),
            allow_all_keys=mcp_server.allow_all_keys,
            available_on_public_internet=bool(getattr(mcp_server, "available_on_public_internet", True)),
            delegate_auth_to_upstream=bool(getattr(mcp_server, "delegate_auth_to_upstream", False)),
            oauth_passthrough=bool(getattr(mcp_server, "oauth_passthrough", False)),
            dcr_bridge=getattr(mcp_server, "dcr_bridge", None),
            created_at=getattr(mcp_server, "created_at", None),
            updated_at=getattr(mcp_server, "updated_at", None),
            tool_name_to_display_name=_deserialize_json_dict(getattr(mcp_server, "tool_name_to_display_name", None)),
            tool_name_to_description=_deserialize_json_dict(getattr(mcp_server, "tool_name_to_description", None)),
            is_byok=bool(getattr(mcp_server, "is_byok", False)),
            byok_description=getattr(mcp_server, "byok_description", None) or [],
            byok_api_key_help_url=getattr(mcp_server, "byok_api_key_help_url", None),
            source_url=getattr(mcp_server, "source_url", None),
            # AWS SigV4 fields
            aws_access_key_id=aws_creds.get("aws_access_key_id"),
            aws_secret_access_key=aws_creds.get("aws_secret_access_key"),
            aws_session_token=aws_creds.get("aws_session_token"),
            aws_region_name=aws_creds.get("aws_region_name"),
            aws_service_name=aws_creds.get("aws_service_name"),
            aws_role_name=aws_creds.get("aws_role_name"),
            aws_session_name=aws_creds.get("aws_session_name"),
            instructions=mcp_server.instructions,
            # Token exchange (OBO) fields: dedicated columns, with the credentials blob as a
            # back-compat fallback for servers persisted before the columns existed.
            token_exchange_endpoint=mcp_server.token_exchange_endpoint
            or (credentials_dict.get("token_exchange_endpoint") if credentials_dict else None),
            audience=mcp_server.audience or (credentials_dict.get("audience") if credentials_dict else None),
            subject_token_type=mcp_server.subject_token_type
            or (credentials_dict.get("subject_token_type") if credentials_dict else None)
            or DEFAULT_SUBJECT_TOKEN_TYPE,
            token_exchange_profile=mcp_server.token_exchange_profile
            or (credentials_dict.get("token_exchange_profile") if credentials_dict else None)
            or "rfc8693",
            timeout=getattr(mcp_server, "timeout", None),
            max_concurrent_requests=getattr(mcp_server, "max_concurrent_requests", None),
        )
        _warn_internal_delegate_pkce_if_applicable(new_server, source="database")
        if persist_discovered_endpoints:
            await self._persist_discovered_obo_token_url(
                server_id=mcp_server.server_id,
                auth_type=auth_type,
                existing_token_url=manual_token_url,
                discovered_token_url=new_server.token_url,
            )
            await self._persist_discovered_oauth_endpoints(
                server_id=mcp_server.server_id,
                auth_type=auth_type,
                existing_issuer=manual_issuer,
                existing_authorization_url=manual_authorization_url,
                existing_token_url=manual_token_url,
                existing_scopes=scopes,
                metadata=gated_oauth_metadata,
                is_issuer_anchored=use_issuer_anchor,
            )
        return new_server

    async def _persist_discovered_obo_token_url(
        self,
        *,
        server_id: str,
        auth_type: Optional[MCPAuthType],
        existing_token_url: Optional[str],
        discovered_token_url: Optional[str],
    ) -> None:
        """Write a freshly discovered OBO token endpoint back onto the DB row.

        ``build_mcp_server_from_table`` resolves ``token_url`` via RFC 9728 -> RFC 8414 for an
        ``oauth2_token_exchange`` server that has none configured, but that resolved value otherwise
        lives only on the returned in-memory object; the row keeps ``token_url=None`` so every rebuild
        re-runs discovery, and a transient upstream outage during a rebuild leaves the server with no
        endpoint until discovery next succeeds. Persisting it makes ``_obo_needs_endpoint_discovery``
        return False on the next build. Fires at most once per server (skipped once the row has a
        value), and is best-effort: a write failure just means discovery runs again next time.
        """
        if auth_type != MCPAuth.oauth2_token_exchange:
            return
        if existing_token_url or not discovered_token_url:
            return
        from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415

        if prisma_client is None:
            return
        try:
            await MCPServerRepository(prisma_client).table.update(
                where={"server_id": server_id},
                data={"token_url": discovered_token_url},
            )
            verbose_logger.debug("Persisted discovered OBO token_url for MCP server %s", server_id)
        except Exception as exc:  # noqa: BLE001 - best-effort; a failed write re-discovers next build
            verbose_logger.warning("Failed to persist discovered OBO token_url for MCP server %s: %s", server_id, exc)

    async def _persist_discovered_oauth_endpoints(
        self,
        *,
        server_id: str,
        auth_type: MCPAuthType | None,
        existing_issuer: str | None,
        existing_authorization_url: str | None,
        existing_token_url: str | None,
        existing_scopes: list[str] | None,
        metadata: MCPOAuthMetadata | None,
        is_issuer_anchored: bool = False,
    ) -> None:
        """Write freshly discovered OAuth endpoints back onto the DB row.

        Same rationale as ``_persist_discovered_obo_token_url`` but for the interactive oauth2
        family: discovered ``authorization_url``/``token_url``/``scopes`` otherwise live only on
        the in-memory registry entry, which is rebuilt on every client connect (the DCR reuse path
        calls ``update_server``) and on every post-write DB reload, so one failed re-discovery
        serves 400 "authorization url is not set" from /authorize until a later rebuild succeeds.
        Only fills row fields that are currently empty, never persists origin-fallback guesses
        (RFC 9728/8414-advertised metadata only), and deliberately skips ``registration_url``
        because ``_dcr_bridge_relays_client_registration`` keys off that column. Best-effort: a
        failed write re-discovers on the next build. Scopes go through ``update_mcp_server`` so
        they merge into the credentials blob without touching the stored client credentials.

        For an issuer-anchored server (``is_issuer_anchored``) the endpoints are re-derived from the
        §3.3-validated issuer document on every build, so they are NOT persisted into the endpoint
        columns: persisting them would make the next build see populated endpoints and treat them as
        authoritative stored values, defeating the "endpoints come solely from the issuer" invariant.
        Only the resource-driven scopes are persisted for such servers.
        """
        if auth_type not in _UPSTREAM_OAUTH_DISCOVERY_AUTH_TYPES:
            return
        if metadata is None or metadata.from_origin_fallback:
            return
        issuer_update = (
            {"issuer": metadata.discovered_issuer} if metadata.discovered_issuer and not existing_issuer else {}
        )
        authorization_url_update = (
            {"authorization_url": metadata.authorization_url}
            if metadata.authorization_url and not existing_authorization_url and not is_issuer_anchored
            else {}
        )
        token_url_update = (
            {"token_url": metadata.token_url}
            if metadata.token_url and not existing_token_url and not is_issuer_anchored
            else {}
        )
        scopes_update = {"credentials": {"scopes": metadata.scopes}} if metadata.scopes and not existing_scopes else {}
        updates: dict[str, object] = {
            **issuer_update,
            **authorization_url_update,
            **token_url_update,
            **scopes_update,
        }
        if not updates:
            return
        from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415  # db.py imports this module at load
            update_mcp_server,
        )
        from litellm.proxy._types import UpdateMCPServerRequest  # noqa: PLC0415  # heavy module; import at call time
        from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime value, set after startup

        if prisma_client is None:
            return
        try:
            await update_mcp_server(
                prisma_client=prisma_client,
                data=UpdateMCPServerRequest.model_validate({"server_id": server_id, **updates}),
                touched_by="mcp_oauth_discovery",
            )
            verbose_logger.info(
                "Persisted discovered OAuth endpoints for MCP server %s: %s",
                server_id,
                sorted(updates),
            )
        except Exception as exc:  # noqa: BLE001 - best-effort; a failed write re-discovers next build
            verbose_logger.warning(
                "Failed to persist discovered OAuth endpoints for MCP server %s: %s",
                server_id,
                exc,
            )

    async def _maybe_register_openapi_tools(self, server: MCPServer, *, initialize_mapping: bool = True):
        """Register OpenAPI tools if the server has a spec_path configured."""
        if server.spec_path:
            verbose_logger.info(f"Loading OpenAPI spec from {server.spec_path} for server {server.name}")
            await self._register_openapi_tools(
                spec_path=server.spec_path,
                server=server,
                base_url=server.url or "",
            )
            if initialize_mapping:
                self.initialize_tool_name_to_mcp_server_name_mapping()

    async def add_server(self, mcp_server: LiteLLM_MCPServerTable):
        # The runtime registry is the allowlist for tool calls and health
        # probes (which spawn the underlying transport, including stdio
        # subprocesses). Match the eligibility set used by the bulk DB
        # filter in reload_servers_from_database() — NULL is legacy and
        # "approved" is a legacy alias for "active".
        if mcp_server.approval_status not in (None, "active", "approved"):
            return
        try:
            if mcp_server.server_id not in self.registry:
                # Callers hand us a record returned by the db.py read/write
                # helpers, which already decrypt global env var values (the
                # `credentials` field is the only one still encrypted here).
                # Re-decrypting plaintext would zero the values, so build with
                # env_vars_are_encrypted=False.
                new_server = await self.build_mcp_server_from_table(mcp_server, env_vars_are_encrypted=False)
                self._assign_unique_short_prefix(new_server)
                self.registry[mcp_server.server_id] = new_server
                await self._maybe_register_openapi_tools(new_server)
                verbose_logger.debug(f"Added MCP Server: {new_server.name}")

        except Exception as e:
            verbose_logger.debug(f"Failed to add MCP server: {str(e)}")
            raise e

    async def update_server(self, mcp_server: LiteLLM_MCPServerTable):
        # If a previously-active server has been moved out of the active
        # state, evict any stale registry entry so subsequent tool calls and
        # health probes can't reach it.
        if mcp_server.approval_status not in (None, "active", "approved"):
            evicted = self.registry.pop(mcp_server.server_id, None)
            if evicted is None and mcp_server.server_name:
                evicted = self.registry.pop(mcp_server.server_name, None)
            if evicted is not None:
                self._cleanup_server_tool_routing_artifacts(evicted)
            return
        try:
            if mcp_server.server_id in self.registry:
                # See add_server: db.py helpers already decrypted env var
                # values, so don't decrypt them a second time here.
                new_server = await self.build_mcp_server_from_table(mcp_server, env_vars_are_encrypted=False)
                # Carry the previously-resolved short prefix across so the
                # tool names stay stable for clients holding cached lists.
                existing_prefix = self.registry[mcp_server.server_id].short_prefix
                if existing_prefix and not new_server.short_prefix:
                    new_server.short_prefix = existing_prefix
                _carry_forward_resolved_oauth_endpoints(
                    new_server=new_server,
                    previous_server=self.registry[mcp_server.server_id],
                )
                self._assign_unique_short_prefix(new_server)
                self.registry[mcp_server.server_id] = new_server
                await self._maybe_register_openapi_tools(new_server)
                verbose_logger.debug(f"Updated MCP Server: {new_server.name}")

        except Exception as e:
            verbose_logger.debug(f"Failed to udpate MCP server: {str(e)}")
            raise e

    def get_all_mcp_server_ids(self) -> set[str]:
        """
        Get all MCP server IDs
        """
        all_servers = list(self.get_registry().values())
        return {server.server_id for server in all_servers}

    def get_allow_all_keys_server_ids(self) -> list[str]:
        """Return server IDs that bypass per-key restrictions."""
        return [server.server_id for server in self.get_registry().values() if server.allow_all_keys is True]

    @staticmethod
    def get_byom_submitted_servers_cache_key(user_id: str) -> str:
        return f"byom_submitted_servers:{user_id}"

    async def invalidate_byom_submitted_servers_cache(self, user_id: str | None) -> None:
        if not user_id:
            return
        try:
            from litellm.proxy.proxy_server import user_api_key_cache

            await user_api_key_cache.async_delete_cache(key=self.get_byom_submitted_servers_cache_key(user_id))
        except Exception as e:  # noqa: BLE001
            verbose_logger.warning(f"Failed to invalidate BYOM submitted MCP server cache: {str(e)}")

    async def _get_active_submitted_mcp_server_ids_for_user(
        self, user_api_key_auth: UserAPIKeyAuth | None
    ) -> list[str]:
        submitter_user_id = getattr(user_api_key_auth, "user_id", None) if user_api_key_auth else None
        if not submitter_user_id:
            return []

        try:
            from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415
                get_active_submitted_mcp_server_ids_for_user,
            )
            from litellm.proxy.proxy_server import prisma_client, user_api_key_cache
        except Exception as e:  # noqa: BLE001
            verbose_logger.warning(f"Failed to load BYOM submitted MCP server cache dependencies: {str(e)}")
            return []

        byom_cache_key = self.get_byom_submitted_servers_cache_key(submitter_user_id)
        submitted_server_ids: list[str] | None = None
        try:
            cached_submitted_server_ids = await user_api_key_cache.async_get_cache(key=byom_cache_key)
            if cached_submitted_server_ids is not None:
                submitted_server_ids = cast(list[str], cached_submitted_server_ids)
        except Exception as e:  # noqa: BLE001
            verbose_logger.warning(f"Failed to read BYOM submitted MCP server cache: {str(e)}")

        if submitted_server_ids is None:
            if prisma_client is None:
                submitted_server_ids = []
            else:
                try:
                    submitted_server_ids = await get_active_submitted_mcp_server_ids_for_user(
                        prisma_client, submitter_user_id
                    )
                except Exception as e:  # noqa: BLE001
                    verbose_logger.warning(f"Failed to read BYOM submitted MCP servers from database: {str(e)}")
                    submitted_server_ids = []
            try:
                await user_api_key_cache.async_set_cache(
                    key=byom_cache_key,
                    value=submitted_server_ids,
                    ttl=60,
                )
            except Exception as e:  # noqa: BLE001
                verbose_logger.warning(f"Failed to write BYOM submitted MCP server cache: {str(e)}")

        return [server_id for server_id in submitted_server_ids if self.get_mcp_server_by_id(server_id) is not None]

    async def get_allowed_mcp_servers(self, user_api_key_auth: Optional[UserAPIKeyAuth] = None) -> list[str]:
        """
        Get the allowed MCP Servers for the user.

        Priority:
        1. If object_permission.mcp_servers is explicitly set, use it (even for admins)
        2. If admin and no object_permission, return all servers
        3. Otherwise, use standard permission checks
        """
        from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view

        allow_all_server_ids = self.get_allow_all_keys_server_ids()

        # The key explicitly opted out of every MCP server. Return zero before
        # layering on allow_all_keys or submitted servers so the opt-out is absolute.
        key_object_permission = user_api_key_auth.object_permission if user_api_key_auth else None
        if key_object_permission is not None and (
            SpecialMCPServerNames.no_mcp_servers.value in (key_object_permission.mcp_servers or [])
        ):
            return []

        # Check if object_permission.mcp_servers is explicitly set (not None, empty list is valid)
        has_explicit_object_permission = key_object_permission is not None and (
            key_object_permission.mcp_servers is not None
        )
        if has_explicit_object_permission:
            verbose_logger.debug(f"Object permission mcp_servers explicitly set: {key_object_permission.mcp_servers}")

        # BYOM creator visibility never widens a key that was explicitly scoped:
        # only keys without their own mcp_servers list get submitted servers unioned in.
        submitted_server_ids = (
            []
            if has_explicit_object_permission
            else await self._get_active_submitted_mcp_server_ids_for_user(user_api_key_auth)
        )

        try:
            # If admin but NO explicit object permission, get all servers
            if user_api_key_auth and _user_has_admin_view(user_api_key_auth) and not has_explicit_object_permission:
                verbose_logger.debug("Admin user without explicit object_permission - returning all servers")
                return list(self.get_registry().keys())

            # Get allowed servers from object permissions (respects object_permission even for admins)
            allowed_mcp_servers = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth)
            verbose_logger.debug(f"Allowed MCP Servers for user api key auth: {allowed_mcp_servers}")
            combined_servers = set(allowed_mcp_servers)
            # Only skip allow_all_keys servers when the request is inside a toolset
            # scope.  toolset_mcp_route / dynamic_mcp_route set _mcp_active_toolset_id
            # before calling the handler — that ContextVar is the reliable signal.
            # Using op.mcp_toolsets==[] would false-positive on DB-default rows where
            # Postgres initialises the column to ARRAY[]::TEXT[].
            from litellm.proxy._experimental.mcp_server.mcp_context import (  # noqa: PLC0415
                _mcp_active_toolset_id,
            )

            in_toolset_scope = _mcp_active_toolset_id.get() is not None
            if not in_toolset_scope:
                combined_servers.update(allow_all_server_ids)
                combined_servers.update(submitted_server_ids)

            # For anonymous callers (no user_id, no role), also surface any
            # servers the operator has opted into upstream-delegated auth.
            # These servers handle their own auth at the upstream level, so
            # LiteLLM granting access here does not bypass any security gate.
            is_anonymous = not (
                user_api_key_auth
                and (
                    getattr(user_api_key_auth, "user_id", None)
                    or getattr(user_api_key_auth, "user_role", None)
                    or getattr(user_api_key_auth, "api_key", None)
                )
            )
            if is_anonymous:
                delegate_server_ids = [
                    server.server_id
                    for server in self.get_registry().values()
                    if (
                        getattr(server, "auth_type", None) == MCPAuth.oauth2
                        and getattr(server, "delegate_auth_to_upstream", False) is True
                        # M2M servers must not be exposed anonymously: an
                        # unauthenticated caller would get LiteLLM to proxy tool
                        # calls using its stored client_credentials. Resolve the flow
                        # rather than reading has_client_credentials so an unstamped
                        # M2M-shape row (null column, verbatim-read as non-M2M) still
                        # fails closed here, matching the anonymous-delegate auth gate.
                        and MCPServerManager.effective_oauth2_flow(server) != "client_credentials"
                    )
                    or getattr(server, "auth_type", None) == MCPAuth.true_passthrough
                ]
                combined_servers.update(delegate_server_ids)

            if len(combined_servers) == 0:
                verbose_logger.debug("No allowed MCP Servers found for user api key auth.")
            return list(combined_servers)
        except Exception:  # noqa: BLE001
            verbose_logger.exception(
                "Failed to get allowed MCP servers; team-level object_permission "
                "grants may be dropped. Falling back to global and submitted servers."
            )
            return list(dict.fromkeys(allow_all_server_ids + submitted_server_ids))

    async def resolve_toolset_tool_permissions(
        self,
        toolset_ids: list[str],
    ) -> dict[str, list[str]]:
        """
        Resolve a list of toolset IDs into a mcp_tool_permissions dict.

        Returns: {server_id: [tool_name, ...]} — the union of all tools across
        the given toolsets.  Results are cached via ``user_api_key_cache`` (a
        Redis-backed ``DualCache`` in production) so that cache entries are
        shared across workers and cold-cache DB hits are minimised.
        """
        from litellm.proxy._experimental.mcp_server.toolset_db import list_mcp_toolsets
        from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

        if not toolset_ids or prisma_client is None:
            return {}

        cache_key = "toolset_perms:" + ",".join(sorted(toolset_ids))
        cached = await user_api_key_cache.async_get_cache(key=cache_key)
        if cached is not None:
            return cached

        try:
            toolsets = await list_mcp_toolsets(prisma_client, toolset_ids=toolset_ids)
            tool_permissions: dict[str, list[str]] = {}
            for toolset in toolsets:
                for tool in toolset.tools:
                    raw_name = tool["tool_name"]
                    server = self.get_mcp_server_by_id(tool["server_id"])
                    unprefixed = strip_known_server_prefix(raw_name, server)
                    tool_permissions.setdefault(tool["server_id"], [])
                    if unprefixed not in tool_permissions[tool["server_id"]]:
                        tool_permissions[tool["server_id"]].append(unprefixed)
            await user_api_key_cache.async_set_cache(
                key=cache_key,
                value=tool_permissions,
                ttl=get_management_object_ttl(user_api_key_cache),
            )
            return tool_permissions
        except Exception as e:
            verbose_logger.warning(f"Failed to resolve toolset permissions: {str(e)}")
            return {}

    def invalidate_toolset_cache(self, toolset_id: Optional[str] = None) -> None:
        """Evict cached toolset permission entries.

        Called after create/update/delete of a toolset so stale data is not served.
        The in-memory layer of ``user_api_key_cache`` is cleared immediately;
        Redis entries expire naturally after the configured TTL.
        Pass toolset_id to evict only entries containing that ID, or None to clear all.
        """
        # Clear the in-memory layer of the shared DualCache for affected keys.
        # We can't enumerate Redis keys by pattern, so Redis entries expire via TTL.
        try:
            from litellm.proxy.proxy_server import user_api_key_cache

            in_mem = getattr(user_api_key_cache, "in_memory_cache", None)
            if in_mem is None:
                return
            cache_dict = getattr(in_mem, "cache_dict", {})
            if toolset_id is None:
                keys_to_remove = [k for k in cache_dict if k.startswith("toolset_")]
            else:
                # Evict permission-cache entries that reference this toolset ID.
                # Also evict ALL name-cache entries (toolset_name:*): we can't map
                # toolset_id → toolset_name without a DB call, and the name may have
                # changed in an update anyway.
                keys_to_remove = [
                    k
                    for k in cache_dict
                    if (k.startswith("toolset_perms:") and toolset_id in k) or k.startswith("toolset_name:")
                ]
            for k in keys_to_remove:
                cache_dict.pop(k, None)
        except Exception as e:
            verbose_logger.warning(f"invalidate_toolset_cache: failed to evict in-memory entries: {e}")

    async def get_toolset_by_name_cached(
        self,
        prisma_client: Any,
        toolset_name: str,
    ) -> Optional[Any]:
        """Return a toolset by name, cached in ``user_api_key_cache`` (Redis-backed
        ``DualCache`` in production) to avoid a DB hit on every routed request.

        Serialisation note: the cache value is stored as a plain JSON-safe dict via
        ``model_dump(mode="json")`` so that Redis round-trips correctly in multi-worker
        deployments.  On a cache hit we reconstruct the ``MCPToolset`` Pydantic object
        so callers can always use attribute access (e.g. ``toolset.toolset_id``).
        """
        from litellm.proxy.proxy_server import user_api_key_cache
        from litellm.types.mcp_server.mcp_toolset import MCPToolset

        cache_key = f"toolset_name:{toolset_name}"
        cached = await user_api_key_cache.async_get_cache(key=cache_key)
        if cached is not None:
            # Sentinel value used to cache "not found" so we don't re-query for
            # names that don't exist.
            if cached == "__not_found__":
                return None
            # Redis deserialises JSON back as a plain dict — reconstruct the model.
            if isinstance(cached, dict):
                return MCPToolset(**cached)
            return cached

        from litellm.proxy._experimental.mcp_server.toolset_db import (
            get_mcp_toolset_by_name,
        )

        toolset = await get_mcp_toolset_by_name(prisma_client, toolset_name)
        await user_api_key_cache.async_set_cache(
            key=cache_key,
            value=(toolset.model_dump(mode="json") if toolset is not None else "__not_found__"),
            ttl=get_management_object_ttl(user_api_key_cache),
        )
        return toolset

    def filter_server_ids_by_ip(self, server_ids: list[str], client_ip: Optional[str]) -> list[str]:
        """
        Filter server IDs by client IP — external callers only see public servers.

        Returns server_ids unchanged when client_ip is None (no filtering).
        """
        filtered, _ = self.filter_server_ids_by_ip_with_info(server_ids, client_ip)
        return filtered

    def filter_server_ids_by_ip_with_info(
        self, server_ids: list[str], client_ip: Optional[str]
    ) -> tuple[list[str], int]:
        """
        Filter server IDs by client IP — external callers only see public servers.

        Returns (filtered_ids, ip_blocked_count) where ip_blocked_count is the number
        of servers that were blocked because the client IP is not allowed to access them.
        Returns server_ids unchanged (with 0 blocked) when client_ip is None.
        """
        if client_ip is None:
            return server_ids, 0
        allowed = []
        blocked = 0
        for sid in server_ids:
            s = self.get_mcp_server_by_id(sid)
            if s is not None and self._is_server_accessible_from_ip(s, client_ip):
                allowed.append(sid)
            elif s is not None:
                blocked += 1
        return allowed, blocked

    async def get_tools_for_server(self, server_id: str) -> list[MCPTool]:
        """
        Get the tools for a given server
        """
        try:
            server = self.get_mcp_server_by_id(server_id)
            if server is None:
                verbose_logger.warning(f"MCP Server {server_id} not found")
                return []
            return await self._get_tools_from_server(server)
        except Exception as e:
            verbose_logger.warning(f"Failed to get tools from server {server_id}: {str(e)}")
            return []

    async def list_tools(
        self,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_server_auth_headers: Optional[dict[str, Union[str, dict[str, str]]]] = None,
    ) -> list[MCPTool]:
        """
        List all tools available across all MCP Servers.

        Args:
            user_api_key_auth: User authentication
            mcp_auth_header: MCP auth header (deprecated)
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}
            mcp_protocol_version: Optional MCP protocol version from request header

        Returns:
            List[MCPTool]: Combined list of tools from all servers
        """
        allowed_mcp_servers = await self.get_allowed_mcp_servers(user_api_key_auth)

        verbose_logger.debug("SERVER MANAGER LISTING TOOLS")

        async def _fetch_server_tools(server_id: str) -> list[MCPTool]:
            """Fetch tools from a single server with error handling."""
            server = self.get_mcp_server_by_id(server_id)
            if server is None:
                verbose_logger.warning(f"MCP Server {server_id} not found")
                return []

            # Get server-specific auth header if available
            server_auth_header: Optional[Union[str, dict[str, str]]] = None
            if mcp_server_auth_headers:
                from litellm.proxy._experimental.mcp_server.utils import (
                    lookup_mcp_server_auth_in_headers,
                )

                server_auth_header = lookup_mcp_server_auth_in_headers(
                    mcp_server_auth_headers,
                    alias=server.alias,
                    server_name=server.server_name,
                )

            # Fall back to deprecated mcp_auth_header if no server-specific header found
            if server_auth_header is None:
                server_auth_header = mcp_auth_header

            try:
                tools = await self._get_tools_from_server(
                    server=server,
                    mcp_auth_header=server_auth_header,
                    user_api_key_auth=user_api_key_auth,
                )
                return tools
            except Exception as e:
                verbose_logger.warning(
                    f"Failed to list tools from server {server.name}: {str(e)}. Continuing with other servers."
                )
                return []

        # Fetch tools from all servers in parallel
        tasks = [_fetch_server_tools(server_id) for server_id in allowed_mcp_servers]
        results = await asyncio.gather(*tasks)

        # Flatten results into single list
        list_tools_result: list[MCPTool] = [tool for tools in results for tool in tools]

        verbose_logger.info(f"Successfully fetched {len(list_tools_result)} tools total from all servers")
        return list_tools_result

    #########################################################
    # Methods that call the upstream MCP servers
    #########################################################
    @staticmethod
    def _extract_bearer_token(
        oauth2_headers: Optional[dict[str, str]],
        raw_headers: Optional[dict[str, str]],
    ) -> Optional[str]:
        """Extract the bare Bearer token from oauth2_headers or raw_headers.

        Returns the token string without the ``Bearer `` prefix, or ``None``
        if no Authorization header is found.
        """
        auth_value: Optional[str] = None
        if oauth2_headers and "Authorization" in oauth2_headers:
            auth_value = oauth2_headers["Authorization"]
        elif raw_headers:
            # raw_headers may have lowercase keys depending on the ASGI server
            normalized = {k.lower(): v for k, v in raw_headers.items()}
            auth_value = normalized.get("authorization")
        if auth_value:
            if auth_value.startswith("Bearer "):
                return auth_value[len("Bearer ") :]
            return auth_value
        return None

    def _obo_subject_token(
        self,
        server: MCPServer,
        raw_headers: Optional[dict[str, str]],
    ) -> Optional[str]:
        """The caller's bearer as the token_exchange (OBO) subject token, for that mode only.

        Prompts/resources discovery and reads on a token_exchange server must exchange the caller's
        token like the tools paths do, not connect with no credential. Other modes never read the
        inbound bearer, so return None to avoid forwarding it.
        """
        if server.auth_type != MCPAuth.oauth2_token_exchange:
            return None
        return self._extract_bearer_token(None, raw_headers)

    def _build_stdio_env(
        self,
        server: MCPServer,
        raw_headers: Optional[dict[str, str]] = None,
    ) -> Optional[dict[str, str]]:
        """Resolve stdio env values, supporting header-driven placeholders."""

        if server.transport != MCPTransport.stdio or not server.env:
            return None

        resolved_env: dict[str, str] = {}
        normalized_headers = {k.lower(): v for k, v in (raw_headers or {}).items()}

        for env_key, env_value in server.env.items():
            stripped_value = env_value.strip()
            match = self._STDIO_ENV_TEMPLATE_PATTERN.match(stripped_value)
            if match:
                header_name = match.group(1)
                header_value = normalized_headers.get(header_name.lower())
                if header_value is None:
                    continue
                resolved_env[env_key] = header_value
            else:
                resolved_env[env_key] = env_value

        return resolved_env

    def _references_per_user_env_var(self, server: MCPServer) -> bool:
        """True when ``server.static_headers`` reference a per-user ``${NAME}`` env var.

        Such placeholders can only be filled from a calling user's stored values,
        so a userless probe (health check / instructions prefetch) would forward
        the literal ``${NAME}`` upstream and get rejected. Callers skip the probe
        and report ``unknown`` instead of a misleading ``unhealthy``.
        """
        static_headers = server.static_headers
        env_vars = getattr(server, "env_vars", None)
        if not static_headers or not env_vars:
            return False
        _global_values, user_specs = parse_admin_env_vars(env_vars)
        user_var_names = {spec["name"] for spec in user_specs}
        if not user_var_names:
            return False
        referenced = collect_env_var_references(strings=static_headers.values())
        return bool(referenced & user_var_names)

    async def _resolve_static_headers_with_env_vars(
        self,
        server: MCPServer,
        user_api_key_auth: Optional[UserAPIKeyAuth],
        *,
        raise_on_missing: bool = True,
    ) -> Optional[dict[str, str]]:
        """Return server.static_headers with ``${NAME}`` interpolated.

        Globals come from ``server.env_vars`` entries with ``scope=="global"``.
        Per-user values come from the ``LiteLLM_MCPUserEnvVars`` row for the
        calling user.

        When ``raise_on_missing`` is ``True`` (the tool-*call* path), raises
        ``MCPMissingUserEnvVarsError`` if ``static_headers`` reference a per-user
        variable the calling user has not yet supplied — converted into a
        user-facing 412 by the REST layer.

        When ``raise_on_missing`` is ``False`` (the tool-*list* path), missing
        per-user vars are non-blocking: we interpolate whatever is available and
        leave unfilled ``${NAME}`` references untouched, so the server's tools
        still appear in the listing. The user only hits the friendly error when
        they actually invoke a tool that needs the missing value.
        """
        static_headers = server.static_headers
        env_vars = getattr(server, "env_vars", None)
        if not static_headers and not env_vars:
            return static_headers

        global_values, user_specs = parse_admin_env_vars(env_vars)
        # An empty-valued global is treated as unset: it must not mask a per-user
        # var the user still has to supply, nor override a value the user did
        # supply. The unresolved ${NAME} is then left untouched, like any other
        # undefined reference.
        global_values = {name: value for name, value in global_values.items() if value}
        user_var_names = {spec["name"] for spec in user_specs}

        # If no env vars are configured, return static_headers as-is.
        if not global_values and not user_specs:
            return static_headers

        # Figure out which user-scoped vars are actually referenced. A var that
        # also carries a global value is always covered by that global (globals
        # win in the merge below), so it can never be genuinely "missing" even if
        # the user hasn't filled it in -- only vars without a global fallback do.
        referenced = collect_env_var_references(strings=(static_headers or {}).values())
        referenced_user_vars = referenced & user_var_names
        required_user_vars = {name for name in referenced_user_vars if name not in global_values}

        user_values: dict[str, str] = {}
        if required_user_vars:
            try:
                user_values = await self._load_user_env_vars(server, user_api_key_auth)
            except Exception as exc:
                # On the tool-call path a DB failure must surface as a real
                # server error, not a misleading "set up your credentials" 412.
                # On the listing path we stay best-effort and leave the
                # unfilled ${NAME} references untouched so tools still appear.
                if raise_on_missing:
                    raise
                verbose_logger.warning(
                    "MCPServerManager: best-effort user env var load failed for server=%s: %s",
                    server.server_id,
                    exc,
                )

            if raise_on_missing:
                missing = sorted(name for name in required_user_vars if not user_values.get(name))
                if missing:
                    # A cached negative must never produce a 412: cache
                    # invalidation is process-local, so a user who just stored
                    # values on another worker would otherwise be told their
                    # credentials are missing until the entry expires. Confirm
                    # against the DB before raising.
                    user_values = await self._load_user_env_vars(server, user_api_key_auth, force_refresh=True)
                    missing = sorted(name for name in required_user_vars if not user_values.get(name))
                if missing:
                    raise MCPMissingUserEnvVarsError(
                        server_id=server.server_id,
                        server_name=server.server_name or server.name,
                        missing=missing,
                        setup_url=build_env_var_setup_url(server.server_id),
                    )

        # Only honor stored user values for currently user-scoped vars, and let
        # admin globals win, so a stale row from when a var was user-scoped can
        # never override the global value the admin set after switching it.
        scoped_user_values = {name: value for name, value in user_values.items() if name in user_var_names}
        merged_vars: dict[str, str] = {**scoped_user_values, **global_values}
        if not static_headers:
            return static_headers
        return interpolate_headers(static_headers, merged_vars)

    async def _load_user_env_vars(
        self,
        server: MCPServer,
        user_api_key_auth: Optional[UserAPIKeyAuth],
        *,
        force_refresh: bool = False,
    ) -> dict[str, str]:
        """Look up the calling user's env var values for ``server``.

        Returns an empty dict when no user is available. Results are cached in a
        short-lived in-memory map keyed by (user_id, server_id) so the tool-call
        and tool-listing paths avoid a DB round-trip per request within the TTL
        window; the cache is invalidated when the user stores or clears values.
        Pass ``force_refresh`` to bypass the cache read and re-fetch from the DB
        (used before raising a "missing credentials" error so a process-local
        stale entry cannot mask values stored on another worker). A missing DB
        connection and any other DB error propagate so the caller can decide
        between failing the request (tool-call path) and staying best-effort
        (listing path); they must never be mistaken for "user has no values",
        which would send the user a misleading "set up your credentials" 412.
        """
        if user_api_key_auth is None:
            return {}
        user_id = getattr(user_api_key_auth, "user_id", None)
        if not user_id:
            return {}

        cache_key = (user_id, server.server_id)
        if not force_refresh:
            cached = _user_env_vars_cache.get(cache_key)
            if cached is not None:
                values, ts = cached
                if time.monotonic() - ts < _USER_ENV_VARS_CACHE_TTL:
                    return values

        from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415

        if prisma_client is None:
            raise RuntimeError(
                "MCP per-user env vars require a database connection, but none "
                "is configured. Connect a database to your proxy to use per-user "
                "MCP env vars."
            )
        from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415
            get_user_env_vars,
        )

        values = await get_user_env_vars(prisma_client, user_id, server.server_id)
        _write_user_env_vars_cache(user_id, server.server_id, values)
        return values

    async def _resolve_v2_auth(
        self,
        *,
        server: MCPServer,
        spec: ServerSpec,
        provider: UpstreamCredentialProvider,
        subject_token: Optional[str],
        user_api_key_auth: Optional[UserAPIKeyAuth],
        extra_headers: Optional[dict[str, str]],
    ) -> tuple[Optional[httpx.Auth], Optional[dict[str, str]]]:
        """Resolve a v2-owned server's upstream credential into ``(resolved_auth, extra_headers)``.

        On a missing/rejected per-user credential this raises the mode's discovery challenge
        (authorization_code's browser-OAuth 401, token_exchange's RFC 9728 challenge) or maps any
        other ``CredError`` onto its public HTTP status; it never returns an error as a value.
        """
        match await provider.resolve_credentials(to_subject(user_api_key_auth, subject_token), spec):
            case Ok(auth):
                # NoOpAuth has no header_name and so never conflicts.
                header_name = getattr(auth, "header_name", None)
                conflicts = bool(
                    header_name and extra_headers and any(key.lower() == header_name.lower() for key in extra_headers)
                )
                if not conflicts:
                    return auth, extra_headers
                if isinstance(spec.config, (TokenExchangeConfig, AuthorizationCodeConfig)):
                    # The resolver owns the per-user credential here (token_exchange's exchanged
                    # token, authorization_code's stored token). It is authoritative: a guardrail such
                    # as MCPJWTSigner, static_headers, or any other injected Authorization must NOT
                    # shadow it (otherwise the upstream gets e.g. the signer's JWT instead of the
                    # exchanged token and rejects it). Drop the conflicting header so the resolved
                    # token reaches upstream.
                    return auth, _without_authorization(extra_headers)
                # Other modes: an Authorization already supplied via extra_headers (a forwarded caller
                # header or static_headers) is intentional and wins; v1 applies those last.
                return None, extra_headers
            case Error(err):
                if err.tag == "unauthorized" and isinstance(spec.config, AuthorizationCodeConfig):
                    # authorization_code's missing per-user token -> the per-server browser-OAuth
                    # challenge, built here where the full MCPServer is in hand.
                    raise_user_oauth_challenge(server, root_path=get_server_root_path())
                if err.tag == "unauthorized" and isinstance(spec.config, TokenExchangeConfig):
                    # token_exchange (OBO): a missing/rejected subject token -> the RFC 9728 challenge
                    # pointing at the IdP the client must SSO with to obtain one, rather than an opaque
                    # 401. No gateway-side browser flow. An IdP step-up rejection (Entra Conditional
                    # Access) threads its claims blob into the challenge for the client to satisfy.
                    raise_token_exchange_challenge(
                        server,
                        root_path=get_server_root_path(),
                        claims=err.unauthorized.claims,
                    )
                raise_public(err)

    async def preflight_token_exchange(
        self,
        server: MCPServer,
        oauth2_headers: Optional[dict[str, str]],
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> None:
        """Run the OBO exchange for a caller-supplied subject at the transport edge.

        Single-server routes call this before the MCP session opens, where an HTTP status and
        ``WWW-Authenticate`` still reach the client. A rejected subject raises the RFC 9728
        challenge and any other ``CredError`` maps onto its public HTTP status, so an exchange
        failure surfaces as a failure instead of the session continuing into an empty tool list.
        A successful exchange is cached by the exchanger, so the session's list/call reuses it.
        """
        if server.auth_type != MCPAuth.oauth2_token_exchange:
            return
        subject_token = self._extract_bearer_token(oauth2_headers, None)
        if not subject_token:
            return
        spec = to_server_spec(server)
        if spec is None or not isinstance(spec.config, TokenExchangeConfig):
            return
        match await self._cred_provider.resolve_credentials(to_subject(user_api_key_auth, subject_token), spec):
            case Ok(_):
                return
            case Error(err):
                if err.tag == "unauthorized":
                    raise_token_exchange_challenge(
                        server,
                        root_path=get_server_root_path(),
                        claims=err.unauthorized.claims,
                    )
                raise_public(err)

    async def _create_mcp_client(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, dict[str, str]]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        stdio_env: Optional[dict[str, str]] = None,
        subject_token: Optional[str] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        cred_provider: Optional[UpstreamCredentialProvider] = None,
    ) -> MCPClient:
        """
        Create an MCPClient instance for the given server.

        Auth resolution (single place for all auth logic):
        1. ``mcp_auth_header`` — per-request/per-user override
        2. OAuth2 Token Exchange (OBO) — exchange user token for scoped token
        3. OAuth2 client_credentials token — auto-fetched and cached
        4. ``server.authentication_token`` — static token from config/DB

        Args:
            server: The server configuration.
            mcp_auth_header: Optional per-request auth override.
            extra_headers: Additional headers to forward.
            stdio_env: Environment variables for stdio transport.
            subject_token: Optional user JWT for token exchange (OBO) flow.
            user_api_key_auth: Optional auth context for sampling callbacks.

        Returns:
            Configured MCP client instance.
        """
        transport = server.transport or MCPTransport.sse
        spec = None if transport == MCPTransport.stdio else to_server_spec(server)
        provider = cred_provider or self._cred_provider
        # A caller-supplied per-request override (mcp_auth_header / x-mcp-*) defers to the v1 path
        # so it wins - except for the modes the v2 resolver owns per-caller (authorization_code's
        # stored token, token_exchange's RFC 8693 minted token, and the passthrough modes'
        # forwarded caller token). A caller must not be able to substitute another user's stored
        # credential, nor silently disable the OBO exchange and forward an arbitrary bearer
        # upstream, so we keep the v2 spec and ignore the override for these; the REST tools
        # preview supplies its not-yet-persisted token through the resolver (cred_provider),
        # never this path.
        if (
            spec is not None
            and mcp_auth_header
            and not isinstance(spec.config, (AuthorizationCodeConfig, PassthroughConfig, TokenExchangeConfig))
        ):
            spec = None
        auth_value = (
            await resolve_mcp_auth(server, mcp_auth_header, subject_token=subject_token) if spec is None else None
        )

        # Create sampling and elicitation callbacks for this client
        sampling_cb = _create_sampling_callback(user_api_key_auth=user_api_key_auth) if server.allow_sampling else None
        elicitation_cb = _create_elicitation_callback() if server.allow_elicitation else None

        # Handle stdio transport
        if transport == MCPTransport.stdio:
            resolved_env = (
                stdio_env if stdio_env is not None else (dict(server.env) if server.env is not None else None)
            )

            # Ensure npm-based STDIO MCP servers have a writable cache dir.
            # In containers the default (~/.npm or /app/.npm) may not exist
            # or be read-only, causing npx to fail with ENOENT.
            if resolved_env is not None and "NPM_CONFIG_CACHE" not in resolved_env:
                resolved_env["NPM_CONFIG_CACHE"] = MCP_NPM_CACHE_DIR
            # Defense-in-depth: block commands not in the allowlist.
            # The Pydantic validator blocks new servers; this catches legacy
            # config/DB records predating the allowlist.
            if server.command:
                base_command = os.path.basename(server.command)
                # Strip .exe/.cmd/.bat/.com suffix for Windows compatibility
                base_command_no_ext = base_command.lower()
                for ext in [".exe", ".cmd", ".bat", ".com"]:
                    if base_command.lower().endswith(ext):
                        base_command_no_ext = base_command[: -len(ext)].lower()
                        break
                if (
                    base_command.lower() not in MCP_STDIO_ALLOWED_COMMANDS
                    and base_command_no_ext not in MCP_STDIO_ALLOWED_COMMANDS
                ):
                    raise HTTPException(
                        status_code=403,
                        detail=f"MCP stdio command '{server.command}' is not in the allowlist ({sorted(MCP_STDIO_ALLOWED_COMMANDS)}). "
                        f"Add it to LITELLM_MCP_STDIO_EXTRA_COMMANDS to allow this command.",
                    )

            stdio_config: Optional[MCPStdioConfig] = None
            if server.command and server.args is not None:
                stdio_config = MCPStdioConfig(
                    command=server.command,
                    args=server.args,
                    env=resolved_env,
                )

            return MCPClient(
                server_url="",  # Not used for stdio
                transport_type=transport,
                auth_type=server.auth_type,
                auth_value=auth_value,
                timeout=(server.timeout if server.timeout is not None else MCP_CLIENT_TIMEOUT),
                stdio_config=stdio_config,
                extra_headers=extra_headers,
                sampling_callback=sampling_cb,
                elicitation_callback=elicitation_cb,
            )
        else:
            # For HTTP/SSE transports
            server_url = server.url or ""

            if spec is not None:
                inbound_token = subject_token
                if isinstance(spec.config, PassthroughConfig):
                    inbound_token, extra_headers = _take_forwarded_authorization(extra_headers)
                    per_server_token = _passthrough_token_from_mcp_auth_header(mcp_auth_header)
                    if per_server_token is not None:
                        inbound_token = per_server_token
                resolved_auth, extra_headers = await self._resolve_v2_auth(
                    server=server,
                    spec=spec,
                    provider=provider,
                    subject_token=inbound_token,
                    user_api_key_auth=user_api_key_auth,
                    extra_headers=extra_headers,
                )
                return MCPClient(
                    server_url=server_url,
                    transport_type=transport,
                    auth_type=server.auth_type,
                    timeout=(server.timeout if server.timeout is not None else MCP_CLIENT_TIMEOUT),
                    extra_headers=extra_headers,
                    resolved_auth=resolved_auth,
                    sampling_callback=sampling_cb,
                    elicitation_callback=elicitation_cb,
                )

            # Create SigV4 auth if configured
            aws_auth = None
            if server.auth_type == MCPAuth.aws_sigv4:
                aws_auth = MCPSigV4Auth(
                    aws_access_key_id=server.aws_access_key_id,
                    aws_secret_access_key=server.aws_secret_access_key,
                    aws_session_token=server.aws_session_token,
                    aws_region_name=server.aws_region_name,
                    aws_service_name=server.aws_service_name,
                    aws_role_name=server.aws_role_name,
                    aws_session_name=server.aws_session_name,
                )

            return MCPClient(
                server_url=server_url,
                transport_type=transport,
                auth_type=server.auth_type,
                auth_value=auth_value,
                timeout=(server.timeout if server.timeout is not None else MCP_CLIENT_TIMEOUT),
                extra_headers=extra_headers,
                aws_auth=aws_auth,
                sampling_callback=sampling_cb,
                elicitation_callback=elicitation_cb,
            )

    async def _get_tools_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, dict[str, str]]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[dict[str, str]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        oauth2_headers: Optional[dict[str, str]] = None,
    ) -> list[MCPTool]:
        """
        Helper method to get tools from a single MCP server with prefixed names.

        Args:
            server (MCPServer): The server to query tools from
            mcp_auth_header: Optional auth header for MCP server

        Returns:
            List[MCPTool]: List of tools available on the server with prefixed names
        """
        from litellm.proxy._experimental.mcp_server.tool_registry import (
            global_mcp_tool_registry,
        )

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"_get_tools_from_server for {server.name}...")

        client = None

        try:
            # Tool *listing* must not be blocked by missing per-user env vars —
            # the server's tools should still appear so the client connects. The
            # friendly "missing vars" error is raised only on the tool-*call*
            # path (see _call_regular_mcp_tool).
            resolved_static_headers = await self._resolve_static_headers_with_env_vars(
                server, user_api_key_auth, raise_on_missing=False
            )
            if resolved_static_headers:
                if extra_headers is None:
                    extra_headers = {}
                extra_headers.update(resolved_static_headers)

            # MCPJWTSigner: inject signed JWT for tools/list (list path skips pre_call_hook).
            # Skip entirely when the signer is not configured (avoid an unnecessary
            # dict copy on every list call), when the server has its own static
            # Authorization header, when a per-user mcp_auth_header has already
            # been resolved, or when the caller already supplied an Authorization
            # entry in extra_headers (e.g. a per-user OAuth token resolved
            # upstream) — admin-configured static auth and per-user OAuth must
            # take precedence so the signer doesn't silently overwrite e.g. an
            # upstream API key or a user's OAuth token (MCPClient._get_auth_headers
            # applies extra_headers after writing Authorization from auth_value, so
            # an injected JWT would otherwise clobber the per-user token).
            if user_api_key_auth is not None and not server.spec_path:
                from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
                    get_mcp_jwt_signer,
                    inject_mcp_jwt_headers_for_upstream,
                )

                static_headers = server.static_headers or {}
                has_static_authorization = any(
                    isinstance(k, str) and k.lower() == "authorization" for k in static_headers.keys()
                )
                has_extra_authorization = bool(extra_headers) and any(
                    isinstance(k, str) and k.lower() == "authorization" for k in (extra_headers or {}).keys()
                )

                if (
                    get_mcp_jwt_signer() is not None
                    and not has_static_authorization
                    and not mcp_auth_header
                    and not has_extra_authorization
                ):
                    extra_headers = await inject_mcp_jwt_headers_for_upstream(
                        user_api_key_dict=user_api_key_auth,
                        extra_headers=extra_headers,
                        raw_headers=raw_headers,
                        for_list_tools=True,
                    )

            stdio_env = self._build_stdio_env(server, raw_headers)

            # token_exchange (OBO) discovery needs the caller's token too: list it with the user's own
            # token (mirrors the call path), not v1's deleted client_credentials fallback. Other modes
            # never read the inbound bearer, so leave subject_token None to avoid forwarding it.
            subject_token = (
                self._extract_bearer_token(oauth2_headers, raw_headers)
                if server.auth_type == MCPAuth.oauth2_token_exchange
                else None
            )

            client = await self._create_mcp_client(
                server=server,
                mcp_auth_header=mcp_auth_header,
                extra_headers=extra_headers,
                stdio_env=stdio_env,
                subject_token=subject_token,
                user_api_key_auth=user_api_key_auth,
            )

            ## HANDLE OPENAPI TOOLS
            if server.spec_path:
                # OpenAPI tools were stored in the registry under the prefix
                # active at registration time — fetch by that same prefix.
                _tools = global_mcp_tool_registry.list_tools(tool_prefix=get_server_prefix(server))
                tools = global_mcp_tool_registry.convert_tools_to_mcp_sdk_tool_type(_tools)
                # OpenAPI tools are stored in the registry with their prefix already
                # applied (e.g. "test_petstore-getinventory").  Do NOT pass them
                # through _create_prefixed_tools — that would add the prefix a second
                # time producing "test_petstore-test_petstore-getinventory".
                if not add_prefix:
                    prefix = get_server_prefix(server)
                    sep = MCP_TOOL_PREFIX_SEPARATOR
                    tools = [
                        (
                            t.model_copy(update={"name": t.name[len(prefix) + len(sep) :]})
                            if t.name.startswith(f"{prefix}{sep}")
                            else t
                        )
                        for t in tools
                    ]
                return tools
            else:
                tools = await self._fetch_tools_with_timeout(client, server.name)
                self._remember_upstream_initialize_instructions(server, client)

            prefixed_or_original_tools = self._create_prefixed_tools(tools, server, add_prefix=add_prefix)

            return prefixed_or_original_tools

        except MCPUpstreamAuthError as upstream_auth_error:
            # Pass-through 401 must surface to single-server routes so the
            # client triggers the upstream OAuth flow. The multi-server
            # aggregator catches this explicitly to keep absorbing.
            if server.is_dcr_bridge and upstream_auth_error.www_authenticate is not None:
                raise MCPUpstreamAuthError(
                    status_code=upstream_auth_error.status_code,
                    www_authenticate=None,
                    server_name=upstream_auth_error.server_name,
                ) from upstream_auth_error
            raise
        except HTTPException as e:
            # A v2 resolver auth challenge (token_exchange's RFC 9728 401, authorization_code's
            # browser-OAuth 401, or a 403) is raised at client-build time, inside this try. Route it
            # through the same MCPUpstreamAuthError channel as pass-through so single-server routes
            # surface the challenge (the client re-authenticates) while the aggregator keeps absorbing.
            # Non-auth HTTP errors stay absorbed so one misconfigured server can't blank the listing.
            if e.status_code in (401, 403):
                headers = e.headers or {}
                challenge_header = headers.get("WWW-Authenticate") or headers.get("www-authenticate")
                raise MCPUpstreamAuthError(
                    status_code=e.status_code,
                    www_authenticate=None if server.is_dcr_bridge else challenge_header,
                    server_name=server.name,
                ) from e
            verbose_logger.warning(f"Failed to get tools from server {server.name}: {str(e)}")
            return []
        except Exception as e:
            verbose_logger.warning(f"Failed to get tools from server {server.name}: {str(e)}")
            return []

    async def get_prompts_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, dict[str, str]]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[dict[str, str]] = None,
    ) -> list[Prompt]:
        """
        Helper method to get prompts from a single MCP server with prefixed names.

        Args:
            server (MCPServer): The server to query prompts from
            mcp_auth_header: Optional auth header for MCP server

        Returns:
            List[Prompt]: List of prompts available on the server with prefixed names
        """

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"get_prompts_from_server for {server.name}...")

        client = None

        try:
            if server.static_headers:
                if extra_headers is None:
                    extra_headers = {}
                extra_headers.update(server.static_headers)

            stdio_env = self._build_stdio_env(server, raw_headers)
            subject_token = self._obo_subject_token(server, raw_headers)

            client = await self._create_mcp_client(
                server=server,
                mcp_auth_header=mcp_auth_header,
                extra_headers=extra_headers,
                stdio_env=stdio_env,
                subject_token=subject_token,
            )

            prompts = await client.list_prompts()

            prefixed_or_original_prompts = self._create_prefixed_prompts(prompts, server, add_prefix=add_prefix)

            return prefixed_or_original_prompts

        except Exception as e:
            verbose_logger.warning(f"Failed to get prompts from server {server.name}: {str(e)}")
            return []

    async def get_resources_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, dict[str, str]]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[dict[str, str]] = None,
    ) -> list[Resource]:
        """Fetch available resources from a single MCP server."""

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"get_resources_from_server for {server.name}...")

        client = None

        try:
            if server.static_headers:
                if extra_headers is None:
                    extra_headers = {}
                extra_headers.update(server.static_headers)

            stdio_env = self._build_stdio_env(server, raw_headers)
            subject_token = self._obo_subject_token(server, raw_headers)

            client = await self._create_mcp_client(
                server=server,
                mcp_auth_header=mcp_auth_header,
                extra_headers=extra_headers,
                stdio_env=stdio_env,
                subject_token=subject_token,
            )

            resources = await client.list_resources()

            prefixed_resources = self._create_prefixed_resources(resources, server, add_prefix=add_prefix)

            return prefixed_resources

        except Exception as e:
            verbose_logger.warning(f"Failed to get resources from server {server.name}: {str(e)}")
            return []

    async def get_resource_templates_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, dict[str, str]]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[dict[str, str]] = None,
    ) -> list[ResourceTemplate]:
        """Fetch available resource templates from a single MCP server."""

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"get_resource_templates_from_server for {server.name}...")

        client = None

        try:
            if server.static_headers:
                if extra_headers is None:
                    extra_headers = {}
                extra_headers.update(server.static_headers)

            stdio_env = self._build_stdio_env(server, raw_headers)
            subject_token = self._obo_subject_token(server, raw_headers)

            client = await self._create_mcp_client(
                server=server,
                mcp_auth_header=mcp_auth_header,
                extra_headers=extra_headers,
                stdio_env=stdio_env,
                subject_token=subject_token,
            )

            resource_templates = await client.list_resource_templates()

            prefixed_templates = self._create_prefixed_resource_templates(
                resource_templates, server, add_prefix=add_prefix
            )

            return prefixed_templates

        except Exception as e:
            verbose_logger.warning(f"Failed to get resource templates from server {server.name}: {str(e)}")
            return []

    async def read_resource_from_server(
        self,
        server: MCPServer,
        url: AnyUrl,
        mcp_auth_header: Optional[Union[str, dict[str, str]]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        raw_headers: Optional[dict[str, str]] = None,
    ) -> ReadResourceResult:
        """Read resource contents from a specific MCP server."""

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"read_resource_from_server for {server.name}...")

        if server.static_headers:
            if extra_headers is None:
                extra_headers = {}
            extra_headers.update(server.static_headers)

        stdio_env = self._build_stdio_env(server, raw_headers)
        subject_token = self._obo_subject_token(server, raw_headers)

        client = await self._create_mcp_client(
            server=server,
            mcp_auth_header=mcp_auth_header,
            extra_headers=extra_headers,
            stdio_env=stdio_env,
            subject_token=subject_token,
        )

        return await client.read_resource(url)

    async def get_prompt_from_server(
        self,
        server: MCPServer,
        prompt_name: str,
        arguments: Optional[dict[str, Any]] = None,
        mcp_auth_header: Optional[Union[str, dict[str, str]]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        raw_headers: Optional[dict[str, str]] = None,
    ) -> GetPromptResult:
        """Fetch a specific prompt definition from a single MCP server."""

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"get_prompt_from_server for {server.name}...")

        if server.static_headers:
            if extra_headers is None:
                extra_headers = {}
            extra_headers.update(server.static_headers)

        stdio_env = self._build_stdio_env(server, raw_headers)
        subject_token = self._obo_subject_token(server, raw_headers)

        client = await self._create_mcp_client(
            server=server,
            mcp_auth_header=mcp_auth_header,
            extra_headers=extra_headers,
            stdio_env=stdio_env,
            subject_token=subject_token,
        )

        get_prompt_request_params = GetPromptRequestParams(
            name=prompt_name,
            arguments=arguments,
        )
        return await client.get_prompt(get_prompt_request_params)

    @staticmethod
    def _is_same_authority_metadata_url(url: str, server_url: str) -> bool:
        """
        Whether ``url`` shares scheme, host, and port with ``server_url``.

        Same-authority metadata URLs are produced by our well-known discovery
        construction and by resource servers that publish protected-resource
        metadata on the resource origin. These must keep working for
        administrator-configured internal MCP servers, so they are fetched
        directly. Cross-origin URLs are fetched through ``async_safe_get``.
        """
        try:
            target = urlparse(url)
            base = urlparse(server_url)
        except Exception:
            return False

        if target.scheme not in ("http", "https") or not target.hostname:
            return False

        target_port = target.port or (443 if target.scheme == "https" else 80)
        base_port = base.port or (443 if base.scheme == "https" else 80)
        return (
            base.scheme == target.scheme
            and (base.hostname or "").lower() == target.hostname.lower()
            and base_port == target_port
        )

    async def _fetch_oauth_discovery_url(self, url: str, server_url: str) -> Any:
        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.MCP,
            params={"timeout": MCP_METADATA_TIMEOUT},
        )
        if self._is_same_authority_metadata_url(url, server_url):
            # Same-authority URLs may point at administrator-configured
            # internal MCP servers. Do not run them through user URL
            # validation, but also do not follow redirects because the
            # redirect target would not inherit the same-authority guarantee.
            return await client.get(url, follow_redirects=False)
        return await async_safe_get(client, url)

    async def _descovery_metadata(
        self,
        server_url: str,
        *,
        allow_origin_fallback: bool = True,
    ) -> Optional[MCPOAuthMetadata]:
        """Discover OAuth metadata by following RFC 9728 (protected resource metadata discovery).

        ``allow_origin_fallback`` controls the last-resort guess that treats the resource server's own
        origin as its authorization server when nothing is advertised. The browser ``oauth2`` flow keeps
        it (a human sees the redirect), but token_exchange (OBO) sets it False so the gateway never
        exchanges a subject token against an endpoint it inferred rather than one explicitly configured
        or authoritatively advertised via RFC 9728 / RFC 8414.
        """

        try:
            client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)
            response = await client.get(server_url)
            response.raise_for_status()
            (
                authorization_servers,
                resource_scopes,
            ) = await self._attempt_well_known_discovery(server_url)
            metadata = await self._fetch_authorization_server_metadata(authorization_servers, server_url)
            if metadata is None and not resource_scopes and authorization_servers and response.status_code == 200:
                verbose_logger.warning(
                    "MCP OAuth discovery for %s received 200 OK without RFC 9728 challenge and no discoverable authorization metadata.",
                    server_url,
                )
            if metadata is None and resource_scopes:
                return MCPOAuthMetadata(scopes=resource_scopes)
            if metadata is not None and resource_scopes:
                metadata.scopes = resource_scopes
            return metadata
        except HTTPStatusError as exc:
            verbose_logger.debug(
                "MCP OAuth discovery for %s received status error: %s",
                server_url,
                exc,
            )

            header_value: Optional[str] = None
            if exc.response is not None:
                header_value = exc.response.headers.get("WWW-Authenticate") or exc.response.headers.get(
                    "www-authenticate"
                )

            resource_metadata_url, scopes = self._parse_www_authenticate_header(header_value)

            authorization_servers = []
            resource_scopes = None
            if resource_metadata_url:
                (
                    authorization_servers,
                    resource_scopes,
                ) = await self._fetch_oauth_metadata_from_resource(resource_metadata_url, server_url)
            else:
                (
                    authorization_servers,
                    resource_scopes,
                ) = await self._attempt_well_known_discovery(server_url)

            metadata = None
            used_origin_fallback = False
            if allow_origin_fallback and not authorization_servers:
                try:
                    parsed_url = urlparse(server_url)
                    if parsed_url.scheme and parsed_url.netloc:
                        authorization_servers = [f"{parsed_url.scheme}://{parsed_url.netloc}"]
                        used_origin_fallback = True
                except Exception:
                    authorization_servers = []

            if authorization_servers:
                metadata = await self._fetch_authorization_server_metadata(authorization_servers, server_url)
                if metadata is not None and used_origin_fallback:
                    metadata.from_origin_fallback = True

            preferred_scopes = scopes or resource_scopes
            if metadata is None and preferred_scopes:
                metadata = MCPOAuthMetadata(scopes=preferred_scopes)
            elif metadata is not None and preferred_scopes:
                metadata.scopes = preferred_scopes

            return metadata
        except Exception as exc:  # pragma: no cover - network/transient issues
            verbose_logger.debug("MCP OAuth discovery failed for %s: %s", server_url, exc)
            return None

    def _parse_www_authenticate_header(self, header_value: Optional[str]) -> tuple[Optional[str], Optional[list[str]]]:
        if not header_value:
            return None, None

        _, _, params_section = header_value.partition(" ")
        params_section = params_section or header_value

        param_pattern = re.compile(r"([a-zA-Z0-9_]+)\s*=\s*\"?([^\",]+)\"?")
        params: dict[str, str] = {
            match.group(1).lower(): match.group(2).strip() for match in param_pattern.finditer(params_section)
        }

        resource_metadata_url = params.get("resource_metadata")

        scope_value = params.get("scope")
        scopes_list = [s for s in (scope_value.split() if scope_value else []) if s]
        scopes = scopes_list or None

        return resource_metadata_url, scopes

    async def _fetch_oauth_metadata_from_resource(
        self, resource_metadata_url: str, server_url: str
    ) -> tuple[list[str], Optional[list[str]]]:
        if not resource_metadata_url:
            return [], None

        try:
            response = await self._fetch_oauth_discovery_url(resource_metadata_url, server_url)
            response.raise_for_status()
            data = response.json()
        except SSRFError as exc:
            verbose_logger.warning(
                "MCP OAuth discovery: refusing to fetch resource metadata from %s "
                "(rejected by SSRF guard for server %s): %s",
                resource_metadata_url,
                server_url,
                exc,
            )
            return [], None
        except Exception as exc:  # pragma: no cover - network issues
            verbose_logger.debug(
                "Failed to fetch MCP OAuth metadata from %s: %s",
                resource_metadata_url,
                exc,
            )
            return [], None

        raw_servers = data.get("authorization_servers")
        if isinstance(raw_servers, list):
            authorization_servers = [entry for entry in raw_servers if isinstance(entry, str) and entry.strip() != ""]
        else:
            authorization_servers = []

        scopes = self._extract_scopes(data.get("scopes_supported") or data.get("scopes"))

        return authorization_servers, scopes

    async def _attempt_well_known_discovery(self, server_url: str) -> tuple[list[str], Optional[list[str]]]:
        try:
            parsed = urlparse(server_url)
        except Exception:
            return [], None

        if not parsed.scheme or not parsed.netloc:
            return [], None

        base = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path or ""
        path = path.strip("/")

        candidate_urls: list[str] = []
        if path:
            candidate_urls.append(f"{base}/.well-known/oauth-protected-resource/{path}")
        candidate_urls.append(f"{base}/.well-known/oauth-protected-resource")

        for url in candidate_urls:
            (
                authorization_servers,
                scopes,
            ) = await self._fetch_oauth_metadata_from_resource(url, server_url)
            if authorization_servers:
                return authorization_servers, scopes

        return [], None

    async def _fetch_authorization_server_metadata(
        self, authorization_servers: list[str], server_url: str
    ) -> Optional[MCPOAuthMetadata]:
        for issuer in authorization_servers:
            metadata = await self._fetch_single_authorization_server_metadata(issuer, server_url)
            if metadata is not None:
                return metadata
        return None

    async def _fetch_issuer_anchored_oauth_metadata(
        self, issuer: str, server_url: Optional[str]
    ) -> Optional[MCPOAuthMetadata]:
        """RFC 8414 issuer-anchored discovery for the OAuth endpoints, with resource-driven scopes.

        Fetch authorization-server metadata from the admin-configured issuer's own origin and adopt
        its ``token_endpoint``/``registration_endpoint`` only when the document self-attests that same
        issuer (RFC 8414 §3.3). Because the trust anchor is the pinned issuer rather than anything the
        MCP resource advertises, the endpoints are authoritative for that issuer and cannot be
        substituted by a compromised resource. Fails closed (returns None) on a §3.3 mismatch or a
        fetch failure. The issuer is passed as its own ``server_url`` so the endpoint fetch is treated
        as same-authority and is not subject to the resource-scoped SSRF shortcut.

        Scopes are NOT taken from the issuer document. Per the MCP authorization spec Scope Selection
        Strategy and RFC 9728, the scopes a client requests are resource-driven (the WWW-Authenticate
        challenge or the protected-resource ``scopes_supported``), so the resource's advertised scopes
        are fetched separately and used; the resource can influence only the requested scope, which
        the authorization server and user consent bound (RFC 6749 §3.3), never the token endpoint.
        """
        metadata = await self._fetch_single_authorization_server_metadata(issuer, issuer, require_issuer=issuer)
        if metadata is None:
            verbose_logger.warning(
                "MCP OAuth issuer-anchored discovery for issuer %s yielded no metadata whose issuer "
                "matched (RFC 8414 §3.3); OAuth endpoints stay unresolved until a rebuild succeeds",
                issuer,
            )
            return None
        resource_metadata = (
            await self._descovery_metadata(server_url, allow_origin_fallback=False) if server_url else None
        )
        resource_scopes = resource_metadata.scopes if resource_metadata else None
        return metadata.model_copy(update={"scopes": resource_scopes})

    async def _fetch_single_authorization_server_metadata(
        self, issuer_url: str, server_url: str, require_issuer: Optional[str] = None
    ) -> Optional[MCPOAuthMetadata]:
        try:
            parsed = urlparse(issuer_url)
        except Exception:
            return None

        if not parsed.scheme or not parsed.netloc:
            return None

        base = f"{parsed.scheme}://{parsed.netloc}"
        path = (parsed.path or "").strip("/")

        candidate_urls: list[str] = []
        if path:
            candidate_urls.append(f"{base}/.well-known/oauth-authorization-server/{path}")
            candidate_urls.append(f"{base}/.well-known/openid-configuration/{path}")
            candidate_urls.append(f"{issuer_url.rstrip('/')}/.well-known/openid-configuration")
        candidate_urls.append(f"{base}/.well-known/oauth-authorization-server")
        candidate_urls.append(f"{base}/.well-known/openid-configuration")
        candidate_urls.append(issuer_url.rstrip("/"))

        for url in candidate_urls:
            try:
                response = await self._fetch_oauth_discovery_url(url, server_url)
                response.raise_for_status()
                data = response.json()
            except SSRFError as exc:
                verbose_logger.warning(
                    "MCP OAuth discovery: refusing to fetch authorization-server "
                    "metadata from %s (rejected by SSRF guard for server %s): %s",
                    url,
                    server_url,
                    exc,
                )
                continue
            except Exception as exc:  # pragma: no cover - network issues
                verbose_logger.debug(
                    "Failed to fetch authorization metadata from %s: %s",
                    url,
                    exc,
                )
                continue

            claimed_issuer = data.get("issuer")
            verbose_logger.debug(
                "Authorization server metadata from %s: issuer=%s grant_types_supported=%s "
                "token_endpoint_auth_methods_supported=%s",
                url,
                claimed_issuer,
                data.get("grant_types_supported"),
                data.get("token_endpoint_auth_methods_supported"),
            )
            if require_issuer is not None and not _issuer_matches(claimed_issuer, require_issuer):
                verbose_logger.warning(
                    "MCP OAuth issuer-anchored discovery: metadata at %s self-attests issuer %r, which "
                    "does not match the configured issuer %r (RFC 8414 §3.3); rejecting so a compromised "
                    "resource cannot substitute an attacker authorization server",
                    url,
                    claimed_issuer,
                    require_issuer,
                )
                continue

            scopes = self._extract_scopes(data.get("scopes_supported"))
            metadata = MCPOAuthMetadata(
                scopes=scopes,
                authorization_url=data.get("authorization_endpoint"),
                token_url=data.get("token_endpoint"),
                registration_url=data.get("registration_endpoint"),
                discovered_issuer=claimed_issuer if isinstance(claimed_issuer, str) and claimed_issuer else None,
            )

            if any(
                [
                    metadata.scopes,
                    metadata.authorization_url,
                    metadata.token_url,
                    metadata.registration_url,
                ]
            ):
                return metadata

        return self._build_azure_authorization_server_metadata(parsed)

    @staticmethod
    def _build_azure_authorization_server_metadata(
        parsed_issuer_url: Any,
    ) -> Optional[MCPOAuthMetadata]:
        path_parts = [part for part in (parsed_issuer_url.path or "").split("/") if part]
        if parsed_issuer_url.netloc not in _AZURE_ENTRA_HOSTS or len(path_parts) != 2 or path_parts[1] != "v2.0":
            return None

        tenant = path_parts[0]
        base = f"{parsed_issuer_url.scheme}://{parsed_issuer_url.netloc}/{tenant}"
        return MCPOAuthMetadata(
            authorization_url=f"{base}/oauth2/v2.0/authorize",
            token_url=f"{base}/oauth2/v2.0/token",
        )

    @staticmethod
    def _decrypt_credential_field(
        encrypted_value: Optional[str],
        key: str,
        credentials_are_encrypted: bool,
    ) -> Optional[str]:
        """Decrypt a single credential field, or return as-is if not encrypted."""
        if not encrypted_value:
            return None
        if credentials_are_encrypted:
            return decrypt_value_helper(
                value=encrypted_value,
                key=key,
                exception_type="debug",
                return_original_value=True,
            )
        return encrypted_value

    def _extract_aws_credentials(
        self,
        credentials_dict: Optional[dict[str, str]],
        credentials_are_encrypted: bool,
    ) -> dict[str, Optional[str]]:
        """Extract and decrypt AWS SigV4 credential fields from credentials dict."""
        if not credentials_dict:
            return {}
        return {
            "aws_access_key_id": self._decrypt_credential_field(
                credentials_dict.get("aws_access_key_id"),
                "aws_access_key_id",
                credentials_are_encrypted,
            ),
            "aws_secret_access_key": self._decrypt_credential_field(
                credentials_dict.get("aws_secret_access_key"),
                "aws_secret_access_key",
                credentials_are_encrypted,
            ),
            "aws_session_token": self._decrypt_credential_field(
                credentials_dict.get("aws_session_token"),
                "aws_session_token",
                credentials_are_encrypted,
            ),
            "aws_region_name": credentials_dict.get("aws_region_name"),
            "aws_service_name": credentials_dict.get("aws_service_name"),
            "aws_role_name": credentials_dict.get("aws_role_name"),
            "aws_session_name": credentials_dict.get("aws_session_name"),
        }

    def _extract_scopes(self, scopes_value: Any) -> Optional[list[str]]:
        if isinstance(scopes_value, str):
            scopes = [s.strip() for s in scopes_value.split() if s.strip()]
            return scopes or None
        if isinstance(scopes_value, list):
            scopes = [s for s in scopes_value if isinstance(s, str) and s.strip()]
            return scopes or None
        return None

    async def _fetch_tools_with_timeout(
        self,
        client: MCPClient,
        server_name: str,
    ) -> list[MCPTool]:
        """
        Fetch tools from MCP client with timeout and error handling.

        Uses anyio.fail_after() instead of asyncio.wait_for() to avoid conflicts
        with the MCP SDK's anyio TaskGroup. See GitHub issue #20715 for details.

        An upstream HTTP 401 is converted into :class:`MCPUpstreamAuthError`
        instead of being swallowed to an empty tool list, regardless of the
        server's auth_type. Callers route it by surface: the single-server HTTP
        routes turn it into a 401 + ``WWW-Authenticate`` challenge so standards-
        compliant MCP clients trigger the upstream OAuth flow, while the
        multi-server ``/mcp`` aggregator absorbs it to an empty list so one
        unauthenticated server doesn't fail the whole listing. Only a 401
        (missing/invalid credential) drives the re-auth challenge; a 403
        (authenticated but forbidden, e.g. insufficient scope) is not a re-auth
        signal and, like other non-auth errors, returns an empty list.

        Args:
            client: MCP client instance
            server_name: Name of the server for logging

        Returns:
            List of tools from the server
        """
        try:
            with anyio.fail_after(MCP_TOOL_LISTING_TIMEOUT):
                tools = await client.list_tools(raise_on_error=True)
                verbose_logger.debug(f"Tools from {server_name}: {tools}")
                return tools
        except TimeoutError:
            verbose_logger.warning(f"Timeout while listing tools from {server_name}")
            return []
        except asyncio.CancelledError:
            verbose_logger.warning(f"Task cancelled while listing tools from {server_name}")
            return []
        except ConnectionError as e:
            verbose_logger.warning(f"Connection error while listing tools from {server_name}: {str(e)}")
            return []
        except Exception as e:
            auth_info = _extract_upstream_auth_failure(e)
            if auth_info is not None and auth_info[0] == 401:
                _, www_authenticate = auth_info
                verbose_logger.info(f"Upstream auth failure from MCP server {server_name}: HTTP 401")
                raise MCPUpstreamAuthError(
                    status_code=401,
                    www_authenticate=www_authenticate,
                    server_name=server_name,
                ) from e
            verbose_logger.warning(f"Error listing tools from {server_name}: {str(e)}")
            return []

    _SHORT_PREFIX_MAX_REHASH_ATTEMPTS = 1024

    def _assign_unique_short_prefix(
        self,
        server: MCPServer,
        registry: Optional[dict[str, MCPServer]] = None,
    ) -> None:
        """Resolve and cache a collision-free short tool prefix on ``server``.

        Called at registration time for every MCP server entering the
        registry.  Mutates ``server.short_prefix`` in place.  No-ops when
        ``LITELLM_USE_SHORT_MCP_TOOL_PREFIX`` is disabled, when the server
        has no ``server_id`` (synthetic temp-server objects), or when a
        prefix is already cached.

        Collision strategy: take the natural hash; if it's already used by
        a *different* server in the combined registry, rehash with an
        incrementing attempt counter until we find an unused slot.  The
        attempt counter is folded into the hash so the resulting prefix is
        still deterministic for a given (server_id, set-of-other-server-ids)
        pair within one process.
        """
        if not is_short_mcp_tool_prefix_enabled():
            return
        if server.short_prefix:
            return
        if not server.server_id:
            return

        used: dict[str, str] = {}
        registry_for_collision_check = registry or self.get_registry()
        for other in registry_for_collision_check.values():
            if other.server_id == server.server_id:
                continue
            if other.short_prefix:
                used[other.short_prefix] = other.server_id

        for attempt in range(self._SHORT_PREFIX_MAX_REHASH_ATTEMPTS):
            candidate = compute_short_server_prefix(server.server_id, attempt=attempt)
            if candidate not in used:
                server.short_prefix = candidate
                if attempt > 0:
                    verbose_logger.info(
                        "MCP short-prefix collision resolved for server %s: "
                        "natural hash collided with %s, using rehashed prefix "
                        "%s (attempt=%d).",
                        server.server_id,
                        used.get(
                            compute_short_server_prefix(server.server_id, attempt=0),
                            "<unknown>",
                        ),
                        candidate,
                        attempt,
                    )
                return

        raise RuntimeError(
            f"Unable to assign a unique short MCP tool prefix for server "
            f"{server.server_id} after {self._SHORT_PREFIX_MAX_REHASH_ATTEMPTS} "
            "attempts; the 3-character prefix space is too crowded."
        )

    def _create_prefixed_tools(self, tools: list[MCPTool], server: MCPServer, add_prefix: bool = True) -> list[MCPTool]:
        """
        Create prefixed tools and update tool mapping.

        Args:
            tools: List of original tools from server
            server: Server instance

        Returns:
            List of tools with prefixed names
        """
        prefixed_tools = []
        prefix = get_server_prefix(server)

        for tool in tools:
            tool_copy = tool.model_copy(deep=True)

            original_name = tool_copy.name
            prefixed_name = add_server_prefix_to_name(original_name, prefix)

            name_to_use = prefixed_name if add_prefix else original_name

            # Preserve all tool fields including metadata/_meta by avoiding mutation
            tool_copy.name = name_to_use
            prefixed_tools.append(tool_copy)

            # Register every known prefix form (alias, server_name, server_id,
            # short ID) so call_tool can resolve regardless of which form a
            # caller / cached client is using.
            self.tool_name_to_mcp_server_name_mapping[original_name] = prefix
            for known_prefix in iter_known_server_prefixes(server):
                qualified = add_server_prefix_to_name(original_name, known_prefix)
                self.tool_name_to_mcp_server_name_mapping[qualified] = prefix

        verbose_logger.info(f"Successfully fetched {len(prefixed_tools)} tools from server {server.name}")
        return prefixed_tools

    def _create_prefixed_prompts(
        self, prompts: list[Prompt], server: MCPServer, add_prefix: bool = True
    ) -> list[Prompt]:
        """
        Create prefixed prompts and update prompt mapping.

        Args:
            prompts: List of original prompts from server
            server: Server instance

        Returns:
            List of prompts with prefixed names
        """
        prefixed_prompts = []
        prefix = get_server_prefix(server)

        for prompt in prompts:
            prefixed_name = add_server_prefix_to_name(prompt.name, prefix)

            name_to_use = prefixed_name if add_prefix else prompt.name

            prompt.name = name_to_use
            prefixed_prompts.append(prompt)

        verbose_logger.info(f"Successfully fetched {len(prefixed_prompts)} prompts from server {server.name}")
        return prefixed_prompts

    def _create_prefixed_resources(
        self, resources: list[Resource], server: MCPServer, add_prefix: bool = True
    ) -> list[Resource]:
        """Prefix resource names and track origin server for read requests."""

        prefixed_resources: list[Resource] = []
        prefix = get_server_prefix(server)

        for resource in resources:
            name_to_use = add_server_prefix_to_name(resource.name, prefix) if add_prefix else resource.name
            resource.name = name_to_use
            prefixed_resources.append(resource)

        verbose_logger.info(f"Successfully fetched {len(prefixed_resources)} resources from server {server.name}")
        return prefixed_resources

    def _create_prefixed_resource_templates(
        self,
        resource_templates: list[ResourceTemplate],
        server: MCPServer,
        add_prefix: bool = True,
    ) -> list[ResourceTemplate]:
        """Prefix resource template names for multi-server scenarios."""

        prefixed_templates: list[ResourceTemplate] = []
        prefix = get_server_prefix(server)

        for resource_template in resource_templates:
            name_to_use = (
                add_server_prefix_to_name(resource_template.name, prefix) if add_prefix else resource_template.name
            )
            resource_template.name = name_to_use
            prefixed_templates.append(resource_template)

        verbose_logger.info(
            f"Successfully fetched {len(prefixed_templates)} resource templates from server {server.name}"
        )
        return prefixed_templates

    def check_allowed_or_banned_tools(self, tool_name: str, server: MCPServer) -> bool:
        """
        Check if the tool is allowed or banned for the given server
        """
        from litellm.proxy._experimental.mcp_server.utils import (
            server_applies_tool_allowlist,
        )

        if server_applies_tool_allowlist(server):
            if not server.allowed_tools:
                return False
            return tool_name in server.allowed_tools or f"{server.name}-{tool_name}" in server.allowed_tools
        if server.disallowed_tools:
            return (
                tool_name not in server.disallowed_tools and f"{server.name}-{tool_name}" not in server.disallowed_tools
            )
        return True

    def validate_allowed_params(self, tool_name: str, arguments: dict[str, Any], server: MCPServer) -> None:
        """
        Filter arguments to only include allowed parameters for the given tool.

        Args:
            tool_name: Name of the tool (with or without prefix)
            arguments: Dictionary of arguments to filter
            server: MCPServer configuration

        Returns:
            Filtered dictionary containing only allowed parameters

        Raises:
            HTTPException: If allowed_params is configured for this tool but arguments contain disallowed params
        """
        from litellm.proxy._experimental.mcp_server.utils import (
            split_server_prefix_from_name,
        )

        # If no allowed_params configured, return all arguments
        if not server.allowed_params:
            return

        # Get the unprefixed tool name to match against config
        unprefixed_tool_name, _ = split_server_prefix_from_name(tool_name)

        # Check both prefixed and unprefixed tool names
        allowed_params_list = server.allowed_params.get(tool_name) or server.allowed_params.get(unprefixed_tool_name)

        # If this tool doesn't have allowed_params specified, allow all params
        if allowed_params_list is None:
            return None

        # Filter arguments to only include allowed parameters
        disallowed_params = [param for param in arguments.keys() if param not in allowed_params_list]

        if disallowed_params:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": f"Parameters {disallowed_params} are not allowed for tool {tool_name}. "
                    f"Allowed parameters: {allowed_params_list}. "
                    f"Contact proxy admin to allow these parameters."
                },
            )

    async def check_tool_permission_for_key_team(
        self,
        tool_name: str,
        server: MCPServer,
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> None:
        """
        Check if a tool is allowed based on key/team object_permission.mcp_tool_permissions.
        Uses MCPRequestHandler.is_tool_allowed_for_server for consistent inheritance logic.
        Raises HTTPException if tool is not allowed.

        Args:
            tool_name: Name of the tool to check
            server: MCPServer object
            user_api_key_auth: User authentication

        Raises:
            HTTPException: If tool is not allowed for this key/team
        """
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        if not user_api_key_auth:
            return

        # Check if tool is allowed
        is_allowed = await MCPRequestHandler.is_tool_allowed_for_server(
            tool_name=tool_name,
            server_id=server.server_id,
            user_api_key_auth=user_api_key_auth,
        )

        if not is_allowed:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": f"Tool '{tool_name}' is not allowed for your key/team on server '{server.name}'. Contact proxy admin for access."
                },
            )

    async def _call_openapi_tool_handler(
        self,
        server: MCPServer,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        """
        Call an OpenAPI tool handler directly.

        For OpenAPI servers, instead of using MCP protocol, we call the tool handler
        that was registered during OpenAPI spec parsing. This handler makes direct
        HTTP requests to the API.

        Args:
            tool_name: The full tool name (with prefix) to call
            arguments: Tool arguments to pass to the handler

        Returns:
            CallToolResult with the response from the API
        """
        from mcp.types import TextContent

        from litellm.proxy._experimental.mcp_server.tool_registry import (
            global_mcp_tool_registry,
        )

        # Get the tool from the registry
        tool = global_mcp_tool_registry.get_tool(f"{server.name}-{tool_name}")
        if tool is None:
            # Tool not found in registry
            error_msg = f"OpenAPI tool {tool_name} not found in registry"
            verbose_logger.error(error_msg)
            return CallToolResult(
                content=[TextContent(type="text", text=error_msg)],
                isError=True,
            )

        try:
            # Call the tool handler with the arguments
            # The handler is an async function that makes the HTTP request
            handler_result = await tool.handler(**arguments)

            # Convert the handler result (string response) to CallToolResult format
            result = CallToolResult(
                content=[TextContent(type="text", text=str(handler_result))],
                isError=False,
            )

            return result

        except Exception as e:
            error_msg = f"Error calling OpenAPI tool {tool_name}: {str(e)}"
            verbose_logger.error(error_msg)
            return CallToolResult(
                content=[TextContent(type="text", text=error_msg)],
                isError=True,
            )

    async def pre_call_tool_check(
        self,
        name: str,
        arguments: dict[str, Any],
        server_name: str,
        user_api_key_auth: Optional[UserAPIKeyAuth],
        proxy_logging_obj: ProxyLogging,
        server: MCPServer,
        raw_headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Run pre-call checks and guardrail hooks for an MCP tool call.

        Returns a dict that may contain:
        - "arguments": hook-modified tool arguments (only if changed)
        - "extra_headers": headers injected by pre_mcp_call guardrail hooks
        """
        ## check if the tool is allowed or banned for the given server
        if not self.check_allowed_or_banned_tools(name, server):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": f"Tool {name} is not allowed for server {server.name}. Contact proxy admin to allow this tool."
                },
            )

        ## check tool-level permissions from object_permission
        await self.check_tool_permission_for_key_team(
            tool_name=name,
            server=server,
            user_api_key_auth=user_api_key_auth,
        )

        ## filter parameters based on allowed_params configuration
        self.validate_allowed_params(
            tool_name=name,
            arguments=arguments,
            server=server,
        )

        # Extract incoming Bearer token from raw request headers so
        # guardrails like MCPJWTSigner can verify + re-sign it (FR-5).
        normalized_raw = {k.lower(): v for k, v in (raw_headers or {}).items()}
        incoming_bearer_token: Optional[str] = None
        auth_hdr = normalized_raw.get("authorization", "")
        if auth_hdr.lower().startswith("bearer "):
            incoming_bearer_token = auth_hdr[len("bearer ") :]

        pre_hook_kwargs = {
            "name": name,
            "arguments": arguments,
            "server_name": server_name,
            "mcp_rate_limit_server_name": server.alias or server.server_name or server.name,
            "user_api_key_auth": user_api_key_auth,
            "user_api_key_user_id": (getattr(user_api_key_auth, "user_id", None) if user_api_key_auth else None),
            "user_api_key_team_id": (getattr(user_api_key_auth, "team_id", None) if user_api_key_auth else None),
            "user_api_key_end_user_id": (
                getattr(user_api_key_auth, "end_user_id", None) if user_api_key_auth else None
            ),
            "user_api_key_hash": (getattr(user_api_key_auth, "api_key_hash", None) if user_api_key_auth else None),
            "incoming_bearer_token": incoming_bearer_token,
        }

        # Create MCP request object for processing
        mcp_request_obj = proxy_logging_obj._create_mcp_request_object_from_kwargs(pre_hook_kwargs)

        # Convert to LLM format for existing guardrail compatibility
        synthetic_llm_data = proxy_logging_obj._convert_mcp_to_llm_format(mcp_request_obj, pre_hook_kwargs)

        hook_result: dict[str, Any] = {}
        try:
            # Use standard pre_call_hook
            modified_data = await proxy_logging_obj.pre_call_hook(
                user_api_key_dict=user_api_key_auth,  # type: ignore
                data=synthetic_llm_data,
                call_type=CallTypes.call_mcp_tool.value,
            )
            if modified_data:
                # Convert response back to MCP format and apply modifications
                modified_kwargs = proxy_logging_obj._convert_mcp_hook_response_to_kwargs(modified_data, pre_hook_kwargs)
                if modified_kwargs.get("arguments") != arguments:
                    hook_result["arguments"] = modified_kwargs["arguments"]
                if modified_kwargs.get("extra_headers"):
                    hook_result["extra_headers"] = modified_kwargs["extra_headers"]

        except (
            BlockedPiiEntityError,
            GuardrailRaisedException,
            HTTPException,
        ) as e:
            # Re-raise guardrail exceptions to properly fail the MCP call
            verbose_logger.error(f"Guardrail blocked MCP tool call pre call: {str(e)}")
            raise e

        return hook_result

    def _create_during_hook_task(
        self,
        name: str,
        arguments: dict[str, Any],
        server_name_from_prefix: Optional[str],
        user_api_key_auth: Optional[UserAPIKeyAuth],
        proxy_logging_obj: ProxyLogging,
        start_time: datetime.datetime,
    ):
        """Create and return a during hook task for MCP tool calls."""
        from litellm.types.llms.base import HiddenParams
        from litellm.types.mcp import MCPDuringCallRequestObject

        request_obj = MCPDuringCallRequestObject(
            tool_name=name,
            arguments=arguments,
            server_name=server_name_from_prefix,
            start_time=start_time.timestamp() if start_time else None,
            hidden_params=HiddenParams(),
        )

        during_hook_kwargs = {
            "name": name,
            "arguments": arguments,
            "server_name": server_name_from_prefix,
            "user_api_key_auth": user_api_key_auth,
        }

        synthetic_llm_data = proxy_logging_obj._convert_mcp_to_llm_format(request_obj, during_hook_kwargs)

        return asyncio.create_task(
            proxy_logging_obj.during_call_hook(
                user_api_key_dict=user_api_key_auth,
                data=synthetic_llm_data,
                call_type=CallTypes.call_mcp_tool.value,
            )
        )

    def _get_call_semaphore(self, mcp_server: MCPServer) -> Optional[asyncio.Semaphore]:
        limit = mcp_server.max_concurrent_requests
        if limit is None or limit <= 0:
            return None
        cached = self._server_call_semaphores.get(mcp_server.server_id)
        if cached is not None and cached[0] == limit:
            return cached[1]
        semaphore = asyncio.Semaphore(limit)
        self._server_call_semaphores[mcp_server.server_id] = (limit, semaphore)
        return semaphore

    @asynccontextmanager
    async def _limit_outbound_concurrency(self, mcp_server: MCPServer) -> AsyncIterator[None]:
        semaphore = self._get_call_semaphore(mcp_server)
        if semaphore is None:
            yield
            return
        async with semaphore:
            yield

    async def _obo_call_tool_with_retry(
        self,
        *,
        client: MCPClient,
        call_tool_params: MCPCallToolRequestParams,
        host_progress_callback: Optional[Callable],
        mcp_server: MCPServer,
        server_auth_header: str | dict[str, str] | None,
        extra_headers: Optional[dict[str, str]],
        stdio_env: Optional[dict[str, str]],
        subject_token: Optional[str],
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> CallToolResult:
        """Call a token_exchange (OBO) tool; on an upstream 401/403 re-mint the token once and retry.

        The exchanged token is baked into the client at build time, so the retry invalidates the
        cached exchange and rebuilds the client (which re-exchanges). One retry only: a non-auth
        failure or a second auth failure degrades to the normal ``isError`` result, and a re-exchange
        that now fails surfaces its own 401 challenge from ``_create_mcp_client``.
        """
        try:
            return await client.call_tool(
                call_tool_params, host_progress_callback=host_progress_callback, raise_on_error=True
            )
        except Exception as exc:
            if _extract_upstream_auth_failure(exc) is None:
                return MCPClient.error_tool_result(exc)
            spec = to_server_spec(mcp_server)
            if spec is not None:
                await self._cred_provider.invalidate_credentials(to_subject(user_api_key_auth, subject_token), spec)
            retry_client = await self._create_mcp_client(
                server=mcp_server,
                mcp_auth_header=server_auth_header,
                extra_headers=extra_headers,
                stdio_env=stdio_env,
                subject_token=subject_token,
                user_api_key_auth=user_api_key_auth,
            )
            return await retry_client.call_tool(call_tool_params, host_progress_callback=host_progress_callback)

    async def _call_regular_mcp_tool(
        self,
        mcp_server: MCPServer,
        original_tool_name: str,
        arguments: dict[str, Any],
        tasks: list,
        mcp_auth_header: Optional[str],
        mcp_server_auth_headers: Optional[dict[str, dict[str, str]]],
        oauth2_headers: Optional[dict[str, str]],
        raw_headers: Optional[dict[str, str]],
        proxy_logging_obj: Optional[ProxyLogging],
        host_progress_callback: Optional[Callable] = None,
        hook_extra_headers: Optional[dict[str, str]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> CallToolResult:
        """
        Call a regular MCP tool using the MCP client.

        Args:
            mcp_server: The MCP server configuration
            original_tool_name: The original tool name (without prefix)
            arguments: Tool arguments
            tasks: List of async tasks to append to (for during hooks)
            mcp_auth_header: MCP auth header (deprecated)
            mcp_server_auth_headers: Optional dict of server-specific auth headers
            oauth2_headers: Optional OAuth2 headers
            raw_headers: Optional raw headers from the request
            proxy_logging_obj: Optional ProxyLogging object for hook integration
            host_progress_callback: Optional callback for progress updates
            hook_extra_headers: Optional headers injected by pre_mcp_call guardrail
                hooks. Merged last (highest priority) into outbound request headers.

        Returns:
            CallToolResult from the MCP server

        Raises:
            BlockedPiiEntityError: If PII is blocked by guardrails
            GuardrailRaisedException: If guardrails block the call
            HTTPException: If an HTTP error occurs
        """
        # Get server-specific auth header if available (case-insensitive)
        # FIX: Added case-insensitive matching to handle auth header keys that may not match
        # the exact case of server alias/name (e.g., '1litellmagcgateway' vs '1LiteLLMAGCGateway')
        server_auth_header: Optional[Union[dict[str, str], str]] = None
        if mcp_server_auth_headers:
            # Normalize keys for case-insensitive lookup
            from litellm.proxy._experimental.mcp_server.utils import (
                lookup_mcp_server_auth_in_headers,
            )

            server_auth_header = lookup_mcp_server_auth_in_headers(
                mcp_server_auth_headers,
                alias=mcp_server.alias,
                server_name=mcp_server.server_name,
            )

        # Fall back to deprecated mcp_auth_header if no server-specific header found
        if server_auth_header is None:
            server_auth_header = mcp_auth_header

        # Extract subject token for OAuth2 Token Exchange (OBO) flow
        subject_token: Optional[str] = None
        extra_headers: Optional[dict[str, str]] = None
        if mcp_server.auth_type == MCPAuth.oauth2_token_exchange:
            subject_token = self._extract_bearer_token(oauth2_headers, raw_headers)
        elif mcp_server.auth_type == MCPAuth.oauth2:
            if mcp_server.has_client_credentials:
                # For M2M OAuth servers, Authorization must come from token fetch.
                extra_headers = None
            else:
                extra_headers = oauth2_headers
                # Migrated authorization_code: the v2 resolver injects the stored per-user
                # token, so drop the caller-forwarded Authorization (apply-if-absent would
                # otherwise let it shadow the resolved token). Delegate keeps it. Centralized
                # via _should_strip_caller_authorization to match _prepare_mcp_server_headers.
                if extra_headers and _should_strip_caller_authorization(
                    mcp_server=mcp_server,
                    raw_headers=raw_headers,
                    user_api_key_auth=user_api_key_auth,
                ):
                    extra_headers = _without_authorization(extra_headers)
        elif mcp_server.is_true_passthrough or mcp_server.is_oauth_delegate:
            extra_headers = _client_forwarded_authorization_headers(
                mcp_server=mcp_server,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                user_api_key_auth=user_api_key_auth,
            )

        if mcp_server.extra_headers and raw_headers:
            if extra_headers is None:
                extra_headers = {}

            normalized_raw_headers = {str(k).lower(): v for k, v in raw_headers.items() if isinstance(k, str)}
            strip_caller_authorization = _should_strip_caller_authorization(
                mcp_server=mcp_server,
                raw_headers=raw_headers,
                user_api_key_auth=user_api_key_auth,
            )

            for header in mcp_server.extra_headers:
                if not isinstance(header, str):
                    continue
                if header.lower() == "authorization" and strip_caller_authorization:
                    continue
                header_value = normalized_raw_headers.get(header.lower())
                if header_value is None:
                    continue
                extra_headers[header] = header_value

        # Interpolate env vars into static_headers. Raises
        # MCPMissingUserEnvVarsError when the calling user has not filled in
        # a required per-user variable — the REST layer converts that into
        # a friendly 412 with a setup URL.
        resolved_static_headers = await self._resolve_static_headers_with_env_vars(mcp_server, user_api_key_auth)
        if resolved_static_headers:
            if extra_headers is None:
                extra_headers = {}
            extra_headers.update(resolved_static_headers)

        if hook_extra_headers:
            if extra_headers is None:
                extra_headers = {}
            if "Authorization" in hook_extra_headers:
                if "Authorization" in extra_headers:
                    verbose_logger.warning(
                        "MCPServerManager: hook_extra_headers 'Authorization' will overwrite "
                        "the existing Authorization header from static_headers. "
                        "The hook JWT will take precedence."
                    )
                elif server_auth_header is not None:
                    # server_auth_header is passed separately to _create_mcp_client as
                    # auth_value.  Both will reach the upstream server — warn so admins
                    # know two Authorization credentials are being sent.
                    verbose_logger.warning(
                        "MCPServerManager: hook_extra_headers injects 'Authorization' while "
                        "server '%s' already has a configured authentication_token. "
                        "Both credentials will be sent; the hook header is in extra_headers "
                        "and the server token is in auth_value — the upstream server decides "
                        "which one wins.  Consider unsetting authentication_token if you want "
                        "the hook JWT to be the sole credential.",
                        mcp_server.server_name or mcp_server.name,
                    )
            extra_headers.update(hook_extra_headers)

        # Reset to None if no headers were actually added
        if extra_headers is not None and len(extra_headers) == 0:
            extra_headers = None

        stdio_env = self._build_stdio_env(mcp_server, raw_headers)

        client = await self._create_mcp_client(
            server=mcp_server,
            mcp_auth_header=server_auth_header,
            extra_headers=extra_headers,
            stdio_env=stdio_env,
            subject_token=subject_token,
            user_api_key_auth=user_api_key_auth,
        )

        call_tool_params = MCPCallToolRequestParams(
            name=original_tool_name,
            arguments=arguments,
        )

        if mcp_server.auth_type == MCPAuth.oauth2_token_exchange and subject_token:
            # OBO: the exchanged token may have been revoked/rotated upstream since it was cached, so
            # an upstream 401 gets one re-mint + retry. Gated to this mode; all others keep the plain
            # single call below.
            async def _obo_call_tool_limited():
                async with self._limit_outbound_concurrency(mcp_server):
                    return await self._obo_call_tool_with_retry(
                        client=client,
                        call_tool_params=call_tool_params,
                        host_progress_callback=host_progress_callback,
                        mcp_server=mcp_server,
                        server_auth_header=server_auth_header,
                        extra_headers=extra_headers,
                        stdio_env=stdio_env,
                        subject_token=subject_token,
                        user_api_key_auth=user_api_key_auth,
                    )

            tool_call_coro = _obo_call_tool_limited()
        else:
            # Scoped to the two client-forwarded token modes this stack introduced; legacy
            # oauth2 + delegate_auth_to_upstream (is_oauth_passthrough) is being removed, so it is not
            # added here even though the list path still relays for it.
            relays_upstream_auth = mcp_server.is_true_passthrough or mcp_server.is_oauth_delegate
            server_label = mcp_server.name or mcp_server.server_name or mcp_server.alias or ""

            async def _call_tool_via_client(client, params):
                async with self._limit_outbound_concurrency(mcp_server):
                    if not relays_upstream_auth:
                        return await client.call_tool(params, host_progress_callback=host_progress_callback)
                    # The client-forwarded modes carry the caller's own upstream token, so an upstream
                    # 401 (expired/invalid token) is the caller's to resolve: relay it as
                    # MCPUpstreamAuthError so single-server REST callers turn it into a 401 +
                    # WWW-Authenticate and re-run the upstream OAuth flow. Only 401 is a re-auth signal
                    # (mirrors the list path and MCPUpstreamAuthError's contract); a 403 is a genuine
                    # authorization failure that re-auth won't fix, so it takes the non-auth branch and
                    # stays a visible warning. raise_on_error only re-raises transport failures
                    # (tool-level isError results are still returned normally); a non-auth failure keeps
                    # the same isError degradation the default path produces.
                    try:
                        return await client.call_tool(
                            params, host_progress_callback=host_progress_callback, raise_on_error=True
                        )
                    except Exception as e:
                        auth_info = _extract_upstream_auth_failure(e)
                        if auth_info is None or auth_info[0] != 401:
                            # A genuine (non-auth or 403-forbidden) upstream/transport failure.
                            # raise_on_error demoted the client-layer log to debug, so surface it here at
                            # warning level to keep the outage visible; the caller still gets the graceful
                            # isError result the default masking path would have produced. Log the
                            # exception type only, never str(e), which for an httpx error embeds the
                            # upstream URL (a credential can hide in it).
                            verbose_logger.warning(
                                "Pass-through MCP tool call failed against %s (non-auth, %s)",
                                server_label,
                                type(e).__name__,
                            )
                            return client.error_tool_result(e)
                        _, www_authenticate = auth_info
                        raise MCPUpstreamAuthError(
                            status_code=401,
                            www_authenticate=www_authenticate,
                            server_name=server_label,
                        ) from e

            tool_call_coro = _call_tool_via_client(client, call_tool_params)

        tasks.append(asyncio.create_task(tool_call_coro))

        _timeout = mcp_server.timeout if mcp_server.timeout is not None else MCP_CLIENT_TIMEOUT
        try:
            mcp_responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=_timeout)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "timeout",
                    "message": f"MCP tool call timed out after {_timeout}s",
                },
            )
        except (
            BlockedPiiEntityError,
            GuardrailRaisedException,
            HTTPException,
        ) as e:
            verbose_logger.error(f"Guardrail blocked MCP tool call during result check: {str(e)}")
            raise e

        # If proxy_logging_obj is None, the tool call result is at index 0
        # If proxy_logging_obj is not None, the tool call result is at index 1 (after the during hook task)
        result_index = 1 if proxy_logging_obj else 0
        result = mcp_responses[result_index]
        self._remember_upstream_initialize_instructions(mcp_server, client)

        return cast(CallToolResult, result)

    def _resolve_mcp_server_for_tool_call(
        self,
        server_name: str,
        name: str,
    ) -> MCPServer:
        """Resolve MCP server for call_tool (prefixed name, registry, fallback)."""
        prefixed_tool_name = add_server_prefix_to_name(name, server_name)
        mcp_server = self._get_mcp_server_from_tool_name(prefixed_tool_name)
        resolved_by_server_name_only = False
        normalized_server_name = normalize_server_name(server_name)

        def _candidate_matches_server_name(candidate: MCPServer) -> bool:
            for identifier in (
                candidate.alias,
                candidate.server_name,
                candidate.name,
            ):
                if identifier and normalize_server_name(identifier) == (normalized_server_name):
                    return True
            return False

        if mcp_server is None:
            for candidate in self.get_registry().values():
                if _candidate_matches_server_name(candidate):
                    mcp_server = candidate
                    resolved_by_server_name_only = True
                    break
        if mcp_server is None:
            fallback = self._get_mcp_server_from_tool_name(name)
            if fallback is not None and (not server_name or _candidate_matches_server_name(fallback)):
                mcp_server = fallback
        if mcp_server is None:
            raise ValueError(f"Tool {name} not found")

        if resolved_by_server_name_only:
            tool_known = (
                name in self.tool_name_to_mcp_server_name_mapping
                or prefixed_tool_name in self.tool_name_to_mcp_server_name_mapping
            )
            if not tool_known:
                raise ValueError(f"Tool {name} not found")

        return mcp_server

    async def has_user_oauth_token(self, server: MCPServer, user_api_key_auth: Optional[UserAPIKeyAuth]) -> bool:
        """Whether the v2 resolver can produce a per-user token for this server right now.

        This is the preemptive 401's existence check, routed through the same resolver that drives
        the egress so every authorization_code resolution (egress and the discovery challenge) runs
        through v2. Returns False for a server the resolver does not own (a None spec).
        """
        spec = to_server_spec(server)
        if spec is None:
            return False
        return await self._cred_provider.has_user_token(to_subject(user_api_key_auth, None), spec)

    async def invalidate_user_oauth_token_cache(self, user_id: str, server_id: str) -> None:
        """Drop every cached token for ``(user_id, server_id)`` after the credential row changes
        (re-auth, revoke, config-change purge): the v2 chain's cache and the legacy per-user token
        cache, so the next resolve reads the new row instead of serving the replaced token until its
        cache TTL, whichever path resolves it. This is the single invalidation point for per-user
        OAuth tokens; callers must not evict individual caches directly. Best-effort: a cache-drop
        failure is logged, never raised, because the DB write already succeeded and the TTL remains
        the backstop.
        """
        try:
            await self._per_user_oauth_token_store.invalidate(user_id, server_id)
        except Exception as exc:  # noqa: BLE001 - cache drop is best-effort; TTL is the backstop
            verbose_logger.warning(
                "Failed to invalidate cached MCP OAuth token for user=%s server=%s: %s", user_id, server_id, exc
            )
        try:
            await self._per_user_token_cache.delete(user_id, server_id)
        except Exception as exc:  # noqa: BLE001 - cache drop is best-effort; TTL is the backstop
            verbose_logger.warning(
                "Failed to drop legacy cached MCP OAuth token for user=%s server=%s: %s", user_id, server_id, exc
            )

    async def _resolve_oauth2_headers_for_tool_call(
        self,
        mcp_server: MCPServer,
        oauth2_headers: Optional[dict[str, str]],
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> Optional[dict[str, str]]:
        """Look up per-user OAuth headers when the client did not supply a token."""
        if not mcp_server.needs_user_oauth_token or oauth2_headers or user_api_key_auth is None:
            return oauth2_headers

        if to_server_spec(mcp_server) is not None:
            # Migrated to v2: the resolver owns this server's per-user token (inject or fail-closed
            # 401). Building it into extra_headers here would let the v2 graft defer to it and
            # shadow the resolver, double-resolving and hiding the per-server challenge.
            return oauth2_headers

        user_id = getattr(user_api_key_auth, "user_id", None)
        if not user_id:
            return oauth2_headers

        try:
            from litellm.proxy._experimental.mcp_server.server import (  # noqa: PLC0415
                _get_user_oauth_extra_headers_from_db,
            )

            stored_headers = await _get_user_oauth_extra_headers_from_db(
                server=mcp_server,
                user_api_key_auth=user_api_key_auth,
            )
            if stored_headers:
                return stored_headers
        except Exception as _lookup_exc:
            verbose_logger.debug(
                "call_tool: per-user token lookup failed for user=%s server=%s: %s",
                user_id,
                mcp_server.server_id,
                _lookup_exc,
            )
        return oauth2_headers

    async def _gather_openapi_tool_tasks(
        self,
        tasks: list[Any],
        proxy_logging_obj: Optional[ProxyLogging],
    ) -> CallToolResult:
        """Await OpenAPI tool tasks and return the tool call result."""
        try:
            mcp_responses = await asyncio.gather(*tasks)
            result_index = 1 if proxy_logging_obj else 0
            return cast(CallToolResult, mcp_responses[result_index])
        except (
            BlockedPiiEntityError,
            GuardrailRaisedException,
            HTTPException,
        ) as e:
            verbose_logger.error(f"Guardrail blocked MCP tool call during result check: {str(e)}")
            raise e

    async def call_tool(
        self,
        server_name: str,
        name: str,
        arguments: dict[str, Any],
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_server_auth_headers: Optional[dict[str, dict[str, str]]] = None,
        proxy_logging_obj: Optional[ProxyLogging] = None,
        oauth2_headers: Optional[dict[str, str]] = None,
        raw_headers: Optional[dict[str, str]] = None,
        host_progress_callback: Optional[Callable] = None,
    ) -> CallToolResult:
        """
        Call a tool with the given name and arguments

        Args:
            server_name: Server name
            name: Tool name
            arguments: Tool arguments
            user_api_key_auth: User authentication
            mcp_auth_header: MCP auth header (deprecated)
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}
            proxy_logging_obj: Optional ProxyLogging object for hook integration


        Returns:
            CallToolResult from the MCP server
        """
        start_time = datetime.datetime.now()
        mcp_server = self._resolve_mcp_server_for_tool_call(server_name, name)

        # Resolved before any hook runs so a missing BYOK credential (401) never
        # leaves during-hook side effects (audit logging, rate-limit bookkeeping)
        # recorded against a call that ultimately fails.
        mcp_auth_header = await _resolve_byok_mcp_auth_header(
            mcp_server,
            user_api_key_auth,
            mcp_auth_header,
        )

        #########################################################
        # Pre MCP Tool Call Hook
        # Allow validation and modification of tool calls before execution
        # Using standard pre_call_hook
        #########################################################
        hook_result: dict[str, Any] = {}
        if proxy_logging_obj:
            hook_result = await self.pre_call_tool_check(
                name=name,
                arguments=arguments,
                server_name=server_name,
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging_obj,
                server=mcp_server,
                raw_headers=raw_headers,
            )
            if "arguments" in hook_result:
                arguments = hook_result["arguments"]

        # Prepare tasks for during hooks
        tasks = []
        if proxy_logging_obj:
            during_hook_task = self._create_during_hook_task(
                name=name,
                arguments=arguments,
                server_name_from_prefix=server_name,
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging_obj,
                start_time=start_time,
            )
            tasks.append(during_hook_task)

        oauth2_headers = await self._resolve_oauth2_headers_for_tool_call(mcp_server, oauth2_headers, user_api_key_auth)

        # For OpenAPI servers, call the tool handler directly instead of via MCP client
        if mcp_server.spec_path:
            verbose_logger.debug("Calling OpenAPI tool %s directly via HTTP handler", name)
            if hook_result.get("extra_headers"):
                verbose_logger.warning(
                    "pre_mcp_call hook returned extra_headers for OpenAPI-backed "
                    "MCP server '%s' — header injection is not supported for "
                    "OpenAPI servers; headers will be ignored. Use SSE/HTTP "
                    "transport to enable hook header injection.",
                    server_name,
                )

            auth_header_value = (
                _format_byok_openapi_auth_header(mcp_server, mcp_auth_header) if mcp_auth_header else None
            )
            forwarded_headers = _openapi_forwarded_extra_headers(mcp_server, raw_headers, user_api_key_auth)

            async def _call_openapi_via_handler():
                from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
                    _request_auth_header,
                    _request_extra_headers,
                )

                auth_token = _request_auth_header.set(auth_header_value)
                extra_token = _request_extra_headers.set(forwarded_headers)
                try:
                    async with self._limit_outbound_concurrency(mcp_server):
                        return await self._call_openapi_tool_handler(mcp_server, name, arguments)
                finally:
                    _request_auth_header.reset(auth_token)
                    _request_extra_headers.reset(extra_token)

            tasks.append(asyncio.create_task(_call_openapi_via_handler()))
        else:
            return await self._call_regular_mcp_tool(
                mcp_server=mcp_server,
                original_tool_name=name,
                arguments=arguments,
                tasks=tasks,
                mcp_auth_header=mcp_auth_header,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                proxy_logging_obj=proxy_logging_obj,
                host_progress_callback=host_progress_callback,
                hook_extra_headers=hook_result.get("extra_headers"),
                user_api_key_auth=user_api_key_auth,
            )

        return await self._gather_openapi_tool_tasks(tasks, proxy_logging_obj)

    #########################################################
    # End of Methods that call the upstream MCP servers
    #########################################################

    def initialize_tool_name_to_mcp_server_name_mapping(self):
        """
        On startup, initialize the tool name to MCP server name mapping
        """
        try:
            if asyncio.get_running_loop():
                asyncio.create_task(self._initialize_tool_name_to_mcp_server_name_mapping())
        except RuntimeError as e:  # no running event loop
            verbose_logger.exception(
                f"No running event loop - skipping tool name to MCP server name mapping initialization: {str(e)}"
            )

    async def _initialize_tool_name_to_mcp_server_name_mapping(self):
        """
        Call list_tools for each server and update the tool name to MCP server name mapping
        Note: This now handles prefixed tool names
        """
        for server in self.get_registry().values():
            if server.needs_user_oauth_token:
                # Skip OAuth2 servers that rely on user-provided tokens
                continue
            try:
                tools = await self._get_tools_from_server(server)
            except MCPUpstreamAuthError as e:
                # Pass-through servers expect a user-supplied bearer token;
                # at startup we have none, so an upstream 401 is normal.
                # Swallow it so we keep mapping the remaining servers.
                verbose_logger.debug(
                    f"Skipping tool name mapping for server {server.name} due to upstream auth error: {str(e)}"
                )
                continue
            except Exception as e:
                verbose_logger.warning(
                    f"Failed to get tools from server {server.name} during tool name mapping initialization: {str(e)}"
                )
                continue
            for tool in tools:
                # The tool.name here is already prefixed from _get_tools_from_server
                # Extract original name for mapping
                original_name, _ = split_server_prefix_from_name(tool.name)
                self.tool_name_to_mcp_server_name_mapping[original_name] = server.name
                self.tool_name_to_mcp_server_name_mapping[tool.name] = server.name

    def _get_mcp_server_from_tool_name(self, tool_name: str) -> Optional[MCPServer]:
        """
        Get the MCP Server from the tool name (handles both prefixed and non-prefixed names)

        Args:
            tool_name: Tool name (can be prefixed or non-prefixed)

        Returns:
            MCPServer if found, None otherwise
        """
        registry_servers = list(self.get_registry().values())

        # Build prefix → server lookup covering every known form a tool name
        # may take (alias / server_name / server_id / short ID).  This is what
        # makes the short-prefix mode work without breaking historical names.
        prefix_to_server: dict[str, MCPServer] = {}
        for server in registry_servers:
            for known_prefix in iter_known_server_prefixes(server):
                normalised = normalize_server_name(known_prefix)
                prefix_to_server.setdefault(normalised, server)

        # First try with the original tool name
        if tool_name in self.tool_name_to_mcp_server_name_mapping:
            server_name = self.tool_name_to_mcp_server_name_mapping[tool_name]
            normalised_lookup = normalize_server_name(server_name)
            if normalised_lookup in prefix_to_server:
                return prefix_to_server[normalised_lookup]
            for server in registry_servers:
                if normalize_server_name(server.name) == normalised_lookup:
                    return server

        # If not found and tool name is prefixed, extract the prefix and
        # match against any known form.
        if is_tool_name_prefixed(tool_name, known_server_prefixes=set(prefix_to_server.keys())):
            (
                original_tool_name,
                server_name_from_prefix,
            ) = split_server_prefix_from_name(tool_name)
            normalised_prefix = normalize_server_name(server_name_from_prefix)
            matched_server = prefix_to_server.get(normalised_prefix)
            if matched_server is not None and (
                original_tool_name in self.tool_name_to_mcp_server_name_mapping
                or tool_name in self.tool_name_to_mcp_server_name_mapping
            ):
                return matched_server

        return None

    async def reload_servers_from_database(self):
        """Re-synchronize the in-memory MCP server registry with the database."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_prisma_client_or_throw,
        )

        verbose_logger.debug("Loading MCP servers from database into registry...")
        self._upstream_initialize_instructions_by_server_id.clear()
        self._upstream_initialize_instructions_probed_at.clear()

        # perform authz check to filter the mcp servers user has access to
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        # Load only "active", legacy "approved", and NULL (no approval workflow) rows.
        # Pending/rejected servers are excluded at the DB level so we never load them.
        from litellm.proxy._experimental.mcp_server.db import LiteLLM_MCPServerTable

        raw_rows = await MCPServerRepository(prisma_client).table.find_many(
            where={
                "OR": [
                    {"approval_status": None},
                    {"approval_status": {"in": ["active", "approved"]}},
                ]
            }
        )
        db_mcp_servers = [LiteLLM_MCPServerTable(**r.model_dump()) for r in raw_rows]
        verbose_logger.info(f"Found {len(db_mcp_servers)} MCP servers in database")

        previous_registry = self.registry
        new_registry: dict[str, MCPServer] = {}

        # Stage one: build every server.  Stage two assigns short prefixes
        # against the *full* set so dedup is deterministic regardless of
        # iteration order.
        for server in db_mcp_servers:
            try:
                existing_server = previous_registry.get(server.server_id)

                if (
                    existing_server is not None
                    and existing_server.updated_at is not None
                    and server.updated_at is not None
                    and existing_server.updated_at == server.updated_at
                ):
                    # Re-use existing server instance to avoid re-running build_mcp_server_from_table()
                    # which can perform network discovery for OAuth2 servers.
                    new_registry[server.server_id] = existing_server
                    continue

                _warn_on_server_name_fields(
                    server_id=server.server_id,
                    alias=getattr(server, "alias", None),
                    server_name=getattr(server, "server_name", None),
                )
                verbose_logger.debug(f"Building server from DB: {server.server_id} ({server.server_name})")
                # raw_rows come straight from the DB, so their global env var
                # values (like credentials) are still encrypted here, unlike the
                # already-decrypted records add_server/update_server are handed.
                # Decrypt them while building the registry entry.
                new_server = await self.build_mcp_server_from_table(server, env_vars_are_encrypted=True)
                # Carry the cached short_prefix from the previous registry entry
                # (if any) so the prefix is stable across reloads.
                if existing_server is not None and existing_server.short_prefix:
                    new_server.short_prefix = existing_server.short_prefix
                _carry_forward_resolved_oauth_endpoints(new_server=new_server, previous_server=existing_server)
                new_registry[server.server_id] = new_server
            except Exception as e:
                verbose_logger.exception(
                    "Skipping MCP server %s (%s) during DB reload: %s",
                    server.server_id,
                    getattr(server, "alias", None),
                    e,
                )

        # Assign short prefixes against the full candidate set without
        # publishing the staged registry to concurrent callers.
        registered_registry: dict[str, MCPServer] = {}
        registered_openapi_tools = False
        for server_id, new_server in new_registry.items():
            try:
                self._assign_unique_short_prefix(new_server, registry=new_registry)
                # Register OpenAPI tools *after* the final short prefix is assigned
                # so the tools are stored in the global registry under the same
                # prefix that lookups will use.
                await self._maybe_register_openapi_tools(new_server, initialize_mapping=False)
                registered_registry[server_id] = new_server
                if new_server.spec_path:
                    registered_openapi_tools = True
            except Exception as e:
                verbose_logger.exception(
                    "Skipping MCP server %s (%s) during DB reload: %s",
                    new_server.server_id,
                    getattr(new_server, "alias", None),
                    e,
                )

        self.registry = registered_registry
        if registered_openapi_tools:
            self.initialize_tool_name_to_mcp_server_name_mapping()

        verbose_logger.debug("MCP registry refreshed (%s servers in registry)", len(registered_registry))

    def get_mcp_servers_from_ids(self, server_ids: list[str]) -> list[MCPServer]:
        servers = []
        registry = self.get_registry()
        for server in registry.values():
            if server.server_id in server_ids:
                servers.append(server)
        return servers

    def _get_general_settings(self) -> dict[str, Any]:
        """Get general_settings, importing lazily to avoid circular imports."""
        try:
            from litellm.proxy.proxy_server import (
                general_settings as proxy_general_settings,
            )

            return proxy_general_settings
        except ImportError:
            # Fallback if proxy_server not available
            return {}

    def _is_server_accessible_from_ip(self, server: MCPServer, client_ip: Optional[str]) -> bool:
        """
        Check if a server is accessible from the given client IP.

        - If client_ip is None, no IP filtering is applied (internal callers).
        - If the server has available_on_public_internet=True, it's always accessible.
        - Otherwise, only internal/private IPs can access it.
        """
        if client_ip is None:
            return True
        if server.available_on_public_internet:
            return True
        # Check backwards compat: litellm.public_mcp_servers
        public_ids = set(litellm.public_mcp_servers or [])
        if server.server_id in public_ids:
            return True
        # Non-public server: only accessible from internal IPs
        general_settings = self._get_general_settings()
        internal_networks = IPAddressUtils.parse_internal_networks(general_settings.get("mcp_internal_ip_ranges"))
        return IPAddressUtils.is_internal_ip(client_ip, internal_networks)

    def get_mcp_server_by_id(self, server_id: str) -> Optional[MCPServer]:
        """
        Get the MCP Server from the server id
        """
        registry = self.get_registry()
        for server in registry.values():
            if server.server_id == server_id:
                return server
        return None

    def get_public_mcp_servers(self) -> list[MCPServer]:
        """
        Return the MCP servers published to the AI Hub via /v1/mcp/make_public.

        Default (litellm.public_mcp_hub_strict_whitelist=True): mirrors
        /public/model_hub and /public/agent_hub — gates strictly on the
        litellm.public_mcp_servers whitelist. Returns an empty list when no
        servers have been published. The per-server available_on_public_internet
        flag is unrelated — it governs IP-based access in
        _is_server_accessible_from_ip, not hub visibility.

        Legacy (litellm.public_mcp_hub_strict_whitelist=False): preserves the
        pre-fix behavior where any server with available_on_public_internet=True
        is also included. Intended as a one-release migration window for
        deployments that relied on the OR-with-default semantics; will be
        removed in a future release.
        """
        if litellm.public_mcp_hub_strict_whitelist:
            if litellm.public_mcp_servers is None:
                return []
            public_ids = set(litellm.public_mcp_servers)
            return [server for server in self.get_registry().values() if server.server_id in public_ids]

        public_ids = set(litellm.public_mcp_servers or [])
        return [
            server
            for server in self.get_registry().values()
            if server.available_on_public_internet or server.server_id in public_ids
        ]

    def expand_permission_list(self, identifiers: list[str]) -> list[str]:
        """
        Expand a permission list of server_ids/names/aliases into concrete
        server_ids against the current region's config + DB registry union.

        Entries that match a server_id pass through unchanged. Entries that
        match an alias/server_name/name are replaced with every matching
        server_id (duplicate names grant access to all matches). Entries
        that resolve to nothing pass through as-is and a debug log is
        emitted so admins can diagnose stale/typo permission entries — the
        downstream access-check denies them when compared against the
        concrete request server_id.
        """
        if not identifiers:
            return []
        registry = self.get_registry()
        expanded: set[str] = set()
        for identifier in identifiers:
            if identifier in registry:
                expanded.add(identifier)
                continue
            matches: list[str] = [
                server_id
                for server_id, server in registry.items()
                if server.alias == identifier or server.server_name == identifier or server.name == identifier
            ]
            if matches:
                expanded.update(matches)
            else:
                # %r quotes and escapes control chars so an admin-controlled
                # identifier with newlines cannot forge log lines.
                verbose_logger.debug(
                    "MCP permission entry %r does not resolve to any known "
                    "server (config + DB union). Passing through — the "
                    "downstream access check will deny it if it's stale.",
                    identifier,
                )
                expanded.add(identifier)
        return list(expanded)

    def expand_tool_permissions(
        self,
        tool_permissions: Optional[dict[str, list[str]]],
    ) -> dict[str, list[str]]:
        """
        Rewrite an ``mcp_tool_permissions`` dict keyed by id/name/alias so
        every key is a concrete server_id where possible. Tool lists from
        keys that point at the same server are unioned, matching the
        "duplicate names grant access to all matches" semantics of
        ``expand_permission_list``.

        Required so name-based keys don't silently drop their tool
        restrictions when the lookup uses the resolved server_id. Unresolved
        keys pass through via ``expand_permission_list`` so stale id-keyed
        restrictions still apply when the same string is used for lookup.
        """
        if not tool_permissions:
            return {}
        result: dict[str, list[str]] = {}
        for key, tools in tool_permissions.items():
            for server_id in self.expand_permission_list([key]):
                result.setdefault(server_id, []).extend(tools or [])
        return result

    def get_mcp_server_by_name(self, server_name: str, client_ip: Optional[str] = None) -> Optional[MCPServer]:
        """
        Get the MCP Server from the server name.

        Uses priority-based matching to avoid collisions:
        1. First pass: exact alias match (highest priority)
        2. Second pass: exact server_name match
        3. Third pass: exact name match (lowest priority)

        Args:
            server_name: The server name to look up.
            client_ip: Optional client IP for access control. When provided,
                       non-public servers are hidden from external IPs.
        """
        registry = self.get_registry()
        # Pass 1: Match by alias (highest priority)
        for server in registry.values():
            if server.alias == server_name:
                if not self._is_server_accessible_from_ip(server, client_ip):
                    return None
                return server
        # Pass 2: Match by server_name
        for server in registry.values():
            if server.server_name == server_name:
                if not self._is_server_accessible_from_ip(server, client_ip):
                    return None
                return server
        # Pass 3: Match by name (lowest priority)
        for server in registry.values():
            if server.name == server_name:
                if not self._is_server_accessible_from_ip(server, client_ip):
                    return None
                return server
        return None

    def get_filtered_registry(self, client_ip: Optional[str] = None) -> dict[str, MCPServer]:
        """
        Get registry filtered by client IP access control.

        Args:
            client_ip: Optional client IP. When provided, non-public servers
                       are hidden from external IPs. When None, returns all servers.
        """
        registry = self.get_registry()
        if client_ip is None:
            return registry
        return {k: v for k, v in registry.items() if self._is_server_accessible_from_ip(v, client_ip)}

    def _generate_stable_server_id(
        self,
        server_name: str,
        url: str,
        transport: str,
        auth_type: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> str:
        """
        Generate a stable server ID based on server parameters using a hash function.

        This is critical to ensure the server_id is stable across server restarts.
        Some users store MCPs on the config.yaml and permission management is based on server_ids.

        Eg a key might have mcp_servers = ["1234"], if the server_id changes across restarts, the key will no longer have access to the MCP.

        Args:
            server_name: Name of the server
            url: Server URL
            transport: Transport type (sse, http, etc.)
            auth_type: Authentication type (optional)
            alias: Server alias (optional)

        Returns:
            A deterministic server ID string
        """
        # Create a string from all the identifying parameters
        params_string = f"{server_name}|{url}|{transport}|{auth_type or ''}|{alias or ''}"

        # Generate SHA-256 hash
        hash_object = hashlib.sha256(params_string.encode("utf-8"))
        hash_hex = hash_object.hexdigest()

        # Take first 32 characters and format as UUID-like string
        return hash_hex[:32]

    async def health_check_server(
        self, server_id: str, mcp_auth_header: Optional[str] = None
    ) -> LiteLLM_MCPServerTable:
        """
        Perform a health check on a specific MCP server.

        Args:
            server_id: The ID of the server to health check
            mcp_auth_header: Optional authentication header for the MCP server

        Returns:
            Dict containing health check results
        """
        from datetime import datetime

        server = self.get_mcp_server_by_id(server_id)
        if not server:
            verbose_logger.warning(f"MCP Server {server_id} not found")
            return LiteLLM_MCPServerTable(
                server_id=server_id,
                server_name=None,
                transport=MCPTransport.http,  # Default transport for not found servers
                status="unknown",
                health_check_error="Server not found",
                last_health_check=datetime.now(),
            )

        status: Literal["healthy", "unhealthy", "unknown"] = "unknown"
        health_check_error = None

        # Check if we should skip health check based on auth configuration
        should_skip_health_check = False

        # Skip if server requires per-user authentication (OAuth2 or passthrough auth)
        if server.requires_per_user_auth:
            should_skip_health_check = True
        # Skip if auth_type is not none and authentication_token is missing
        # (except aws_sigv4 which uses its own credential fields)
        elif (
            server.auth_type
            and server.auth_type != MCPAuth.none
            and server.auth_type != MCPAuth.aws_sigv4
            and not server.authentication_token
        ):
            should_skip_health_check = True
        # Skip if static_headers reference a per-user env var: a userless probe
        # can't fill ${NAME} and would forward the literal placeholder upstream,
        # flipping the server to unhealthy even though real user calls succeed.
        elif self._references_per_user_env_var(server):
            should_skip_health_check = True

        if not should_skip_health_check:
            resolved_static_headers = await self._resolve_static_headers_with_env_vars(
                server=server,
                user_api_key_auth=None,
                raise_on_missing=False,
            )
            extra_headers = dict(resolved_static_headers) if resolved_static_headers else {}

            client = await self._create_mcp_client(
                server=server,
                mcp_auth_header=None,
                extra_headers=extra_headers,
                stdio_env=None,
            )

            try:

                async def _noop(session):
                    return "ok"

                # Add timeout wrapper to prevent hanging
                await asyncio.wait_for(client.run_with_session(_noop), timeout=MCP_HEALTH_CHECK_TIMEOUT)
                self._remember_upstream_initialize_instructions(server, client)
                status = "healthy"
            except asyncio.TimeoutError:
                health_check_error = f"Health check timed out after {MCP_HEALTH_CHECK_TIMEOUT} seconds"
                status = "unhealthy"
            except asyncio.CancelledError:
                health_check_error = "Health check was cancelled"
                status = "unknown"
            except Exception as e:
                health_check_error = str(e)
                status = "unhealthy"

        return LiteLLM_MCPServerTable(
            server_id=server.server_id,
            server_name=server.server_name,
            alias=server.alias,
            description=(server.mcp_info.get("description") if server.mcp_info else None),
            url=server.url,
            transport=server.transport,
            auth_type=server.auth_type,
            created_at=server.created_at,
            updated_at=server.updated_at,
            teams=[],
            mcp_access_groups=server.access_groups or [],
            allowed_tools=server.allowed_tools or [],
            extra_headers=server.extra_headers or [],
            mcp_info=server.mcp_info,
            static_headers=server.static_headers,
            env_vars=self._env_vars_to_models(server.env_vars),
            status=status,
            last_health_check=datetime.now(),
            health_check_error=health_check_error,
            command=getattr(server, "command", None),
            args=getattr(server, "args", None) or [],
            env=getattr(server, "env", None) or {},
            issuer=server.issuer,
            authorization_url=server.authorization_url,
            token_url=server.token_url,
            registration_url=server.registration_url,
            oauth2_flow=server.oauth2_flow,
            dcr_bridge=server.dcr_bridge,
            token_exchange_endpoint=server.token_exchange_endpoint,
            audience=server.audience,
            subject_token_type=server.subject_token_type,
            token_exchange_profile=server.token_exchange_profile,
            allow_all_keys=server.allow_all_keys,
            instructions=server.instructions,
            timeout=server.timeout,
            max_concurrent_requests=server.max_concurrent_requests,
        )

    async def get_all_mcp_servers_with_health_and_teams(
        self,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        server_ids: Optional[list[str]] = None,
    ) -> list[LiteLLM_MCPServerTable]:
        """
        Get all MCP servers that the user has access to, with health status and team information.

        Args:
            user_api_key_auth: User authentication info for access control
            server_ids: Optional list of server IDs to filter. If provided, only these servers
                       will be checked (subject to access control). If None, all accessible servers are checked.

        Returns:
            List of MCP server objects with health and team data
        """

        # Get allowed server IDs
        allowed_server_ids = await self.get_allowed_mcp_servers(user_api_key_auth)

        # Filter by requested server_ids if provided
        if server_ids:
            # Only check servers that are both requested AND accessible
            target_server_ids = [sid for sid in server_ids if sid in allowed_server_ids]
        else:
            # Check all accessible servers
            target_server_ids = allowed_server_ids

        return await self._run_health_checks(target_server_ids)

    async def get_all_allowed_mcp_servers(
        self,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> list[LiteLLM_MCPServerTable]:
        """
        Get all MCP servers that the user has access to.

        Args:
            user_api_key_auth: User authentication info for access control

        Returns:
            List of MCP server objects without health status
        """
        # Get allowed server IDs
        allowed_server_ids = await self.get_allowed_mcp_servers(user_api_key_auth)

        list_mcp_servers: list[LiteLLM_MCPServerTable] = []

        for server_id in allowed_server_ids:
            server = self.get_mcp_server_by_id(server_id)
            if not server:
                verbose_logger.warning(f"MCP Server {server_id} not found in registry")
                continue

            mcp_server_table = self._build_mcp_server_table(server)
            list_mcp_servers.append(mcp_server_table)

        return list_mcp_servers

    @staticmethod
    def _env_vars_to_models(
        env_vars: Optional[list[dict[str, Any]]],
    ) -> Optional[list[MCPEnvVar]]:
        if env_vars is None:
            return None
        return [MCPEnvVar.model_validate(env_var) for env_var in env_vars]

    def _build_mcp_server_table(self, server: MCPServer) -> LiteLLM_MCPServerTable:
        return LiteLLM_MCPServerTable(
            server_id=server.server_id,
            server_name=server.server_name,
            alias=server.alias,
            description=(server.mcp_info.get("description") if server.mcp_info else None),
            url=server.url,
            spec_path=server.spec_path,
            transport=server.transport,
            auth_type=server.auth_type,
            created_at=server.created_at,
            updated_at=server.updated_at,
            teams=[],
            mcp_access_groups=server.access_groups or [],
            allowed_tools=server.allowed_tools or [],
            tool_name_to_display_name=server.tool_name_to_display_name,
            tool_name_to_description=server.tool_name_to_description,
            extra_headers=server.extra_headers or [],
            mcp_info=server.mcp_info,
            static_headers=server.static_headers,
            env_vars=self._env_vars_to_models(server.env_vars),
            status=None,  # No health check performed
            last_health_check=None,  # No health check performed
            health_check_error=None,
            command=getattr(server, "command", None),
            args=getattr(server, "args", None) or [],
            env=getattr(server, "env", None) or {},
            issuer=server.issuer,
            authorization_url=server.authorization_url,
            token_url=server.token_url,
            registration_url=server.registration_url,
            oauth2_flow=server.oauth2_flow,
            token_exchange_endpoint=server.token_exchange_endpoint,
            audience=server.audience,
            subject_token_type=server.subject_token_type,
            token_exchange_profile=server.token_exchange_profile,
            allow_all_keys=server.allow_all_keys,
            available_on_public_internet=server.available_on_public_internet,
            delegate_auth_to_upstream=server.delegate_auth_to_upstream,
            oauth_passthrough=getattr(server, "oauth_passthrough", False),
            dcr_bridge=server.dcr_bridge,
            is_byok=server.is_byok,
            byok_description=server.byok_description,
            byok_api_key_help_url=server.byok_api_key_help_url,
            source_url=server.source_url,
            instructions=server.instructions,
            timeout=server.timeout,
            max_concurrent_requests=server.max_concurrent_requests,
        )

    async def get_all_mcp_servers_unfiltered(self) -> list[LiteLLM_MCPServerTable]:
        """Return all MCP servers from registry without applying access controls."""

        registry = self.get_registry()
        if not registry:
            return []

        servers: list[LiteLLM_MCPServerTable] = []
        for server in registry.values():
            servers.append(self._build_mcp_server_table(server))
        return servers

    async def get_all_mcp_servers_with_health_unfiltered(
        self, server_ids: Optional[list[str]] = None
    ) -> list[LiteLLM_MCPServerTable]:
        """Return health info for all servers in registry regardless of user access."""

        registry = self.get_registry()
        if not registry:
            return []

        if server_ids:
            target_server_ids = [sid for sid in server_ids if sid in registry]
        else:
            target_server_ids = list(registry.keys())

        if not target_server_ids:
            return []

        return await self._run_health_checks(target_server_ids)

    async def _run_health_checks(self, target_server_ids: list[str]) -> list[LiteLLM_MCPServerTable]:
        if not target_server_ids:
            return []

        tasks = [self.health_check_server(server_id) for server_id in target_server_ids]
        results = await asyncio.gather(*tasks)
        return [server for server in results if server is not None]


global_mcp_server_manager: MCPServerManager = MCPServerManager()
