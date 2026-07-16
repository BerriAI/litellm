"""
1. Allow proxy admin to perform create, update, and delete operations on MCP servers in the db.
2. Allows users to view the mcp servers they have access to.

Endpoints here:
- GET `/v1/mcp/server` - Returns all of the configured mcp servers in the db filtered by requestor's access
- GET `/v1/mcp/server/{server_id}` - Returns the the specific mcp server in the db given `server_id` filtered by requestor's access
- POST `/v1/mcp/server` - Add a new external mcp server.
- PUT `/v1/mcp/server` -  Edits an existing mcp server.
- DELETE `/v1/mcp/server/{server_id}` - Deletes the mcp server given `server_id`.
- GET `/v1/mcp/tools - lists all the tools available for a key
- GET `/v1/mcp/access_groups` - lists all available MCP access groups
- GET `/v1/mcp/discover` - Returns curated list of well-known MCP servers for discovery UI
- GET `/v1/mcp/openapi-registry` - Returns well-known OpenAPI APIs with OAuth 2.0 metadata

"""

import functools
import importlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Set

from fastapi import (
    APIRouter,
    Depends,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse

try:
    from prisma.errors import RecordNotFoundError, UniqueViolationError
except ImportError:
    RecordNotFoundError = Exception  # type: ignore
    UniqueViolationError = Exception  # type: ignore

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._experimental.mcp_server.utils import (
    build_env_var_setup_url,
    collect_env_var_references,
    LITELLM_MCP_SERVER_DESCRIPTION,
    LITELLM_MCP_SERVER_NAME,
    get_server_prefix,
    parse_admin_env_vars,
)
from litellm.proxy._experimental.mcp_server.utils import (
    validate_and_normalize_mcp_server_payload as _base_validate_and_normalize_mcp_server_payload,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.management_helpers.audit_logs import get_audit_log_changed_by
from litellm.repositories.table_repositories import (
    MCPServerRepository,
    MCPUserCredentialsRepository,
)

router = APIRouter(prefix="/v1/mcp", tags=["mcp"])

MCP_AVAILABLE: bool = True

TEMPORARY_MCP_SERVER_TTL_SECONDS = 300
TEMPORARY_MCP_SERVER_REDIS_KEY_PREFIX = "litellm:mcp:temporary_server"


def does_mcp_server_exist(mcp_server_records: Iterable[Any], mcp_server_id: str) -> bool:
    """
    Check if the mcp server with the given id exists in the iterable of mcp servers.

    Defined at module level (outside ``if MCP_AVAILABLE``) so it can be imported
    on Python < 3.10 where the ``mcp`` package is unavailable.
    """
    for mcp_server_record in mcp_server_records:
        if mcp_server_record.server_id == mcp_server_id:
            return True
    return False


DEFAULT_MCP_REGISTRY_VERSION = "1.0.0"

try:
    importlib.import_module("mcp")
except ImportError as e:
    verbose_logger.debug(f"MCP module not found: {e}")
    MCP_AVAILABLE = False

if MCP_AVAILABLE:
    try:
        from mcp.shared.tool_name_validation import (
            validate_tool_name,  # pyright: ignore[reportAssignmentType]
        )
    except ImportError:
        from pydantic import BaseModel

        class _ToolNameValidationResult(BaseModel):
            is_valid: bool = True
            warnings: list = []

        def validate_tool_name(name: str) -> _ToolNameValidationResult:  # type: ignore[misc]
            return _ToolNameValidationResult()

    from litellm.proxy._experimental.mcp_server.db import (
        approve_mcp_server,
        create_mcp_server,
        delete_mcp_server,
        delete_user_credential,
        delete_user_env_vars,
        get_all_mcp_servers_for_user,
        get_mcp_server,
        get_mcp_servers,
        get_mcp_submissions,
        get_user_env_vars,
        get_user_env_vars_bulk,
        get_user_oauth_credential,
        list_user_oauth_credentials,
        mcp_oauth_token_identity,
        merge_user_env_vars,
        purge_user_oauth_credentials_for_server,
        reject_mcp_server,
        store_user_credential,
        store_user_oauth_credential,
        update_mcp_server,
    )
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _raise_if_not_oauth2,
        authorize_with_server,
        exchange_token_with_server,
        get_request_base_url,
        register_client_with_server,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._experimental.mcp_server.ui_session_utils import (
        build_effective_auth_contexts,
    )
    from litellm.proxy._types import (
        LiteLLM_MCPServerTable,
        LitellmUserRoles,
        MakeMCPServersPublicRequest,
        MCPApprovalStatus,
        MCPOAuthUserCredentialRequest,
        MCPOAuthUserCredentialStatus,
        MCPSubmissionsSummary,
        MCPTransport,
        MCPUserCredentialListItem,
        MCPUserCredentialRequest,
        MCPUserCredentialResponse,
        MCPUserEnvVarSpec,
        MCPUserEnvVarsRequest,
        MCPUserEnvVarsStatus,
        NewMCPServerRequest,
        RejectMCPServerRequest,
        SpecialMCPServerName,
        UpdateMCPServerRequest,
        UserAPIKeyAuth,
        UserMCPManagementMode,
    )
    from litellm.proxy.auth.user_api_key_auth import (
        _user_api_key_auth_builder,
        user_api_key_auth,
    )
    from litellm.proxy.common_utils.http_parsing_utils import (
        _read_request_body,
        populate_request_with_path_params,
    )
    from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view
    from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
    from litellm.types.mcp import MCPAuth, MCPCredentials
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    @dataclass
    class _TemporaryMCPServerEntry:
        server: MCPServer
        expires_at: datetime

    def _validate_mcp_server_name_fields(payload: Any) -> None:
        candidates: List[tuple[str, Optional[str]]] = []

        server_name = getattr(payload, "server_name", None)
        alias = getattr(payload, "alias", None)

        if server_name:
            candidates.append(("server_name", server_name))
        if alias:
            candidates.append(("alias", alias))

        for field_name, value in candidates:
            if not value:
                continue

            validation_result = validate_tool_name(value)
            if validation_result.is_valid:
                continue

            error_messages_text = f"Invalid MCP tool prefix '{value}' provided via {field_name}"
            if validation_result.warnings:
                error_messages_text = error_messages_text + "\n" + "\n".join(validation_result.warnings)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": error_messages_text},
            )

    def validate_and_normalize_mcp_server_payload(payload: Any) -> None:
        _base_validate_and_normalize_mcp_server_payload(payload)
        _validate_mcp_server_name_fields(payload)

    def stamp_omitted_oauth2_flow(payload: NewMCPServerRequest) -> None:
        """Fallback only: fill in oauth2_flow when an oauth2 create omits it.

        An explicit oauth2_flow from the caller (the dashboard's flow selector, a REST
        body, config.yaml) always wins and is never touched. The shape check below runs
        solely for oauth2 creates that leave the field unset, so those rows still
        persist a flow instead of relying on read-time inference.

        The create payload carries the plaintext credentials, so the M2M-vs-interactive
        decision is reliable here in a way it is not at read time (credentials are
        encrypted at rest and redacted in responses). The client_credentials shape
        mirrors the legacy inference in MCPServerManager._resolve_oauth2_flow; every
        other oauth2 configuration is the authorization_code grant, including
        delegate_auth_to_upstream, where the client runs that grant upstream.
        """
        if payload.auth_type != MCPAuth.oauth2:
            return
        if payload.oauth2_flow:
            return
        credentials = payload.credentials or {}
        has_m2m_shape = bool(
            payload.token_url
            and credentials.get("client_id")
            and credentials.get("client_secret")
            and not payload.authorization_url
        )
        payload.oauth2_flow = "client_credentials" if has_m2m_shape else "authorization_code"

    _VALID_MCP_REQUIRED_FIELDS: frozenset = frozenset(NewMCPServerRequest.model_fields)

    def _validate_mcp_required_fields(payload: Any) -> None:
        """Validate submission payload against admin-configured mcp_required_fields."""
        from litellm.proxy.proxy_server import (
            general_settings as proxy_general_settings,
        )

        required_fields: Optional[List[str]] = proxy_general_settings.get("mcp_required_fields")
        if not required_fields:
            return

        # Fail fast on unknown field names — a typo in the config would silently
        # block every submission with a confusing "missing fields" error.
        unknown = [f for f in required_fields if f not in _VALID_MCP_REQUIRED_FIELDS]
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": f"mcp_required_fields contains unknown field names: {unknown}. "
                    "Check general_settings.mcp_required_fields in your proxy config."
                },
            )

        # Mirror the UI's compliance checks (MCPStandardsSettings.tsx FIELD_GROUPS):
        # auth_type requires a real value — "none" is treated as absent.
        _AUTH_TYPE_SENTINEL = "none"

        def _field_present(field_name: str) -> bool:
            value = getattr(payload, field_name, None)
            if value is None:
                return False
            # Treat empty string and empty list as absent (mirrors UI compliance check)
            if isinstance(value, (str, list)) and not value:
                return False
            if field_name == "auth_type" and value == _AUTH_TYPE_SENTINEL:
                return False
            return True

        missing = [f for f in required_fields if not _field_present(f)]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": f"Submission is missing required fields: {missing}. "
                    "Configure required fields via general_settings.mcp_required_fields."
                },
            )

    def _is_public_registry_enabled() -> bool:
        from litellm.proxy.proxy_server import (
            general_settings as proxy_general_settings,
        )

        return bool(proxy_general_settings.get("enable_mcp_registry"))

    def _build_registry_remote_url(base_url: str, path: str) -> str:
        normalized_base = base_url.rstrip("/")
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"{normalized_base}{normalized_path}"

    def _build_mcp_registry_server_name(server: MCPServer) -> str:
        if server.alias:
            return server.alias
        if server.server_name:
            return server.server_name
        return server.server_id

    def _build_mcp_registry_entry_for_server(server: MCPServer, base_url: str) -> Dict[str, Any]:
        server_name = _build_mcp_registry_server_name(server)
        title = server_name
        description = server_name
        version = DEFAULT_MCP_REGISTRY_VERSION

        server_prefix = get_server_prefix(server)
        if not server_prefix:
            raise ValueError("MCP server prefix is missing")
        remote_url = _build_registry_remote_url(base_url, f"/{server_prefix}/mcp")

        return {
            "name": server_name,
            "title": title,
            "description": description,
            "version": version,
            "remotes": [
                {
                    "type": "streamable-http",
                    "url": remote_url,
                }
            ],
        }

    def _build_builtin_registry_entry(base_url: str) -> Dict[str, Any]:
        remote_url = _build_registry_remote_url(base_url, "/mcp")
        return {
            "name": LITELLM_MCP_SERVER_NAME,
            "title": LITELLM_MCP_SERVER_NAME,
            "description": LITELLM_MCP_SERVER_DESCRIPTION,
            "version": DEFAULT_MCP_REGISTRY_VERSION,
            "remotes": [
                {
                    "type": "streamable-http",
                    "url": remote_url,
                }
            ],
        }

    _temporary_mcp_servers: Dict[str, _TemporaryMCPServerEntry] = {}

    def _prune_expired_temporary_mcp_servers() -> None:
        if not _temporary_mcp_servers:
            return

        now = datetime.utcnow()
        expired_ids = [server_id for server_id, entry in _temporary_mcp_servers.items() if entry.expires_at <= now]
        for server_id in expired_ids:
            _temporary_mcp_servers.pop(server_id, None)

    def _cache_temporary_mcp_server(server: MCPServer, ttl_seconds: int) -> MCPServer:
        ttl_seconds = max(1, ttl_seconds)
        _prune_expired_temporary_mcp_servers()
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        _temporary_mcp_servers[server.server_id] = _TemporaryMCPServerEntry(
            server=server,
            expires_at=expires_at,
        )
        return server

    async def _cache_temporary_mcp_server_in_redis(server: MCPServer, ttl_seconds: int) -> None:
        """
        Best-effort write-through to Redis so temporary MCP OAuth sessions are
        shared across proxy instances. Keep local in-memory cache as fallback.
        """
        if litellm.cache is None or not hasattr(litellm.cache, "cache"):
            return
        cache_backend = getattr(litellm.cache, "cache", None)
        if cache_backend is None or not hasattr(cache_backend, "async_set_cache"):
            return

        payload: Dict[str, Any] = server.model_dump(mode="json")
        payload_json = json.dumps(payload)
        try:
            encrypted_payload = encrypt_value_helper(payload_json)
        except Exception as e:
            verbose_proxy_logger.debug(f"Failed to encrypt temporary MCP server payload for Redis cache: {str(e)}")
            return

        if not isinstance(encrypted_payload, str):
            verbose_proxy_logger.debug("Encrypted temporary MCP payload is not a string; skipping Redis cache write")
            return

        try:
            await cache_backend.async_set_cache(
                key=f"{TEMPORARY_MCP_SERVER_REDIS_KEY_PREFIX}:{server.server_id}",
                value=encrypted_payload,
                ttl=max(1, ttl_seconds),
            )
        except Exception as e:
            verbose_proxy_logger.debug(f"Failed to write temporary MCP server to Redis cache: {str(e)}")

    async def _get_temporary_mcp_server_from_redis(
        server_id: str,
    ) -> Optional[MCPServer]:
        """
        Best-effort read from Redis shared cache. Returns None on miss/errors.

        Values must be encrypted strings (same contract as _cache_temporary_mcp_server_in_redis);
        legacy plaintext dict payloads are rejected.
        """
        if litellm.cache is None or not hasattr(litellm.cache, "cache"):
            return None
        cache_backend = getattr(litellm.cache, "cache", None)
        if cache_backend is None or not hasattr(cache_backend, "async_get_cache"):
            return None

        try:
            cached_server = await cache_backend.async_get_cache(
                key=f"{TEMPORARY_MCP_SERVER_REDIS_KEY_PREFIX}:{server_id}"
            )
        except Exception as e:
            verbose_proxy_logger.debug(f"Failed reading temporary MCP server from Redis cache: {str(e)}")
            return None

        if not isinstance(cached_server, str):
            verbose_proxy_logger.debug(
                "Temporary MCP Redis cache value must be an encrypted string; rejecting non-string payload"
            )
            return None

        decrypted_json = decrypt_value_helper(
            value=cached_server,
            key="temporary_mcp_server",
            exception_type="debug",
        )
        if decrypted_json is None:
            return None
        try:
            loaded = json.loads(decrypted_json)
        except Exception as e:
            verbose_proxy_logger.debug(f"Invalid decrypted temporary MCP payload in Redis cache: {str(e)}")
            return None
        if not isinstance(loaded, dict):
            return None
        payload_dict: Dict[str, Any] = loaded

        try:
            return MCPServer(**payload_dict)
        except Exception as e:
            verbose_proxy_logger.debug(f"Invalid temporary MCP server payload in Redis cache: {str(e)}")
            return None

    async def get_cached_temporary_mcp_server(
        server_id: str,
    ) -> Optional[MCPServer]:
        _prune_expired_temporary_mcp_servers()
        entry = _temporary_mcp_servers.get(server_id)
        if entry is None:
            redis_server = await _get_temporary_mcp_server_from_redis(server_id)
            if redis_server is None:
                return None
            # Intentionally avoid repopulating local cache from Redis to prevent
            # extending effective lifetime beyond the remaining Redis TTL.
            return redis_server
        return entry.server

    def _redact_mcp_credentials(
        mcp_server: LiteLLM_MCPServerTable,
    ) -> LiteLLM_MCPServerTable:
        """Return a copy of the MCP server object with credentials removed."""

        try:
            redacted_server = mcp_server.model_copy(deep=True)
        except AttributeError:
            redacted_server = mcp_server.copy(deep=True)  # type: ignore[attr-defined]

        if hasattr(redacted_server, "credentials"):
            setattr(redacted_server, "credentials", None)

        return redacted_server

    def _redact_mcp_credentials_list(
        mcp_servers: Iterable[LiteLLM_MCPServerTable],
    ) -> List[LiteLLM_MCPServerTable]:
        return [_redact_mcp_credentials(server) for server in mcp_servers]

    def _user_is_full_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
        """True only for ``PROXY_ADMIN``; ``PROXY_ADMIN_VIEW_ONLY`` returns False.

        Global env var secrets pre-fill the admin edit form, so a full admin
        must see them, but a read-only admin gets the same redacted view as
        any other non-managing caller.
        """
        return user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN

    def _is_restricted_virtual_key_request(user_api_key_dict: UserAPIKeyAuth) -> bool:
        """Best-effort detection for route-restricted virtual keys.

        We treat a requestor as a "restricted" virtual key if `allowed_routes`
        is a non-empty list. This matches the auth gate that blocks routes with
        the error: "Virtual key is not allowed to call this route...".
        """

        allowed_routes = getattr(user_api_key_dict, "allowed_routes", None)
        return isinstance(allowed_routes, list) and len(allowed_routes) > 0

    def _sanitize_mcp_server_for_non_admin(
        mcp_server: LiteLLM_MCPServerTable,
    ) -> LiteLLM_MCPServerTable:
        """Strip credential-bearing fields for non-admin viewers.

        Non-admin users may legitimately need to discover MCP servers
        their team has access to (so they can pick one in the UI), but
        they must never see fields that can carry bearer tokens or
        upstream API keys. ``_redact_mcp_credentials`` already clears
        the explicit ``credentials`` field; this layers on top to catch
        the URL+headers+env vectors that the virtual-key sanitizer also
        strips. Reset values match each field's declared default on
        ``LiteLLM_MCPServerTable`` (``None`` for Optional fields,
        ``[]``/``{}`` for required list/dict fields).
        """
        sanitized = _redact_mcp_credentials(mcp_server)
        # URL is the highest-impact vector: many MCP integrations embed
        # the upstream API key directly in the path. spec_path can carry
        # similar tokens in the OpenAPI spec URL.
        sanitized.url = None
        sanitized.spec_path = None
        sanitized.static_headers = None
        sanitized.extra_headers = []
        sanitized.allowed_response_headers = []
        sanitized.env = {}
        sanitized.command = None
        sanitized.args = []
        sanitized.issuer = None
        sanitized.authorization_url = None
        sanitized.token_url = None
        sanitized.registration_url = None
        sanitized.token_exchange_endpoint = None
        sanitized.audience = None
        sanitized.subject_token_type = None
        sanitized.token_exchange_profile = None
        # Drop env vars entirely rather than only blanking global values: the
        # names alone (DB_PASSWORD, GITHUB_API_KEY, ...) leak what secrets the
        # admin configured. Non-admins get the per-user vars they must fill in
        # from the dedicated /user-env-vars/status endpoint instead.
        sanitized.env_vars = None
        return sanitized

    def _sanitize_mcp_server_list_for_non_admin(
        mcp_servers: Iterable[LiteLLM_MCPServerTable],
    ) -> List[LiteLLM_MCPServerTable]:
        return [_sanitize_mcp_server_for_non_admin(s) for s in mcp_servers]

    def _sanitize_mcp_server_for_virtual_key(
        mcp_server: LiteLLM_MCPServerTable,
    ) -> LiteLLM_MCPServerTable:
        """Return a minimally sufficient MCP server view for virtual keys.

        Security model:
        - Virtual keys should be able to *discover* accessible servers.
        - They should NOT receive sensitive configuration details like upstream
          URLs, env vars, headers, commands/args, access-group names, or
          credentials.
        """

        sanitized = _redact_mcp_credentials(mcp_server)

        # Remove potentially sensitive config + identity fields.
        sanitized.url = None
        sanitized.static_headers = None
        sanitized.env = {}
        sanitized.command = None
        sanitized.args = []
        sanitized.extra_headers = []
        sanitized.allowed_response_headers = []
        sanitized.allowed_tools = []
        sanitized.mcp_access_groups = []
        sanitized.teams = []
        sanitized.env_vars = None

        sanitized.issuer = None
        sanitized.authorization_url = None
        sanitized.token_url = None
        sanitized.registration_url = None
        sanitized.token_exchange_endpoint = None
        sanitized.audience = None
        sanitized.subject_token_type = None
        sanitized.token_exchange_profile = None

        sanitized.health_check_error = None
        sanitized.last_health_check = None

        sanitized.created_by = None
        sanitized.updated_by = None
        sanitized.created_at = None
        sanitized.updated_at = None

        # `mcp_info` is arbitrary metadata; keep only an explicit safe subset.
        is_public = False
        if isinstance(sanitized.mcp_info, dict):
            is_public = bool(sanitized.mcp_info.get("is_public"))
        sanitized.mcp_info = {"is_public": True} if is_public else None

        return sanitized

    def _sanitize_mcp_server_list_for_virtual_key(
        mcp_servers: Iterable[LiteLLM_MCPServerTable],
    ) -> List[LiteLLM_MCPServerTable]:
        return [_sanitize_mcp_server_for_virtual_key(server) for server in mcp_servers]

    def _inherit_credentials_from_existing_server(
        payload: NewMCPServerRequest,
    ) -> NewMCPServerRequest:
        if not payload.server_id or payload.credentials:
            return payload

        existing_server = global_mcp_server_manager.get_mcp_server_by_id(payload.server_id)
        if existing_server is None:
            return payload

        inherited_credentials: MCPCredentials = {}
        if existing_server.authentication_token:
            inherited_credentials["auth_value"] = existing_server.authentication_token
        if existing_server.client_id:
            inherited_credentials["client_id"] = existing_server.client_id
        if existing_server.client_secret:
            inherited_credentials["client_secret"] = existing_server.client_secret
        if existing_server.scopes:
            inherited_credentials["scopes"] = existing_server.scopes
        # AWS SigV4 fields
        if existing_server.aws_access_key_id:
            inherited_credentials["aws_access_key_id"] = existing_server.aws_access_key_id
        if existing_server.aws_secret_access_key:
            inherited_credentials["aws_secret_access_key"] = existing_server.aws_secret_access_key
        if existing_server.aws_session_token:
            inherited_credentials["aws_session_token"] = existing_server.aws_session_token
        if existing_server.aws_region_name:
            inherited_credentials["aws_region_name"] = existing_server.aws_region_name
        if existing_server.aws_service_name:
            inherited_credentials["aws_service_name"] = existing_server.aws_service_name

        if not inherited_credentials:
            return payload

        try:
            return payload.model_copy(update={"credentials": inherited_credentials})
        except AttributeError:
            pass

        payload_dict: Dict[str, Any]
        try:
            payload_dict = payload.model_dump()  # type: ignore[attr-defined]
        except AttributeError:
            payload_dict = payload.dict()  # type: ignore[attr-defined]
        payload_dict["credentials"] = inherited_credentials
        return NewMCPServerRequest(**payload_dict)

    def _build_temporary_mcp_server_record(
        payload: NewMCPServerRequest,
        created_by: Optional[str],
    ) -> LiteLLM_MCPServerTable:
        now = datetime.utcnow()
        server_id = payload.server_id or str(uuid.uuid4())
        server_name = payload.server_name or payload.alias or server_id
        return LiteLLM_MCPServerTable(
            server_id=server_id,
            server_name=server_name,
            alias=payload.alias,
            description=payload.description,
            url=payload.url,
            transport=payload.transport,
            auth_type=payload.auth_type,
            credentials=payload.credentials,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            updated_by=created_by,
            teams=[],
            mcp_access_groups=payload.mcp_access_groups,
            allowed_tools=payload.allowed_tools or [],
            extra_headers=payload.extra_headers or [],
            allowed_response_headers=payload.allowed_response_headers or [],
            mcp_info=payload.mcp_info,
            static_headers=payload.static_headers,
            command=payload.command,
            args=payload.args,
            env=payload.env,
            issuer=payload.issuer,
            authorization_url=payload.authorization_url,
            token_url=payload.token_url,
            registration_url=payload.registration_url,
            allow_all_keys=payload.allow_all_keys,
            available_on_public_internet=payload.available_on_public_internet,
            timeout=payload.timeout,
            max_concurrent_requests=payload.max_concurrent_requests,
        )

    def get_prisma_client_or_throw(message: str):
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": message},
            )
        return prisma_client

    # Router to fetch all MCP tools available for the current key

    @router.get(
        "/tools",
        tags=["mcp"],
        dependencies=[Depends(user_api_key_auth)],
    )
    async def get_mcp_tools(
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Get all MCP tools available for the current key, including those from access groups
        """
        from litellm.proxy._experimental.mcp_server.server import _list_mcp_tools

        tools = await _list_mcp_tools(
            user_api_key_auth=user_api_key_dict,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )
        dumped_tools = [dict(tool) for tool in tools]

        return {"tools": dumped_tools}

    @router.get(
        "/access_groups",
        tags=["mcp"],
        dependencies=[Depends(user_api_key_auth)],
    )
    async def get_mcp_access_groups(
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Get all available MCP access groups from the database AND config
        """
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy.proxy_server import prisma_client

        access_groups = set()

        # Get from config-loaded servers
        for server in global_mcp_server_manager.config_mcp_servers.values():
            if server.access_groups:
                access_groups.update(server.access_groups)

        # Get from DB
        if prisma_client is not None:
            try:
                mcp_servers = await MCPServerRepository(prisma_client).table.find_many()
                for server in mcp_servers:
                    if hasattr(server, "mcp_access_groups") and server.mcp_access_groups:
                        access_groups.update(server.mcp_access_groups)
            except Exception as e:
                verbose_proxy_logger.debug(f"Error getting MCP access groups: {e}")

        # Convert to sorted list
        access_groups_list = sorted(list(access_groups))
        return {"access_groups": access_groups_list}

    @router.get(
        "/network/client-ip",
        tags=["mcp"],
        dependencies=[Depends(user_api_key_auth)],
        description="Returns the caller's IP address as seen by the proxy.",
    )
    async def get_client_ip(request: Request):
        from litellm.proxy.auth.ip_address_utils import IPAddressUtils

        client_ip = IPAddressUtils.get_mcp_client_ip(request)
        return {"ip": client_ip}

    @router.get(
        "/registry.json",
        tags=["mcp"],
        description="MCP registry endpoint. Spec: https://github.com/modelcontextprotocol/registry",
    )
    async def get_mcp_registry(request: Request):
        if not _is_public_registry_enabled():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MCP registry is not enabled",
            )

        from litellm.proxy.auth.ip_address_utils import IPAddressUtils

        client_ip = IPAddressUtils.get_mcp_client_ip(request)

        verbose_proxy_logger.debug("MCP registry request from IP=%s", client_ip)

        base_url = get_request_base_url(request)
        registry_servers: List[Dict[str, Any]] = []
        registry_servers.append({"server": _build_builtin_registry_entry(base_url)})

        # Centralized IP-based filtering: external callers only see public servers
        registered_servers = list(global_mcp_server_manager.get_filtered_registry(client_ip).values())

        registered_servers.sort(key=_build_mcp_registry_server_name)

        for server in registered_servers:
            try:
                entry = _build_mcp_registry_entry_for_server(server, base_url)
            except Exception as e:
                verbose_proxy_logger.debug(
                    f"Skipping MCP server {getattr(server, 'server_id', 'unknown')} in registry: {e}"
                )
                continue
            registry_servers.append({"server": entry})

        return {"servers": registry_servers}

    ## FastAPI Routes
    def _get_user_mcp_management_mode() -> UserMCPManagementMode:
        from litellm.proxy.proxy_server import (
            general_settings as proxy_general_settings,
        )

        mode = proxy_general_settings.get("user_mcp_management_mode")
        if mode == "view_all":
            return "view_all"
        return "restricted"

    async def _get_team_scoped_mcp_server_list(
        team_id: str,
    ) -> List[LiteLLM_MCPServerTable]:
        """
        Return MCP servers scoped to a team: team's allowed servers + allow_all_keys servers.
        Used by the Create Key UI to populate the MCP server dropdown.
        """
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.management_helpers.object_permission_utils import (
            _get_allow_all_keys_server_ids,
            _get_team_allowed_mcp_servers,
        )
        from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

        team_obj = await get_team_object(
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            check_db_only=True,
        )

        team_server_ids = await _get_team_allowed_mcp_servers(team_obj)
        allow_all_server_ids = _get_allow_all_keys_server_ids()
        all_allowed_ids = team_server_ids | allow_all_server_ids

        if not all_allowed_ids:
            return []

        # Collect servers from registry
        servers: List[LiteLLM_MCPServerTable] = []
        for server_id in all_allowed_ids:
            server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
            if server is not None:
                mcp_server_table = global_mcp_server_manager._build_mcp_server_table(server)
                servers.append(mcp_server_table)

        return _redact_mcp_credentials_list(servers)

    async def _resolve_accessible_mcp_servers(
        user_api_key_dict: UserAPIKeyAuth,
    ) -> List[LiteLLM_MCPServerTable]:
        """The server set the dashboard grid shows (GET /v1/mcp/server, no team
        filter), returned unredacted. Callers that surface this to a client must
        apply their own redaction; the per-user env-var status endpoint relies on
        the raw env_vars and only ever returns is_set booleans, never secrets.

        Sharing this resolution keeps the red "missing user fields" card status
        aligned with the cards actually rendered: an admin in view_all mode sees
        every server even when their key carries no per-server MCP grant.
        """
        if _get_user_mcp_management_mode() == "view_all" and not _is_restricted_virtual_key_request(user_api_key_dict):
            return await global_mcp_server_manager.get_all_mcp_servers_unfiltered()

        aggregated: Dict[str, LiteLLM_MCPServerTable] = {}
        for auth_context in await build_effective_auth_contexts(user_api_key_dict):
            for server in await global_mcp_server_manager.get_all_allowed_mcp_servers(user_api_key_auth=auth_context):
                aggregated.setdefault(server.server_id, server)
        return list(aggregated.values())

    @router.get(
        "/server",
        description="Returns the mcp server list with associated teams",
        dependencies=[Depends(user_api_key_auth)],
        response_model=List[LiteLLM_MCPServerTable],
    )
    async def fetch_all_mcp_servers(
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        team_id: Optional[str] = Query(
            None,
            description="Filter MCP servers by team scope. When provided, returns only "
            "servers the team has access to plus globally available (allow_all_keys) servers. "
            "Used by the Create Key UI to show team-scoped MCP servers.",
        ),
    ):
        """
        Get all of the configured mcp servers for the user in the db with their associated teams
        ```
        curl --location 'http://localhost:4000/v1/mcp/server' \
        --header 'Authorization: Bearer your_api_key_here'

        # Filter by team scope (for Create Key UI)
        curl --location 'http://localhost:4000/v1/mcp/server?team_id=team-123' \
        --header 'Authorization: Bearer your_api_key_here'
        ```
        """

        # If team_id is provided, return team-scoped servers + allow_all_keys servers
        is_restricted_virtual_key = _is_restricted_virtual_key_request(user_api_key_dict)
        if team_id is not None and isinstance(team_id, str) and team_id.strip():
            # Restricted virtual keys must not use the team_id filter to
            # bypass their own access limitations.
            if is_restricted_virtual_key:
                raise HTTPException(
                    status_code=403,
                    detail="Restricted virtual keys cannot query team-scoped MCP servers.",
                )

            # Only proxy admins may query another team's MCP servers.
            # Non-admins must belong to the requested team.
            sanitized_team_id = team_id.strip()
            is_admin = _user_has_admin_view(user_api_key_dict)
            if not is_admin:
                from litellm.proxy.auth.auth_checks import get_team_object
                from litellm.proxy.proxy_server import (
                    prisma_client,
                    user_api_key_cache,
                )

                team_obj = await get_team_object(
                    team_id=sanitized_team_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    check_db_only=True,
                )
                user_in_team = any(
                    m.user_id is not None and m.user_id == user_api_key_dict.user_id
                    for m in team_obj.members_with_roles
                )
                if not user_in_team:
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have permission to view MCP servers for this team.",
                    )

            redacted_mcp_servers = await _get_team_scoped_mcp_server_list(sanitized_team_id)
        else:
            servers = await _resolve_accessible_mcp_servers(user_api_key_dict)
            redacted_mcp_servers = _redact_mcp_credentials_list(servers)

        # augment the mcp servers with public status
        if litellm.public_mcp_servers is not None:
            for server in redacted_mcp_servers:
                if server.server_id in litellm.public_mcp_servers:
                    if server.mcp_info is None:
                        server.mcp_info = {}
                    server.mcp_info["is_public"] = True

        # Annotate has_user_credential for BYOK servers (single batched query)
        from litellm.proxy.proxy_server import prisma_client as _byok_prisma_client

        user_id = user_api_key_dict.user_id or ""
        if user_id and _byok_prisma_client is not None:
            byok_server_ids = [s.server_id for s in redacted_mcp_servers if getattr(s, "is_byok", False)]
            if byok_server_ids:
                cred_rows = await MCPUserCredentialsRepository(_byok_prisma_client).table.find_many(
                    where={"user_id": user_id, "server_id": {"in": byok_server_ids}}
                )
                cred_set = {r.server_id for r in cred_rows}
                for server in redacted_mcp_servers:
                    if getattr(server, "is_byok", False):
                        server.has_user_credential = server.server_id in cred_set

        # Virtual keys only get a sanitized discovery view.
        if is_restricted_virtual_key:
            return _sanitize_mcp_server_list_for_virtual_key(redacted_mcp_servers)

        # only a full PROXY_ADMIN sees credential-bearing fields; everyone else
        # goes through the non-admin sanitizer
        if not _user_is_full_admin(user_api_key_dict):
            return _sanitize_mcp_server_list_for_non_admin(redacted_mcp_servers)

        return redacted_mcp_servers

    @router.get(
        "/server/health",
        description="Health check for MCP servers",
        dependencies=[Depends(user_api_key_auth)],
    )
    async def health_check_servers(
        server_ids: Optional[List[str]] = Query(
            None,
            description="Server IDs to check. If not provided, checks all accessible servers.",
        ),
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Perform health checks on one or more MCP servers.

        Parameters:
        - server_ids: Optional list of server IDs. If not provided, checks all accessible servers.

        Returns:
        - Health check results for requested servers

        ```
        # Check all accessible servers
        curl --location 'http://localhost:4000/v1/mcp/server/health' \
        --header 'Authorization: Bearer your_api_key_here'

        # Check specific servers
        curl --location 'http://localhost:4000/v1/mcp/server/health?server_ids=server-1&server_ids=server-2' \
        --header 'Authorization: Bearer your_api_key_here'
        ```
        """
        user_mcp_management_mode = _get_user_mcp_management_mode()

        if user_mcp_management_mode == "view_all":
            servers = await global_mcp_server_manager.get_all_mcp_servers_with_health_unfiltered(server_ids=server_ids)
            return [{"server_id": server.server_id, "status": server.status} for server in servers]

        auth_contexts = await build_effective_auth_contexts(user_api_key_dict)

        server_status_map: Dict[str, Optional[Literal["healthy", "unhealthy", "unknown"]]] = {}
        for auth_context in auth_contexts:
            servers = await global_mcp_server_manager.get_all_mcp_servers_with_health_and_teams(
                user_api_key_auth=auth_context,
                server_ids=server_ids,
            )
            for server in servers:
                if server.server_id not in server_status_map:
                    server_status_map[server.server_id] = server.status

        return [{"server_id": server_id, "status": status} for server_id, status in server_status_map.items()]

    @router.post(
        "/server/register",
        description="Submit a new MCP server for admin review (non-admin users). Mirrors POST /guardrails/register.",
        dependencies=[Depends(user_api_key_auth)],
        response_model=LiteLLM_MCPServerTable,
        status_code=status.HTTP_201_CREATED,
    )
    @management_endpoint_wrapper
    async def register_mcp_server(
        payload: NewMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Allow team members to submit an MCP server for admin review.
        Creates the server with approval_status=pending_review.
        Requires a team-scoped API key.
        """
        if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "PROXY_ADMIN users should use POST /v1/mcp/server to create servers directly instead of the submission workflow."
                },
            )

        if not user_api_key_dict.team_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Registration requires an API key associated with a team. Use a team-scoped key."},
            )

        # stdio servers spawn a local subprocess on the proxy host with the
        # configured command + args, so accepting them from non-admin callers
        # would let a team member propose a server config that an admin could
        # rubber-stamp into local code execution. Restrict stdio submission to
        # the admin POST /v1/mcp/server path or to config.yaml.
        if payload.transport == MCPTransport.stdio:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": (
                        "stdio MCP servers cannot be submitted via the user "
                        "registration workflow. Ask a proxy admin to add this "
                        "server via POST /v1/mcp/server or to declare it in "
                        "config.yaml."
                    )
                },
            )

        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")

        validate_and_normalize_mcp_server_payload(payload)
        stamp_omitted_oauth2_flow(payload)
        _validate_mcp_required_fields(payload)

        payload.approval_status = MCPApprovalStatus.pending_review
        payload.submitted_by = user_api_key_dict.user_id
        payload.submitted_at = datetime.now(timezone.utc)

        try:
            new_mcp_server = await create_mcp_server(
                prisma_client,
                payload,
                touched_by=user_api_key_dict.user_id or user_api_key_dict.team_id,
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error registering mcp server: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": f"Error registering mcp server: {str(e)}"},
            )
        # Do NOT add to runtime registry — pending servers are not active
        return _redact_mcp_credentials(new_mcp_server)

    @router.get(
        "/server/submissions",
        description="Returns all MCP servers submitted by non-admin users (admin review queue). Mirrors GET /guardrails/submissions.",
        dependencies=[Depends(user_api_key_auth)],
        response_model=MCPSubmissionsSummary,
    )
    @management_endpoint_wrapper
    async def get_mcp_server_submissions(
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Admin-only endpoint to view all user-submitted MCP servers pending review.
        """
        if user_api_key_dict.user_role not in (
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Admin access required to view MCP server submissions."},
            )

        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")

        submissions = await get_mcp_submissions(prisma_client)
        submissions.items = _redact_mcp_credentials_list(submissions.items)
        if not _user_is_full_admin(user_api_key_dict):
            submissions.items = _sanitize_mcp_server_list_for_non_admin(submissions.items)
        return submissions

    @router.put(
        "/server/{server_id}/approve",
        description="Approve a pending MCP server submission (admin only). Mirrors PUT /guardrails/{id}/approve.",
        dependencies=[Depends(user_api_key_auth)],
        response_model=LiteLLM_MCPServerTable,
    )
    @management_endpoint_wrapper
    async def approve_mcp_server_submission(
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Admin approves a pending or previously-rejected MCP server — sets approval_status=active and loads it into the runtime registry.
        """
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Admin access required to approve MCP server submissions."},
            )

        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")

        existing = await get_mcp_server(prisma_client, server_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"MCP server '{server_id}' not found."},
            )
        if existing.approval_status == MCPApprovalStatus.active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "MCP server is already active."},
            )

        approved = await approve_mcp_server(
            prisma_client,
            server_id,
            touched_by=user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
        )
        await global_mcp_server_manager.invalidate_byom_submitted_servers_cache(approved.submitted_by)
        await global_mcp_server_manager.reload_servers_from_database()

        return _redact_mcp_credentials(approved)

    @router.put(
        "/server/{server_id}/reject",
        description="Reject a pending MCP server submission (admin only). Mirrors PUT /guardrails/{id}/reject.",
        dependencies=[Depends(user_api_key_auth)],
        response_model=LiteLLM_MCPServerTable,
    )
    @management_endpoint_wrapper
    async def reject_mcp_server_submission(
        server_id: str,
        payload: RejectMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Admin rejects a pending MCP server — sets approval_status=rejected with optional review_notes.
        """
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Admin access required to reject MCP server submissions."},
            )

        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")

        existing = await get_mcp_server(prisma_client, server_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"MCP server '{server_id}' not found."},
            )
        if existing.approval_status == MCPApprovalStatus.rejected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "MCP server is already rejected."},
            )

        was_active = existing.approval_status == MCPApprovalStatus.active
        rejected = await reject_mcp_server(
            prisma_client,
            server_id,
            touched_by=user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
            review_notes=payload.review_notes,
        )
        # Only evict from the runtime registry if the server was previously active
        if was_active:
            await global_mcp_server_manager.reload_servers_from_database()
        return _redact_mcp_credentials(rejected)

    @router.get(
        "/server/{server_id}",
        description="Returns the mcp server info",
        dependencies=[Depends(user_api_key_auth)],
        response_model=LiteLLM_MCPServerTable,
    )
    async def fetch_mcp_server(
        request: Request,
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Get the info on the mcp server specified by the `server_id`
        Parameters:
        - server_id: str - Required. The unique identifier of the mcp server to get info on.
        ```
        curl --location 'http://localhost:4000/v1/mcp/server/server_id' \
        --header 'Authorization: Bearer your_api_key_here'
        ```
        """
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")

        # check to see if server exists (DB first, then registry for config-based servers)
        mcp_server = await get_mcp_server(prisma_client, server_id)
        from_db = mcp_server is not None

        if mcp_server is None:
            # Fallback: check registry (config-based servers) - list endpoint uses get_registry()
            from litellm.proxy.auth.ip_address_utils import IPAddressUtils

            client_ip = IPAddressUtils.get_mcp_client_ip(request)
            registry_server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
            if registry_server is not None and not global_mcp_server_manager._is_server_accessible_from_ip(
                registry_server, client_ip
            ):
                registry_server = None
            if registry_server is None:
                # Try lookup by server_name or alias (client may use display name in URL)
                registry_server = global_mcp_server_manager.get_mcp_server_by_name(server_id, client_ip=client_ip)
            if registry_server is not None:
                mcp_server = global_mcp_server_manager._build_mcp_server_table(registry_server)

        if mcp_server is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"MCP Server with id {server_id} not found"},
            )

        # Implement authz restriction from requested user
        is_admin_view = _user_has_admin_view(user_api_key_dict)
        is_restricted_virtual_key = _is_restricted_virtual_key_request(user_api_key_dict)

        if not is_admin_view:
            # Perform authz check BEFORE any health check (avoid side-effects for
            # unauthorized callers).
            if from_db:
                mcp_server_records = await get_all_mcp_servers_for_user(prisma_client, user_api_key_dict)
                exists = does_mcp_server_exist(mcp_server_records, server_id)
            else:
                # Registry/config server: use same access logic as list endpoint
                allowed_server_ids = await global_mcp_server_manager.get_allowed_mcp_servers(user_api_key_dict)
                exists = mcp_server.server_id in allowed_server_ids

            if not exists:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": (
                            f"User does not have permission to view mcp server with id {server_id}. "
                            "You can only view mcp servers that you have access to."
                        )
                    },
                )

        # At this point caller is authorized to view the server.
        if from_db:
            await global_mcp_server_manager.add_server(mcp_server)

        # Perform health check on the server using server manager
        try:
            health_result = await global_mcp_server_manager.health_check_server(server_id)
            # Update the server object with health check results
            mcp_server.status = health_result.status if health_result.status else "unknown"
            mcp_server.last_health_check = health_result.last_health_check
            mcp_server.health_check_error = health_result.health_check_error
        except Exception as e:
            verbose_proxy_logger.debug(f"Error performing health check on server {server_id}: {e}")
            mcp_server.status = "unknown"
            mcp_server.last_health_check = datetime.now()
            mcp_server.health_check_error = str(e)

        redacted = _redact_mcp_credentials(mcp_server)
        if is_restricted_virtual_key:
            return _sanitize_mcp_server_for_virtual_key(redacted)
        # only a full PROXY_ADMIN sees credential-bearing fields; everyone else
        # goes through the non-admin sanitizer
        if not _user_is_full_admin(user_api_key_dict):
            return _sanitize_mcp_server_for_non_admin(redacted)
        return redacted

    @router.post(
        "/server",
        description="Allows creation of mcp servers",
        dependencies=[Depends(user_api_key_auth)],
        response_model=LiteLLM_MCPServerTable,
        status_code=status.HTTP_201_CREATED,
    )
    @management_endpoint_wrapper
    async def add_mcp_server(
        payload: NewMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        litellm_changed_by: Optional[str] = Header(
            None,
            description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
        ),
    ):
        """
        Allow users to add a new external mcp server.
        """
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")

        # Validate and normalize payload fields
        validate_and_normalize_mcp_server_payload(payload)
        stamp_omitted_oauth2_flow(payload)

        # AuthZ - restrict only proxy admins to create mcp servers
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "User does not have permission to create mcp servers. You can only create mcp servers if you are a PROXY_ADMIN."
                },
            )

        # Block reserved special server IDs
        if (
            SpecialMCPServerName.all_team_servers == payload.server_id
            or SpecialMCPServerName.all_proxy_servers == payload.server_id
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"MCP Server with id {payload.server_id} is special and cannot be used."},
            )

        if payload.server_id is not None:
            # fail if the mcp server with id already exists
            mcp_server = await get_mcp_server(prisma_client, payload.server_id)
            if mcp_server is not None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": f"MCP Server with id {payload.server_id} already exists. Cannot create another."},
                )

        # TODO: audit log for create

        # Admin-created servers are always active — clear any submission lifecycle
        # fields the caller may have provided to prevent fake entries appearing in
        # the submissions queue.
        payload.approval_status = MCPApprovalStatus.active
        payload.submitted_by = None
        payload.submitted_at = None

        # The database write is the commit point: if it fails nothing was
        # persisted and the request is a genuine failure.
        try:
            new_mcp_server = await create_mcp_server(
                prisma_client,
                payload,
                touched_by=user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error creating mcp server: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": f"Error creating mcp server: {str(e)}"},
            )

        # Registry refresh is best-effort: the row is already committed, so a
        # failure here (e.g. an unrelated malformed row in the table) must not
        # surface as a 500 and orphan the created server, which would push the
        # caller to retry and create duplicates.
        try:
            await global_mcp_server_manager.add_server(new_mcp_server)
            await global_mcp_server_manager.reload_servers_from_database()
        except Exception as e:
            verbose_proxy_logger.exception(
                f"MCP server {new_mcp_server.server_id} created but in-memory registry refresh failed: {str(e)}"
            )

        return _redact_mcp_credentials(new_mcp_server)

    @router.post(
        "/server/oauth/session",
        description="Temporarily cache an MCP server in memory without writing to the database",
        dependencies=[Depends(user_api_key_auth)],
        status_code=status.HTTP_200_OK,
    )
    @management_endpoint_wrapper
    async def add_session_mcp_server(
        payload: NewMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        litellm_changed_by: Optional[str] = Header(
            None,
            description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
        ),
    ):
        """
        Cache MCP server info in memory for a short duration (~5 minutes).

        This endpoint does not write to the database. If the same server_id is provided
        again while the cache entry is active, it will refresh the cached data + TTL.
        """

        # Validate and normalize payload fields (alias/server name rules)
        validate_and_normalize_mcp_server_payload(payload)
        stamp_omitted_oauth2_flow(payload)

        # Restrict to proxy admins similar to the persistent create endpoint
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "User does not have permission to create temporary mcp servers. You can only create temporary mcp servers if you are a PROXY_ADMIN."
                },
            )

        created_by = user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME
        payload_with_credentials = _inherit_credentials_from_existing_server(payload)
        temp_record = _build_temporary_mcp_server_record(
            payload_with_credentials,
            created_by,
        )

        try:
            temporary_server = await global_mcp_server_manager.build_mcp_server_from_table(
                temp_record,
                credentials_are_encrypted=False,
                persist_discovered_endpoints=False,
            )
            _cache_temporary_mcp_server(
                temporary_server,
                ttl_seconds=TEMPORARY_MCP_SERVER_TTL_SECONDS,
            )
            await _cache_temporary_mcp_server_in_redis(
                temporary_server,
                ttl_seconds=TEMPORARY_MCP_SERVER_TTL_SECONDS,
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error caching temporary mcp server: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": f"Error caching temporary mcp server: {str(e)}"},
            )

        return _redact_mcp_credentials(temp_record)

    async def _mcp_oauth_user_api_key_auth(request: Request) -> UserAPIKeyAuth:
        """
        Auth dependency for MCP OAuth browser-navigation endpoints (/authorize, /token).

        Tries the Authorization header first. Falls back to decoding the UI
        'token' session cookie (set by SSO login) to extract the API key, which
        allows browser-based OAuth redirects to work without an explicit
        Authorization header.
        """
        import jwt as _jwt

        from litellm.proxy.proxy_server import master_key

        auth_header = request.headers.get("Authorization", "")
        api_key = auth_header  # _get_bearer_token will strip "Bearer " prefix

        if not api_key:
            token_cookie = request.cookies.get("token")
            if token_cookie and master_key:
                try:
                    decoded = _jwt.decode(
                        token_cookie,
                        master_key,
                        algorithms=["HS256"],
                        # UI session cookies may omit exp; don't require it.
                        options={"verify_exp": False, "verify_aud": False},
                    )
                    if decoded.get("login_method") in ("sso", "username_password"):
                        cookie_key = decoded.get("key", "")
                        if cookie_key:
                            api_key = f"Bearer {cookie_key}"
                except _jwt.InvalidTokenError:
                    pass

        # For delegate_auth_to_upstream servers the entire PKCE handshake
        # (both /authorize browser redirect and /token authorization_code
        # exchange) must work without a LiteLLM session.  /authorize is opened
        # in a VS Code webview that may have no cookie; /token is a programmatic
        # POST from VS Code. PKCE security (code_verifier) guarantees the
        # authorization_code exchange cannot be replayed, so anonymous access
        # is safe for that grant only.
        #
        # Importantly, NOT safe for refresh_token grants: ``mcp_token`` will
        # forward the request to the upstream issuer with LiteLLM's stored
        # ``client_secret`` attached, so any caller holding a refresh token
        # issued to this client could mint fresh upstream access tokens through
        # us. Require normal LiteLLM auth for those.
        if not api_key:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415
                global_mcp_server_manager,
            )
            from litellm.proxy.auth.auth_utils import (  # noqa: PLC0415
                get_request_route,
            )

            server_id = request.path_params.get("server_id", "")
            if server_id:
                _s = global_mcp_server_manager.get_mcp_server_by_id(server_id)
                if not _s:
                    _s = global_mcp_server_manager.get_mcp_server_by_name(server_id)
                if (
                    _s
                    and getattr(_s, "auth_type", None) == MCPAuth.oauth2
                    and getattr(_s, "delegate_auth_to_upstream", False) is True
                    # M2M servers fetch tokens with stored credentials; never
                    # expose their /authorize or /token endpoints anonymously.
                    and not _s.has_client_credentials
                ):
                    # For /token, require PKCE authorization_code; refresh_token
                    # grants must NOT bypass auth (see comment above).
                    path_lower = get_request_route(request).rstrip("/").lower()
                    if path_lower.endswith("/token"):
                        body_data = await _read_request_body(request=request)
                        grant_type = (body_data or {}).get("grant_type", "")
                        if grant_type != "authorization_code":
                            # Fall through to normal LiteLLM auth (will 401 if
                            # no key supplied).
                            pass
                        else:
                            return UserAPIKeyAuth()
                    else:
                        # /authorize and other PKCE-flow GETs are safe to
                        # bypass: PKCE binds the upstream issuer's ``code``
                        # to the original ``code_challenge`` so no anonymous
                        # token can be minted via the redirect alone.
                        return UserAPIKeyAuth()

        request_data = await _read_request_body(request=request)
        request_data = populate_request_with_path_params(request_data=request_data, request=request)

        return await _user_api_key_auth_builder(
            request=request,
            api_key=api_key,
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data=request_data,
        )

    async def _get_cached_temporary_mcp_server_or_404(
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth,
        request: Optional[Request] = None,
    ) -> MCPServer:
        server = await get_cached_temporary_mcp_server(server_id)
        resolved_from_temp_cache = server is not None
        if server is None:
            # Fall back to real DB/config server (e.g. for the user-side OAuth flow
            # which calls these endpoints with a real server_id, not a temp session id).
            from litellm.proxy.auth.ip_address_utils import IPAddressUtils

            client_ip = IPAddressUtils.get_mcp_client_ip(request) if request else None
            server = global_mcp_server_manager.get_mcp_server_by_id(
                server_id
            ) or global_mcp_server_manager.get_mcp_server_by_name(server_id, client_ip=client_ip)
        if server is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"MCP server {server_id} not found"},
            )

        # Per-server access policy mirrors `fetch_mcp_server`: admin-view
        # callers are unrestricted; non-admins must have the server in their
        # allowed-servers set. Temporary cached servers come from the
        # admin-only `/server/oauth/session` setup flow and are not exposed
        # to non-admins.
        if not _user_has_admin_view(user_api_key_dict):
            if resolved_from_temp_cache:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"error": f"Access denied to MCP server {server_id}"},
                )
            allowed_server_ids: Set[str] = set()
            for auth_context in await build_effective_auth_contexts(user_api_key_dict):
                allowed_server_ids.update(await global_mcp_server_manager.get_allowed_mcp_servers(auth_context))
            if server.server_id not in allowed_server_ids:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"error": f"Access denied to MCP server {server_id}"},
                )
        return server

    @router.get(
        "/server/oauth/{server_id}/authorize",
        include_in_schema=False,
        dependencies=[Depends(_mcp_oauth_user_api_key_auth)],
    )
    async def mcp_authorize(
        request: Request,
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(_mcp_oauth_user_api_key_auth),
        client_id: Optional[str] = None,
        redirect_uri: str = Query(...),
        state: str = "",
        code_challenge: Optional[str] = None,
        code_challenge_method: Optional[str] = None,
        response_type: Optional[str] = None,
        scope: Optional[str] = None,
    ):
        mcp_server = await _get_cached_temporary_mcp_server_or_404(server_id, user_api_key_dict, request=request)
        _raise_if_not_oauth2(mcp_server)
        # Use the server's stored client_id when the caller doesn't supply one
        resolved_client_id = mcp_server.client_id or client_id or ""
        if not resolved_client_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "missing_client_id",
                    "message": (
                        "No client_id available for this MCP server. "
                        "Either configure the server with a client_id or supply one in the request."
                    ),
                },
            )
        return await authorize_with_server(
            request=request,
            mcp_server=mcp_server,
            client_id=resolved_client_id,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            response_type=response_type,
            scope=scope,
        )

    @router.post(
        "/server/oauth/{server_id}/token",
        include_in_schema=False,
        dependencies=[Depends(_mcp_oauth_user_api_key_auth)],
    )
    async def mcp_token(
        request: Request,
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(_mcp_oauth_user_api_key_auth),
        grant_type: str = Form(...),
        code: Optional[str] = Form(None),
        redirect_uri: Optional[str] = Form(None),
        client_id: Optional[str] = Form(None),
        client_secret: Optional[str] = Form(None),
        code_verifier: Optional[str] = Form(None),
        refresh_token: Optional[str] = Form(None),
        scope: Optional[str] = Form(None),
    ):
        mcp_server = await _get_cached_temporary_mcp_server_or_404(server_id, user_api_key_dict, request=request)
        _raise_if_not_oauth2(mcp_server)
        resolved_client_id = mcp_server.client_id or client_id or ""
        if not resolved_client_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "missing_client_id",
                    "message": (
                        "No client_id available for this MCP server. "
                        "Either configure the server with a client_id or supply one in the request."
                    ),
                },
            )
        return await exchange_token_with_server(
            request=request,
            mcp_server=mcp_server,
            grant_type=grant_type,
            code=code,
            redirect_uri=redirect_uri,
            client_id=resolved_client_id,
            client_secret=client_secret,
            code_verifier=code_verifier,
            refresh_token=refresh_token,
            scope=scope,
        )

    @router.post(
        "/server/oauth/{server_id}/register",
        include_in_schema=False,
        dependencies=[Depends(user_api_key_auth)],
    )
    async def mcp_register(
        request: Request,
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        mcp_server = await _get_cached_temporary_mcp_server_or_404(server_id, user_api_key_dict, request=request)
        request_data = await _read_request_body(request=request)
        data: dict = {**request_data}

        return await register_client_with_server(
            request=request,
            mcp_server=mcp_server,
            client_name=data.get("client_name", ""),
            grant_types=data.get("grant_types", []),
            response_types=data.get("response_types", []),
            token_endpoint_auth_method=data.get("token_endpoint_auth_method", ""),
            fallback_client_id=server_id,
            persist_credentials=_user_is_full_admin(user_api_key_dict),
        )

    @router.delete(
        "/server/{server_id}",
        description="Allows deleting mcp serves in the db",
        dependencies=[Depends(user_api_key_auth)],
        response_class=JSONResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    @management_endpoint_wrapper
    async def remove_mcp_server(
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        litellm_changed_by: Optional[str] = Header(
            None,
            description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
        ),
    ):
        """
        Delete MCP Server from db and associated MCP related server entities.

        Parameters:
        - server_id: str - Required. The unique identifier of the mcp server to delete.
        ```
        curl -X "DELETE" --location 'http://localhost:4000/v1/mcp/server/server_id' \
        --header 'Authorization: Bearer your_api_key_here'
        ```
        """
        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
        )

        # Authz - restrict only admins to delete mcp servers
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Call not allowed to delete MCP server. User is not a proxy admin. route={}".format(
                        "DELETE /v1/mcp/server"
                    )
                },
            )

        # try to delete the mcp server
        mcp_server_record_deleted = await delete_mcp_server(prisma_client, server_id)

        if mcp_server_record_deleted is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"MCP Server not found, passed server_id={server_id}"},
            )
        global_mcp_server_manager.remove_server(mcp_server_record_deleted)

        # Ensure registry is up to date by reloading from database
        await global_mcp_server_manager.reload_servers_from_database()

        # TODO: Enterprise: Finish audit log trail
        if litellm.store_audit_logs:
            pass

        # TODO: Delete from virtual keys

        # TODO: Delete from teams

        # Update from global mcp store

        return Response(status_code=status.HTTP_202_ACCEPTED)

    @router.post(
        "/server/{server_id}/user-credential",
        description="Store or update the calling user's API key for a BYOK MCP server",
        dependencies=[Depends(user_api_key_auth)],
        response_model=MCPUserCredentialResponse,
    )
    @management_endpoint_wrapper
    async def store_mcp_user_credential(
        server_id: str,
        payload: MCPUserCredentialRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """Store a BYOK credential for the calling user."""
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        mcp_server = await _authorize_and_fetch_mcp_server(prisma_client, user_api_key_dict, server_id)
        if not getattr(mcp_server, "is_byok", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "This MCP server does not support BYOK credentials"},
            )
        user_id = user_api_key_dict.user_id or ""
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "User ID not found in token"},
            )
        if payload.save:
            await store_user_credential(prisma_client, user_id, server_id, payload.credential)
            from litellm.proxy._experimental.mcp_server.server import (
                _invalidate_byok_cred_cache,
            )

            _invalidate_byok_cred_cache(user_id, server_id)
            return MCPUserCredentialResponse(server_id=server_id, has_credential=True)
        # save=False: credential not persisted
        return MCPUserCredentialResponse(server_id=server_id, has_credential=False)

    @router.delete(
        "/server/{server_id}/user-credential",
        description="Delete the calling user's stored API key for a BYOK MCP server",
        dependencies=[Depends(user_api_key_auth)],
        response_model=MCPUserCredentialResponse,
    )
    @management_endpoint_wrapper
    async def delete_mcp_user_credential(
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """Remove the calling user's BYOK credential."""
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        user_id = user_api_key_dict.user_id or ""
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "User ID not found in token"},
            )
        try:
            await delete_user_credential(prisma_client, user_id, server_id)
        except RecordNotFoundError:
            pass  # Already deleted or didn't exist
        from litellm.proxy._experimental.mcp_server.server import (
            _invalidate_byok_cred_cache,
        )

        _invalidate_byok_cred_cache(user_id, server_id)
        return MCPUserCredentialResponse(server_id=server_id, has_credential=False)

    # ── OAuth2 user-credential endpoints ──────────────────────────────────────

    @router.post(
        "/server/{server_id}/oauth-user-credential",
        description="Store the calling user's OAuth2 token for an OpenAPI MCP server",
        dependencies=[Depends(user_api_key_auth)],
        response_model=MCPOAuthUserCredentialStatus,
    )
    @management_endpoint_wrapper
    async def store_mcp_oauth_user_credential(
        server_id: str,
        payload: MCPOAuthUserCredentialRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """Persist the OAuth2 access token obtained by the calling user."""
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        await _authorize_and_fetch_mcp_server(prisma_client, user_api_key_dict, server_id)
        user_id = user_api_key_dict.user_id or ""
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "User ID not found in token"},
            )
        await store_user_oauth_credential(
            prisma_client,
            user_id,
            server_id,
            payload.access_token,
            refresh_token=payload.refresh_token,
            expires_in=payload.expires_in,
            scopes=payload.scopes,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415
            global_mcp_server_manager,
        )

        await global_mcp_server_manager.invalidate_user_oauth_token_cache(user_id, server_id)
        # Read back the persisted record so the response reflects the stored
        # expires_at rather than recomputing it here (which could diverge by
        # milliseconds or if the storage logic ever adds a grace period).
        stored = await get_user_oauth_credential(prisma_client, user_id, server_id)
        expires_at: Optional[str] = stored.get("expires_at") if stored else None
        return MCPOAuthUserCredentialStatus(
            server_id=server_id,
            has_credential=True,
            expires_at=expires_at,
            is_expired=False,
        )

    @router.delete(
        "/server/{server_id}/oauth-user-credential",
        description="Revoke the calling user's stored OAuth2 token for an MCP server",
        dependencies=[Depends(user_api_key_auth)],
        response_model=MCPOAuthUserCredentialStatus,
    )
    @management_endpoint_wrapper
    async def delete_mcp_oauth_user_credential(
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """Revoke/delete the user's OAuth2 credential."""
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        user_id = user_api_key_dict.user_id or ""
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "User ID not found in token"},
            )
        # Only delete if the stored credential is actually an OAuth2 token.
        # This prevents accidentally deleting a BYOK credential if one exists
        # for the same (user_id, server_id) pair.
        cred_to_delete = await get_user_oauth_credential(prisma_client, user_id, server_id)
        if cred_to_delete is not None:
            try:
                await delete_user_credential(prisma_client, user_id, server_id)
            except RecordNotFoundError:
                pass  # Already gone — treat as a successful delete
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415
                global_mcp_server_manager,
            )

            await global_mcp_server_manager.invalidate_user_oauth_token_cache(user_id, server_id)
        return MCPOAuthUserCredentialStatus(
            server_id=server_id,
            has_credential=False,
            is_expired=False,
        )

    @router.get(
        "/server/{server_id}/oauth-user-credential/status",
        description="Check whether the calling user has a stored OAuth2 credential for this MCP server",
        dependencies=[Depends(user_api_key_auth)],
        response_model=MCPOAuthUserCredentialStatus,
    )
    @management_endpoint_wrapper
    async def get_mcp_oauth_user_credential_status(
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """Return credential status (has_credential, expiry) without exposing the token."""
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        user_id = user_api_key_dict.user_id or ""
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "User ID not found in token"},
            )
        cred = await get_user_oauth_credential(prisma_client, user_id, server_id)
        if cred is None:
            return MCPOAuthUserCredentialStatus(server_id=server_id, has_credential=False, is_expired=False)
        expires_at: Optional[str] = cred.get("expires_at")
        is_expired = False
        if expires_at:
            try:
                exp = datetime.fromisoformat(expires_at)
                is_expired = exp < datetime.now(timezone.utc)
            except Exception:
                pass
        return MCPOAuthUserCredentialStatus(
            server_id=server_id,
            has_credential=True,
            expires_at=expires_at,
            is_expired=is_expired,
            connected_at=cred.get("connected_at"),
        )

    @router.get(
        "/user-credentials",
        description="List all OAuth2 MCP credentials stored for the calling user",
        dependencies=[Depends(user_api_key_auth)],
        response_model=List[MCPUserCredentialListItem],
    )
    @management_endpoint_wrapper
    async def list_mcp_user_credentials(
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """Return all servers the calling user has connected via OAuth2."""
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        user_id = user_api_key_dict.user_id or ""
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "User ID not found in token"},
            )
        oauth_creds = await list_user_oauth_credentials(prisma_client, user_id)
        if not oauth_creds:
            return []
        # Fetch server metadata for display names — single batch query instead of N+1.
        server_ids = [c["server_id"] for c in oauth_creds]
        servers = {srv.server_id: srv for srv in await get_mcp_servers(prisma_client, server_ids)}
        items: List[MCPUserCredentialListItem] = []
        for cred in oauth_creds:
            sid = cred["server_id"]
            srv = servers.get(sid)
            expires_at: Optional[str] = cred.get("expires_at")
            items.append(
                MCPUserCredentialListItem(
                    server_id=sid,
                    server_name=getattr(srv, "server_name", None) if srv else None,
                    alias=getattr(srv, "alias", None) if srv else None,
                    credential_type="oauth2",
                    has_credential=True,
                    expires_at=expires_at,  # always pass the raw timestamp; client computes expiry state
                    connected_at=cred.get("connected_at"),
                )
            )
        return items

    # ── Per-user MCP env var endpoints ────────────────────────────────────────

    async def _authorize_and_fetch_mcp_server(
        prisma_client,
        user_api_key_dict: UserAPIKeyAuth,
        server_id: str,
    ) -> LiteLLM_MCPServerTable:
        """Resolve the MCP server a caller may manage their own per-user state for.

        Looks the server up in the DB, then the in-memory registry, so a
        config-defined server (which never gets a DB row) resolves too. Admins
        may reach any server and get a 404 for an unknown id. A non-admin may
        only reach a server in their allowed set and otherwise gets 403 (never
        404, so server ids can't be enumerated), using the same allowed-server
        resolution the MCP gateway enforces on tool calls.
        """
        server = await get_mcp_server(prisma_client, server_id)
        if server is None:
            registry_server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
            if registry_server is not None:
                server = global_mcp_server_manager._build_mcp_server_table(registry_server)

        if _user_has_admin_view(user_api_key_dict):
            if server is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": f"MCP Server {server_id} not found"},
                )
            return server

        allowed_server_ids: set[str] = set()
        for auth_context in await build_effective_auth_contexts(user_api_key_dict):
            allowed_server_ids.update(await global_mcp_server_manager.get_allowed_mcp_servers(auth_context))
        if server is None or server.server_id not in allowed_server_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": (
                        f"User does not have permission to access mcp server with id {server_id}. "
                        "You can only manage mcp servers that you have access to."
                    )
                },
            )
        return server

    def _compute_user_env_var_status(
        *,
        server: LiteLLM_MCPServerTable,
        stored_values: Dict[str, str],
    ) -> MCPUserEnvVarsStatus:
        """Build a status object for one server given the user's stored values.

        Stored credentials are write-only: the response reports only whether
        each value ``is_set`` and never echoes the decrypted secret back, so a
        leaked token can't be used to exfiltrate the raw upstream credential.
        """
        global_values, user_specs = parse_admin_env_vars(getattr(server, "env_vars", None))
        # An empty-valued global is not a usable fallback, so it must not mark a
        # referenced per-user var as covered, matching the empty-global filter in
        # _resolve_static_headers_with_env_vars. Otherwise this endpoint reports no
        # credential needed for a var every tool call still 412s on.
        global_values = {name: value for name, value in global_values.items() if value}

        # A var only blocks when it's referenced by static_headers and has no
        # admin global fallback, mirroring _resolve_static_headers_with_env_vars
        # (globals win the merge) so the status endpoint never asks the user for
        # credentials a tool call wouldn't actually require.
        static_headers = getattr(server, "static_headers", None) or {}
        if isinstance(static_headers, str):
            try:
                static_headers = json.loads(static_headers) or {}
            except (ValueError, TypeError):
                static_headers = {}
        referenced = collect_env_var_references(strings=static_headers.values())
        user_var_names = {spec["name"] for spec in user_specs}
        blocking = {name for name in (referenced & user_var_names) if name not in global_values}

        required: List[MCPUserEnvVarSpec] = []
        missing_count = 0
        for spec in user_specs:
            name = spec["name"]
            if name not in blocking:
                continue
            value = stored_values.get(name)
            is_set = bool(value)
            if not is_set:
                missing_count += 1
            required.append(
                MCPUserEnvVarSpec(
                    name=name,
                    description=spec.get("description"),
                    is_set=is_set,
                )
            )

        return MCPUserEnvVarsStatus(
            server_id=server.server_id,
            server_name=getattr(server, "server_name", None),
            alias=getattr(server, "alias", None),
            required=required,
            missing_count=missing_count,
            setup_url=build_env_var_setup_url(server.server_id) if required else None,
        )

    @router.get(
        "/server/{server_id}/user-env-vars",
        description="Return the calling user's per-user MCP env var status for this server.",
        dependencies=[Depends(user_api_key_auth)],
        response_model=MCPUserEnvVarsStatus,
    )
    @management_endpoint_wrapper
    async def get_mcp_user_env_vars(
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> MCPUserEnvVarsStatus:
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        user_id = user_api_key_dict.user_id or ""
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "User ID not found in token"},
            )
        server = await _authorize_and_fetch_mcp_server(prisma_client, user_api_key_dict, server_id)
        stored = await get_user_env_vars(prisma_client, user_id, server_id)
        return _compute_user_env_var_status(server=server, stored_values=stored)

    @router.post(
        "/server/{server_id}/user-env-vars",
        description=(
            "Store the calling user's per-user MCP env var values for this "
            "server. Submitted values are merged over any previously stored "
            "values, so you only send the fields you want to set or change; a "
            "variable omitted (or sent empty) keeps its stored value. Use "
            "DELETE to clear all stored values."
        ),
        dependencies=[Depends(user_api_key_auth)],
        response_model=MCPUserEnvVarsStatus,
    )
    @management_endpoint_wrapper
    async def store_mcp_user_env_vars(
        server_id: str,
        payload: MCPUserEnvVarsRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> MCPUserEnvVarsStatus:
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        user_id = user_api_key_dict.user_id or ""
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "User ID not found in token"},
            )
        server = await _authorize_and_fetch_mcp_server(prisma_client, user_api_key_dict, server_id)
        # Only known per-user var names declared by the admin are accepted —
        # never persist arbitrary keys the user invents. Submitted values are
        # merged over the existing set so a user updating one credential does
        # not have to re-enter the others (which are write-only and never shown
        # back); an omitted/empty field keeps its stored value.
        _, user_specs = parse_admin_env_vars(getattr(server, "env_vars", None))
        allowed_names = {spec["name"] for spec in user_specs}
        updates = {k: v for k, v in payload.values.items() if k in allowed_names and v != ""}
        merged = await merge_user_env_vars(prisma_client, user_id, server_id, updates, allowed_names)
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            invalidate_user_env_vars_cache,
        )

        invalidate_user_env_vars_cache(user_id, server_id)
        return _compute_user_env_var_status(server=server, stored_values=merged)

    @router.delete(
        "/server/{server_id}/user-env-vars",
        description="Clear the calling user's per-user MCP env var values for this server.",
        dependencies=[Depends(user_api_key_auth)],
        response_model=MCPUserEnvVarsStatus,
    )
    @management_endpoint_wrapper
    async def clear_mcp_user_env_vars(
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> MCPUserEnvVarsStatus:
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        user_id = user_api_key_dict.user_id or ""
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "User ID not found in token"},
            )
        server = await _authorize_and_fetch_mcp_server(prisma_client, user_api_key_dict, server_id)
        await delete_user_env_vars(prisma_client, user_id, server_id)
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            invalidate_user_env_vars_cache,
        )

        invalidate_user_env_vars_cache(user_id, server_id)
        return _compute_user_env_var_status(server=server, stored_values={})

    @router.get(
        "/user-env-vars/status",
        description="Per-user MCP env var status across every server the user can access. "
        "Used by the dashboard to highlight servers with missing per-user vars.",
        dependencies=[Depends(user_api_key_auth)],
        response_model=List[MCPUserEnvVarsStatus],
    )
    @management_endpoint_wrapper
    async def list_mcp_user_env_var_status(
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> List[MCPUserEnvVarsStatus]:
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        user_id = user_api_key_dict.user_id or ""
        if not user_id:
            return []
        accessible = await _resolve_accessible_mcp_servers(user_api_key_dict)
        if not accessible:
            return []
        server_ids = [s.server_id for s in accessible]
        stored_bulk = await get_user_env_vars_bulk(prisma_client, user_id, server_ids)
        statuses: List[MCPUserEnvVarsStatus] = []
        for server in accessible:
            stored = stored_bulk.get(server.server_id, {})
            status_obj = _compute_user_env_var_status(server=server, stored_values=stored)
            if status_obj.required:
                statuses.append(status_obj)
        return statuses

    @router.put(
        "/server",
        description="Allows deleting mcp serves in the db",
        dependencies=[Depends(user_api_key_auth)],
        response_model=LiteLLM_MCPServerTable,
        status_code=status.HTTP_202_ACCEPTED,
    )
    @management_endpoint_wrapper
    async def edit_mcp_server(
        payload: UpdateMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        litellm_changed_by: Optional[str] = Header(
            None,
            description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
        ),
    ):
        """
        Updates the MCP Server in the db.

        Parameters:
        - payload: UpdateMCPServerRequest - Required. The updated mcp server data.
        ```
        curl -X "PUT" --location 'http://localhost:4000/v1/mcp/server' \
        --header 'Authorization: Bearer your_api_key_here'
        ```
        """
        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
        )

        payload_fields_set = set(payload.fields_set())

        # Validate and normalize payload fields
        validate_and_normalize_mcp_server_payload(payload)

        # Authz - restrict only admins to delete mcp servers
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Call not allowed to update MCP server. User is not a proxy admin. route={}".format(
                        "PUT /v1/mcp/server"
                    )
                },
            )

        # Snapshot the pre-update identity so we can detect a mint-relevant change below. The read is
        # advisory (it only feeds the stale-token purge decision), so a failure skips the purge with a
        # warning instead of failing the edit, whose primary job is the update itself.
        try:
            old_server_record = await get_mcp_server(prisma_client, payload.server_id)
            old_server_record_read_failed = False
        except Exception as exc:  # noqa: BLE001 - advisory read; invalidation is best-effort end-to-end
            verbose_logger.warning(
                "MCP server %s: could not snapshot the pre-update record; skipping the stale-token check: %s",
                payload.server_id,
                exc,
            )
            old_server_record = None
            old_server_record_read_failed = True

        if (
            payload.dcr_bridge
            and payload.auth_type is None
            and (old_server_record is not None or old_server_record_read_failed)
        ):
            stored_auth_type = old_server_record.auth_type if old_server_record else None
            stored_auth_type_name = getattr(stored_auth_type, "value", stored_auth_type)
            if stored_auth_type not in (MCPAuth.true_passthrough, MCPAuth.oauth_delegate):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": (
                            "dcr_bridge is only supported for auth_type true_passthrough or "
                            f"oauth_delegate (stored auth_type: {stored_auth_type_name!r}). Include "
                            "the server's auth_type in the update payload or configure one of the "
                            "client-forwarded token modes first."
                        )
                    },
                )

        # try to update the mcp server
        mcp_server_record_updated = await update_mcp_server(
            prisma_client,
            payload,
            touched_by=user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
            fields_set=payload_fields_set,
        )

        if mcp_server_record_updated is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"MCP Server not found, passed server_id={payload.server_id}"},
            )
        await global_mcp_server_manager.update_server(mcp_server_record_updated)

        # Ensure registry is up to date by reloading from database
        await global_mcp_server_manager.reload_servers_from_database()

        # If a field that determines which upstream OAuth token gets minted changed (url/audience, OAuth
        # mode/grant, authorization-server endpoints, or the OAuth client + scopes), every stored per-user
        # token was minted for the old configuration and is stale. Purge them (DB + cache) so the next
        # tool call re-authorizes instead of forwarding a token for a resource/AS/client that no longer
        # matches. Best-effort: a purge failure must not fail the update, whose primary job already
        # succeeded.
        if old_server_record is not None and mcp_oauth_token_identity(old_server_record) != mcp_oauth_token_identity(
            mcp_server_record_updated
        ):
            try:
                purged = await purge_user_oauth_credentials_for_server(prisma_client, payload.server_id)
                if purged:
                    verbose_logger.info(
                        "MCP server %s: purged %d stale per-user OAuth token(s) after a mint-relevant config change",
                        payload.server_id,
                        purged,
                    )
            except Exception as exc:  # noqa: BLE001 - purge is best-effort; the server update already succeeded
                verbose_logger.warning(
                    "MCP server %s: failed to purge stale per-user OAuth tokens after config change: %s",
                    payload.server_id,
                    exc,
                )

        # TODO: Enterprise: Finish audit log trail
        if litellm.store_audit_logs:
            pass

        return _redact_mcp_credentials(mcp_server_record_updated)

    @router.post(
        "/make_public",
        description="Allows making MCP servers public for AI Hub",
        dependencies=[Depends(user_api_key_auth)],
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def make_mcp_servers_public(
        request: MakeMCPServersPublicRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Make MCP servers public for AI Hub
        """
        try:
            # Update the public model groups
            import litellm
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )
            from litellm.proxy.proxy_server import proxy_config

            # Load existing config
            config = await proxy_config.get_config()
            # Check if user has admin permissions
            if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "Only proxy admins can update public mcp servers. Your role={}".format(
                            user_api_key_dict.user_role
                        )
                    },
                )

            if litellm.public_mcp_servers is None:
                litellm.public_mcp_servers = []

            for server_id in request.mcp_server_ids:
                server = global_mcp_server_manager.get_mcp_server_by_id(server_id=server_id)
                if server is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"MCP Server with ID {server_id} not found",
                    )

            litellm.public_mcp_servers = request.mcp_server_ids

            # Update config with new settings
            if "litellm_settings" not in config or config["litellm_settings"] is None:
                config["litellm_settings"] = {}

            config["litellm_settings"]["public_mcp_servers"] = litellm.public_mcp_servers

            # Save the updated config
            await proxy_config.save_config(new_config=config)

            verbose_proxy_logger.debug(
                f"Updated public mcp servers to: {litellm.public_mcp_servers} by user: {user_api_key_dict.user_id}"
            )

            return {
                "message": "Successfully updated public mcp servers",
                "public_mcp_servers": litellm.public_mcp_servers,
                "updated_by": user_api_key_dict.user_id,
            }
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.exception(f"Error making agent public: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # --- MCP Discovery ---

    _MCP_REGISTRY_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mcp_registry.json",
    )

    _mcp_registry_cache: Optional[Dict[str, Any]] = None

    def _load_mcp_registry() -> Dict[str, Any]:
        """Load the curated MCP registry from disk. Cached after first read."""
        global _mcp_registry_cache
        if _mcp_registry_cache is not None:
            return _mcp_registry_cache
        try:
            with open(_MCP_REGISTRY_PATH, "r") as f:
                data: Dict[str, Any] = json.load(f)
        except Exception as e:
            verbose_proxy_logger.warning(f"Failed to load MCP registry from {_MCP_REGISTRY_PATH}: {e}")
            data = {"servers": []}
        _mcp_registry_cache = data
        return data

    @router.get(
        "/discover",
        description="Returns a curated list of well-known MCP servers for discovery UI",
        dependencies=[Depends(user_api_key_auth)],
    )
    async def discover_mcp_servers(
        query: Optional[str] = Query(None, description="Search filter for server names and descriptions"),
        category: Optional[str] = Query(None, description="Filter by category"),
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Returns a curated list of well-known MCP servers that can be added to the proxy.

        Used by the UI to show a discovery grid when adding new MCP servers.
        """
        # Admin Viewer follows the read-parity rule.
        if not _user_has_admin_view(user_api_key_dict):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Only proxy admins can access MCP discovery. Your role={}".format(
                        user_api_key_dict.user_role
                    )
                },
            )

        registry = _load_mcp_registry()
        servers = registry.get("servers", [])

        # Apply query filter
        if query:
            query_lower = query.lower()
            servers = [
                s
                for s in servers
                if query_lower in s.get("name", "").lower()
                or query_lower in s.get("title", "").lower()
                or query_lower in s.get("description", "").lower()
            ]

        # Apply category filter
        if category:
            servers = [s for s in servers if s.get("category", "") == category]

        # Extract unique categories from the full list (before filtering)
        all_servers = registry.get("servers", [])
        categories = sorted(set(s.get("category", "Other") for s in all_servers))

        return {
            "servers": servers,
            "categories": categories,
        }

    # --- OpenAPI Registry ---

    _OPENAPI_REGISTRY_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "openapi_registry.json",
    )

    @functools.lru_cache(maxsize=1)
    def _load_openapi_registry() -> Dict[str, Any]:
        with open(_OPENAPI_REGISTRY_PATH, "r") as f:
            data: Dict[str, Any] = json.load(f)
        return data

    @router.get(
        "/openapi-registry",
        description="Returns well-known OpenAPI APIs with OAuth 2.0 metadata for the OpenAPI MCP picker",
    )
    async def get_openapi_registry(
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        # Admin Viewer follows the read-parity rule.
        if not _user_has_admin_view(user_api_key_dict):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Only proxy admins can access the OpenAPI registry. Your role={}".format(
                        user_api_key_dict.user_role
                    )
                },
            )
        try:
            return _load_openapi_registry()
        except Exception as e:
            verbose_proxy_logger.warning(f"Failed to load OpenAPI registry from {_OPENAPI_REGISTRY_PATH}: {e}")
            return {"apis": []}

    # ---------------------------------------------------------------------------
    # MCP Toolset endpoints
    # ---------------------------------------------------------------------------

    from litellm.proxy._experimental.mcp_server.toolset_db import (
        create_mcp_toolset,
        delete_mcp_toolset,
        get_mcp_toolset,
        list_mcp_toolsets,
        update_mcp_toolset,
    )
    from litellm.types.mcp_server.mcp_toolset import (
        NewMCPToolsetRequest,
        UpdateMCPToolsetRequest,
    )

    @router.post(
        "/toolset",
        description="Create a new MCP toolset (admin only)",
        status_code=status.HTTP_201_CREATED,
    )
    @management_endpoint_wrapper
    async def add_mcp_toolset(
        payload: NewMCPToolsetRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        litellm_changed_by: Optional[str] = Header(None),
    ):
        """Create a named toolset — a curated selection of {server_id, tool_name} pairs."""
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Only proxy admins can create MCP toolsets."},
            )
        touched_by = (
            get_audit_log_changed_by(
                litellm_changed_by=litellm_changed_by,
                user_api_key_dict=user_api_key_dict,
                litellm_proxy_admin_name=LITELLM_PROXY_ADMIN_NAME,
            )
            or LITELLM_PROXY_ADMIN_NAME
        )
        try:
            result = await create_mcp_toolset(prisma_client, payload, touched_by)
        except UniqueViolationError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error": f"A toolset named '{payload.toolset_name}' already exists."},
            )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        global_mcp_server_manager.invalidate_toolset_cache()
        return result

    @router.get(
        "/toolset",
        description="List MCP toolsets accessible to the calling key",
    )
    @management_endpoint_wrapper
    async def fetch_mcp_toolsets(
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """Return toolsets the calling key is allowed to access."""
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        is_admin = _user_has_admin_view(user_api_key_dict)
        op = user_api_key_dict.object_permission
        # mcp_toolsets=None or [] both mean "not restricted by toolsets".
        # For admins: either value → no restriction → return all.
        # For non-admins: either value → no toolsets explicitly granted → return nothing.
        # (An admin whose DB row has mcp_toolsets=[] should still see all toolsets.)
        raw_toolsets = getattr(op, "mcp_toolsets", None) if op else None
        if not raw_toolsets:
            if is_admin:
                return await list_mcp_toolsets(prisma_client)
            return []
        return await list_mcp_toolsets(prisma_client, toolset_ids=raw_toolsets)

    @router.get(
        "/toolset/{toolset_id}",
        description="Get a specific MCP toolset by ID",
    )
    @management_endpoint_wrapper
    async def fetch_mcp_toolset(
        toolset_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        # Non-admin keys may only fetch toolsets they've been explicitly granted.
        if not _user_has_admin_view(user_api_key_dict):
            op = user_api_key_dict.object_permission
            granted = getattr(op, "mcp_toolsets", None) if op else None
            if granted is None or toolset_id not in granted:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"error": "API key does not have access to this toolset."},
                )
        toolset = await get_mcp_toolset(prisma_client, toolset_id)
        if toolset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"Toolset '{toolset_id}' not found."},
            )
        return toolset

    @router.put(
        "/toolset",
        description="Update an existing MCP toolset (admin only)",
    )
    @management_endpoint_wrapper
    async def edit_mcp_toolset(
        payload: UpdateMCPToolsetRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        litellm_changed_by: Optional[str] = Header(None),
    ):
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Only proxy admins can update MCP toolsets."},
            )
        touched_by = (
            get_audit_log_changed_by(
                litellm_changed_by=litellm_changed_by,
                user_api_key_dict=user_api_key_dict,
                litellm_proxy_admin_name=LITELLM_PROXY_ADMIN_NAME,
            )
            or LITELLM_PROXY_ADMIN_NAME
        )
        try:
            result = await update_mcp_toolset(prisma_client, payload, touched_by)
        except UniqueViolationError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": (
                        f"A toolset named '{payload.toolset_name}' already exists."
                        if payload.toolset_name
                        else "A toolset with that name already exists."
                    )
                },
            )
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"Toolset '{payload.toolset_id}' not found."},
            )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        global_mcp_server_manager.invalidate_toolset_cache(getattr(payload, "toolset_id", None))
        return result

    @router.delete(
        "/toolset/{toolset_id}",
        description="Delete an MCP toolset (admin only)",
        status_code=status.HTTP_202_ACCEPTED,
    )
    @management_endpoint_wrapper
    async def remove_mcp_toolset(
        toolset_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        litellm_changed_by: Optional[str] = Header(None),
    ):
        prisma_client = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Only proxy admins can delete MCP toolsets."},
            )
        deleted = await delete_mcp_toolset(prisma_client, toolset_id)
        if deleted is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"Toolset '{toolset_id}' not found."},
            )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        global_mcp_server_manager.invalidate_toolset_cache(toolset_id)
        return Response(status_code=status.HTTP_202_ACCEPTED)
