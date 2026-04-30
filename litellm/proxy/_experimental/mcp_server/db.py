import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Union, cast

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    MCPApprovalStatus,
    MCPSubmissionsSummary,
    NewMCPServerRequest,
    SpecialMCPServerName,
    UpdateMCPServerRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    _get_salt_key,
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy.utils import PrismaClient
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.mcp import MCPCredentials


def _prepare_mcp_server_data(
    data: Union[NewMCPServerRequest, UpdateMCPServerRequest],
) -> Dict[str, Any]:
    """
    Helper function to prepare MCP server data for database operations.
    Handles JSON field serialization for mcp_info and env fields.

    Args:
        data: NewMCPServerRequest or UpdateMCPServerRequest object

    Returns:
        Dict with properly serialized JSON fields
    """
    from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

    # Convert model to dict
    data_dict = data.model_dump(exclude_none=True)
    # Ensure alias is always present in the dict (even if None)
    if "alias" not in data_dict:
        data_dict["alias"] = getattr(data, "alias", None)

    # Handle credentials serialization
    credentials = data_dict.get("credentials")
    if credentials is not None:
        data_dict["credentials"] = encrypt_credentials(
            credentials=credentials, encryption_key=_get_salt_key()
        )
        data_dict["credentials"] = safe_dumps(data_dict["credentials"])

    # Handle static_headers serialization
    if data.static_headers is not None:
        data_dict["static_headers"] = safe_dumps(data.static_headers)

    # Handle mcp_info serialization
    if data.mcp_info is not None:
        data_dict["mcp_info"] = safe_dumps(data.mcp_info)

    # Handle env serialization
    if data.env is not None:
        data_dict["env"] = safe_dumps(data.env)

    # Handle tool name override serialization
    if data.tool_name_to_display_name is not None:
        data_dict["tool_name_to_display_name"] = safe_dumps(
            data.tool_name_to_display_name
        )
    if data.tool_name_to_description is not None:
        data_dict["tool_name_to_description"] = safe_dumps(
            data.tool_name_to_description
        )

    # mcp_access_groups is already List[str], no serialization needed

    # Force include is_byok even when False (exclude_none=True would not drop it,
    # but be explicit to ensure a False value is always written to the DB).
    data_dict["is_byok"] = getattr(data, "is_byok", False)

    return data_dict


def encrypt_credentials(
    credentials: MCPCredentials, encryption_key: Optional[str]
) -> MCPCredentials:
    auth_value = credentials.get("auth_value")
    if auth_value is not None:
        credentials["auth_value"] = encrypt_value_helper(
            value=auth_value,
            new_encryption_key=encryption_key,
        )
    client_id = credentials.get("client_id")
    if client_id is not None:
        credentials["client_id"] = encrypt_value_helper(
            value=client_id,
            new_encryption_key=encryption_key,
        )
    client_secret = credentials.get("client_secret")
    if client_secret is not None:
        credentials["client_secret"] = encrypt_value_helper(
            value=client_secret,
            new_encryption_key=encryption_key,
        )
    # AWS SigV4 credential fields
    aws_access_key_id = credentials.get("aws_access_key_id")
    if aws_access_key_id is not None:
        credentials["aws_access_key_id"] = encrypt_value_helper(
            value=aws_access_key_id,
            new_encryption_key=encryption_key,
        )
    aws_secret_access_key = credentials.get("aws_secret_access_key")
    if aws_secret_access_key is not None:
        credentials["aws_secret_access_key"] = encrypt_value_helper(
            value=aws_secret_access_key,
            new_encryption_key=encryption_key,
        )
    aws_session_token = credentials.get("aws_session_token")
    if aws_session_token is not None:
        credentials["aws_session_token"] = encrypt_value_helper(
            value=aws_session_token,
            new_encryption_key=encryption_key,
        )
    # aws_region_name and aws_service_name are NOT secrets — stored as-is
    return credentials


def decrypt_credentials(
    credentials: MCPCredentials,
) -> MCPCredentials:
    """Decrypt all secret fields in an MCPCredentials dict using the global salt key."""
    secret_fields = [
        "auth_value",
        "client_id",
        "client_secret",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
    ]
    for field in secret_fields:
        value = credentials.get(field)  # type: ignore[literal-required]
        if value is not None and isinstance(value, str):
            credentials[field] = decrypt_value_helper(  # type: ignore[literal-required]
                value=value,
                key=field,
                exception_type="debug",
                return_original_value=True,
            )
    return credentials


async def get_all_mcp_servers(
    prisma_client: PrismaClient,
    approval_status: Optional[str] = None,
) -> List[LiteLLM_MCPServerTable]:
    """
    Returns mcp servers from the db, optionally filtered by approval_status.
    Pass approval_status=None to return all servers regardless of approval state.
    """
    try:
        where: Dict[str, Any] = {}
        if approval_status is not None:
            where["approval_status"] = approval_status
        mcp_servers = await prisma_client.db.litellm_mcpservertable.find_many(
            where=where if where else {}
        )

        return [
            LiteLLM_MCPServerTable(**mcp_server.model_dump())
            for mcp_server in mcp_servers
        ]
    except Exception as e:
        verbose_proxy_logger.debug(
            "litellm.proxy._experimental.mcp_server.db.py::get_all_mcp_servers - {}".format(
                str(e)
            )
        )
        return []


async def get_mcp_server(
    prisma_client: PrismaClient, server_id: str
) -> Optional[LiteLLM_MCPServerTable]:
    """
    Returns the matching mcp server from the db iff exists
    """
    mcp_server: Optional[LiteLLM_MCPServerTable] = (
        await prisma_client.db.litellm_mcpservertable.find_unique(
            where={
                "server_id": server_id,
            }
        )
    )
    return mcp_server


async def get_mcp_servers(
    prisma_client: PrismaClient, server_ids: Iterable[str]
) -> List[LiteLLM_MCPServerTable]:
    """
    Returns the matching mcp servers from the db with the server_ids
    """
    _mcp_servers: List[LiteLLM_MCPServerTable] = (
        await prisma_client.db.litellm_mcpservertable.find_many(
            where={
                "server_id": {"in": server_ids},
            }
        )
    )
    final_mcp_servers: List[LiteLLM_MCPServerTable] = []
    for _mcp_server in _mcp_servers:
        final_mcp_servers.append(LiteLLM_MCPServerTable(**_mcp_server.model_dump()))

    return final_mcp_servers


async def get_mcp_servers_by_verificationtoken(
    prisma_client: PrismaClient, token: str
) -> List[str]:
    """
    Returns the mcp servers from the db for the verification token
    """
    verification_token_record: LiteLLM_TeamTable = (
        await prisma_client.db.litellm_verificationtoken.find_unique(
            where={
                "token": token,
            },
            include={
                "object_permission": True,
            },
        )
    )

    mcp_servers: Optional[List[str]] = []
    if (
        verification_token_record is not None
        and verification_token_record.object_permission is not None
    ):
        mcp_servers = verification_token_record.object_permission.mcp_servers
    return mcp_servers or []


async def get_mcp_servers_by_team(
    prisma_client: PrismaClient, team_id: str
) -> List[str]:
    """
    Returns the mcp servers from the db for the team id
    """
    team_record: LiteLLM_TeamTable = (
        await prisma_client.db.litellm_teamtable.find_unique(
            where={
                "team_id": team_id,
            },
            include={
                "object_permission": True,
            },
        )
    )

    mcp_servers: Optional[List[str]] = []
    if team_record is not None and team_record.object_permission is not None:
        mcp_servers = team_record.object_permission.mcp_servers
    return mcp_servers or []


async def get_all_mcp_servers_for_user(
    prisma_client: PrismaClient,
    user: UserAPIKeyAuth,
) -> List[LiteLLM_MCPServerTable]:
    """
    Get all the mcp servers filtered by the given user has access to.

    Following Least-Privilege Principle - the requestor should only be able to see the mcp servers that they have access to.
    """

    mcp_server_ids: Set[str] = set()
    mcp_servers = []

    # Get the mcp servers for the key
    if user.api_key:
        token_mcp_servers = await get_mcp_servers_by_verificationtoken(
            prisma_client, user.api_key
        )
        mcp_server_ids.update(token_mcp_servers)

        # check for special team membership
        if (
            SpecialMCPServerName.all_team_servers in mcp_server_ids
            and user.team_id is not None
        ):
            team_mcp_servers = await get_mcp_servers_by_team(
                prisma_client, user.team_id
            )
            mcp_server_ids.update(team_mcp_servers)

    if len(mcp_server_ids) > 0:
        mcp_servers = await get_mcp_servers(prisma_client, mcp_server_ids)

    return mcp_servers


async def get_objectpermissions_for_mcp_server(
    prisma_client: PrismaClient, mcp_server_id: str
) -> List[LiteLLM_ObjectPermissionTable]:
    """
    Get all the object permissions records and the associated team and verficiationtoken records that have access to the mcp server
    """
    object_permission_records = (
        await prisma_client.db.litellm_objectpermissiontable.find_many(
            where={
                "mcp_servers": {"has": mcp_server_id},
            },
            include={
                "teams": True,
                "verification_tokens": True,
            },
        )
    )

    return object_permission_records


async def get_virtualkeys_for_mcp_server(
    prisma_client: PrismaClient, server_id: str
) -> List:
    """
    Get all the virtual keys that have access to the mcp server
    """
    virtual_keys = await prisma_client.db.litellm_verificationtoken.find_many(
        where={
            "mcp_servers": {"has": server_id},
        },
    )

    if virtual_keys is None:
        return []
    return virtual_keys


async def delete_mcp_server_from_team(prisma_client: PrismaClient, server_id: str):
    """
    Remove the mcp server from the team
    """
    pass


async def delete_mcp_server_from_virtualkey():
    """
    Remove the mcp server from the virtual key
    """
    pass


async def delete_mcp_server(
    prisma_client: PrismaClient, server_id: str
) -> Optional[LiteLLM_MCPServerTable]:
    """
    Delete the mcp server from the db by server_id

    Returns the deleted mcp server record if it exists, otherwise None
    """
    deleted_server = await prisma_client.db.litellm_mcpservertable.delete(
        where={
            "server_id": server_id,
        },
    )
    return deleted_server


async def create_mcp_server(
    prisma_client: PrismaClient, data: NewMCPServerRequest, touched_by: str
) -> LiteLLM_MCPServerTable:
    """
    Create a new mcp server record in the db
    """
    if data.server_id is None:
        data.server_id = str(uuid.uuid4())

    # Use helper to prepare data with proper JSON serialization
    data_dict = _prepare_mcp_server_data(data)

    # Add audit fields
    data_dict["created_by"] = touched_by
    data_dict["updated_by"] = touched_by

    new_mcp_server = await prisma_client.db.litellm_mcpservertable.create(
        data=data_dict  # type: ignore
    )

    return new_mcp_server


async def update_mcp_server(
    prisma_client: PrismaClient, data: UpdateMCPServerRequest, touched_by: str
) -> LiteLLM_MCPServerTable:
    """
    Update a new mcp server record in the db
    """
    import json

    from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

    # Use helper to prepare data with proper JSON serialization
    data_dict = _prepare_mcp_server_data(data)

    # Pre-fetch existing record once if we need it for auth_type or credential logic
    existing = None
    has_credentials = (
        "credentials" in data_dict and data_dict["credentials"] is not None
    )
    if data.auth_type or has_credentials:
        existing = await prisma_client.db.litellm_mcpservertable.find_unique(
            where={"server_id": data.server_id}
        )

    # Clear stale credentials when auth_type changes but no new credentials provided
    if (
        data.auth_type
        and "credentials" not in data_dict
        and existing
        and existing.auth_type is not None
        and existing.auth_type != data.auth_type
    ):
        data_dict["credentials"] = None

    # Merge credentials: preserve existing fields not present in the update.
    # Without this, a partial credential update (e.g. changing only region)
    # would wipe encrypted secrets that the UI cannot display back.
    if "credentials" in data_dict and data_dict["credentials"] is not None:
        if existing and existing.credentials:
            # Only merge when auth_type is unchanged. Switching auth types
            # (e.g. oauth2 → api_key) should replace credentials entirely
            # to avoid stale secrets from the previous auth type lingering.
            auth_type_unchanged = (
                data.auth_type is None or data.auth_type == existing.auth_type
            )
            if auth_type_unchanged:
                existing_creds = (
                    json.loads(existing.credentials)
                    if isinstance(existing.credentials, str)
                    else dict(existing.credentials)
                )
                new_creds = (
                    json.loads(data_dict["credentials"])
                    if isinstance(data_dict["credentials"], str)
                    else dict(data_dict["credentials"])
                )
                # New values override existing; existing keys not in update are preserved
                merged = {**existing_creds, **new_creds}
                data_dict["credentials"] = safe_dumps(merged)

    # Add audit fields
    data_dict["updated_by"] = touched_by

    updated_mcp_server = await prisma_client.db.litellm_mcpservertable.update(
        where={"server_id": data.server_id}, data=data_dict  # type: ignore
    )

    return updated_mcp_server


async def rotate_mcp_server_credentials_master_key(
    prisma_client: PrismaClient, touched_by: str, new_master_key: str
):
    mcp_servers = await prisma_client.db.litellm_mcpservertable.find_many()

    for mcp_server in mcp_servers:
        credentials = mcp_server.credentials
        if not credentials:
            continue

        credentials_copy = dict(credentials)
        # Decrypt with current key first, then re-encrypt with new key
        decrypted_credentials = decrypt_credentials(
            credentials=cast(MCPCredentials, credentials_copy),
        )
        encrypted_credentials = encrypt_credentials(
            credentials=decrypted_credentials,
            encryption_key=new_master_key,
        )

        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        serialized_credentials = safe_dumps(encrypted_credentials)

        await prisma_client.db.litellm_mcpservertable.update(
            where={"server_id": mcp_server.server_id},
            data={
                "credentials": serialized_credentials,
                "updated_by": touched_by,
            },
        )


async def store_user_credential(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
    credential: str,
) -> None:
    """Store a user credential for a BYOK MCP server."""

    encoded = base64.urlsafe_b64encode(credential.encode()).decode()
    await prisma_client.db.litellm_mcpusercredentials.upsert(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}},
        data={
            "create": {
                "user_id": user_id,
                "server_id": server_id,
                "credential_b64": encoded,
            },
            "update": {"credential_b64": encoded},
        },
    )


async def get_user_credential(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
) -> Optional[str]:
    """Return credential for a user+server pair, or None."""

    row = await prisma_client.db.litellm_mcpusercredentials.find_unique(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
    )
    if row is None:
        return None
    try:
        return base64.urlsafe_b64decode(row.credential_b64).decode()
    except Exception:
        # Fall back to nacl decryption for credentials stored by older code
        return decrypt_value_helper(
            value=row.credential_b64,
            key="byok_credential",
            exception_type="debug",
            return_original_value=False,
        )


async def has_user_credential(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
) -> bool:
    """Return True if the user has a stored credential for this server."""
    row = await prisma_client.db.litellm_mcpusercredentials.find_unique(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
    )
    return row is not None


async def delete_user_credential(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
) -> None:
    """Delete the user's stored credential for a BYOK MCP server."""
    await prisma_client.db.litellm_mcpusercredentials.delete(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
    )


# ── OAuth2 user-credential helpers ────────────────────────────────────────────


async def store_user_oauth_credential(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
    access_token: str,
    refresh_token: Optional[str] = None,
    expires_in: Optional[int] = None,
    scopes: Optional[List[str]] = None,
    skip_byok_guard: bool = False,
) -> None:
    """Persist an OAuth2 access token for a user+server pair.

    The payload is JSON-serialised and stored base64-encoded in the same
    ``credential_b64`` column used by BYOK.  A ``"type": "oauth2"`` key
    differentiates it from plain BYOK API keys.
    """

    expires_at: Optional[str] = None
    if expires_in is not None:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        ).isoformat()

    payload: Dict[str, Any] = {
        "type": "oauth2",
        "access_token": access_token,
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }
    if refresh_token:
        payload["refresh_token"] = refresh_token
    if expires_at:
        payload["expires_at"] = expires_at
    if scopes:
        payload["scopes"] = scopes

    # Guard against silently overwriting a BYOK credential with an OAuth token.
    # BYOK credentials lack a "type" field (or use a non-"oauth2" type).
    # Skip the guard when the caller knows the row is already an OAuth2 credential
    # (e.g. during token refresh), saving an extra DB round-trip.
    if not skip_byok_guard:
        existing = await prisma_client.db.litellm_mcpusercredentials.find_unique(
            where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
        )
        if existing is not None:
            _byok_error = ValueError(
                f"A non-OAuth2 credential already exists for user {user_id} "
                f"and server {server_id}. Refusing to overwrite."
            )
            try:
                raw = json.loads(
                    base64.urlsafe_b64decode(existing.credential_b64).decode()
                )
            except Exception:
                # Credential is not base64+JSON — it's a plain-text BYOK key.
                raise _byok_error
            if raw.get("type") != "oauth2":
                raise _byok_error

    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    await prisma_client.db.litellm_mcpusercredentials.upsert(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}},
        data={
            "create": {
                "user_id": user_id,
                "server_id": server_id,
                "credential_b64": encoded,
            },
            "update": {"credential_b64": encoded},
        },
    )


def is_oauth_credential_expired(cred: Dict[str, Any]) -> bool:
    """Return True if the OAuth2 credential's access_token has expired.

    Checks the ``expires_at`` ISO-format string stored in the credential payload.
    Returns False when ``expires_at`` is absent or unparseable (treat as non-expired).
    """
    expires_at = cred.get("expires_at")
    if not expires_at:
        return False
    try:
        exp_dt = datetime.fromisoformat(expires_at)
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > exp_dt
    except (ValueError, TypeError):
        return False


async def get_user_oauth_credential(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
) -> Optional[Dict[str, Any]]:
    """Return the decoded OAuth2 payload dict for a user+server pair, or None."""

    row = await prisma_client.db.litellm_mcpusercredentials.find_unique(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
    )
    if row is None:
        return None
    try:
        decoded = base64.urlsafe_b64decode(row.credential_b64).decode()
        parsed = json.loads(decoded)
        if isinstance(parsed, dict) and parsed.get("type") == "oauth2":
            return parsed
        # Row exists but is a BYOK (plain string), not an OAuth token
        return None
    except Exception:
        return None


async def list_user_oauth_credentials(
    prisma_client: PrismaClient,
    user_id: str,
) -> List[Dict[str, Any]]:
    """Return all OAuth2 credential payloads for a user, tagged with server_id."""

    rows = await prisma_client.db.litellm_mcpusercredentials.find_many(
        where={"user_id": user_id}
    )
    results: List[Dict[str, Any]] = []
    for row in rows:
        try:
            decoded = base64.urlsafe_b64decode(row.credential_b64).decode()
            parsed = json.loads(decoded)
            if isinstance(parsed, dict) and parsed.get("type") == "oauth2":
                parsed["server_id"] = row.server_id
                results.append(parsed)
        except Exception:
            pass  # Skip non-OAuth rows (BYOK plain strings)
    return results


async def refresh_user_oauth_token(
    prisma_client: PrismaClient,
    user_id: str,
    server: Any,
    cred: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Attempt to refresh a per-user OAuth2 token using its stored refresh_token.

    POSTs to ``server.token_url`` with ``grant_type=refresh_token``.

    On success: persists the new credential via ``store_user_oauth_credential``
    and returns the updated payload dict.
    On failure (network error, invalid_grant, missing refresh_token, …): logs a
    warning and returns ``None`` — the caller is responsible for clearing the
    stale credential and triggering re-authentication.
    """
    refresh_token: Optional[str] = cred.get("refresh_token")
    token_url: Optional[str] = getattr(server, "token_url", None)
    server_id: str = getattr(server, "server_id", "")
    client_id: Optional[str] = getattr(server, "client_id", None)
    client_secret: Optional[str] = getattr(server, "client_secret", None)

    if not refresh_token:
        verbose_proxy_logger.debug(
            "refresh_user_oauth_token: no refresh_token stored for user=%s server=%s",
            user_id,
            server_id,
        )
        return None
    if not token_url:
        verbose_proxy_logger.debug(
            "refresh_user_oauth_token: server=%s has no token_url configured",
            server_id,
        )
        return None

    token_data: Dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    if client_id:
        token_data["client_id"] = client_id
    if client_secret:
        token_data["client_secret"] = client_secret

    try:
        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.Oauth2Check
        )
        response = await async_client.post(
            token_url,
            headers={"Accept": "application/json"},
            data=token_data,
        )
        response.raise_for_status()
        body: Dict[str, Any] = response.json()
    except Exception as exc:
        verbose_proxy_logger.warning(
            "refresh_user_oauth_token: refresh request failed for user=%s server=%s: %s",
            user_id,
            server_id,
            exc,
        )
        return None

    access_token: Optional[str] = body.get("access_token")
    if not access_token:
        verbose_proxy_logger.warning(
            "refresh_user_oauth_token: token response missing access_token for "
            "user=%s server=%s",
            user_id,
            server_id,
        )
        return None

    expires_in: Optional[int] = None
    raw_expires = body.get("expires_in")
    try:
        expires_in = int(raw_expires) if raw_expires is not None else None
    except (TypeError, ValueError):
        pass

    # Rotate refresh token when the provider returns a new one
    new_refresh_token: Optional[str] = body.get("refresh_token") or refresh_token

    raw_scope = body.get("scope")
    scopes: Optional[List[str]] = (
        raw_scope.split() if isinstance(raw_scope, str) and raw_scope else None
    ) or cred.get("scopes")

    await store_user_oauth_credential(
        prisma_client=prisma_client,
        user_id=user_id,
        server_id=server_id,
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=expires_in,
        scopes=scopes,
        skip_byok_guard=True,  # Row is already OAuth2; skip the extra find_unique check
    )

    verbose_proxy_logger.info(
        "refresh_user_oauth_token: refreshed token for user=%s server=%s",
        user_id,
        server_id,
    )
    return await get_user_oauth_credential(prisma_client, user_id, server_id)


async def approve_mcp_server(
    prisma_client: PrismaClient,
    server_id: str,
    touched_by: str,
) -> LiteLLM_MCPServerTable:
    """Set approval_status=active and record reviewed_at."""
    now = datetime.now(timezone.utc)
    updated = await prisma_client.db.litellm_mcpservertable.update(
        where={"server_id": server_id},
        data={
            "approval_status": MCPApprovalStatus.active,
            "reviewed_at": now,
            "updated_by": touched_by,
        },
    )
    return LiteLLM_MCPServerTable(**updated.model_dump())


async def reject_mcp_server(
    prisma_client: PrismaClient,
    server_id: str,
    touched_by: str,
    review_notes: Optional[str] = None,
) -> LiteLLM_MCPServerTable:
    """Set approval_status=rejected, record reviewed_at and review_notes."""
    now = datetime.now(timezone.utc)
    data: Dict[str, Any] = {
        "approval_status": MCPApprovalStatus.rejected,
        "reviewed_at": now,
        "updated_by": touched_by,
    }
    if review_notes is not None:
        data["review_notes"] = review_notes
    updated = await prisma_client.db.litellm_mcpservertable.update(
        where={"server_id": server_id},
        data=data,
    )
    return LiteLLM_MCPServerTable(**updated.model_dump())


async def get_mcp_submissions(
    prisma_client: PrismaClient,
) -> MCPSubmissionsSummary:
    """
    Returns all MCP servers that were submitted by non-admin users (submitted_at IS NOT NULL),
    along with a summary count breakdown by approval_status.
    Mirrors get_guardrail_submissions() from guardrail_endpoints.py.
    """
    rows = await prisma_client.db.litellm_mcpservertable.find_many(
        where={"submitted_at": {"not": None}},
        order={"submitted_at": "desc"},
        take=500,  # safety cap; paginate if needed in a future iteration
    )
    items = [LiteLLM_MCPServerTable(**r.model_dump()) for r in rows]

    pending = sum(
        1 for i in items if i.approval_status == MCPApprovalStatus.pending_review
    )
    active = sum(1 for i in items if i.approval_status == MCPApprovalStatus.active)
    rejected = sum(1 for i in items if i.approval_status == MCPApprovalStatus.rejected)

    return MCPSubmissionsSummary(
        total=len(items),
        pending_review=pending,
        active=active,
        rejected=rejected,
        items=items,
    )
