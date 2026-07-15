import base64
import binascii
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Union, cast

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._experimental.mcp_server.auth.token_endpoint_auth import (
    build_token_endpoint_client_auth,
    normalize_token_endpoint_auth_method,
)
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    MCPApprovalStatus,
    MCPEnvVarScope,
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
from litellm.proxy.utils import PrismaClient
from litellm.repositories.object_permission_repository import ObjectPermissionRepository
from litellm.repositories.table_repositories import (
    MCPServerRepository,
    MCPUserCredentialsRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.mcp import MCPCredentials

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

_AUTH_FLOW_SCOPED_FIELDS: frozenset = frozenset(
    {
        "issuer",
        "authorization_url",
        "token_url",
        "registration_url",
        "oauth2_flow",
        "dcr_bridge",
        "token_exchange_endpoint",
        "audience",
        "subject_token_type",
        "token_exchange_profile",
    }
)

# Token-exchange settings with dedicated columns that also exist on
# ``MCPCredentials`` as a legacy shape (rows and REST callers that predate the
# columns). Every write lifts blob values into the columns and strips them from
# the stored blob, so the read-time ``column or blob`` fallback only serves rows
# the current code has never written — a cleared column can then never be
# silently resurrected by a stale blob copy. These keys are stored plaintext
# (endpoints/identifiers, not secrets), so values lift as-is.
_TOKEN_EXCHANGE_COLUMN_FIELDS: frozenset = frozenset(
    {
        "token_exchange_endpoint",
        "audience",
        "subject_token_type",
        "token_exchange_profile",
    }
)

# The client-forwarded token modes share one stored-credential shape: the admin-declared upstream
# OAuth app (client_id/client_secret) plus the same authorize relay, and neither mints anything the
# gateway keeps. So a switch WITHIN this class must preserve the stored app, unlike a cross-class
# switch (e.g. an oauth2 row whose client may be DCR-minted and is not reusable elsewhere).
_CLIENT_FORWARDED_AUTH_TYPES: frozenset = frozenset({"true_passthrough", "oauth_delegate"})

# Minted token material that must never survive a client rotation on a persisted row.
_MINTED_TOKEN_CREDENTIAL_FIELDS: frozenset = frozenset({"access_token", "refresh_token", "expires_in"})


def _credential_auth_class(auth_type: Optional[str]) -> Optional[str]:
    """Collapse the client-forwarded modes to one credential class; every other auth_type is its own
    class. Used so credential handling keys off whether the stored-credential shape actually changed,
    not off a raw auth_type inequality that treats true_passthrough<->oauth_delegate as a full reset."""
    if auth_type in _CLIENT_FORWARDED_AUTH_TYPES:
        return "client_forwarded"
    return auth_type


def _drop_stale_minted_on_client_rotation(merged: Dict[str, Any], new_creds: Dict[str, Any]) -> Dict[str, Any]:
    """When the update rotates the client, drop stale minted token keys it did not itself set, so an old
    app's access/refresh token never rides forward under the new client. A no-op when no client key changed."""
    if "client_id" not in new_creds and "client_secret" not in new_creds:
        return merged
    return {
        key: value for key, value in merged.items() if key not in _MINTED_TOKEN_CREDENTIAL_FIELDS or key in new_creds
    }


def _is_global_env_var_scope(scope: Any) -> bool:
    """``scope="user"`` entries are placeholders the user fills in; everything
    else (including a missing scope) is an admin-supplied global value."""
    return scope != MCPEnvVarScope.user and scope != "user"


def _encrypt_global_env_var_values(env_vars: Iterable[Dict[str, Any]]) -> None:
    """Encrypt ``scope="global"`` env var values in place before persisting.

    Global values hold admin-supplied secrets (API keys, passwords) that get
    interpolated into headers, so they are encrypted at rest like credentials
    and the per-user ``values_b64`` column. Per-user placeholders are not
    secrets and are stored verbatim.
    """
    for entry in env_vars:
        if not _is_global_env_var_scope(entry.get("scope")):
            continue
        value = entry.get("value")
        if value:
            entry["value"] = encrypt_value_helper(value)


def decrypt_global_env_var_values(env_vars: Optional[Iterable[Any]]) -> None:
    """Decrypt ``scope="global"`` env var values in place after reading the DB.

    Accepts ``MCPEnvVar`` models (``LiteLLM_MCPServerTable``) or plain dicts
    (raw rows / deserialized JSON). Global values are always stored encrypted,
    so a value that no longer decrypts (e.g. after a salt-key change) is dropped
    and a warning is logged rather than forwarding the ciphertext into upstream
    ``${NAME}`` headers, where it would silently fail.
    """
    if not env_vars:
        return
    for entry in env_vars:
        is_dict = isinstance(entry, dict)
        scope = entry.get("scope") if is_dict else getattr(entry, "scope", None)
        if not _is_global_env_var_scope(scope):
            continue
        value = entry.get("value") if is_dict else getattr(entry, "value", None)
        if not value:
            continue
        decrypted = decrypt_value_helper(
            value=value,
            key="mcp_global_env_var",
            exception_type="debug",
            return_original_value=False,
        )
        if decrypted is None:
            name = entry.get("name") if is_dict else getattr(entry, "name", None)
            verbose_proxy_logger.warning(
                "MCP global env var %s failed to decrypt (LITELLM_SALT_KEY "
                "changed?); dropping it so ciphertext is not sent upstream",
                name,
            )
            decrypted = ""
        if is_dict:
            entry["value"] = decrypted
        else:
            entry.value = decrypted


def _decrypt_env_vars_on_returned_row(row: Any) -> None:
    """Decrypt ``scope="global"`` env var values on a row returned by Prisma create/update.

    Prisma may hand back ``env_vars`` either as a parsed list (the common case for
    JSONB columns) or as a raw JSON string (observed for some write paths). The
    in-place decrypt helper only mutates iterables of dicts/models, so a string
    payload would silently skip decryption and ciphertext would leak into the
    registry via ``add_server``/``update_server`` (which trust the caller).
    Parse the string back to a list so the in-place decrypt actually runs, and
    write the decrypted list back onto the row so downstream consumers see plain
    values.
    """
    env_vars = getattr(row, "env_vars", None)
    if env_vars is None:
        return
    if isinstance(env_vars, str):
        try:
            env_vars = json.loads(env_vars)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(env_vars, list):
            return
        try:
            setattr(row, "env_vars", env_vars)
        except (AttributeError, TypeError):
            pass
    decrypt_global_env_var_values(env_vars)


def _reencrypt_global_env_var_values(
    env_vars: Optional[Iterable[Any]], new_encryption_key: str
) -> Optional[List[Dict[str, Any]]]:
    """Re-encrypt ``scope="global"`` env var values for master-key rotation.

    Each global value is decrypted with the current salt key and re-encrypted
    under ``new_encryption_key``. Returns the rebuilt list when at least one
    value was rotated, else ``None`` so the caller can skip the DB write. A
    value that fails to decrypt is left untouched (and logged) so a corrupt
    entry is preserved for recovery rather than overwritten.
    """
    if not env_vars:
        return None
    if isinstance(env_vars, str):
        try:
            env_vars = json.loads(env_vars)
        except (json.JSONDecodeError, TypeError):
            return None
        if not env_vars:
            return None
    rebuilt = [dict(v) for v in env_vars]
    rotated = False
    for entry in rebuilt:
        if not _is_global_env_var_scope(entry.get("scope")):
            continue
        value = entry.get("value")
        if not value:
            continue
        decrypted = decrypt_value_helper(
            value=value,
            key="mcp_global_env_var",
            exception_type="debug",
            return_original_value=False,
        )
        if decrypted is None:
            verbose_proxy_logger.warning(
                "rotate_mcp_server_credentials_master_key: could not decrypt global env var %s, skipping",
                entry.get("name"),
            )
            continue
        entry["value"] = encrypt_value_helper(decrypted, new_encryption_key=new_encryption_key)
        rotated = True
    return rebuilt if rotated else None


def _prepare_mcp_server_data(
    data: Union[NewMCPServerRequest, UpdateMCPServerRequest],
    exclude_unset: bool = False,
    fields_set: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Helper function to prepare MCP server data for database operations.
    Handles JSON field serialization for mcp_info and env fields.

    Args:
        data: NewMCPServerRequest or UpdateMCPServerRequest object
        exclude_unset: When True, only fields the caller explicitly provided are
            included. Used for partial updates (PUT /v1/mcp/server) so omitted
            fields keep their existing DB value instead of being silently reset
            to a Pydantic schema default. ``exclude_none`` is not enough here:
            non-Optional fields (e.g. ``transport=MCPTransport.sse``,
            ``mcp_access_groups=[]``, ``allow_all_keys=False``) are backfilled
            with their default when omitted, and a non-None default survives the
            ``exclude_none`` filter and overwrites the row.

    Returns:
        Dict with properly serialized JSON fields
    """
    from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

    # Convert model to dict.
    # - Partial update (exclude_unset): only caller-provided keys are emitted, so
    #   omitted fields are never written and keep their existing DB value.
    # - Create (exclude_none): drop None-valued fields and let DB defaults apply.
    if exclude_unset:
        if fields_set is None:
            fields_set = data.fields_set()
        data_dict = data.model_dump(exclude_unset=True)
        # ``validate_and_normalize_mcp_server_payload`` always assigns ``alias``
        # on the payload, which marks it as set even when the caller omitted it.
        # Drop it only when the original request omitted alias; an explicit
        # ``alias=None`` is a valid request to clear the stored alias.
        if data_dict.get("alias") is None and "alias" not in fields_set:
            data_dict.pop("alias", None)
        # Prisma ``allowed_tools`` is a required String[]; ``null`` is invalid.
        # The UI sends null to clear a whitelist — treat that as ``[]``.
        if "allowed_tools" in data_dict and data_dict["allowed_tools"] is None:
            data_dict["allowed_tools"] = []
        # Json map fields use ``@default("{}")``; explicit null means clear overrides.
        for json_map_field in (
            "tool_name_to_display_name",
            "tool_name_to_description",
        ):
            if json_map_field in data_dict and data_dict[json_map_field] is None:
                data_dict[json_map_field] = {}
    else:
        data_dict = data.model_dump(exclude_none=True)
        # Ensure alias is always present in the dict (even if None)
        if "alias" not in data_dict:
            data_dict["alias"] = getattr(data, "alias", None)

    # Handle credentials serialization
    credentials = data_dict.get("credentials")
    if credentials is not None:
        # Lift legacy blob-shaped token-exchange settings into their dedicated
        # columns (an explicit top-level value wins, including an explicit
        # null) and strip them from the blob so it never seeds the read-time
        # fallback for rows written by current code.
        for te_field in _TOKEN_EXCHANGE_COLUMN_FIELDS:
            blob_value = credentials.pop(te_field, None)
            if blob_value is not None and te_field not in data_dict:
                data_dict[te_field] = blob_value
        data_dict["credentials"] = encrypt_credentials(credentials=credentials, encryption_key=_get_salt_key())
        data_dict["credentials"] = safe_dumps(data_dict["credentials"])

    # Serialize JSON fields from ``data_dict`` (not ``data``) so the
    # exclude_unset filter is respected. Reading back from ``data`` would
    # reintroduce defaults (e.g. ``env={}``) for fields the caller never set.
    if data_dict.get("static_headers") is not None:
        data_dict["static_headers"] = safe_dumps(data_dict["static_headers"])

    # env_vars is read from ``data_dict`` (not ``data``) like every other JSON
    # column so the exclude_unset filter is respected: a partial update that
    # omits env_vars never overwrites the stored value. Global values are
    # encrypted at rest before serialization.
    env_vars = data_dict.get("env_vars")
    if env_vars is not None:
        serialized_env_vars = [dict(v) for v in env_vars]
        _encrypt_global_env_var_values(serialized_env_vars)
        data_dict["env_vars"] = safe_dumps(serialized_env_vars)

    if data_dict.get("mcp_info") is not None:
        data_dict["mcp_info"] = safe_dumps(data_dict["mcp_info"])

    if data_dict.get("env") is not None:
        data_dict["env"] = safe_dumps(data_dict["env"])

    if "tool_name_to_display_name" in data_dict:
        data_dict["tool_name_to_display_name"] = safe_dumps(data_dict["tool_name_to_display_name"] or {})
    if "tool_name_to_description" in data_dict:
        data_dict["tool_name_to_description"] = safe_dumps(data_dict["tool_name_to_description"] or {})

    # mcp_access_groups is already List[str], no serialization needed

    # On create, force is_byok so a False value is always written to the DB. On
    # partial update, only write it when the caller explicitly provided it.
    if not exclude_unset:
        data_dict["is_byok"] = getattr(data, "is_byok", False)

    return data_dict


def encrypt_credentials(credentials: MCPCredentials, encryption_key: Optional[str]) -> MCPCredentials:
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
        mcp_servers = await MCPServerRepository(prisma_client).table.find_many(where=where if where else {})

        tables = [LiteLLM_MCPServerTable(**mcp_server.model_dump()) for mcp_server in mcp_servers]
        for table in tables:
            decrypt_global_env_var_values(table.env_vars)
        return tables
    except Exception as e:
        verbose_proxy_logger.debug(
            "litellm.proxy._experimental.mcp_server.db.py::get_all_mcp_servers - {}".format(str(e))
        )
        return []


async def get_mcp_server(prisma_client: PrismaClient, server_id: str) -> Optional[LiteLLM_MCPServerTable]:
    """
    Returns the matching mcp server from the db iff exists
    """
    mcp_server: Optional[LiteLLM_MCPServerTable] = await MCPServerRepository(prisma_client).table.find_unique(
        where={
            "server_id": server_id,
        }
    )
    if mcp_server is None:
        return None
    table = LiteLLM_MCPServerTable(**mcp_server.model_dump())
    decrypt_global_env_var_values(table.env_vars)
    return table


async def get_mcp_servers(prisma_client: PrismaClient, server_ids: Iterable[str]) -> List[LiteLLM_MCPServerTable]:
    """
    Returns the matching mcp servers from the db with the server_ids
    """
    _mcp_servers: List[LiteLLM_MCPServerTable] = await MCPServerRepository(prisma_client).table.find_many(
        where={
            "server_id": {"in": server_ids},
        }
    )
    final_mcp_servers: List[LiteLLM_MCPServerTable] = []
    for _mcp_server in _mcp_servers:
        table = LiteLLM_MCPServerTable(**_mcp_server.model_dump())
        decrypt_global_env_var_values(table.env_vars)
        final_mcp_servers.append(table)

    return final_mcp_servers


async def get_mcp_servers_by_verificationtoken(prisma_client: PrismaClient, token: str) -> List[str]:
    """
    Returns the mcp servers from the db for the verification token
    """
    verification_token_record: LiteLLM_TeamTable = await VerificationTokenRepository(prisma_client).table.find_unique(
        where={
            "token": token,
        },
        include={
            "object_permission": True,
        },
    )

    mcp_servers: Optional[List[str]] = []
    if verification_token_record is not None and verification_token_record.object_permission is not None:
        mcp_servers = verification_token_record.object_permission.mcp_servers
    return mcp_servers or []


async def get_mcp_servers_by_team(prisma_client: PrismaClient, team_id: str) -> List[str]:
    """
    Returns the mcp servers from the db for the team id
    """
    team_record: LiteLLM_TeamTable = await TeamRepository(prisma_client).table.find_unique(
        where={
            "team_id": team_id,
        },
        include={
            "object_permission": True,
        },
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
        token_mcp_servers = await get_mcp_servers_by_verificationtoken(prisma_client, user.api_key)
        mcp_server_ids.update(token_mcp_servers)

        # check for special team membership
        if SpecialMCPServerName.all_team_servers in mcp_server_ids and user.team_id is not None:
            team_mcp_servers = await get_mcp_servers_by_team(prisma_client, user.team_id)
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
    object_permission_records = await ObjectPermissionRepository(prisma_client).table.find_many(
        where={
            "mcp_servers": {"has": mcp_server_id},
        },
        include={
            "teams": True,
            "verification_tokens": True,
        },
    )

    return object_permission_records


async def get_virtualkeys_for_mcp_server(prisma_client: PrismaClient, server_id: str) -> List:
    """
    Get all the virtual keys that have access to the mcp server
    """
    virtual_keys = await VerificationTokenRepository(prisma_client).table.find_many(
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
    prisma_client: PrismaClient,
    server_id: str,
    invalidate_token_cache: Optional[Callable[[str, str], Awaitable[None]]] = None,
) -> Optional[LiteLLM_MCPServerTable]:
    """
    Delete the mcp server from the db by server_id

    The server-row delete is the commit point. Per-user credential and env var
    rows have no FK cascade, so they are cleaned up afterwards on a best-effort
    basis: a transient failure there leaves only orphaned rows pointing at a
    now-missing server and must not turn a successful delete into a
    caller-visible error. Each table is cleaned independently so a failure on one
    still attempts the other.

    Each enumerated credential row's user also gets their cached per-user token
    invalidated (legacy cache + v2 store, via invalidate_token_cache, defaulting
    to the manager's shared invalidation): the caches are keyed by
    (user_id, server_id), so without this a re-created server reusing the same
    server_id would serve tokens minted for the deleted server until TTL.

    Returns the deleted mcp server record if it exists, otherwise None
    """
    deleted_server = await MCPServerRepository(prisma_client).table.delete(
        where={
            "server_id": server_id,
        },
    )
    if deleted_server is not None:
        credential_user_ids: List[str] = []
        try:
            credential_rows = await prisma_client.db.litellm_mcpusercredentials.find_many(
                where={"server_id": server_id}
            )
            credential_user_ids = [row.user_id for row in credential_rows]
        except Exception as e:  # noqa: BLE001 - enumeration is best-effort; cached tokens expire by TTL
            verbose_proxy_logger.warning(
                "MCP server %s deleted but per-user credential enumeration failed; cached tokens expire by TTL: %s",
                server_id,
                e,
            )
        for model, label in (
            (prisma_client.db.litellm_mcpusercredentials, "credential"),
            (prisma_client.db.litellm_mcpuserenvvars, "env var"),
        ):
            try:
                await model.delete_many(where={"server_id": server_id})
            except Exception as e:
                verbose_proxy_logger.warning(
                    "MCP server %s deleted but per-user %s cleanup failed; "
                    "orphaned rows can be removed on a later delete: %s",
                    server_id,
                    label,
                    e,
                )
        if credential_user_ids:
            if invalidate_token_cache is None:
                from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                    global_mcp_server_manager,
                )

                invalidate_token_cache = global_mcp_server_manager.invalidate_user_oauth_token_cache
            for user_id in credential_user_ids:
                await invalidate_token_cache(user_id, server_id)
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

    new_mcp_server = await MCPServerRepository(prisma_client).table.create(
        data=data_dict  # type: ignore
    )

    _decrypt_env_vars_on_returned_row(new_mcp_server)
    return new_mcp_server


async def update_mcp_server(
    prisma_client: PrismaClient,
    data: UpdateMCPServerRequest,
    touched_by: str,
    fields_set: Optional[Set[str]] = None,
) -> LiteLLM_MCPServerTable:
    """
    Update a new mcp server record in the db
    """
    import json

    from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

    # Use helper to prepare data with proper JSON serialization.
    # exclude_unset=True makes this a true partial update: fields the caller did
    # not provide are not written, so they keep their existing DB value instead
    # of being reset to a schema default (transport=sse, allow_all_keys=False...).
    data_dict = _prepare_mcp_server_data(data, exclude_unset=True, fields_set=fields_set)

    # Pre-fetch existing record once if we need it for auth_type, url, or credential logic
    existing = None
    has_credentials = "credentials" in data_dict and data_dict["credentials"] is not None
    # An explicit token-exchange column write (set or clear) also migrates the
    # legacy blob copies below, so the existing row is needed for those updates.
    explicit_te_write = bool(_TOKEN_EXCHANGE_COLUMN_FIELDS & data_dict.keys())
    url_provided = "url" in data_dict and data_dict["url"] is not None
    if data.auth_type or has_credentials or explicit_te_write or url_provided:
        existing = await MCPServerRepository(prisma_client).table.find_unique(where={"server_id": data.server_id})

    auth_type_changed = bool(
        data.auth_type
        and existing
        and _credential_auth_class(existing.auth_type) != _credential_auth_class(data.auth_type)
    )
    # A url change re-points the server at a potentially different upstream, so any discovered or
    # trust-on-first-use OAuth endpoints/issuer belong to the old upstream and must re-discover.
    url_changed = bool(url_provided and existing and existing.url != data_dict["url"])

    # Clear stale credentials when auth_type changes but no new credentials provided
    if auth_type_changed and "credentials" not in data_dict:
        data_dict["credentials"] = None

    if auth_type_changed or url_changed:
        data_dict.update({field: None for field in _AUTH_FLOW_SCOPED_FIELDS if field not in data_dict})

    # An explicit column write that does not touch credentials must still migrate
    # the row's legacy blob copies: lift values for columns the caller left
    # untouched, strip every copy from the blob. Without this, clearing a column
    # (e.g. to re-enable RFC 9728/8414 discovery) would leave the blob copy in
    # place, and the next credentials update's migrate-on-write would silently
    # repopulate the column the admin just cleared. (When credentials ARE in the
    # update, the merge below performs the same migration.)
    if explicit_te_write and "credentials" not in data_dict and existing is not None and existing.credentials:
        existing_creds = (
            json.loads(existing.credentials) if isinstance(existing.credentials, str) else dict(existing.credentials)
        )
        if _TOKEN_EXCHANGE_COLUMN_FIELDS & existing_creds.keys():
            for te_field in _TOKEN_EXCHANGE_COLUMN_FIELDS:
                legacy_value = existing_creds.pop(te_field, None)
                if legacy_value is not None and te_field not in data_dict and getattr(existing, te_field, None) is None:
                    data_dict[te_field] = legacy_value
            data_dict["credentials"] = safe_dumps(existing_creds)

    # Merge credentials: preserve existing fields not present in the update.
    # Without this, a partial credential update (e.g. changing only region)
    # would wipe encrypted secrets that the UI cannot display back.
    if "credentials" in data_dict and data_dict["credentials"] is not None:
        if existing and existing.credentials:
            # Only merge when the credential CLASS is unchanged. A cross-class switch
            # (e.g. oauth2 → api_key, or oauth2 → true_passthrough) replaces credentials
            # entirely to avoid stale secrets from the previous class lingering; a switch
            # within the client-forwarded class (true_passthrough ↔ oauth_delegate) keeps
            # the same declared app and so must merge, not replace.
            if not auth_type_changed:
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
                # New values override existing; existing keys not in update are preserved. A client
                # rotation additionally drops the previous app's stale minted token keys.
                merged = _drop_stale_minted_on_client_rotation({**existing_creds, **new_creds}, new_creds)
                # Migrate-on-write for legacy rows: token-exchange settings the
                # old blob shape carried move to their dedicated columns (unless
                # the caller set the column this update, or the row already has
                # one) and are never re-persisted in the blob. Stored plaintext,
                # so the merged value lifts as-is.
                for te_field in _TOKEN_EXCHANGE_COLUMN_FIELDS:
                    legacy_value = merged.pop(te_field, None)
                    if (
                        legacy_value is not None
                        and te_field not in data_dict
                        and getattr(existing, te_field, None) is None
                    ):
                        data_dict[te_field] = legacy_value
                data_dict["credentials"] = safe_dumps(merged)

    # Add audit fields
    data_dict["updated_by"] = touched_by

    # prisma-python rejects a raw ``None`` for a ``Json?`` field ("value is required but not set"); the
    # clear paths above use ``None`` as the merge-skip sentinel, so translate it here to ``Json(None)``,
    # which writes SQL null and reads back as ``None``. Done at the edge so the merge guards stay simple.
    if "credentials" in data_dict and data_dict["credentials"] is None:
        from prisma import Json  # noqa: PLC0415  # local import: prisma may be ungenerated at module load in some tools

        data_dict["credentials"] = Json(None)

    updated_mcp_server = await MCPServerRepository(prisma_client).table.update(
        where={"server_id": data.server_id},
        data=data_dict,  # type: ignore
    )

    _decrypt_env_vars_on_returned_row(updated_mcp_server)
    return updated_mcp_server


async def rotate_mcp_server_credentials_master_key(prisma_client: PrismaClient, touched_by: str, new_master_key: str):
    from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

    mcp_servers = await MCPServerRepository(prisma_client).table.find_many()

    updated = 0
    for mcp_server in mcp_servers:
        update_data: Dict[str, Any] = {}

        credentials = mcp_server.credentials
        if credentials:
            # Decrypt with current key first, then re-encrypt with new key
            decrypted_credentials = decrypt_credentials(
                credentials=cast(MCPCredentials, dict(credentials)),
            )
            encrypted_credentials = encrypt_credentials(
                credentials=decrypted_credentials,
                encryption_key=new_master_key,
            )
            update_data["credentials"] = safe_dumps(encrypted_credentials)

        rotated_env_vars = _reencrypt_global_env_var_values(mcp_server.env_vars, new_master_key)
        if rotated_env_vars is not None:
            update_data["env_vars"] = safe_dumps(rotated_env_vars)

        if not update_data:
            continue

        update_data["updated_by"] = touched_by
        await MCPServerRepository(prisma_client).table.update(
            where={"server_id": mcp_server.server_id},
            data=update_data,
        )
        updated += 1
    verbose_proxy_logger.info(
        "rotate_mcp_server_credentials_master_key: rotated %d MCP server row(s)",
        updated,
    )


def _decode_user_credential(stored: str) -> Optional[str]:
    """Read back a value persisted in ``LiteLLM_MCPUserCredentials.credential_b64``.

    Tries nacl decryption first (current write format).  Falls back to a
    plain ``urlsafe_b64decode`` for rows persisted by older code that wrote
    the credential without encryption.  Returns ``None`` when neither path
    yields a valid string.
    """
    decrypted = decrypt_value_helper(
        value=stored,
        key="mcp_user_credential",
        exception_type="debug",
        return_original_value=False,
    )
    if decrypted is not None:
        return decrypted
    try:
        return base64.urlsafe_b64decode(stored).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError, TypeError):
        return None


def _decode_oauth_payload(stored: str) -> Optional[Dict[str, Any]]:
    """Return the OAuth2 payload dict if ``stored`` holds one, else ``None``.

    A row is considered an OAuth2 credential iff its decoded value parses as
    a JSON object with ``"type": "oauth2"``.  Plain BYOK credentials (which
    share the same column) decode to a non-JSON string and return ``None``.
    """
    decoded = _decode_user_credential(stored)
    if decoded is None:
        return None
    try:
        parsed = json.loads(decoded)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, dict) and parsed.get("type") == "oauth2":
        return parsed
    return None


async def rotate_mcp_user_credentials_master_key(prisma_client: PrismaClient, new_master_key: str):
    """Re-encrypt every ``LiteLLM_MCPUserCredentials`` row with ``new_master_key``.

    Reads each ``credential_b64`` with the current salt key (falling back to
    legacy plain base64 for unmigrated rows) and writes it back encrypted
    under the new master key.  Rows that are unreadable under both paths
    are logged and skipped so one corrupt row does not abort the rotation.
    """
    rows = await MCPUserCredentialsRepository(prisma_client).table.find_many()
    rotated = 0
    skipped = 0
    for row in rows:
        plaintext = _decode_user_credential(row.credential_b64)
        if plaintext is None:
            verbose_proxy_logger.warning(
                "rotate_mcp_user_credentials_master_key: could not decode "
                "credential for user_id=%s server_id=%s, skipping",
                row.user_id,
                row.server_id,
            )
            skipped += 1
            continue
        re_encrypted = encrypt_value_helper(plaintext, new_encryption_key=new_master_key)
        await MCPUserCredentialsRepository(prisma_client).table.update(
            where={
                "user_id_server_id": {
                    "user_id": row.user_id,
                    "server_id": row.server_id,
                }
            },
            data={"credential_b64": re_encrypted},
        )
        rotated += 1
    verbose_proxy_logger.info(
        "rotate_mcp_user_credentials_master_key: rotated %d row(s), skipped %d",
        rotated,
        skipped,
    )


async def rotate_mcp_user_env_vars_master_key(prisma_client: PrismaClient, new_master_key: str):
    """Re-encrypt every ``LiteLLM_MCPUserEnvVars`` row with ``new_master_key``.

    Reads each ``values_b64`` blob with the current salt key and writes it back
    encrypted under the new master key. Rows that fail to decrypt are logged and
    skipped so one corrupt row does not abort the rotation nor overwrite values
    that may still be recoverable.
    """
    rows = await prisma_client.db.litellm_mcpuserenvvars.find_many()
    rotated = 0
    skipped = 0
    for row in rows:
        plaintext = decrypt_value_helper(
            value=row.values_b64,
            key="mcp_user_env_vars",
            exception_type="debug",
            return_original_value=False,
        )
        if plaintext is None:
            verbose_proxy_logger.warning(
                "rotate_mcp_user_env_vars_master_key: could not decrypt env vars for user_id=%s server_id=%s, skipping",
                row.user_id,
                row.server_id,
            )
            skipped += 1
            continue
        re_encrypted = encrypt_value_helper(plaintext, new_encryption_key=new_master_key)
        await prisma_client.db.litellm_mcpuserenvvars.update(
            where={
                "user_id_server_id": {
                    "user_id": row.user_id,
                    "server_id": row.server_id,
                }
            },
            data={"values_b64": re_encrypted},
        )
        rotated += 1
    verbose_proxy_logger.info(
        "rotate_mcp_user_env_vars_master_key: rotated %d row(s), skipped %d",
        rotated,
        skipped,
    )


async def store_user_credential(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
    credential: str,
) -> None:
    """Store a user credential for a BYOK MCP server."""

    encoded = encrypt_value_helper(credential)
    await MCPUserCredentialsRepository(prisma_client).table.upsert(
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

    row = await MCPUserCredentialsRepository(prisma_client).table.find_unique(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
    )
    if row is None:
        return None
    return _decode_user_credential(row.credential_b64)


async def has_user_credential(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
) -> bool:
    """Return True if the user has a stored credential for this server."""
    row = await MCPUserCredentialsRepository(prisma_client).table.find_unique(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
    )
    return row is not None


async def delete_user_credential(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
) -> None:
    """Delete the user's stored credential for a BYOK MCP server."""
    await MCPUserCredentialsRepository(prisma_client).table.delete(
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

    The payload is JSON-serialised and stored encrypted in the same
    ``credential_b64`` column used by BYOK.  A ``"type": "oauth2"`` key
    differentiates it from plain BYOK API keys.
    """

    expires_at: Optional[str] = None
    if expires_in is not None:
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()

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
    # Skip the guard when the caller knows the row is already an OAuth2 credential
    # (e.g. during token refresh), saving an extra DB round-trip.
    if not skip_byok_guard:
        existing = await MCPUserCredentialsRepository(prisma_client).table.find_unique(
            where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
        )
        if existing is not None and _decode_oauth_payload(existing.credential_b64) is None:
            # Existing row is either a BYOK secret or an OAuth2 row that no
            # longer decrypts (e.g. after a salt-key rotation).  In either
            # case, refuse to overwrite — the caller would clobber data
            # that may still be recoverable.
            raise ValueError(
                f"Existing credential for user {user_id} and server "
                f"{server_id} could not be verified as an OAuth2 token. "
                f"Refusing to overwrite."
            )

    encoded = encrypt_value_helper(json.dumps(payload))
    await MCPUserCredentialsRepository(prisma_client).table.upsert(
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


def is_oauth_credential_expired(cred: Dict[str, Any], buffer_seconds: int = 0) -> bool:
    """Return True if the OAuth2 credential's access_token has expired.

    Checks the ``expires_at`` ISO-format string stored in the credential payload.
    Returns False when ``expires_at`` is absent or unparseable (treat as non-expired).
    With ``buffer_seconds`` > 0, a token that is still valid but expires within the
    buffer is also treated as expired, so callers can refresh proactively instead of
    handing back a token that may lapse mid-request.
    """
    expires_at = cred.get("expires_at")
    if not expires_at:
        return False
    try:
        exp_dt = datetime.fromisoformat(expires_at)
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) + timedelta(seconds=buffer_seconds) > exp_dt
    except (ValueError, TypeError):
        return False


async def get_user_oauth_credential(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
) -> Optional[Dict[str, Any]]:
    """Return the decoded OAuth2 payload dict for a user+server pair, or None."""

    row = await MCPUserCredentialsRepository(prisma_client).table.find_unique(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
    )
    if row is None:
        return None
    return _decode_oauth_payload(row.credential_b64)


async def list_user_oauth_credentials(
    prisma_client: PrismaClient,
    user_id: str,
) -> List[Dict[str, Any]]:
    """Return all OAuth2 credential payloads for a user, tagged with server_id."""

    rows = await MCPUserCredentialsRepository(prisma_client).table.find_many(where={"user_id": user_id})
    results: List[Dict[str, Any]] = []
    for row in rows:
        payload = _decode_oauth_payload(row.credential_b64)
        if payload is None:
            continue
        payload["server_id"] = row.server_id
        results.append(payload)
    return results


def _decrypted_credential_field(creds: Dict[str, object], field: str) -> object:
    """Return one credential field decrypted with the global salt key; non-string and legacy
    plaintext values come back unchanged (decrypt_value_helper returns the original on failure)."""
    value = creds.get(field)
    if not isinstance(value, str):
        return value
    return decrypt_value_helper(
        value=value,
        key=field,
        exception_type="debug",
        return_original_value=True,
    )


def mcp_oauth_token_identity(server: object) -> tuple[object, ...]:
    """The upstream-OAuth-token-determining fields of an MCP server: the resource/audience (url, or
    spec_path for OpenAPI servers), the OAuth mode/grant (auth_type, oauth2_flow), the
    authorization-server endpoints, and the OAuth client + scopes. Mirrors the dashboard's
    getOAuthAuthorizationIdentity. When any of these change on a server update, previously stored
    per-user tokens were minted for the old identity and are stale. Excludes transport and
    delegate_auth_to_upstream, which do not affect what token is minted (RFC 8707/8693).

    client_id/client_secret are compared decrypted: stored values are NaCl-encrypted with a fresh
    nonce on every write, so comparing ciphertext would flag every routine save as an identity
    change and purge tokens that are still valid."""
    creds = getattr(server, "credentials", None)
    if isinstance(creds, str):
        try:
            parsed: object = json.loads(creds)
        except ValueError:
            parsed = None
    else:
        parsed = creds
    creds_dict: Dict[str, object] = parsed if isinstance(parsed, dict) else {}
    return (
        getattr(server, "url", None),
        getattr(server, "spec_path", None),
        getattr(server, "auth_type", None),
        getattr(server, "oauth2_flow", None),
        getattr(server, "issuer", None),
        getattr(server, "authorization_url", None),
        getattr(server, "token_url", None),
        getattr(server, "registration_url", None),
        _decrypted_credential_field(creds_dict, "client_id"),
        _decrypted_credential_field(creds_dict, "client_secret"),
        creds_dict.get("scopes"),
    )


async def purge_user_oauth_credentials_for_server(
    prisma_client: PrismaClient,
    server_id: str,
    invalidate_token_cache: Optional[Callable[[str, str], Awaitable[None]]] = None,
) -> int:
    """Delete every stored per-user OAuth token for a server and invalidate each user's cached
    token everywhere it can be served from (the legacy per-user token cache and the v2 per-user OAuth
    token store), so no user keeps a token minted for a superseded configuration. Called when a server
    update changes a mint-relevant field (see mcp_oauth_token_identity). Returns the number of rows
    removed.

    LiteLLM_MCPUserCredentials also stores BYOK API keys in the same column; only rows whose payload
    decodes as an OAuth2 credential (see _decode_oauth_payload) are deleted, because a config change
    only invalidates minted tokens, never a user's own stored key. Rows are therefore deleted per
    (user_id, server_id) pair rather than by a blanket server_id filter. An OAuth row inserted while
    the purge runs for a user not yet enumerated survives; a re-auth completing in the window for an
    already-enumerated user is deleted along with the stale row (the pair delete cannot tell them
    apart), which costs that user one extra re-auth and nothing else.

    invalidate_token_cache is injectable for tests; it defaults to the manager's shared
    invalidate_user_oauth_token_cache, the single invalidation point for per-user tokens."""
    repo = MCPUserCredentialsRepository(prisma_client)
    rows = await repo.table.find_many(where={"server_id": server_id})
    oauth_rows = [row for row in rows if _decode_oauth_payload(row.credential_b64) is not None]
    if not oauth_rows:
        return 0
    deleted_count = await repo.table.delete_many(
        where={"server_id": server_id, "user_id": {"in": [row.user_id for row in oauth_rows]}}
    )
    if invalidate_token_cache is None:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        invalidate_token_cache = global_mcp_server_manager.invalidate_user_oauth_token_cache

    for row in oauth_rows:
        await invalidate_token_cache(row.user_id, server_id)
    if deleted_count != len(oauth_rows):
        verbose_proxy_logger.warning(
            "MCP server %s: purge removed %d OAuth credential row(s) but %d were enumerated; "
            "row(s) were deleted concurrently during the purge",
            server_id,
            deleted_count,
            len(oauth_rows),
        )
    return deleted_count


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

    try:
        client_auth = build_token_endpoint_client_auth(
            auth_method=normalize_token_endpoint_auth_method(getattr(server, "token_endpoint_auth_method", None)),
            client_id=client_id,
            client_secret=client_secret,
        )
        token_data: Dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            **client_auth.body,
        }
        async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
        response = await async_client.post(
            token_url,
            headers={"Accept": "application/json", **client_auth.headers},
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
            "refresh_user_oauth_token: token response missing access_token for user=%s server=%s",
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
    scopes: Optional[List[str]] = (raw_scope.split() if isinstance(raw_scope, str) and raw_scope else None) or cred.get(
        "scopes"
    )

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


async def resolve_valid_user_oauth_token(
    user_id: str,
    server: Any,
    cred: Optional[Dict[str, Any]],
    prisma_client: Optional[PrismaClient] = None,
) -> Optional[Dict[str, Any]]:
    """Return an OAuth2 credential whose access_token is good for the next request.

    Returns the credential unchanged while its token is valid for at least
    ``MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS``. Only when the token is expired (or
    expiring within that buffer) and a refresh_token is stored does it mint a new one
    via ``refresh_user_oauth_token``. Returns None when there is no usable token
    (missing token, expired with no refresh_token, or a failed refresh).

    The refresh_token is only ever sent to the server's token_url inside
    ``refresh_user_oauth_token``; it is never exposed to the caller beyond the cred
    dict it already holds. ``prisma_client`` is fetched lazily and only when a refresh
    actually happens, so the valid-token path never requires a DB handle.
    """
    if not cred or not cred.get("access_token"):
        return None
    if not is_oauth_credential_expired(cred, buffer_seconds=MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS):
        return cred
    if not cred.get("refresh_token"):
        return None
    if prisma_client is None:
        from litellm.proxy.utils import get_prisma_client_or_throw

        prisma_client = get_prisma_client_or_throw("Database not connected. Cannot refresh OAuth token.")
    refreshed = await refresh_user_oauth_token(
        prisma_client=prisma_client,
        user_id=user_id,
        server=server,
        cred=cred,
    )
    if not refreshed or not refreshed.get("access_token"):
        return None
    return refreshed


async def resolve_user_oauth_access_token(
    user_id: str | None,
    server: "MCPServer",
    prefetched_creds: dict[str, dict[str, object]] | None = None,
) -> str | None:
    """Resolve a user's valid OAuth2 access token for a server: Redis cache, else DB + refresh.

    The egress token-resolution core shared by v1's header builder and the v2 ``OAuthTokenStore``
    adapter. Redis fast-path (skipped when ``prefetched_creds`` is supplied), else a DB read through
    ``resolve_valid_user_oauth_token`` (which refreshes an expired token when a ``refresh_token`` is
    stored), re-warming the Redis cache with the per-server TTL. Returns ``None`` when there is no
    usable token; any error is swallowed to ``None`` so a transient failure reads as "not
    authorized" rather than raising.
    """
    server_id = getattr(server, "server_id", None)
    if not user_id or not server_id:
        return None
    try:
        from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
            _compute_per_user_token_ttl,
            mcp_per_user_token_cache,
        )

        if prefetched_creds is None:
            cached_token = await mcp_per_user_token_cache.get(user_id, server_id)
            if cached_token is not None:
                return cached_token

        prisma_client = None
        if prefetched_creds is not None:
            cred = prefetched_creds.get(server_id)
        else:
            from litellm.proxy.utils import get_prisma_client_or_throw

            prisma_client = get_prisma_client_or_throw(
                "Database not connected. Connect a database to use OAuth2 MCP tools."
            )
            cred = await get_user_oauth_credential(prisma_client, user_id, server_id)

        if not cred or not cred.get("access_token"):
            return None

        cred = await resolve_valid_user_oauth_token(
            user_id=user_id,
            server=server,
            cred=cred,
            prisma_client=prisma_client,
        )
        if cred is None:
            # Refresh failed or token expired with no usable refresh_token — clear the stale
            # Redis entry so the next request doesn't reuse it.
            await mcp_per_user_token_cache.delete(user_id, server_id)
            return None

        access_token: str = cred["access_token"]
        if prefetched_creds is None:
            ttl = _compute_per_user_token_ttl(server, _remaining_token_seconds(cred.get("expires_at")))
            await mcp_per_user_token_cache.set(user_id, server_id, access_token, ttl)
        return access_token
    except Exception as e:
        verbose_proxy_logger.warning(
            "resolve_user_oauth_access_token: failed for user=%s server=%s: %s",
            user_id,
            server_id,
            e,
        )
    return None


def _remaining_token_seconds(expires_at: str | None) -> int | None:
    """Seconds until ``expires_at`` (ISO 8601), or None when absent/past/unparseable."""
    if not expires_at:
        return None
    try:
        exp_dt = datetime.fromisoformat(expires_at)
    except (ValueError, TypeError):
        return None
    if exp_dt.tzinfo is None:
        exp_dt = exp_dt.replace(tzinfo=timezone.utc)
    remaining = int((exp_dt - datetime.now(timezone.utc)).total_seconds())
    return remaining if remaining > 0 else None


async def get_active_submitted_mcp_server_ids_for_user(
    prisma_client: PrismaClient,
    user_id: str,
) -> list[str]:
    """Return active BYOM servers submitted by this user (creator visibility)."""
    if not user_id:
        return []

    rows = await MCPServerRepository(prisma_client).table.find_many(
        where={
            "submitted_by": user_id,
            "approval_status": MCPApprovalStatus.active,
        },
    )
    return [row.server_id for row in rows]


async def approve_mcp_server(
    prisma_client: PrismaClient,
    server_id: str,
    touched_by: str,
) -> LiteLLM_MCPServerTable:
    """Set approval_status=active and record reviewed_at."""
    now = datetime.now(timezone.utc)
    updated = await MCPServerRepository(prisma_client).table.update(
        where={"server_id": server_id},
        data={
            "approval_status": MCPApprovalStatus.active,
            "reviewed_at": now,
            "updated_by": touched_by,
        },
    )
    table = LiteLLM_MCPServerTable(**updated.model_dump())
    decrypt_global_env_var_values(table.env_vars)
    return table


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
    updated = await MCPServerRepository(prisma_client).table.update(
        where={"server_id": server_id},
        data=data,
    )
    table = LiteLLM_MCPServerTable(**updated.model_dump())
    decrypt_global_env_var_values(table.env_vars)
    return table


async def get_mcp_submissions(
    prisma_client: PrismaClient,
) -> MCPSubmissionsSummary:
    """
    Returns all MCP servers that were submitted by non-admin users (submitted_at IS NOT NULL),
    along with a summary count breakdown by approval_status.
    Mirrors get_guardrail_submissions() from guardrail_endpoints.py.
    """
    rows = await MCPServerRepository(prisma_client).table.find_many(
        where={"submitted_at": {"not": None}},
        order={"submitted_at": "desc"},
        take=500,  # safety cap; paginate if needed in a future iteration
    )
    items = [LiteLLM_MCPServerTable(**r.model_dump()) for r in rows]
    for item in items:
        decrypt_global_env_var_values(item.env_vars)

    pending = sum(1 for i in items if i.approval_status == MCPApprovalStatus.pending_review)
    active = sum(1 for i in items if i.approval_status == MCPApprovalStatus.active)
    rejected = sum(1 for i in items if i.approval_status == MCPApprovalStatus.rejected)

    return MCPSubmissionsSummary(
        total=len(items),
        pending_review=pending,
        active=active,
        rejected=rejected,
        items=items,
    )


# ── Per-user MCP environment variables ────────────────────────────────────


def _decode_user_env_vars(stored: str) -> Dict[str, str]:
    """Decrypt a ``values_b64`` blob and parse it as a flat ``{name: value}`` dict."""
    decrypted = decrypt_value_helper(
        value=stored,
        key="mcp_user_env_vars",
        exception_type="debug",
        return_original_value=False,
    )
    if decrypted is None:
        if stored:
            verbose_proxy_logger.warning(
                "MCP per-user env vars failed to decrypt (LITELLM_SALT_KEY "
                "changed?); treating as unset so the user is prompted to "
                "re-enter them rather than silently forwarding ciphertext"
            )
        return {}
    try:
        parsed = json.loads(decrypted)
    except (ValueError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


async def get_user_env_vars(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
) -> Dict[str, str]:
    """Return the calling user's env var dict for ``server_id`` (empty if none)."""
    row = await prisma_client.db.litellm_mcpuserenvvars.find_unique(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
    )
    if row is None:
        return {}
    return _decode_user_env_vars(row.values_b64)


async def get_user_env_vars_bulk(
    prisma_client: PrismaClient,
    user_id: str,
    server_ids: Iterable[str],
) -> Dict[str, Dict[str, str]]:
    """Return ``{server_id: {var_name: value}}`` for one user across many servers.

    Servers with no stored row are simply absent from the result.
    """
    ids = list(server_ids)
    if not ids:
        return {}
    rows = await prisma_client.db.litellm_mcpuserenvvars.find_many(where={"user_id": user_id, "server_id": {"in": ids}})
    return {row.server_id: _decode_user_env_vars(row.values_b64) for row in rows}


async def merge_user_env_vars(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
    updates: Dict[str, str],
    allowed_names: Iterable[str],
) -> Dict[str, str]:
    """Merge ``updates`` into the user's stored env vars for ``server_id`` and
    return the resulting set.

    The read-modify-write runs inside a transaction guarded by a
    ``(user_id, server_id)`` advisory lock so two concurrent writes from the
    same user can't drop one update. Names outside ``allowed_names`` are pruned,
    so an admin retiring a user-scoped variable also clears its stored value.
    """
    allowed = set(allowed_names)
    lock_key = int.from_bytes(
        hashlib.blake2b(f"{user_id}:{server_id}".encode(), digest_size=8).digest(),
        "big",
        signed=True,
    )
    async with prisma_client.db.tx() as tx:
        await tx.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)", lock_key)
        row = await tx.litellm_mcpuserenvvars.find_unique(
            where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
        )
        existing = _decode_user_env_vars(row.values_b64) if row is not None else {}
        merged = {k: v for k, v in {**existing, **updates}.items() if k in allowed}
        encoded = encrypt_value_helper(json.dumps(merged))
        await tx.litellm_mcpuserenvvars.upsert(
            where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}},
            data={
                "create": {
                    "user_id": user_id,
                    "server_id": server_id,
                    "values_b64": encoded,
                },
                "update": {"values_b64": encoded},
            },
        )
    return merged


async def delete_user_env_vars(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
) -> None:
    """Remove the calling user's env var values for ``server_id``.

    Uses ``delete_many`` so a missing row is a no-op; real DB errors still
    propagate to the caller instead of being silently swallowed.
    """
    await prisma_client.db.litellm_mcpuserenvvars.delete_many(where={"user_id": user_id, "server_id": server_id})
