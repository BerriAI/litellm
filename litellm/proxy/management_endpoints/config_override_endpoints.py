import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm._uuid import uuid
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads

try:
    from prisma.errors import RecordNotFoundError
except ImportError:
    RecordNotFoundError = Exception

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._types import (
    AUDIT_ACTIONS,
    CommonProxyErrors,
    KeyManagementSystem,
    LiteLLM_AuditLogs,
    LitellmTableNames,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.table_repositories import ConfigOverridesRepository
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.proxy.management_endpoints.config_overrides import (
    ConfigOverrideSettingsResponse,
    HashicorpVaultConfig,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

router: Final = APIRouter()


class _ConfigOverrideRow(Protocol):
    @property
    def config_value(self) -> str | Mapping[str, object] | None: ...


class _ConfigOverridesTableClient(Protocol):
    async def find_unique(self, where: Mapping[str, str]) -> _ConfigOverrideRow | None: ...

    async def upsert(self, where: Mapping[str, str], data: Mapping[str, Mapping[str, str]]) -> object: ...

    async def delete(self, where: Mapping[str, str]) -> object: ...


def _config_overrides_table(prisma_client: "PrismaClient") -> _ConfigOverridesTableClient:
    return ConfigOverridesRepository(prisma_client).table


_AUDIT_REDACTED: Final = "***REDACTED***"


def _redact_config(config: Mapping[str, object] | None) -> dict[str, str]:
    """Strip values from a config snapshot before audit-log emission.

    Hashicorp Vault config carries ``vault_token``, ``approle_secret_id``,
    ``client_key`` etc.  Persisting them verbatim into ``LiteLLM_AuditLogs``
    would let anyone with read access to the audit table harvest the
    proxy's KMS credentials.  Keep keys, redact values.
    """
    if not config:
        return {}
    return {k: _AUDIT_REDACTED for k in config}


def _log_audit_task_exception(task: "asyncio.Task[None]") -> None:
    if task.cancelled():
        return
    exc: Final = task.exception()
    if exc is not None:
        verbose_proxy_logger.warning("Failed to write hashicorp-vault config audit log: %s", exc)


async def _emit_hashicorp_vault_audit_log(
    *,
    action: AUDIT_ACTIONS,
    before_config: Mapping[str, object] | None,
    after_config: Mapping[str, object] | None,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: str | None,
) -> None:
    """Emit an audit-log row for a /config_overrides/hashicorp_vault mutation.

    Mirrors the ``store_audit_logs``-gated pattern from
    ``team_callback_endpoints.py``.  Captured under
    ``LiteLLM_ConfigOverrides`` so the row co-locates with the table it
    mutates.
    """
    from litellm.proxy.management_helpers.audit_logs import (
        create_audit_log_for_update,
        is_audit_logging_enabled,
    )
    from litellm.proxy.proxy_server import litellm_proxy_admin_name

    if not is_audit_logging_enabled():
        return

    task: Final = asyncio.create_task(
        create_audit_log_for_update(
            request_data=LiteLLM_AuditLogs(
                id=str(uuid.uuid4()),
                updated_at=datetime.now(timezone.utc),
                changed_by=litellm_changed_by or user_api_key_dict.user_id or litellm_proxy_admin_name,
                changed_by_api_key=user_api_key_dict.api_key,
                table_name=LitellmTableNames.CONFIG_OVERRIDES_TABLE_NAME,
                object_id="hashicorp_vault",
                action=action,
                updated_values=json.dumps({"config": _redact_config(after_config)}, default=str),
                before_value=json.dumps({"config": _redact_config(before_config)}, default=str),
            )
        )
    )
    task.add_done_callback(_log_audit_task_exception)


# --- Hashicorp Vault constants ---

HASHICORP_ENV_VAR_MAPPING: Final[dict[str, str]] = {
    "vault_addr": "HCP_VAULT_ADDR",
    "vault_token": "HCP_VAULT_TOKEN",
    "approle_role_id": "HCP_VAULT_APPROLE_ROLE_ID",
    "approle_secret_id": "HCP_VAULT_APPROLE_SECRET_ID",
    "approle_mount_path": "HCP_VAULT_APPROLE_MOUNT_PATH",
    "client_cert": "HCP_VAULT_CLIENT_CERT",
    "client_key": "HCP_VAULT_CLIENT_KEY",
    "vault_cert_role": "HCP_VAULT_CERT_ROLE",
    "vault_namespace": "HCP_VAULT_NAMESPACE",
    "vault_mount_name": "HCP_VAULT_MOUNT_NAME",
    "vault_path_prefix": "HCP_VAULT_PATH_PREFIX",
}

HASHICORP_SENSITIVE_FIELDS: Final[set[str]] = {
    "vault_token",
    "approle_secret_id",
    "client_key",
}

_sensitive_masker: Final = SensitiveDataMasker()


# --- Shared helpers ---


def _mask_sensitive_fields(data: Mapping[str, object], sensitive_fields: set[str]) -> dict[str, object]:
    """Mask sensitive fields for API responses. Non-sensitive fields are left as-is."""
    masked: Final[dict[str, object]] = {}
    for key, value in data.items():
        if value is not None and key in sensitive_fields and isinstance(value, str):
            masked[key] = _sensitive_masker._mask_value(value)
        else:
            masked[key] = value
    return masked


def _get_current_env_values(env_var_mapping: dict[str, str]) -> dict[str, str | None]:
    """Read current env var values as fallback when no DB record exists."""
    values: Final = {}
    for field_name, env_var_name in env_var_mapping.items():
        env_value = os.environ.get(env_var_name)
        values[field_name] = env_value
    return values


class _JsonSchemaField(TypedDict, total=False):
    type: ReadOnly[str]
    anyOf: ReadOnly[Sequence["_JsonSchemaField"]]
    description: ReadOnly[str]


def _extract_field_type(field_info: _JsonSchemaField) -> str:
    """Extract the non-null type from a Pydantic v2 JSON schema field."""
    if "type" in field_info:
        return field_info["type"]
    for option in field_info.get("anyOf", []):
        if option.get("type") != "null":
            return option.get("type", "string")
    return "string"


def _build_field_schema(model_class: type[BaseModel]) -> dict[str, object]:
    """Build field_schema dict from a Pydantic model for UI rendering."""
    schema: Final = TypeAdapter(model_class).json_schema(by_alias=True)
    raw_properties: Final[Mapping[str, _JsonSchemaField]] = schema.get("properties", {})
    properties: Final = {}
    for field_name, field_info in raw_properties.items():
        properties[field_name] = {
            "description": field_info.get("description", ""),
            "type": _extract_field_type(field_info),
        }
    return {
        "description": schema.get("description", ""),
        "properties": properties,
    }


def _parse_config_value(raw: str | Mapping[str, object]) -> dict[str, object]:
    """Parse a config_value from DB (may be JSON string or dict)."""
    if isinstance(raw, str):
        return safe_json_loads(raw, default={})
    return dict(raw)


def _set_env_vars(config_data: Mapping[str, object]) -> None:
    """Set HCP_VAULT_* env vars from config data. Unsets vars for missing/None/empty fields."""
    for field_name, env_var_name in HASHICORP_ENV_VAR_MAPPING.items():
        value = config_data.get(field_name)
        if value is not None and value != "":
            os.environ[env_var_name] = str(value)
        else:
            os.environ.pop(env_var_name, None)


def _clear_hashicorp_vault_state(proxy_config: Any) -> None:
    """Clear all Hashicorp Vault state: env vars, secret manager, and change-detection cache."""
    _set_env_vars({})
    if litellm._key_management_system == KeyManagementSystem.HASHICORP_VAULT:
        litellm.secret_manager_client = None
        litellm._key_management_system = None
    proxy_config._last_hashicorp_vault_config = None


# --- Hashicorp Vault endpoints ---


@router.post(
    "/config_overrides/hashicorp_vault",
    tags=["Config Overrides"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_hashicorp_vault_config(
    config: HashicorpVaultConfig,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: str | None = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Update Hashicorp Vault secret manager configuration.
    Sets environment variables, encrypts sensitive fields, and stores in DB.
    Reinitializes the secret manager on this pod.
    """
    from litellm.proxy.proxy_server import prisma_client, proxy_config

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Only admin users can update config overrides",
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    config_data: dict[str, object] = config.model_dump(exclude_none=True)

    # Merge ALL fields the user didn't send: try DB first, fall back to env vars.
    # Omitted field = keep existing; empty string = clear/remove the field.
    existing_record: Final = await _config_overrides_table(prisma_client).find_unique(
        where={"config_type": "hashicorp_vault"}
    )
    existing_decrypted: dict[str, object] | None = None
    env_values: dict[str, str | None] = {}
    if existing_record is not None and existing_record.config_value is not None:
        existing_data: Final = _parse_config_value(existing_record.config_value)
        existing_decrypted = proxy_config._decrypt_db_variables(existing_data)
        for field in HASHICORP_ENV_VAR_MAPPING:
            if field not in config_data and existing_decrypted.get(field):
                config_data[field] = existing_decrypted[field]
    else:
        # No DB record (or DB record with null config_value) — merge from
        # current env vars instead.
        env_values = _get_current_env_values(HASHICORP_ENV_VAR_MAPPING)
        for field in HASHICORP_ENV_VAR_MAPPING:
            if field not in config_data and env_values.get(field):
                config_data[field] = env_values[field]

    # Strip empty strings — they signal "clear this field"
    config_data = {k: v for k, v in config_data.items() if v != ""}

    # Validate that the config has enough fields to initialize
    has_vault_addr: Final = bool(config_data.get("vault_addr"))
    has_token_auth: Final = bool(config_data.get("vault_token"))
    has_approle_auth: Final = bool(config_data.get("approle_role_id") and config_data.get("approle_secret_id"))
    has_tls_cert_auth: Final = bool(config_data.get("client_cert") and config_data.get("client_key"))

    if not has_vault_addr:
        raise HTTPException(
            status_code=400,
            detail="Vault Address is required",
        )

    if not has_token_auth and not has_approle_auth and not has_tls_cert_auth:
        raise HTTPException(
            status_code=400,
            detail="At least one authentication method is required: "
            "provide a Token, both AppRole Role ID and Secret ID, "
            "or both Client Certificate and Client Key",
        )

    # Snapshot current env vars so we can restore on failure
    previous_env: Final = _get_current_env_values(HASHICORP_ENV_VAR_MAPPING)

    # Set env vars and verify the secret manager can initialize before persisting
    _set_env_vars(config_data)

    try:
        proxy_config.initialize_secret_manager(key_management_system="hashicorp_vault")
    except Exception as e:
        _set_env_vars(previous_env)
        verbose_proxy_logger.exception("Error reinitializing Hashicorp Vault secret manager: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize secret manager: {e}",
        )

    # Only persist to DB after successful init
    encrypted_data: Final = proxy_config._encrypt_env_variables(config_data)
    config_value: Final = safe_dumps(encrypted_data)
    await _config_overrides_table(prisma_client).upsert(
        where={"config_type": "hashicorp_vault"},
        data={
            "create": {
                "config_type": "hashicorp_vault",
                "config_value": config_value,
            },
            "update": {
                "config_value": config_value,
            },
        },
    )

    # Update change-detection cache so the background reload doesn't redundantly re-init
    proxy_config._last_hashicorp_vault_config = safe_json_loads(config_value)

    # Mutating the proxy's KMS config affects every secret retrieval going
    # forward — emit an audit-log row so the action is traceable even
    # though the secret_manager_client itself was just swapped under us.
    # Action keys off row existence (a row with NULL ``config_value`` is
    # still an update). ``before_config`` falls back to env vars when the
    # row was absent or its ``config_value`` was NULL.
    before_config: Final = existing_decrypted if existing_decrypted is not None else env_values
    action: Final[AUDIT_ACTIONS] = "updated" if existing_record is not None else "created"
    await _emit_hashicorp_vault_audit_log(
        action=action,
        before_config=before_config,
        after_config=config_data,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=litellm_changed_by,
    )

    return {
        "message": "Hashicorp Vault configuration updated successfully",
        "status": "success",
    }


@router.get(
    "/config_overrides/hashicorp_vault",
    tags=["Config Overrides"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ConfigOverrideSettingsResponse,
)
async def get_hashicorp_vault_config(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get current Hashicorp Vault configuration.
    Returns decrypted values from DB, or falls back to current env vars.
    """
    from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view
    from litellm.proxy.proxy_server import prisma_client, proxy_config

    # Admin Viewer follows the read-parity rule.
    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail="Only admin users can view config overrides",
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    field_schema: Final = _build_field_schema(HashicorpVaultConfig)

    # Try to load from DB
    db_record: Final = await _config_overrides_table(prisma_client).find_unique(
        where={"config_type": "hashicorp_vault"}
    )

    if db_record is not None and db_record.config_value is not None:
        config_data: Final = _parse_config_value(db_record.config_value)

        # Decrypt then mask sensitive fields so plaintext secrets are never sent to the UI
        decrypted_data: Final[Mapping[str, object]] = proxy_config._decrypt_db_variables(config_data)
        masked_data: Final = _mask_sensitive_fields(decrypted_data, HASHICORP_SENSITIVE_FIELDS)

        return ConfigOverrideSettingsResponse(
            config_type="hashicorp_vault",
            values=masked_data,
            field_schema=field_schema,
        )

    # Fallback to env vars — also mask sensitive values
    env_values: Final = _get_current_env_values(HASHICORP_ENV_VAR_MAPPING)
    masked_env_values: Final = _mask_sensitive_fields(env_values, HASHICORP_SENSITIVE_FIELDS)

    return ConfigOverrideSettingsResponse(
        config_type="hashicorp_vault",
        values=masked_env_values,
        field_schema=field_schema,
    )


@router.delete(
    "/config_overrides/hashicorp_vault",
    tags=["Config Overrides"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_hashicorp_vault_config(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: str | None = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """Delete Hashicorp Vault configuration. Idempotent."""
    from litellm.proxy.proxy_server import prisma_client, proxy_config

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Only admin users can delete config overrides",
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    # Capture the prior config before delete so the audit-log row can
    # show *what* was removed (keys only — values get redacted).
    existing_record: Final = await _config_overrides_table(prisma_client).find_unique(
        where={"config_type": "hashicorp_vault"}
    )
    before_config: dict[str, object] | None = None
    if existing_record is not None and existing_record.config_value is not None:
        try:
            before_config = proxy_config._decrypt_db_variables(_parse_config_value(existing_record.config_value))
        except Exception:
            before_config = None

    # Delete DB record if it exists — ignore if not found
    deleted = False
    try:
        await _config_overrides_table(prisma_client).delete(where={"config_type": "hashicorp_vault"})
        deleted = True
    except RecordNotFoundError:
        verbose_proxy_logger.debug("No existing Hashicorp Vault config record to delete")

    _clear_hashicorp_vault_state(proxy_config)

    # Only emit audit log if a row was actually removed; an idempotent
    # delete on a non-existent row produces no security-relevant change.
    if deleted:
        await _emit_hashicorp_vault_audit_log(
            action="deleted",
            before_config=before_config,
            after_config=None,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )

    return {
        "message": "Hashicorp Vault configuration deleted successfully",
        "status": "success",
    }


@router.post(
    "/config_overrides/hashicorp_vault/test_connection",
    tags=["Config Overrides"],
    dependencies=[Depends(user_api_key_auth)],
)
async def test_hashicorp_vault_connection(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Test the connection to the currently configured Hashicorp Vault.
    Uses the already-initialized secret manager client. Does not modify any state.
    """
    from litellm.secret_managers.hashicorp_secret_manager import (
        HashicorpSecretManager,
    )

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Only admin users can test Vault connection",
        )

    client: Final = litellm.secret_manager_client
    if not isinstance(client, HashicorpSecretManager):
        raise HTTPException(
            status_code=400,
            detail="Hashicorp Vault is not configured. Save a configuration first.",
        )

    # Step 1: Authenticate (exercises AppRole login, TLS cert login, or direct token)
    try:
        headers: Final[dict[str, str]] = await asyncio.to_thread(client._get_request_headers)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Vault authentication failed: {e}",
        )

    # Step 2: Verify the token is valid via token/lookup-self
    try:
        async_client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.SecretManager)
        lookup_url: Final = f"{client.vault_addr}/v1/auth/token/lookup-self"
        if client.vault_namespace:
            headers["X-Vault-Namespace"] = client.vault_namespace
        response: Final = await async_client.get(lookup_url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Vault token validation failed: {e}",
        )

    return {
        "status": "success",
        "message": f"Successfully connected to Vault at {client.vault_addr}",
    }
