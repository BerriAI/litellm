import json
import os
from typing import Any, Dict, Set

from fastapi import APIRouter, Depends, HTTPException
from prisma.errors import RecordNotFoundError
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import CommonProxyErrors, KeyManagementSystem, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.management_endpoints.config_overrides import (
    ConfigOverrideSettingsResponse,
    HashicorpVaultConfig,
)

router = APIRouter()

# --- Hashicorp Vault constants ---

HASHICORP_ENV_VAR_MAPPING: Dict[str, str] = {
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

HASHICORP_SENSITIVE_FIELDS: Set[str] = {
    "vault_token",
    "approle_role_id",
    "approle_secret_id",
    "client_key",
}

_sensitive_masker = SensitiveDataMasker()


# --- Shared helpers ---


def _mask_sensitive_fields(
    data: Dict[str, Any], sensitive_fields: Set[str]
) -> Dict[str, Any]:
    """Mask sensitive fields for API responses. Non-sensitive fields are left as-is."""
    masked = {}
    for key, value in data.items():
        if value is not None and key in sensitive_fields and isinstance(value, str):
            masked[key] = _sensitive_masker._mask_value(value)
        else:
            masked[key] = value
    return masked


def _get_current_env_values(env_var_mapping: Dict[str, str]) -> Dict[str, Any]:
    """Read current env var values as fallback when no DB record exists."""
    values = {}
    for field_name, env_var_name in env_var_mapping.items():
        env_value = os.environ.get(env_var_name)
        values[field_name] = env_value
    return values


def _extract_field_type(field_info: Dict[str, Any]) -> str:
    """Extract the non-null type from a Pydantic v2 JSON schema field."""
    if "type" in field_info:
        return field_info["type"]
    for option in field_info.get("anyOf", []):
        if option.get("type") != "null":
            return option.get("type", "string")
    return "string"


def _build_field_schema(model_class: type) -> Dict[str, Any]:
    """Build field_schema dict from a Pydantic model for UI rendering."""
    schema = TypeAdapter(model_class).json_schema(by_alias=True)
    properties = {}
    for field_name, field_info in schema.get("properties", {}).items():
        properties[field_name] = {
            "description": field_info.get("description", ""),
            "type": _extract_field_type(field_info),
        }
    return {
        "description": schema.get("description", ""),
        "properties": properties,
    }


def _parse_config_value(raw: Any) -> Dict[str, Any]:
    """Parse a config_value from DB (may be JSON string or dict)."""
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def _set_env_vars(config_data: Dict[str, Any]) -> None:
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

    config_data = config.model_dump(exclude_none=True)

    # Merge ALL fields the user didn't send: try DB first, fall back to env vars.
    # Omitted field = keep existing; empty string = clear/remove the field.
    existing_record = await prisma_client.db.litellm_configoverrides.find_unique(
        where={"config_type": "hashicorp_vault"}
    )
    if existing_record is not None and existing_record.config_value is not None:
        existing_data = _parse_config_value(existing_record.config_value)
        existing_decrypted = proxy_config._decrypt_db_variables(existing_data)
        for field in HASHICORP_ENV_VAR_MAPPING:
            if field not in config_data and existing_decrypted.get(field):
                config_data[field] = existing_decrypted[field]
    else:
        # No DB record yet — merge from current env vars
        env_values = _get_current_env_values(HASHICORP_ENV_VAR_MAPPING)
        for field in HASHICORP_ENV_VAR_MAPPING:
            if field not in config_data and env_values.get(field):
                config_data[field] = env_values[field]

    # Strip empty strings — they signal "clear this field"
    config_data = {k: v for k, v in config_data.items() if v != ""}

    # Validate that the config has enough fields to initialize
    has_vault_addr = bool(config_data.get("vault_addr"))
    has_token_auth = bool(config_data.get("vault_token"))
    has_approle_auth = bool(
        config_data.get("approle_role_id") and config_data.get("approle_secret_id")
    )
    has_tls_cert_auth = bool(
        config_data.get("client_cert") and config_data.get("client_key")
    )

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
    previous_env = _get_current_env_values(HASHICORP_ENV_VAR_MAPPING)

    # Set env vars and verify the secret manager can initialize before persisting
    _set_env_vars(config_data)

    try:
        proxy_config.initialize_secret_manager(
            key_management_system="hashicorp_vault"
        )
    except Exception as e:
        _set_env_vars(previous_env)
        verbose_proxy_logger.exception(
            "Error reinitializing Hashicorp Vault secret manager: %s", str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize secret manager",
        )

    # Only persist to DB after successful init
    encrypted_data = proxy_config._encrypt_env_variables(config_data)
    config_value = json.dumps(encrypted_data)
    await prisma_client.db.litellm_configoverrides.upsert(
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
    proxy_config._last_hashicorp_vault_config = json.loads(config_value)

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
    from litellm.proxy.proxy_server import prisma_client, proxy_config

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Only admin users can view config overrides",
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    field_schema = _build_field_schema(HashicorpVaultConfig)

    # Try to load from DB
    db_record = await prisma_client.db.litellm_configoverrides.find_unique(
        where={"config_type": "hashicorp_vault"}
    )

    if db_record is not None and db_record.config_value is not None:
        config_data = _parse_config_value(db_record.config_value)

        # Decrypt then mask sensitive fields so plaintext secrets are never sent to the UI
        decrypted_data = proxy_config._decrypt_db_variables(config_data)
        masked_data = _mask_sensitive_fields(
            decrypted_data, HASHICORP_SENSITIVE_FIELDS
        )

        return ConfigOverrideSettingsResponse(
            config_type="hashicorp_vault",
            values=masked_data,
            field_schema=field_schema,
        )

    # Fallback to env vars — also mask sensitive values
    env_values = _get_current_env_values(HASHICORP_ENV_VAR_MAPPING)
    masked_env_values = _mask_sensitive_fields(
        env_values, HASHICORP_SENSITIVE_FIELDS
    )

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

    # Delete DB record if it exists — ignore if not found
    try:
        await prisma_client.db.litellm_configoverrides.delete(
            where={"config_type": "hashicorp_vault"}
        )
    except RecordNotFoundError:
        verbose_proxy_logger.debug(
            "No existing Hashicorp Vault config record to delete"
        )

    _clear_hashicorp_vault_state(proxy_config)

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

    client = litellm.secret_manager_client
    if not isinstance(client, HashicorpSecretManager):
        raise HTTPException(
            status_code=400,
            detail="Hashicorp Vault is not configured. Save a configuration first.",
        )

    # Step 1: Authenticate (exercises AppRole login, TLS cert login, or direct token)
    try:
        headers = client._get_request_headers()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail="Vault authentication failed",
        )

    # Step 2: Verify the token is valid via token/lookup-self
    try:
        sync_client = _get_httpx_client()
        lookup_url = f"{client.vault_addr}/v1/auth/token/lookup-self"
        if client.vault_namespace:
            headers["X-Vault-Namespace"] = client.vault_namespace
        response = sync_client.get(lookup_url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail="Vault token validation failed",
        )

    return {
        "status": "success",
        "message": f"Successfully connected to Vault at {client.vault_addr}",
    }
