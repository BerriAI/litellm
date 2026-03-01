import json
import os
from typing import Any, Dict, Set

from fastapi import APIRouter, Depends, HTTPException
from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
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


# --- Shared helpers (reusable by future config types) ---


def _encrypt_sensitive_fields(
    data: Dict[str, Any], sensitive_fields: Set[str]
) -> Dict[str, Any]:
    """Encrypt sensitive fields in a config dict. Non-sensitive fields are left as-is."""
    encrypted = {}
    for key, value in data.items():
        if value is not None and key in sensitive_fields and isinstance(value, str):
            encrypted[key] = encrypt_value_helper(value)
        else:
            encrypted[key] = value
    return encrypted


def _decrypt_sensitive_fields(
    data: Dict[str, Any], sensitive_fields: Set[str]
) -> Dict[str, Any]:
    """Decrypt sensitive fields in a config dict. Non-sensitive fields are left as-is."""
    decrypted = {}
    for key, value in data.items():
        if value is not None and key in sensitive_fields and isinstance(value, str):
            decrypted_value = decrypt_value_helper(
                value,
                key=key,
                exception_type="debug",
                return_original_value=True,
            )
            decrypted[key] = decrypted_value
        else:
            decrypted[key] = value
    return decrypted


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
    """Set HCP_VAULT_* env vars from config data. Unsets vars for missing/None fields."""
    for field_name, env_var_name in HASHICORP_ENV_VAR_MAPPING.items():
        value = config_data.get(field_name)
        if value is not None:
            os.environ[env_var_name] = str(value)
        else:
            os.environ.pop(env_var_name, None)


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
            detail={"error": "Only admin users can update config overrides"},
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    config_data = config.model_dump(exclude_none=True)

    # Validate that the config has enough fields to initialize
    has_vault_addr = bool(config_data.get("vault_addr"))
    has_token_auth = bool(config_data.get("vault_token"))
    has_approle_auth = bool(
        config_data.get("approle_role_id") and config_data.get("approle_secret_id")
    )

    if not has_vault_addr:
        raise HTTPException(
            status_code=400,
            detail={"error": "vault_addr is required"},
        )

    if not has_token_auth and not has_approle_auth:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "At least one authentication method is required: "
                "provide vault_token, or both approle_role_id and approle_secret_id"
            },
        )

    # Encrypt sensitive fields before storing in DB
    encrypted_data = _encrypt_sensitive_fields(config_data, HASHICORP_SENSITIVE_FIELDS)

    # Upsert to DB first — only mutate env vars after DB write succeeds
    await prisma_client.db.litellm_configoverrides.upsert(
        where={"config_type": "hashicorp_vault"},
        data={
            "create": {
                "config_type": "hashicorp_vault",
                "config_value": json.dumps(encrypted_data),
            },
            "update": {
                "config_value": json.dumps(encrypted_data),
            },
        },
    )

    # Set environment variables after DB write succeeds
    _set_env_vars(config_data)

    # Reinitialize the secret manager on this pod
    try:
        proxy_config.initialize_secret_manager(
            key_management_system="hashicorp_vault"
        )
    except Exception as e:
        verbose_proxy_logger.exception(
            "Error reinitializing Hashicorp Vault secret manager: %s", str(e)
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Config saved but failed to initialize secret manager: {str(e)}"
            },
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
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": "Only admin users can view config overrides"},
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    field_schema = _build_field_schema(HashicorpVaultConfig)

    # Try to load from DB
    db_record = await prisma_client.db.litellm_configoverrides.find_unique(
        where={"config_type": "hashicorp_vault"}
    )

    if db_record is not None and db_record.config_value is not None:
        config_data = _parse_config_value(db_record.config_value)

        # Decrypt sensitive fields
        decrypted_data = _decrypt_sensitive_fields(
            config_data, HASHICORP_SENSITIVE_FIELDS
        )

        return ConfigOverrideSettingsResponse(
            config_type="hashicorp_vault",
            values=decrypted_data,
            field_schema=field_schema,
        )

    # Fallback to env vars
    env_values = _get_current_env_values(HASHICORP_ENV_VAR_MAPPING)

    return ConfigOverrideSettingsResponse(
        config_type="hashicorp_vault",
        values=env_values,
        field_schema=field_schema,
    )
