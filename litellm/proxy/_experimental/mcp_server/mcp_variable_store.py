"""Storage backend for per-user MCP variables.

Per-user MCP variables are stored as a single encrypted JSON blob per user. By
default the blob lives in the encrypted ``LiteLLM_MCPUserVariables`` DB column.
When an admin configures a credential store (e.g. HashiCorp Vault) the blob is
stored there instead — keyed by ``{prefix}{user_id}`` — and the DB table is
unused for variables (the store is exclusive when configured).

Configuration (first match wins):
  - ``general_settings.mcp_variable_store`` in config.yaml, or
  - ``LITELLM_MCP_VARIABLE_STORE`` environment variable.

Accepted values:
  - unset / "database" / "db" -> encrypted DB column (default)
  - "hashicorp_vault" (aliases "hashicorp", "vault") -> HashiCorp Vault, configured
    via the standard ``HCP_VAULT_*`` environment variables.
  - "key_management_system" (aliases "kms", "global") -> reuse the proxy's
    globally-configured ``litellm.secret_manager_client`` (must be write-capable).

The secret key prefix defaults to ``litellm/mcp/user/`` and may be overridden with
``LITELLM_MCP_VARIABLE_STORE_PREFIX``. The stored value is the same encrypted JSON
blob used by the DB backend, so the value is opaque even to credential-store admins.
"""

import json
import os
from typing import Any, Dict, Optional

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server import db as _db
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.secret_managers.base_secret_manager import BaseSecretManager

_DEFAULT_PATH_PREFIX = "litellm/mcp/user/"

# Cache the resolved (provider, manager) so we don't rebuild it per request.
_cached_provider: Optional[str] = None
_cached_manager: Optional[BaseSecretManager] = None


def _resolve_provider() -> Optional[str]:
    """Return the normalised store provider, or ``None`` for DB storage."""
    provider: Optional[str] = None
    try:
        from litellm.proxy.proxy_server import general_settings  # noqa: PLC0415

        if general_settings:
            provider = general_settings.get("mcp_variable_store")
    except Exception:
        provider = None
    if not provider:
        provider = os.environ.get("LITELLM_MCP_VARIABLE_STORE")
    if not provider:
        return None
    normalised = str(provider).strip().lower()
    if normalised in ("", "database", "db"):
        return None
    return normalised


def _build_manager(provider: str) -> Optional[BaseSecretManager]:
    if provider in ("hashicorp_vault", "hashicorp", "vault"):
        from litellm.secret_managers.hashicorp_secret_manager import (  # noqa: PLC0415
            HashicorpSecretManager,
        )

        return HashicorpSecretManager()
    if provider in ("key_management_system", "kms", "global"):
        import litellm  # noqa: PLC0415

        client = getattr(litellm, "secret_manager_client", None)
        if isinstance(client, BaseSecretManager):
            return client
        verbose_logger.warning(
            "mcp_variable_store=key_management_system but no write-capable secret "
            "manager is configured; falling back to DB storage for MCP variables."
        )
        return None
    verbose_logger.warning(
        "Unknown mcp_variable_store provider %r; falling back to DB storage for "
        "MCP variables.",
        provider,
    )
    return None


def _get_manager() -> Optional[BaseSecretManager]:
    """Return the configured secret manager, or ``None`` when using DB storage."""
    global _cached_provider, _cached_manager
    provider = _resolve_provider()
    if provider is None:
        return None
    if provider == _cached_provider and _cached_manager is not None:
        return _cached_manager
    manager = _build_manager(provider)
    _cached_provider = provider
    _cached_manager = manager
    return manager


def _secret_key(user_id: str) -> str:
    prefix = os.environ.get("LITELLM_MCP_VARIABLE_STORE_PREFIX") or _DEFAULT_PATH_PREFIX
    return f"{prefix.rstrip('/')}/{user_id}"


def reset_cache() -> None:
    """Test hook: drop the cached secret manager so config changes take effect."""
    global _cached_provider, _cached_manager
    _cached_provider = None
    _cached_manager = None


async def get_user_variables(prisma_client: Any, user_id: str) -> Dict[str, str]:
    """Return the user's global variable dict from the active backend."""
    manager = _get_manager()
    if manager is None:
        return await _db.get_user_variables(prisma_client, user_id)
    raw = await manager.async_read_secret(secret_name=_secret_key(user_id))
    if not raw:
        return {}
    return _db._decode_user_variables(raw)


async def store_user_variables(
    prisma_client: Any, user_id: str, values: Dict[str, str]
) -> None:
    """Persist (overwrite) the user's global variable dict to the active backend."""
    manager = _get_manager()
    if manager is None:
        await _db.store_user_variables(prisma_client, user_id, values)
        return
    encoded = encrypt_value_helper(json.dumps(values))
    await manager.async_write_secret(
        secret_name=_secret_key(user_id),
        secret_value=encoded,
        description="LiteLLM per-user MCP variables",
    )


async def delete_user_variables(prisma_client: Any, user_id: str) -> None:
    """Remove the user's global variable blob from the active backend."""
    manager = _get_manager()
    if manager is None:
        await _db.delete_user_variables(prisma_client, user_id)
        return
    await manager.async_delete_secret(secret_name=_secret_key(user_id))
