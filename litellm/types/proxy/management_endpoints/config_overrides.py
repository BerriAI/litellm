from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HashicorpVaultConfig(BaseModel):
    """Configuration for Hashicorp Vault secret manager integration."""

    vault_addr: Optional[str] = Field(
        default=None,
        description="The address of the Vault server (e.g., https://vault.example.com:8200)",
    )
    vault_token: Optional[str] = Field(
        default=None,
        description="Token for Vault token-based authentication",
    )
    approle_role_id: Optional[str] = Field(
        default=None,
        description="Role ID for Vault AppRole authentication",
    )
    approle_secret_id: Optional[str] = Field(
        default=None,
        description="Secret ID for Vault AppRole authentication",
    )
    approle_mount_path: Optional[str] = Field(
        default=None,
        description="Mount path for the AppRole auth method (default: approle)",
    )
    client_cert: Optional[str] = Field(
        default=None,
        description="Path to the client TLS certificate for Vault",
    )
    client_key: Optional[str] = Field(
        default=None,
        description="Path to the client TLS private key for Vault",
    )
    vault_namespace: Optional[str] = Field(
        default=None,
        description="Vault namespace (for multi-tenant Vault, sent as X-Vault-Namespace header)",
    )
    vault_mount_name: Optional[str] = Field(
        default=None,
        description="KV engine mount name (default: secret)",
    )
    vault_path_prefix: Optional[str] = Field(
        default=None,
        description="Optional path prefix for secrets (e.g., myapp -> secret/data/myapp/{secret_name})",
    )


class ConfigOverrideSettingsResponse(BaseModel):
    """Response model for config override settings GET endpoints."""

    config_type: str = Field(description="The type of config override")
    values: Dict[str, Any] = Field(
        description="Current configuration values (sensitive fields decrypted)"
    )
    field_schema: Dict[str, Any] = Field(
        description="Schema information for UI rendering"
    )
