from typing import Any

from pydantic import BaseModel, Field


class HashicorpVaultConfig(BaseModel):
    """Configuration for Hashicorp Vault secret manager integration."""

    vault_addr: str | None = Field(
        default=None,
        description="The address of the Vault server (e.g., https://vault.example.com:8200)",
    )
    vault_token: str | None = Field(
        default=None,
        description="Token for Vault token-based authentication",
    )
    approle_role_id: str | None = Field(
        default=None,
        description="Role ID for Vault AppRole authentication",
    )
    approle_secret_id: str | None = Field(
        default=None,
        description="Secret ID for Vault AppRole authentication",
    )
    approle_mount_path: str | None = Field(
        default=None,
        description="Mount path for the AppRole auth method (default: approle)",
    )
    client_cert: str | None = Field(
        default=None,
        description="Path to the client TLS certificate for Vault",
    )
    client_key: str | None = Field(
        default=None,
        description="Path to the client TLS private key for Vault",
    )
    vault_cert_role: str | None = Field(
        default=None,
        description="Certificate role name for TLS cert authentication",
    )
    vault_namespace: str | None = Field(
        default=None,
        description="Vault namespace (for multi-tenant Vault, sent as X-Vault-Namespace header)",
    )
    vault_mount_name: str | None = Field(
        default=None,
        description="KV engine mount name (default: secret)",
    )
    vault_path_prefix: str | None = Field(
        default=None,
        description="Optional path prefix for secrets (e.g., myapp -> secret/data/myapp/{secret_name})",
    )


class ConfigOverrideSettingsResponse(BaseModel):
    """Response model for config override settings GET endpoints."""

    config_type: str = Field(description="The type of config override")
    values: dict[str, Any] = Field(description="Current configuration values (sensitive fields decrypted)")
    field_schema: dict[str, Any] = Field(description="Schema information for UI rendering")
