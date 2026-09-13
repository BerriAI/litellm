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


class CyberArkConfig(BaseModel):
    """Configuration for CyberArk Conjur secret manager integration."""

    cyberark_api_base: str | None = Field(
        default=None,
        description="The address of the CyberArk Conjur server (e.g., https://conjur.example.com)",
    )
    cyberark_account: str | None = Field(
        default=None,
        description="The Conjur organization account name",
    )
    cyberark_username: str | None = Field(
        default=None,
        description="The Conjur username (login) to authenticate as",
    )
    cyberark_api_key: str | None = Field(
        default=None,
        description="API key for Conjur API-key authentication",
    )
    client_cert: str | None = Field(
        default=None,
        description="Path to the client TLS certificate for certificate-based authentication",
    )
    client_key: str | None = Field(
        default=None,
        description="Path to the client TLS private key for certificate-based authentication",
    )
    ssl_verify: str | None = Field(
        default=None,
        description="Set to false to disable SSL verification (e.g., for self-signed certificates)",
    )
    refresh_interval: str | None = Field(
        default=None,
        description="Auth token cache TTL in seconds (default: 300)",
    )


class ConfigOverrideSettingsResponse(BaseModel):
    """Response model for config override settings GET endpoints."""

    config_type: str = Field(description="The type of config override")
    values: dict[str, Any] = Field(description="Current configuration values (sensitive fields decrypted)")
    field_schema: dict[str, Any] = Field(description="Schema information for UI rendering")
