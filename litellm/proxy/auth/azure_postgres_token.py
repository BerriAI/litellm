import os
from typing import Final, Protocol
from urllib.parse import quote

AZURE_POSTGRES_SCOPE: Final = "https://ossrdbms-aad.database.windows.net/.default"


class _AccessToken(Protocol):
    @property
    def token(self) -> str: ...


class _TokenCredential(Protocol):
    def get_token(self, *scopes: str) -> _AccessToken: ...


def _build_azure_postgres_credential(
    azure_client_id: str | None = None,
    azure_tenant_id: str | None = None,
    azure_client_secret: str | None = None,
) -> _TokenCredential:
    try:
        from azure.identity import (
            ClientSecretCredential,
            DefaultAzureCredential,
            ManagedIdentityCredential,
        )
    except ImportError:
        raise ImportError(
            "azure-identity is required for Azure PostgreSQL passwordless auth. "
            "Install it with: pip install azure-identity"
        )

    _client_id: Final = azure_client_id or os.environ.get("AZURE_CLIENT_ID")
    _tenant_id: Final = azure_tenant_id or os.environ.get("AZURE_TENANT_ID")
    _client_secret: Final = azure_client_secret or os.environ.get("AZURE_CLIENT_SECRET")
    _federated_token_file: Final = os.environ.get("AZURE_FEDERATED_TOKEN_FILE")

    if _client_id and _tenant_id and _client_secret:
        return ClientSecretCredential(
            client_id=_client_id,
            tenant_id=_tenant_id,
            client_secret=_client_secret,
        )
    if _federated_token_file:
        return DefaultAzureCredential()
    if _client_id:
        return ManagedIdentityCredential(client_id=_client_id)
    return DefaultAzureCredential()


def generate_azure_postgres_auth_token(
    credential: _TokenCredential | None = None,
    azure_client_id: str | None = None,
    azure_tenant_id: str | None = None,
    azure_client_secret: str | None = None,
) -> str:
    _credential: Final = credential or _build_azure_postgres_credential(
        azure_client_id=azure_client_id,
        azure_tenant_id=azure_tenant_id,
        azure_client_secret=azure_client_secret,
    )

    return quote(_credential.get_token(AZURE_POSTGRES_SCOPE).token, safe="")
