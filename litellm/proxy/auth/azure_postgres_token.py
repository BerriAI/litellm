import os
from typing import Any
from urllib.parse import quote

AZURE_POSTGRES_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"


def _build_azure_postgres_credential(
    azure_client_id: str | None = None,
    azure_tenant_id: str | None = None,
    azure_client_secret: str | None = None,
) -> Any:
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

    _client_id = azure_client_id or os.environ.get("AZURE_CLIENT_ID")
    _tenant_id = azure_tenant_id or os.environ.get("AZURE_TENANT_ID")
    _client_secret = azure_client_secret or os.environ.get("AZURE_CLIENT_SECRET")
    _federated_token_file = os.environ.get("AZURE_FEDERATED_TOKEN_FILE")

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
    credential: Any | None = None,
    azure_client_id: str | None = None,
    azure_tenant_id: str | None = None,
    azure_client_secret: str | None = None,
) -> str:
    if credential is None:
        credential = _build_azure_postgres_credential(
            azure_client_id=azure_client_id,
            azure_tenant_id=azure_tenant_id,
            azure_client_secret=azure_client_secret,
        )

    return quote(credential.get_token(AZURE_POSTGRES_SCOPE).token, safe="")
