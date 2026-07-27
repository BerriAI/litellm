from urllib.parse import quote

from litellm.secret_managers.get_azure_ad_token_provider import (
    AzureTokenCredential,
    build_azure_identity_credential,
)

AZURE_POSTGRES_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"


def generate_azure_postgres_auth_token(
    credential: AzureTokenCredential | None = None,
    azure_client_id: str | None = None,
    azure_tenant_id: str | None = None,
    azure_client_secret: str | None = None,
) -> str:
    azure_credential = credential or build_azure_identity_credential(
        azure_client_id=azure_client_id,
        azure_tenant_id=azure_tenant_id,
        azure_client_secret=azure_client_secret,
    )
    return quote(azure_credential.get_token(AZURE_POSTGRES_SCOPE).token, safe="")
