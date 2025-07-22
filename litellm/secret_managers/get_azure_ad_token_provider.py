import os
from typing import Any, Callable, Optional, Union

from litellm.types.secret_managers.get_azure_ad_token_provider import (
    AzureCredentialType,
)


def get_azure_ad_token_provider(
    azure_scope: Optional[str] = None,
    azure_credential: Optional[AzureCredentialType] = None,
) -> Callable[[], str]:
    """
    Get Azure AD token provider based on Service Principal with Secret workflow.

    Based on: https://github.com/openai/openai-python/blob/main/examples/azure_ad.py
    See Also:
        https://learn.microsoft.com/en-us/python/api/overview/azure/identity-readme?view=azure-python#service-principal-with-secret;
        https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.clientsecretcredential?view=azure-python.

    Args:
        azure_scope (str, optional): The Azure scope to request token for.
                                    Defaults to environment variable AZURE_SCOPE or
                                    "https://cognitiveservices.azure.com/.default".

    Returns:
        Callable that returns a temporary authentication token.
    """
    import azure.identity as identity
    from azure.identity import (
        CertificateCredential,
        ClientSecretCredential,
        DefaultAzureCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )

    if azure_scope is None:
        azure_scope = (
            os.environ.get("AZURE_SCOPE")
            or "https://cognitiveservices.azure.com/.default"
        )

    cred: str = (
        azure_credential.value if azure_credential else None
        or os.environ.get("AZURE_CREDENTIAL", AzureCredentialType.ClientSecretCredential)
        or AzureCredentialType.ClientSecretCredential
    )
    credential: Optional[
        Union[
            ClientSecretCredential,
            ManagedIdentityCredential,
            CertificateCredential,
            DefaultAzureCredential,
            Any,
        ]
    ] = None
    if cred == AzureCredentialType.ClientSecretCredential:
        credential = ClientSecretCredential(
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
            tenant_id=os.environ["AZURE_TENANT_ID"],
        )
    elif cred == AzureCredentialType.ManagedIdentityCredential:
        credential = ManagedIdentityCredential(client_id=os.environ["AZURE_CLIENT_ID"])
    elif cred == AzureCredentialType.CertificateCredential:
        credential = CertificateCredential(
            client_id=os.environ["AZURE_CLIENT_ID"],
            tenant_id=os.environ["AZURE_TENANT_ID"],
            certificate_path=os.environ["AZURE_CERTIFICATE_PATH"],
        )
    elif cred == AzureCredentialType.DefaultAzureCredential:
        # DefaultAzureCredential doesn't require explicit environment variables
        # It automatically discovers credentials from the environment (managed identity, CLI, etc.)
        credential = DefaultAzureCredential()
    else:
        cred_cls = getattr(identity, cred)
        credential = cred_cls()

    if credential is None:
        raise ValueError("No credential provided")

    return get_bearer_token_provider(credential, azure_scope)
