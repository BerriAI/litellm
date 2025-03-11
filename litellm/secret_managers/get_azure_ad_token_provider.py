import os
from typing import Callable


def get_azure_ad_token_provider() -> Callable[[], str]:
    """
    Get Azure AD token provider based on Service Principal with Secret workflow.

    Based on: https://github.com/openai/openai-python/blob/main/examples/azure_ad.py
    See Also:
        https://learn.microsoft.com/en-us/python/api/overview/azure/identity-readme?view=azure-python#service-principal-with-secret;
        https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.clientsecretcredential?view=azure-python.

    Returns:
        Callable that returns a temporary authentication token.
    """
    import azure.identity as identity
    from azure.identity import get_bearer_token_provider

    azure_scope = os.environ.get(
        "AZURE_SCOPE", "https://cognitiveservices.azure.com/.default"
    )
    cred = os.environ.get("AZURE_CREDENTIAL", "ClientSecretCredential")

    cred_cls = getattr(identity, cred)
    # ClientSecretCredential, DefaultAzureCredential, AzureCliCredential
    if cred == "ClientSecretCredential":
        credential = cred_cls(
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
            tenant_id=os.environ["AZURE_TENANT_ID"],
        )
    elif cred == "ManagedIdentityCredential":
        credential = cred_cls(client_id=os.environ["AZURE_CLIENT_ID"])
    else:
        credential = cred_cls()

    return get_bearer_token_provider(credential, azure_scope)
