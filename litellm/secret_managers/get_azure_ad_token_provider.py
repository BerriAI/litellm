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
    from azure.identity import ClientSecretCredential, get_bearer_token_provider

    try:
        credential = ClientSecretCredential(
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
            tenant_id=os.environ["AZURE_TENANT_ID"],
        )
    except KeyError as e:
        raise ValueError(
            "Missing environment variable required by Azure AD workflow."
        ) from e

    return get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default",
    )
