import os
from typing import Callable

from azure.identity import DefaultAzureCredential
from azure.identity import get_bearer_token_provider


def get_azure_ad_token_provider() -> Callable[[], str]:
    """
    Get Azure AD token provider based on Service Principal with Secret workflow.

    Based on: https://github.com/openai/openai-python/blob/main/examples/azure_ad.py
    See Also: https://learn.microsoft.com/en-us/python/api/overview/azure/identity-readme?view=azure-python#service-principal-with-secret.
    """
    required_keys = {"AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_CLIENT_SECRET"}
    missing_keys = required_keys.difference(os.environ)
    if missing_keys:
        raise ValueError(f"Missing environment variables required by Azure AD workflow: {missing_keys}.")
    return get_bearer_token_provider(
        # DefaultAzureCredential access environment variables:
        # AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
