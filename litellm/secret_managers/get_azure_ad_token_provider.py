import os
from typing import TYPE_CHECKING, Callable, Optional

from litellm._logging import verbose_logger
from litellm.types.secret_managers.get_azure_ad_token_provider import (
    AzureCredentialType,
)

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential


def infer_credential_type_from_environment() -> AzureCredentialType:
    if (
        os.environ.get("AZURE_CLIENT_ID")
        and os.environ.get("AZURE_CLIENT_SECRET")
        and os.environ.get("AZURE_TENANT_ID")
    ):
        return AzureCredentialType.ClientSecretCredential
    elif (
        os.environ.get("AZURE_CLIENT_ID")
        and os.environ.get("AZURE_TENANT_ID")
        and os.environ.get("AZURE_FEDERATED_TOKEN_FILE")
    ):
        return AzureCredentialType.WorkloadIdentityCredential
    elif os.environ.get("AZURE_CLIENT_ID"):
        return AzureCredentialType.ManagedIdentityCredential
    elif (
        os.environ.get("AZURE_CLIENT_ID")
        and os.environ.get("AZURE_TENANT_ID")
        and os.environ.get("AZURE_CERTIFICATE_PATH")
        and os.environ.get("AZURE_CERTIFICATE_PASSWORD")
    ):
        return AzureCredentialType.CertificateCredential
    elif os.environ.get("AZURE_CERTIFICATE_PASSWORD"):
        return AzureCredentialType.CertificateCredential
    elif os.environ.get("AZURE_CERTIFICATE_PATH"):
        return AzureCredentialType.CertificateCredential
    else:
        return AzureCredentialType.DefaultAzureCredential


def get_azure_credential(
    azure_credential: Optional[AzureCredentialType] = None,
) -> "TokenCredential":
    """
    Build the Azure credential used for AD token acquisition and for authenticating
    to Azure secret managers (e.g. Key Vault).

    The credential type is chosen from, in order: the explicit ``azure_credential``
    argument, the ``AZURE_CREDENTIAL`` env var, then inference from the environment.
    ``WorkloadIdentityCredential`` supports AKS workload identity federation, reading
    ``AZURE_CLIENT_ID``, ``AZURE_TENANT_ID`` and ``AZURE_FEDERATED_TOKEN_FILE``.

    See Also:
        https://learn.microsoft.com/en-us/python/api/overview/azure/identity-readme?view=azure-python#service-principal-with-secret;
        https://azure.github.io/azure-workload-identity/docs/quick-start.html.
    """
    import azure.identity as identity
    from azure.identity import (
        CertificateCredential,
        ClientSecretCredential,
        DefaultAzureCredential,
        ManagedIdentityCredential,
        WorkloadIdentityCredential,
    )

    cred: str = (
        azure_credential.value
        if azure_credential
        else None or os.environ.get("AZURE_CREDENTIAL") or infer_credential_type_from_environment()
    )
    verbose_logger.info(f"For Azure credential, choosing credential type: {cred}")

    if cred == AzureCredentialType.ClientSecretCredential:
        return ClientSecretCredential(
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
            tenant_id=os.environ["AZURE_TENANT_ID"],
        )
    elif cred == AzureCredentialType.ManagedIdentityCredential:
        return ManagedIdentityCredential(client_id=os.environ["AZURE_CLIENT_ID"])
    elif cred == AzureCredentialType.WorkloadIdentityCredential:
        return WorkloadIdentityCredential(
            client_id=os.environ["AZURE_CLIENT_ID"],
            tenant_id=os.environ["AZURE_TENANT_ID"],
            token_file_path=os.environ["AZURE_FEDERATED_TOKEN_FILE"],
        )
    elif cred == AzureCredentialType.CertificateCredential:
        if os.getenv("AZURE_CERTIFICATE_PASSWORD"):
            return CertificateCredential(
                client_id=os.environ["AZURE_CLIENT_ID"],
                tenant_id=os.environ["AZURE_TENANT_ID"],
                certificate_path=os.environ["AZURE_CERTIFICATE_PATH"],
                password=os.environ["AZURE_CERTIFICATE_PASSWORD"],
            )
        return CertificateCredential(
            client_id=os.environ["AZURE_CLIENT_ID"],
            tenant_id=os.environ["AZURE_TENANT_ID"],
            certificate_path=os.environ["AZURE_CERTIFICATE_PATH"],
        )
    elif cred == AzureCredentialType.DefaultAzureCredential:
        # DefaultAzureCredential doesn't require explicit environment variables
        # It automatically discovers credentials from the environment (managed identity, CLI, etc.)
        return DefaultAzureCredential()

    cred_cls = getattr(identity, cred)
    return cred_cls()


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
    from azure.identity import get_bearer_token_provider

    if azure_scope is None:
        azure_scope = os.environ.get("AZURE_SCOPE") or "https://cognitiveservices.azure.com/.default"

    credential = get_azure_credential(azure_credential)

    return get_bearer_token_provider(credential, azure_scope)
