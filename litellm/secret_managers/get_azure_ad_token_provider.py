import os
from typing import Any, Callable, Dict, Optional, Union

from litellm._logging import verbose_logger
from litellm.types.secret_managers.get_azure_ad_token_provider import (
    AzureCredentialType,
)


def infer_credential_type_from_environment() -> AzureCredentialType:
    if (
        os.environ.get("AZURE_CLIENT_ID")
        and os.environ.get("AZURE_CLIENT_SECRET")
        and os.environ.get("AZURE_TENANT_ID")
    ):
        return AzureCredentialType.ClientSecretCredential
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


def get_azure_ad_token_provider(
    azure_scope: Optional[str] = None,
    azure_credential: Optional[AzureCredentialType] = None,
    azure_default_credential_options: Optional[Dict[str, Any]] = None,
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
        azure_credential (AzureCredentialType, optional): The credential type to use.
        azure_default_credential_options (dict, optional): Options to pass to DefaultAzureCredential.
            Supports all DefaultAzureCredential kwargs including:
            - managed_identity_client_id: Client ID for user-assigned managed identity
            - exclude_cli_credential: Skip Azure CLI credential
            - exclude_managed_identity_credential: Skip managed identity
            - exclude_environment_credential: Skip environment credential
            - etc. (see azure-identity docs)

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
        azure_credential.value
        if azure_credential
        else None
        or os.environ.get("AZURE_CREDENTIAL")
        or infer_credential_type_from_environment()
    )
    verbose_logger.info(
        f"For Azure AD Token Provider, choosing credential type: {cred}"
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
        if os.getenv("AZURE_CERTIFICATE_PASSWORD"):
            credential = CertificateCredential(
                client_id=os.environ["AZURE_CLIENT_ID"],
                tenant_id=os.environ["AZURE_TENANT_ID"],
                certificate_path=os.environ["AZURE_CERTIFICATE_PATH"],
                password=os.environ["AZURE_CERTIFICATE_PASSWORD"],
            )
        else:
            credential = CertificateCredential(
                client_id=os.environ["AZURE_CLIENT_ID"],
                tenant_id=os.environ["AZURE_TENANT_ID"],
                certificate_path=os.environ["AZURE_CERTIFICATE_PATH"],
            )
    elif cred == AzureCredentialType.DefaultAzureCredential:
        # DefaultAzureCredential doesn't require explicit environment variables
        # It automatically discovers credentials from the environment (managed identity, CLI, etc.)
        # 
        # Configuration via azure_default_credential_options from litellm_params (YAML or SDK call)
        # 
        # Default timeout for CLI-based credentials to prevent slow failures on machines
        # without Azure credentials configured
        dac_options = dict(azure_default_credential_options or {})
        if "process_timeout" not in dac_options:
            dac_options["process_timeout"] = 10  # 10 second timeout for CLI credentials
        
        if dac_options:
            verbose_logger.debug(
                f"Creating DefaultAzureCredential with options: {list(dac_options.keys())}"
            )
            credential = DefaultAzureCredential(**dac_options)
        else:
            credential = DefaultAzureCredential()
    else:
        cred_cls = getattr(identity, cred)
        credential = cred_cls()

    if credential is None:
        raise ValueError("No credential provided")

    return get_bearer_token_provider(credential, azure_scope)
