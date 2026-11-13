import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.single_flight import SingleFlight
from litellm.types.secret_managers.get_azure_ad_token_provider import (
    AzureCredentialType,
)

_HELD_TOKEN_EXPIRY_MARGIN_SECONDS: Final = 60

if TYPE_CHECKING:
    from azure.core.credentials import (
        AccessToken,
        AccessTokenInfo,
        SupportsTokenInfo,
        TokenCredential,
        TokenRequestOptions,
    )
    from typing_extensions import TypeIs  # noqa: TID251  # tells the two azure-core credential protocols apart


def infer_credential_type_from_environment() -> AzureCredentialType:
    if (
        os.environ.get("AZURE_CLIENT_ID")
        and os.environ.get("AZURE_CLIENT_SECRET")
        and os.environ.get("AZURE_TENANT_ID")
    ):
        return AzureCredentialType.ClientSecretCredential
    elif os.environ.get("AZURE_FEDERATED_TOKEN_FILE"):
        return AzureCredentialType.DefaultAzureCredential
    elif os.environ.get("AZURE_CLIENT_ID"):
        return AzureCredentialType.ManagedIdentityCredential
    elif (
        (
            os.environ.get("AZURE_CLIENT_ID")
            and os.environ.get("AZURE_TENANT_ID")
            and os.environ.get("AZURE_CERTIFICATE_PATH")
            and os.environ.get("AZURE_CERTIFICATE_PASSWORD")
        )
        or os.environ.get("AZURE_CERTIFICATE_PASSWORD")
        or os.environ.get("AZURE_CERTIFICATE_PATH")
    ):
        return AzureCredentialType.CertificateCredential
    else:
        return AzureCredentialType.DefaultAzureCredential


@dataclass(slots=True)
class _HeldToken:
    token: "AccessToken | AccessTokenInfo | None" = None

    def still_valid(self) -> str | None:
        held: Final = self.token
        if held is None or held.expires_on - time.time() <= _HELD_TOKEN_EXPIRY_MARGIN_SECONDS:
            return None
        return held.token


@dataclass(slots=True)
class _TokenInfoWatcher:
    credential: "SupportsTokenInfo"
    held: _HeldToken

    def get_token_info(self, *scopes: str, options: "TokenRequestOptions | None" = None) -> "AccessTokenInfo":
        token: Final = self.credential.get_token_info(*scopes, options=options)
        self.held.token = token
        return token

    def close(self) -> None:
        self.credential.close()

    def __enter__(self) -> "SupportsTokenInfo":
        return self.credential.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.credential.__exit__(exc_type, exc_value, traceback)


@dataclass(slots=True)
class _TokenWatcher:
    credential: "TokenCredential"
    held: _HeldToken

    def get_token(
        self,
        *scopes: str,
        claims: str | None = None,
        tenant_id: str | None = None,
        enable_cae: bool = False,
        **kwargs: object,  # kwargs-ok: azure-core forwards its request options verbatim to the wrapped credential
    ) -> "AccessToken":
        token: Final = self.credential.get_token(
            *scopes, claims=claims, tenant_id=tenant_id, enable_cae=enable_cae, **kwargs
        )
        self.held.token = token
        return token


def _supports_token_info(
    credential: "TokenCredential | SupportsTokenInfo",
) -> "TypeIs[SupportsTokenInfo]":
    return hasattr(credential, "get_token_info")


def single_flight_bearer_token_provider(
    credential: "TokenCredential | SupportsTokenInfo", scope: str
) -> Callable[[], str]:
    from azure.identity import get_bearer_token_provider

    held: Final = _HeldToken()
    watched: Final = (
        _TokenInfoWatcher(credential, held) if _supports_token_info(credential) else _TokenWatcher(credential, held)
    )
    provider: Final = get_bearer_token_provider(watched, scope)
    token_request: Final[SingleFlight[str]] = SingleFlight()

    def bearer_token() -> str:
        return token_request.run(provider, while_busy=held.still_valid())

    return bearer_token


def get_azure_ad_token_provider(
    azure_scope: str | None = None,
    azure_credential: AzureCredentialType | None = None,
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
    from azure import identity
    from azure.identity import (
        CertificateCredential,
        ChainedTokenCredential,
        ClientSecretCredential,
        DefaultAzureCredential,
        ManagedIdentityCredential,
        WorkloadIdentityCredential,
    )

    if azure_scope is None:
        azure_scope = os.environ.get("AZURE_SCOPE") or "https://cognitiveservices.azure.com/.default"

    cred: Final[str] = (
        azure_credential.value
        if azure_credential
        else None or os.environ.get("AZURE_CREDENTIAL") or infer_credential_type_from_environment()
    )
    verbose_logger.info("For Azure AD Token Provider, choosing credential type: %s", cred)
    credential: TokenCredential | SupportsTokenInfo | None = None
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
        credential = DefaultAzureCredential()
    elif cred == AzureCredentialType.DeploymentIdentityCredential:
        # DefaultAzureCredential cannot express this: excluding its developer credentials still
        # leaves one managed identity link, which AZURE_CLIENT_ID pins to a user assigned identity,
        # so a host running as a system assigned identity never gets asked
        workload_client_id: Final = os.environ.get("AZURE_CLIENT_ID")
        workload_tenant_id: Final = os.environ.get("AZURE_TENANT_ID")
        workload_token_file: Final = os.environ.get("AZURE_FEDERATED_TOKEN_FILE")
        credential = ChainedTokenCredential(
            *(
                (
                    WorkloadIdentityCredential(
                        client_id=workload_client_id,
                        tenant_id=workload_tenant_id,
                        token_file_path=workload_token_file,
                    ),
                )
                if workload_client_id and workload_tenant_id and workload_token_file
                else ()
            ),
            *((ManagedIdentityCredential(client_id=workload_client_id),) if workload_client_id else ()),
            ManagedIdentityCredential(),
        )
    else:
        cred_cls: Final = getattr(identity, cred)
        credential = cred_cls()

    if credential is None:
        raise ValueError("No credential provided")

    return single_flight_bearer_token_provider(credential, azure_scope)
