# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
from .authorization_code import AuthorizationCodeCredential
from .azure_powershell import AzurePowerShellCredential
from .browser import InteractiveBrowserCredential
from .certificate import CertificateCredential
from .chained import ChainedTokenCredential
from .client_secret import ClientSecretCredential
from .default import DefaultAzureCredential
from .environment import EnvironmentCredential
from .managed_identity import ManagedIdentityCredential
from .on_behalf_of import OnBehalfOfCredential
from .shared_cache import SharedTokenCacheCredential
from .azd_cli import AzureDeveloperCliCredential
from .azure_cli import AzureCliCredential
from .device_code import DeviceCodeCredential
from .user_password import UsernamePasswordCredential
from .vscode import VisualStudioCodeCredential
from .client_assertion import ClientAssertionCredential
from .workload_identity import WorkloadIdentityCredential


__all__ = [
    "AuthorizationCodeCredential",
    "AzureCliCredential",
    "AzureDeveloperCliCredential",
    "AzurePowerShellCredential",
    "CertificateCredential",
    "ChainedTokenCredential",
    "ClientAssertionCredential",
    "ClientSecretCredential",
    "DefaultAzureCredential",
    "DeviceCodeCredential",
    "EnvironmentCredential",
    "InteractiveBrowserCredential",
    "ManagedIdentityCredential",
    "OnBehalfOfCredential",
    "SharedTokenCacheCredential",
    "AzureCliCredential",
    "UsernamePasswordCredential",
    "WorkloadIdentityCredential",
    "VisualStudioCodeCredential",
]
