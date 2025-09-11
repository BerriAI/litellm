# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
"""Credentials for asynchronous Azure SDK clients."""

from ._credentials import (
    AuthorizationCodeCredential,
    AzureDeveloperCliCredential,
    AzureCliCredential,
    AzurePowerShellCredential,
    CertificateCredential,
    ChainedTokenCredential,
    ClientSecretCredential,
    DefaultAzureCredential,
    EnvironmentCredential,
    ManagedIdentityCredential,
    OnBehalfOfCredential,
    SharedTokenCacheCredential,
    VisualStudioCodeCredential,
    ClientAssertionCredential,
    WorkloadIdentityCredential,
)
from ._bearer_token_provider import get_bearer_token_provider


__all__ = [
    "AuthorizationCodeCredential",
    "AzureDeveloperCliCredential",
    "AzureCliCredential",
    "AzurePowerShellCredential",
    "CertificateCredential",
    "ClientSecretCredential",
    "DefaultAzureCredential",
    "EnvironmentCredential",
    "ManagedIdentityCredential",
    "OnBehalfOfCredential",
    "ChainedTokenCredential",
    "SharedTokenCacheCredential",
    "VisualStudioCodeCredential",
    "ClientAssertionCredential",
    "WorkloadIdentityCredential",
    "get_bearer_token_provider",
]
