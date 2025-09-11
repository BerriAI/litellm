# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# -------------------------------------
from ._enums import (
    KeyCurveName,
    KeyExportEncryptionAlgorithm,
    KeyOperation,
    KeyRotationPolicyAction,
    KeyType,
)
from ._shared.client_base import ApiVersion
from ._models import (
    DeletedKey,
    JsonWebKey,
    KeyAttestation,
    KeyProperties,
    KeyReleasePolicy,
    KeyRotationLifetimeAction,
    KeyRotationPolicy,
    KeyVaultKey,
    KeyVaultKeyIdentifier,
    ReleaseKeyResult,
)
from ._client import KeyClient

__all__ = [
    "ApiVersion",
    "KeyClient",
    "JsonWebKey",
    "KeyAttestation",
    "KeyVaultKey",
    "KeyVaultKeyIdentifier",
    "KeyCurveName",
    "KeyExportEncryptionAlgorithm",
    "KeyOperation",
    "KeyRotationPolicyAction",
    "KeyType",
    "DeletedKey",
    "KeyProperties",
    "KeyReleasePolicy",
    "KeyRotationLifetimeAction",
    "KeyRotationPolicy",
    "ReleaseKeyResult",
]

from ._version import VERSION

__version__ = VERSION
