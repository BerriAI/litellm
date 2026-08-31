"""
Secret Manager Handler

Handles retrieving secrets from different secret management systems.
"""

import base64
import os
from collections.abc import Mapping
from typing import Any, Final, Generic, Protocol, TypeVar

from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import print_verbose
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem

_ClientT = TypeVar("_ClientT")


class _SecretManagerClientView(TypedDict, Generic[_ClientT]):
    """Typed read of the untyped secret manager handle configured for this key manager."""

    client: ReadOnly[_ClientT]


class _AzureKeyVaultSecret(Protocol):
    @property
    def value(self) -> str | None: ...


class _AzureKeyVaultClient(Protocol):
    def get_secret(self, name: str) -> _AzureKeyVaultSecret: ...


class _GoogleKmsDecryptResponse(Protocol):
    @property
    def plaintext(self) -> bytes: ...


class _GoogleKmsClient(Protocol):
    def decrypt(self, request: Mapping[str, object]) -> _GoogleKmsDecryptResponse: ...


class _AwsKmsClient(Protocol):
    def decrypt(self, CiphertextBlob: bytes) -> Mapping[str, bytes]: ...


class _GoogleSecretManagerClient(Protocol):
    def get_secret_from_google_secret_manager(self, secret_name: str) -> str | None: ...


class _SyncSecretReader(Protocol):
    def sync_read_secret(self, secret_name: str) -> str | None: ...


class _InfisicalSecret(Protocol):
    @property
    def secret_value(self) -> str | None: ...


class _InfisicalClient(Protocol):
    def get_secret(self, secret_name: str) -> _InfisicalSecret: ...


def _is_base64(s: str) -> bool:
    """Check if a string is valid base64."""
    import binascii

    try:
        return base64.b64encode(base64.b64decode(s)).decode() == s
    except binascii.Error:
        return False


def get_secret_from_manager(
    client: Any,
    key_manager: str,
    secret_name: str,
    key_management_settings: KeyManagementSettings | None = None,
) -> str | None:
    """
    Get a secret from the configured secret manager.

    Args:
        client: The secret manager client instance
        key_manager: The type of key manager (e.g., "azure_key_vault", "google_kms", etc.)
        secret_name: The name/path of the secret to retrieve
        key_management_settings: Optional settings for the key management system

    Returns:
        The secret value as a string, or None if not found

    Raises:
        ValueError: If the secret cannot be retrieved or required parameters are missing
        Exception: For other errors during secret retrieval
    """
    secret = None
    raw_view: Final[_SecretManagerClientView[object]] = {"client": client}
    client_object: Final = raw_view["client"]

    if (
        key_manager == KeyManagementSystem.AZURE_KEY_VAULT.value
        or type(client_object).__module__ + "." + type(client_object).__name__
        == "azure.keyvault.secrets._client.SecretClient"
    ):  # support Azure Secret Client - from azure.keyvault.secrets import SecretClient
        azure_view: Final[_SecretManagerClientView[_AzureKeyVaultClient]] = {"client": client}
        azure_client: Final = azure_view["client"]
        secret = azure_client.get_secret(secret_name).value

    elif (
        key_manager == KeyManagementSystem.GOOGLE_KMS.value
        or client_object.__class__.__name__ == "KeyManagementServiceClient"
    ):
        encrypted_secret: Final = os.getenv(secret_name)
        if encrypted_secret is None:
            raise ValueError("Google KMS requires the encrypted secret to be in the environment!")
        b64_flag: Final = _is_base64(encrypted_secret)
        if b64_flag is True:  # if passed in as encoded b64 string
            ciphertext: Final = base64.b64decode(encrypted_secret)
        else:
            raise ValueError(
                "Google KMS requires the encrypted secret to be encoded in base64"
            )  # fix for this vulnerability https://huntr.com/bounties/ae623c2f-b64b-4245-9ed4-f13a0a5824ce
        google_kms_view: Final[_SecretManagerClientView[_GoogleKmsClient]] = {"client": client}
        google_kms_client: Final = google_kms_view["client"]
        google_kms_response: Final = google_kms_client.decrypt(
            request={
                "name": litellm._google_kms_resource_name,
                "ciphertext": ciphertext,
            }
        )
        secret = google_kms_response.plaintext.decode("utf-8")  # assumes the original value was encoded with utf-8

    elif key_manager == KeyManagementSystem.AWS_KMS.value:
        """
        Only check the tokens which start with 'aws_kms/'. This prevents latency impact caused by checking all keys.
        """
        encrypted_value: Final = os.getenv(secret_name, None)
        if encrypted_value is None:
            raise Exception(f"AWS KMS - Encrypted Value of Key={secret_name} is None")
        # Decode the base64 encoded ciphertext
        ciphertext_blob: Final = base64.b64decode(encrypted_value)

        # Perform the decryption
        aws_kms_view: Final[_SecretManagerClientView[_AwsKmsClient]] = {"client": client}
        aws_kms_client: Final = aws_kms_view["client"]
        aws_kms_response: Final = aws_kms_client.decrypt(CiphertextBlob=ciphertext_blob)

        # Extract and decode the plaintext
        plaintext: Final = aws_kms_response["Plaintext"]
        secret = plaintext.decode("utf-8")
        if isinstance(secret, str):
            secret = secret.strip()

    elif key_manager == KeyManagementSystem.AWS_SECRET_MANAGER.value:
        from litellm.secret_managers.aws_secret_manager_v2 import (
            AWSSecretsManagerV2,
        )

        if isinstance(client, AWSSecretsManagerV2):
            primary_secret_name = None
            if key_management_settings is not None:
                primary_secret_name = key_management_settings.primary_secret_name

            secret = client.sync_read_secret(
                secret_name=secret_name,
                primary_secret_name=primary_secret_name,
            )
            print_verbose(f"get_secret_value_response: [set={secret is not None}]")

    elif key_manager == KeyManagementSystem.GOOGLE_SECRET_MANAGER.value:
        try:
            google_secret_manager_view: Final[_SecretManagerClientView[_GoogleSecretManagerClient]] = {"client": client}
            google_secret_manager_client: Final = google_secret_manager_view["client"]
            secret = google_secret_manager_client.get_secret_from_google_secret_manager(secret_name)
            print_verbose(f"secret from google secret manager: [set={secret is not None}]")
            if secret is None:
                raise ValueError(f"No secret found in Google Secret Manager for {secret_name}")
        except Exception as e:
            print_verbose(f"An error occurred - {e}")
            raise e

    elif key_manager == KeyManagementSystem.HASHICORP_VAULT.value:
        try:
            hashicorp_view: Final[_SecretManagerClientView[_SyncSecretReader]] = {"client": client}
            hashicorp_client: Final = hashicorp_view["client"]
            secret = hashicorp_client.sync_read_secret(secret_name=secret_name)
            if secret is None:
                raise ValueError(f"No secret found in Hashicorp Secret Manager for {secret_name}")
        except Exception as e:
            print_verbose(f"An error occurred - {e}")
            raise e

    elif key_manager == KeyManagementSystem.CYBERARK.value:
        try:
            cyberark_view: Final[_SecretManagerClientView[_SyncSecretReader]] = {"client": client}
            cyberark_client: Final = cyberark_view["client"]
            secret = cyberark_client.sync_read_secret(secret_name=secret_name)
            if secret is None:
                raise ValueError(f"No secret found in CyberArk Secret Manager for {secret_name}")
        except Exception as e:
            print_verbose(f"An error occurred - {e}")
            raise e

    elif key_manager == KeyManagementSystem.CUSTOM.value:
        # Check if client is a CustomSecretManager instance
        from litellm.integrations.custom_secret_manager import CustomSecretManager

        if isinstance(client, CustomSecretManager):
            secret = client.sync_read_secret(
                secret_name=secret_name,
                optional_params=(key_management_settings.model_dump() if key_management_settings else None),
            )
            if secret is None:
                raise ValueError(f"No secret found in Custom Secret Manager for {secret_name}")
        else:
            raise ValueError(
                "Custom secret manager client must be an instance of CustomSecretManager, "
                f"got {type(client_object).__name__}"
            )

    elif key_manager == "local":
        secret = os.getenv(secret_name)

    else:  # assume the default is infisicial client
        infisical_view: Final[_SecretManagerClientView[_InfisicalClient]] = {"client": client}
        infisical_client: Final = infisical_view["client"]
        secret = infisical_client.get_secret(secret_name).secret_value

    return secret
