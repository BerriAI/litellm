"""
Secret Manager Handler

Handles retrieving secrets from different secret management systems.
"""
import base64
import os
from typing import Any, Optional

import litellm
from litellm._logging import print_verbose
from litellm.types.secret_managers.main import KeyManagementSystem


def _is_base64(s):
    """Check if a string is valid base64."""
    import binascii
    try:
        return base64.b64encode(base64.b64decode(s)).decode() == s
    except binascii.Error:
        return False


def get_secret_from_manager( # noqa: PLR0915
    client: Any,
    key_manager: str,
    secret_name: str,
    key_management_settings: Optional[Any] = None,
) -> Optional[str]:
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
    
    if (
        key_manager == KeyManagementSystem.AZURE_KEY_VAULT.value
        or type(client).__module__ + "." + type(client).__name__
        == "azure.keyvault.secrets._client.SecretClient"
    ):  # support Azure Secret Client - from azure.keyvault.secrets import SecretClient
        secret = client.get_secret(secret_name).value
        
    elif (
        key_manager == KeyManagementSystem.GOOGLE_KMS.value
        or client.__class__.__name__ == "KeyManagementServiceClient"
    ):
        encrypted_secret: Any = os.getenv(secret_name)
        if encrypted_secret is None:
            raise ValueError("Google KMS requires the encrypted secret to be in the environment!")
        b64_flag = _is_base64(encrypted_secret)
        if b64_flag is True:  # if passed in as encoded b64 string
            encrypted_secret = base64.b64decode(encrypted_secret)
            ciphertext = encrypted_secret
        else:
            raise ValueError(
                "Google KMS requires the encrypted secret to be encoded in base64"
            )  # fix for this vulnerability https://huntr.com/bounties/ae623c2f-b64b-4245-9ed4-f13a0a5824ce
        response = client.decrypt(
            request={
                "name": litellm._google_kms_resource_name,
                "ciphertext": ciphertext,
            }
        )
        secret = response.plaintext.decode("utf-8")  # assumes the original value was encoded with utf-8
        
    elif key_manager == KeyManagementSystem.AWS_KMS.value:
        """
        Only check the tokens which start with 'aws_kms/'. This prevents latency impact caused by checking all keys.
        """
        encrypted_value = os.getenv(secret_name, None)
        if encrypted_value is None:
            raise Exception("AWS KMS - Encrypted Value of Key={} is None".format(secret_name))
        # Decode the base64 encoded ciphertext
        ciphertext_blob = base64.b64decode(encrypted_value)

        # Set up the parameters for the decrypt call
        params = {"CiphertextBlob": ciphertext_blob}
        # Perform the decryption
        response = client.decrypt(**params)

        # Extract and decode the plaintext
        plaintext = response["Plaintext"]
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
            print_verbose(f"get_secret_value_response: {secret}")
            
    elif key_manager == KeyManagementSystem.GOOGLE_SECRET_MANAGER.value:
        try:
            secret = client.get_secret_from_google_secret_manager(secret_name)
            print_verbose(f"secret from google secret manager:  {secret}")
            if secret is None:
                raise ValueError(f"No secret found in Google Secret Manager for {secret_name}")
        except Exception as e:
            print_verbose(f"An error occurred - {str(e)}")
            raise e
            
    elif key_manager == KeyManagementSystem.HASHICORP_VAULT.value:
        try:
            secret = client.sync_read_secret(secret_name=secret_name)
            if secret is None:
                raise ValueError(f"No secret found in Hashicorp Secret Manager for {secret_name}")
        except Exception as e:
            print_verbose(f"An error occurred - {str(e)}")
            raise e
            
    elif key_manager == KeyManagementSystem.CYBERARK.value:
        try:
            secret = client.sync_read_secret(secret_name=secret_name)
            if secret is None:
                raise ValueError(f"No secret found in CyberArk Secret Manager for {secret_name}")
        except Exception as e:
            print_verbose(f"An error occurred - {str(e)}")
            raise e
            
    elif key_manager == KeyManagementSystem.CUSTOM.value:
        # Check if client is a CustomSecretManager instance
        from litellm.integrations.custom_secret_manager import CustomSecretManager
        
        if isinstance(client, CustomSecretManager):
            secret = client.sync_read_secret(
                secret_name=secret_name,
                optional_params=key_management_settings.model_dump() if key_management_settings else None,
            )
            if secret is None:
                raise ValueError(f"No secret found in Custom Secret Manager for {secret_name}")
        else:
            raise ValueError(
                f"Custom secret manager client must be an instance of CustomSecretManager, got {type(client).__name__}"
            )
        
    elif key_manager == "local":
        secret = os.getenv(secret_name)
        
    else:  # assume the default is infisicial client
        secret = client.get_secret(secret_name).secret_value
        
    return secret

