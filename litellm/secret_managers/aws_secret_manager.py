"""
This is a file for the AWS Secret Manager Integration

Relevant issue: https://github.com/BerriAI/litellm/issues/1883

Requires:
* `os.environ["AWS_REGION_NAME"],
* `pip install boto3>=1.28.57`
"""

import ast
import base64
import os
import re
from typing import Any, Final

import litellm
from litellm.proxy._types import KeyManagementSystem


def validate_environment():
    if "AWS_REGION_NAME" not in os.environ:
        raise ValueError("Missing required environment variable - AWS_REGION_NAME")


def load_aws_kms(use_aws_kms: bool | None):
    if use_aws_kms is None or use_aws_kms is False:
        return
    try:
        import boto3

        validate_environment()

        # Create a Secrets Manager client
        kms_client: Final = boto3.client("kms", region_name=os.getenv("AWS_REGION_NAME"))

        litellm.secret_manager_client = kms_client
        litellm._key_management_system = KeyManagementSystem.AWS_KMS

    except Exception as e:
        raise e


class AWSKeyManagementService_V2:
    """
    V2 Clean Class for decrypting keys from AWS KeyManagementService
    """

    def __init__(self) -> None:
        self.validate_environment()
        self.kms_client = self.load_aws_kms(use_aws_kms=True)

    def validate_environment(
        self,
    ):
        if "AWS_REGION_NAME" not in os.environ:
            raise ValueError("Missing required environment variable - AWS_REGION_NAME")

        ## CHECK IF LICENSE IN ENV ## - premium feature
        is_litellm_license_in_env: bool = False

        if (
            os.getenv("LITELLM_LICENSE", None) is not None
            or os.getenv("LITELLM_SECRET_AWS_KMS_LITELLM_LICENSE", None) is not None
        ):
            is_litellm_license_in_env = True
        if is_litellm_license_in_env is False:
            raise ValueError(
                "AWSKeyManagementService V2 is an Enterprise Feature. Please add a valid LITELLM_LICENSE to your envionment."
            )

    def load_aws_kms(self, use_aws_kms: bool | None):
        if use_aws_kms is None or use_aws_kms is False:
            return
        try:
            import boto3

            validate_environment()

            # Create a Secrets Manager client
            kms_client: Final = boto3.client("kms", region_name=os.getenv("AWS_REGION_NAME"))

            return kms_client
        except Exception as e:
            raise e

    def decrypt_value(self, secret_name: str) -> Any:
        if self.kms_client is None:
            raise ValueError("kms_client is None")
        encrypted_value = os.getenv(secret_name, None)
        if encrypted_value is None:
            raise Exception(f"AWS KMS - Encrypted Value of Key={secret_name} is None")
        if isinstance(encrypted_value, str) and encrypted_value.startswith("aws_kms/"):
            encrypted_value = encrypted_value.replace("aws_kms/", "")

        # Decode the base64 encoded ciphertext
        ciphertext_blob: Final = base64.b64decode(encrypted_value)

        # Set up the parameters for the decrypt call
        params: Final = {"CiphertextBlob": ciphertext_blob}
        # Perform the decryption
        response: Final = self.kms_client.decrypt(**params)

        # Extract and decode the plaintext
        plaintext: Final = response["Plaintext"]
        secret = plaintext.decode("utf-8")
        if isinstance(secret, str):
            secret = secret.strip()
        try:
            secret_value_as_bool: Final = ast.literal_eval(secret)
            if isinstance(secret_value_as_bool, bool):
                return secret_value_as_bool
        except Exception:
            pass

        return secret


"""
- look for all values in the env with `aws_kms/<hashed_key>` 
- decrypt keys 
- rewrite env var with decrypted key (). Note: this environment variable will only be available to the current process and any child processes spawned from it. Once the Python script ends, the environment variable will not persist.
"""


def decrypt_env_var() -> dict[str, Any]:
    # setup client class
    aws_kms: Final = AWSKeyManagementService_V2()
    # iterate through env - for `aws_kms/`
    new_values: Final = {}
    for k, v in os.environ.items():
        if (k is not None and isinstance(k, str) and k.lower().startswith("litellm_secret_aws_kms")) or (
            v is not None and isinstance(v, str) and v.startswith("aws_kms/")
        ):
            decrypted_value = aws_kms.decrypt_value(secret_name=k)
            # reset env var
            k = re.sub("litellm_secret_aws_kms_", "", k, flags=re.IGNORECASE)
            new_values[k] = decrypted_value

    return new_values
