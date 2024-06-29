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
from typing import Any, Dict, Optional

import litellm
from litellm.proxy._types import KeyManagementSystem


def validate_environment():
    if "AWS_REGION_NAME" not in os.environ:
        raise ValueError("Missing required environment variable - AWS_REGION_NAME")


def load_aws_secret_manager(use_aws_secret_manager: Optional[bool]):
    if use_aws_secret_manager is None or use_aws_secret_manager == False:
        return
    try:
        import boto3
        from botocore.exceptions import ClientError

        validate_environment()

        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name="secretsmanager", region_name=os.getenv("AWS_REGION_NAME")
        )

        litellm.secret_manager_client = client
        litellm._key_management_system = KeyManagementSystem.AWS_SECRET_MANAGER

    except Exception as e:
        raise e


def load_aws_kms(use_aws_kms: Optional[bool]):
    if use_aws_kms is None or use_aws_kms is False:
        return
    try:
        import boto3

        validate_environment()

        # Create a Secrets Manager client
        kms_client = boto3.client("kms", region_name=os.getenv("AWS_REGION_NAME"))

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
        if os.getenv("LITELLM_LICENSE", None) is None:
            raise ValueError(
                "AWSKeyManagementService V2 is an Enterprise Feature. Please add a valid LITELLM_LICENSE to your envionment."
            )

    def load_aws_kms(self, use_aws_kms: Optional[bool]):
        if use_aws_kms is None or use_aws_kms is False:
            return
        try:
            import boto3

            validate_environment()

            # Create a Secrets Manager client
            kms_client = boto3.client("kms", region_name=os.getenv("AWS_REGION_NAME"))

            return kms_client
        except Exception as e:
            raise e

    def decrypt_value(self, secret_name: str) -> Any:
        if self.kms_client is None:
            raise ValueError("kms_client is None")
        encrypted_value = os.getenv(secret_name, None)
        if encrypted_value is None:
            raise Exception(
                "AWS KMS - Encrypted Value of Key={} is None".format(secret_name)
            )
        if isinstance(encrypted_value, str) and encrypted_value.startswith("aws_kms/"):
            encrypted_value = encrypted_value.replace("aws_kms/", "")

        # Decode the base64 encoded ciphertext
        ciphertext_blob = base64.b64decode(encrypted_value)

        # Set up the parameters for the decrypt call
        params = {"CiphertextBlob": ciphertext_blob}
        # Perform the decryption
        response = self.kms_client.decrypt(**params)

        # Extract and decode the plaintext
        plaintext = response["Plaintext"]
        secret = plaintext.decode("utf-8")
        if isinstance(secret, str):
            secret = secret.strip()
        try:
            secret_value_as_bool = ast.literal_eval(secret)
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


def decrypt_env_var() -> Dict[str, Any]:
    # setup client class
    aws_kms = AWSKeyManagementService_V2()
    # iterate through env - for `aws_kms/`
    new_values = {}
    for k, v in os.environ.items():
        if v is not None and isinstance(v, str) and v.startswith("aws_kms/"):
            decrypted_value = aws_kms.decrypt_value(secret_name=k)
            # reset env var
            new_values[k] = decrypted_value

    return new_values
