"""
This is a file for the AWS Secret Manager Integration

Relevant issue: https://github.com/BerriAI/litellm/issues/1883

Requires:
* `os.environ["AWS_REGION_NAME"], 
* `pip install boto3>=1.28.57`
"""

import litellm
import os
from typing import Optional
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
