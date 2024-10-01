"""
This is a file for the Google KMS integration

Relevant issue: https://github.com/BerriAI/litellm/issues/1235

Requires:
* `os.environ["GOOGLE_APPLICATION_CREDENTIALS"], os.environ["GOOGLE_KMS_RESOURCE_NAME"]`
* `pip install google-cloud-kms`
"""

import os
from typing import Optional

import litellm
from litellm.proxy._types import KeyManagementSystem


def validate_environment():
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise ValueError(
            "Missing required environment variable - GOOGLE_APPLICATION_CREDENTIALS"
        )
    if "GOOGLE_KMS_RESOURCE_NAME" not in os.environ:
        raise ValueError(
            "Missing required environment variable - GOOGLE_KMS_RESOURCE_NAME"
        )


def load_google_kms(use_google_kms: Optional[bool]):
    if use_google_kms is None or use_google_kms is False:
        return
    try:
        from google.cloud import kms_v1  # type: ignore

        validate_environment()

        # Create the KMS client
        client = kms_v1.KeyManagementServiceClient()
        litellm.secret_manager_client = client
        litellm._key_management_system = KeyManagementSystem.GOOGLE_KMS
        litellm._google_kms_resource_name = os.getenv("GOOGLE_KMS_RESOURCE_NAME")
    except Exception as e:
        raise e
