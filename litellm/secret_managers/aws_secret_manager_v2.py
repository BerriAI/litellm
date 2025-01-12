"""
This is a file for the AWS Secret Manager Integration

Handles Async Operations for:
- Read Secret
- Write Secret
- Delete Secret

Relevant issue: https://github.com/BerriAI/litellm/issues/1883

Requires:
* `os.environ["AWS_REGION_NAME"], 
* `pip install boto3>=1.28.57`
"""

import json
import os
from typing import Any, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.proxy._types import KeyManagementSystem
from litellm.types.llms.custom_http import httpxSpecialProvider

from .base_secret_manager import BaseSecretManager


class AWSSecretsManagerV2(BaseAWSLLM, BaseSecretManager):
    @classmethod
    def validate_environment(cls):
        if "AWS_REGION_NAME" not in os.environ:
            raise ValueError("Missing required environment variable - AWS_REGION_NAME")

    @classmethod
    def load_aws_secret_manager(cls, use_aws_secret_manager: Optional[bool]):
        """
        Initialize AWSSecretsManagerV2 and sets litellm.secret_manager_client = AWSSecretsManagerV2() and litellm._key_management_system = KeyManagementSystem.AWS_SECRET_MANAGER
        """
        if use_aws_secret_manager is None or use_aws_secret_manager is False:
            return
        try:

            cls.validate_environment()
            litellm.secret_manager_client = cls()
            litellm._key_management_system = KeyManagementSystem.AWS_SECRET_MANAGER

        except Exception as e:
            raise e

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Async function to read a secret from AWS Secrets Manager

        Returns:
            str: Secret value
        Raises:
            ValueError: If the secret is not found or an HTTP error occurs
        """
        endpoint_url, headers, body = self._prepare_request(
            action="GetSecretValue",
            secret_name=secret_name,
            optional_params=optional_params,
        )

        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.SecretManager,
            params={"timeout": timeout},
        )

        try:
            response = await async_client.post(
                url=endpoint_url, headers=headers, data=body.decode("utf-8")
            )
            response.raise_for_status()
            return response.json()["SecretString"]
        except httpx.TimeoutException:
            raise ValueError("Timeout error occurred")
        except Exception as e:
            verbose_logger.exception(
                "Error reading secret from AWS Secrets Manager: %s", str(e)
            )
        return None

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Sync function to read a secret from AWS Secrets Manager

        Done for backwards compatibility with existing codebase, since get_secret is a sync function
        """

        # self._prepare_request uses these env vars, we cannot read them from AWS Secrets Manager. If we do we'd get stuck in an infinite loop
        if secret_name in [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION_NAME",
            "AWS_REGION",
            "AWS_BEDROCK_RUNTIME_ENDPOINT",
        ]:
            return os.getenv(secret_name)

        endpoint_url, headers, body = self._prepare_request(
            action="GetSecretValue",
            secret_name=secret_name,
            optional_params=optional_params,
        )

        sync_client = _get_httpx_client(
            params={"timeout": timeout},
        )

        try:
            response = sync_client.post(
                url=endpoint_url, headers=headers, data=body.decode("utf-8")
            )
            return response.json()["SecretString"]
        except httpx.TimeoutException:
            raise ValueError("Timeout error occurred")
        except httpx.HTTPStatusError as e:
            verbose_logger.exception(
                "Error reading secret from AWS Secrets Manager: %s",
                str(e.response.text),
            )
        except Exception as e:
            verbose_logger.exception(
                "Error reading secret from AWS Secrets Manager: %s", str(e)
            )
        return None

    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: Optional[str] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """
        Async function to write a secret to AWS Secrets Manager

        Args:
            secret_name: Name of the secret
            secret_value: Value to store (can be a JSON string)
            description: Optional description for the secret
            optional_params: Additional AWS parameters
            timeout: Request timeout
        """
        import uuid

        # Prepare the request data
        data = {"Name": secret_name, "SecretString": secret_value}
        if description:
            data["Description"] = description

        data["ClientRequestToken"] = str(uuid.uuid4())

        endpoint_url, headers, body = self._prepare_request(
            action="CreateSecret",
            secret_name=secret_name,
            secret_value=secret_value,
            optional_params=optional_params,
            request_data=data,  # Pass the complete request data
        )

        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.SecretManager,
            params={"timeout": timeout},
        )

        try:
            response = await async_client.post(
                url=endpoint_url, headers=headers, data=body.decode("utf-8")
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as err:
            raise ValueError(f"HTTP error occurred: {err.response.text}")
        except httpx.TimeoutException:
            raise ValueError("Timeout error occurred")

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """
        Async function to delete a secret from AWS Secrets Manager

        Args:
            secret_name: Name of the secret to delete
            recovery_window_in_days: Number of days before permanent deletion (default: 7)
            optional_params: Additional AWS parameters
            timeout: Request timeout

        Returns:
            dict: Response from AWS Secrets Manager containing deletion details
        """
        # Prepare the request data
        data = {
            "SecretId": secret_name,
            "RecoveryWindowInDays": recovery_window_in_days,
        }

        endpoint_url, headers, body = self._prepare_request(
            action="DeleteSecret",
            secret_name=secret_name,
            optional_params=optional_params,
            request_data=data,
        )

        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.SecretManager,
            params={"timeout": timeout},
        )

        try:
            response = await async_client.post(
                url=endpoint_url, headers=headers, data=body.decode("utf-8")
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as err:
            raise ValueError(f"HTTP error occurred: {err.response.text}")
        except httpx.TimeoutException:
            raise ValueError("Timeout error occurred")

    def _prepare_request(
        self,
        action: str,  # "GetSecretValue" or "PutSecretValue"
        secret_name: str,
        secret_value: Optional[str] = None,
        optional_params: Optional[dict] = None,
        request_data: Optional[dict] = None,
    ) -> tuple[str, Any, bytes]:
        """Prepare the AWS Secrets Manager request"""
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        optional_params = optional_params or {}
        boto3_credentials_info = self._get_boto_credentials_from_optional_params(
            optional_params
        )

        # Get endpoint
        _, endpoint_url = self.get_runtime_endpoint(
            api_base=None,
            aws_bedrock_runtime_endpoint=boto3_credentials_info.aws_bedrock_runtime_endpoint,
            aws_region_name=boto3_credentials_info.aws_region_name,
        )
        endpoint_url = endpoint_url.replace("bedrock-runtime", "secretsmanager")

        # Use provided request_data if available, otherwise build default data
        if request_data:
            data = request_data
        else:
            data = {"SecretId": secret_name}
            if secret_value and action == "PutSecretValue":
                data["SecretString"] = secret_value

        body = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": f"secretsmanager.{action}",
        }

        # Sign request
        request = AWSRequest(
            method="POST", url=endpoint_url, data=body, headers=headers
        )
        SigV4Auth(
            boto3_credentials_info.credentials,
            "secretsmanager",
            boto3_credentials_info.aws_region_name,
        ).add_auth(request)
        prepped = request.prepare()

        return endpoint_url, prepped.headers, body


# if __name__ == "__main__":
#     print("loading aws secret manager v2")
#     aws_secret_manager_v2 = AWSSecretsManagerV2()

#     print("writing secret to aws secret manager v2")
#     asyncio.run(aws_secret_manager_v2.async_write_secret(secret_name="test_secret_3", secret_value="test_value_2"))
#     print("reading secret from aws secret manager v2")
