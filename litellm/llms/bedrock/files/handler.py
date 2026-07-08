import asyncio
from collections.abc import Mapping
from typing import Any, Coroutine, Optional, Tuple, Union

import httpx

from litellm import LlmProviders
from litellm.litellm_core_utils.cloud_storage_security import (
    BEDROCK_MANAGED_S3_PREFIXES,
    should_allow_legacy_cloud_file_ids,
    validate_managed_cloud_file_id,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.openai import (
    FileContentRequest,
    HttpxBinaryResponseContent,
)

from ..base_aws_llm import BaseAWSLLM


class BedrockFilesHandler(BaseAWSLLM):
    """
    Handles downloading files from S3 for Bedrock batch processing.

    This implementation downloads files from S3 buckets where Bedrock
    stores batch output files.
    """

    def __init__(self):
        super().__init__()
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=LlmProviders.BEDROCK,
        )

    def _extract_s3_uri_from_file_id(self, file_id: str) -> str:
        from .transformation import extract_s3_uri_from_file_id

        return extract_s3_uri_from_file_id(file_id)

    def _parse_s3_uri(
        self,
        s3_uri: str,
        configured_bucket_name: str,
        allow_legacy_cloud_file_ids: bool = False,
    ) -> Tuple[str, str]:
        """
        Parse S3 URI to extract bucket name and object key.

        Args:
            s3_uri: S3 URI (e.g., "s3://bucket-name/path/to/file")

        Returns:
            Tuple of (bucket_name, object_key)
        """
        return validate_managed_cloud_file_id(
            file_id=s3_uri,
            scheme="s3://",
            configured_bucket_name=configured_bucket_name,
            allowed_object_prefixes=BEDROCK_MANAGED_S3_PREFIXES,
            allow_legacy_cloud_file_ids=allow_legacy_cloud_file_ids,
        )

    def _get_configured_s3_bucket_name(self, litellm_params: Mapping[str, object]) -> str:
        from .transformation import get_configured_s3_bucket_name

        return get_configured_s3_bucket_name(litellm_params)

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> HttpxBinaryResponseContent:
        """
        Download file content from S3 bucket for Bedrock files.

        Args:
            file_content_request: Contains file_id (encoded or S3 URI)
            optional_params: Optional parameters containing AWS credentials
            timeout: Request timeout
            max_retries: Max retry attempts

        Returns:
            HttpxBinaryResponseContent: Binary content wrapped in compatible response format
        """
        import boto3
        from botocore.credentials import Credentials

        file_id = file_content_request.get("file_id")
        if not file_id:
            raise ValueError("file_id is required in file_content_request")

        # Extract S3 URI from file ID
        s3_uri = self._extract_s3_uri_from_file_id(file_id)
        configured_bucket_name = self._get_configured_s3_bucket_name(optional_params)
        bucket_name, object_key = self._parse_s3_uri(
            s3_uri=s3_uri,
            configured_bucket_name=configured_bucket_name,
            allow_legacy_cloud_file_ids=should_allow_legacy_cloud_file_ids(optional_params),
        )

        # Get AWS credentials
        aws_region_name = self._get_aws_region_name(optional_params=optional_params, model="")
        credentials: Credentials = self.get_credentials(
            aws_access_key_id=optional_params.get("aws_access_key_id"),
            aws_secret_access_key=optional_params.get("aws_secret_access_key"),
            aws_session_token=optional_params.get("aws_session_token"),
            aws_region_name=aws_region_name,
            aws_session_name=optional_params.get("aws_session_name"),
            aws_profile_name=optional_params.get("aws_profile_name"),
            aws_role_name=optional_params.get("aws_role_name"),
            aws_web_identity_token=optional_params.get("aws_web_identity_token"),
            aws_sts_endpoint=optional_params.get("aws_sts_endpoint"),
        )

        # Create S3 client
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            region_name=aws_region_name,
            verify=self._get_ssl_verify(),
        )

        # Download file from S3
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            file_content = response["Body"].read()
        except Exception as e:
            raise ValueError(f"Failed to download file from S3: {s3_uri}. Error: {str(e)}")

        # Create mock HTTP response
        mock_response = httpx.Response(
            status_code=200,
            content=file_content,
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request(method="GET", url=s3_uri),
        )

        return HttpxBinaryResponseContent(response=mock_response)

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: Optional[str],
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[HttpxBinaryResponseContent, Coroutine[Any, Any, HttpxBinaryResponseContent]]:
        """
        Download file content from S3 bucket for Bedrock files.
        Supports both sync and async operations.

        Args:
            _is_async: Whether to run asynchronously
            file_content_request: Contains file_id (encoded or S3 URI)
            api_base: API base (unused for S3 operations)
            optional_params: Optional parameters containing AWS credentials
            timeout: Request timeout
            max_retries: Max retry attempts

        Returns:
            HttpxBinaryResponseContent or Coroutine: Binary content wrapped in compatible response format
        """
        if _is_async:
            return self.afile_content(
                file_content_request=file_content_request,
                optional_params=optional_params,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            return asyncio.run(
                self.afile_content(
                    file_content_request=file_content_request,
                    optional_params=optional_params,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )
