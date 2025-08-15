import asyncio
from typing import Any, Coroutine, Optional, Union, TYPE_CHECKING

import httpx

from litellm import LlmProviders
from litellm.integrations.gcs_bucket.gcs_bucket_base import (
    GCSBucketBase,
    GCSLoggingConfig,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.openai import CreateFileRequest, OpenAIFileObject
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


from .transformation import BedrockJsonlFilesTransformation

bedrock_files_transformation = BedrockJsonlFilesTransformation()


class BedrockFilesHandler(BaseLLMHTTPHandler):
    """
    Handles Calling Bedrock in OpenAI Files API format v1/files/*

    This implementation uploads files on GCS Buckets
    """

    def __init__(self):
        super().__init__()
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=LlmProviders.BEDROCK,
        )

    async def async_create_file(
        self,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        litellm_params: dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> OpenAIFileObject:
        try:
            # First, prepare the S3 upload URL and object key
            bedrock_files_transformation.get_complete_file_url(
                api_base=api_base,
                api_key=None,
                litellm_params=litellm_params,
                data=create_file_data,
            )
            
            s3_upload_params = bedrock_files_transformation.transform_openai_file_content_to_bedrock_file_content(
                create_file_data=create_file_data,
                litellm_params=litellm_params,
            )

            # Actually upload the file to S3 using boto3
            try:
                import boto3
                from botocore.exceptions import ClientError
            except ImportError:
                raise ImportError("Missing boto3 to upload to S3. Run 'pip install boto3'.")

            # Get AWS credentials from the params
            import litellm
            s3_params = getattr(litellm, "s3_callback_params", {}) or {}
            
            # Create S3 client using credentials from BaseAWSLLM
            credentials = bedrock_files_transformation.get_credentials(
                aws_access_key_id=s3_params.get("s3_aws_access_key_id"),
                aws_secret_access_key=s3_params.get("s3_aws_secret_access_key"),
                aws_session_token=s3_params.get("s3_aws_session_token"),
                aws_region_name=s3_upload_params["region"],
                aws_session_name=s3_params.get("s3_aws_session_name"),
                aws_profile_name=s3_params.get("s3_aws_profile_name"),
                aws_role_name=s3_params.get("s3_aws_role_name"),
                aws_web_identity_token=s3_params.get("s3_aws_web_identity_token"),
                aws_sts_endpoint=s3_params.get("s3_aws_sts_endpoint"),
            )

            s3_client = boto3.client(
                "s3",
                region_name=s3_upload_params["region"],
                endpoint_url=s3_params.get("s3_endpoint_url"),
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                aws_session_token=credentials.token,
            )

            # Upload the file content to S3
            response = s3_client.put_object(
                Bucket=s3_upload_params["bucket"],
                Key=s3_upload_params["key"],
                Body=s3_upload_params["content"],
                ContentType=s3_upload_params["content_type"],
            )
            
            print(f"S3 upload response: {response}")

            # Create a response object that mimics httpx.Response
            class S3UploadResponse:
                def __init__(self, response_metadata):
                    self.status_code = 200 if response_metadata.get("HTTPStatusCode") == 200 else response_metadata.get("HTTPStatusCode", 500)
                    self.text = "S3 upload successful" if self.status_code == 200 else "S3 upload failed"
            
            upload_response = S3UploadResponse(response.get("ResponseMetadata", {}))
                    
        except Exception as e:
            raise self._handle_error(e=e, provider_config=bedrock_files_transformation)

        return bedrock_files_transformation.transform_s3_bucket_response_to_openai_file_object(
            model=None,
            raw_response=upload_response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
            model_id=s3_upload_params["model_id"],
        )

    def create_file(
        self,
        _is_async: bool,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        logging_obj: LiteLLMLoggingObj,
        max_retries: Optional[int],
    ) -> Union[OpenAIFileObject, Coroutine[Any, Any, OpenAIFileObject]]:
        """
        Creates a file on VertexAI GCS Bucket

        Only supported for Async litellm.acreate_file
        """

        if _is_async:
            return self.async_create_file(
                create_file_data=create_file_data,
                api_base=api_base,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            return asyncio.run(
                self.async_create_file(
                    create_file_data=create_file_data,
                    api_base=api_base,
                    litellm_params=litellm_params,
                    logging_obj=logging_obj,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )