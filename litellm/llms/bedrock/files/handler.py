import asyncio
import time
from typing import TYPE_CHECKING, Any, Coroutine, Dict, Optional, Union

import httpx

from litellm import LlmProviders
from litellm._logging import verbose_logger
from litellm.llms.base_llm.files.transformation import BaseFilesConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.llms.openai import CreateFileRequest, OpenAIFileObject

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

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
        transformed_request: Union[bytes, str, dict],
        litellm_params: dict,
        provider_config: BaseFilesConfig,
        headers: dict,
        api_base: str,
        logging_obj: LiteLLMLoggingObj,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> OpenAIFileObject:
        try:
            # Convert the transformed request to the expected format
            if isinstance(transformed_request, dict):
                # Ensure we have the required fields for CreateFileRequest
                file_data = transformed_request.get("file")
                if file_data is None:
                    raise ValueError("file field is required")
                create_file_data = CreateFileRequest(
                    file=file_data,
                    purpose=transformed_request.get("purpose", "batch"),
                    **{
                        k: v
                        for k, v in transformed_request.items()
                        if k not in ["file", "purpose"]
                    },
                )
            else:
                raise ValueError("Expected dict for file creation")

            # First, prepare the S3 upload URL and object key
            bedrock_files_transformation.get_complete_file_url(
                api_base=api_base,
                api_key=None,
                litellm_params=litellm_params,
                data=create_file_data,
            )

            s3_upload_params_raw = bedrock_files_transformation.transform_openai_file_content_to_bedrock_file_content(
                create_file_data=create_file_data,
                litellm_params=litellm_params,
            )
            s3_upload_params: Dict[str, Any] = (
                s3_upload_params_raw if isinstance(s3_upload_params_raw, dict) else {}
            )

            # Actually upload the file to S3 using boto3
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "Missing boto3 to upload to S3. Run 'pip install boto3'."
                )

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

            verbose_logger.debug(f"S3 upload response: {response}")

            # Create a response object that mimics httpx.Response
            class S3UploadResponse:
                def __init__(self, response_metadata):
                    self.status_code = (
                        200
                        if response_metadata.get("HTTPStatusCode") == 200
                        else response_metadata.get("HTTPStatusCode", 500)
                    )
                    self.text = (
                        "S3 upload successful"
                        if self.status_code == 200
                        else "S3 upload failed"
                    )

            upload_response = S3UploadResponse(response.get("ResponseMetadata", {}))

        except Exception as e:
            # provider_config is not used in error handling, so we can pass a dummy object
            # Using type ignore since this is just for error handling
            raise self._handle_error(e=e, provider_config=object())  # type: ignore

        return bedrock_files_transformation.transform_s3_bucket_response_to_openai_file_object(
            model=None,
            raw_response=upload_response,  # type: ignore
            logging_obj=logging_obj,
            litellm_params=litellm_params,
            model_id=s3_upload_params["model_id"],
        )

    def create_file(
        self,
        create_file_data: CreateFileRequest,
        litellm_params: dict,
        provider_config: Optional[BaseFilesConfig],
        headers: dict,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        logging_obj: Optional[LiteLLMLoggingObj] = None,
        _is_async: bool = False,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Union[OpenAIFileObject, Coroutine[Any, Any, OpenAIFileObject]]:
        """
        Creates a file on VertexAI GCS Bucket

        Only supported for Async litellm.acreate_file
        """

        if _is_async:
            return self.async_create_file(
                transformed_request=dict(create_file_data),
                litellm_params=litellm_params,
                provider_config=provider_config,  # type: ignore
                headers=headers,
                api_base=api_base or "",
                logging_obj=logging_obj
                or LiteLLMLoggingObj(
                    model="bedrock",
                    messages=[],
                    stream=False,
                    call_type="file",
                    start_time=time.time(),
                    litellm_call_id="",
                    function_id="",
                ),
                timeout=timeout,
            )
        else:
            return asyncio.run(
                self.async_create_file(
                    transformed_request=dict(create_file_data),
                    litellm_params=litellm_params,
                    provider_config=provider_config or BaseFilesConfig(),  # type: ignore
                    headers=headers,
                    api_base=api_base or "",
                    logging_obj=logging_obj
                    or LiteLLMLoggingObj(
                        model="bedrock",
                        messages=[],
                        stream=False,
                        call_type="file",
                        start_time=time.time(),
                        litellm_call_id="",
                        function_id="",
                    ),
                    timeout=timeout,
                )
            )
