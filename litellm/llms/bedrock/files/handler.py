import asyncio
import time
from typing import TYPE_CHECKING, Any, Coroutine, Dict, Optional, Union

import httpx

from litellm import LlmProviders
from litellm._logging import verbose_logger
from litellm.integrations.s3_v2 import S3Logger
from litellm.llms.base_llm.chat.transformation import BaseLLMException
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

    This implementation uploads files to S3 using the S3Logger class
    """

    def __init__(self):
        super().__init__()
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=LlmProviders.BEDROCK,
        )
        # Initialize S3Logger for file uploads
        self.s3_logger = None

    def _get_s3_logger(self) -> S3Logger:
        """Get or create S3Logger instance for file uploads"""
        if self.s3_logger is None:
            # Get S3 config from global litellm settings
            import litellm

            s3_params = getattr(litellm, "s3_callback_params", {}) or {}

            self.s3_logger = S3Logger(
                s3_bucket_name=s3_params.get("s3_bucket_name"),
                s3_region_name=s3_params.get("s3_region_name"),
                s3_api_version=s3_params.get("s3_api_version"),
                s3_use_ssl=s3_params.get("s3_use_ssl", True),
                s3_verify=s3_params.get("s3_verify"),
                s3_endpoint_url=s3_params.get("s3_endpoint_url"),
                s3_aws_access_key_id=s3_params.get("s3_aws_access_key_id"),
                s3_aws_secret_access_key=s3_params.get("s3_aws_secret_access_key"),
                s3_aws_session_token=s3_params.get("s3_aws_session_token"),
                s3_aws_session_name=s3_params.get("s3_aws_session_name"),
                s3_aws_profile_name=s3_params.get("s3_aws_profile_name"),
                s3_aws_role_name=s3_params.get("s3_aws_role_name"),
                s3_aws_web_identity_token=s3_params.get("s3_aws_web_identity_token"),
                s3_aws_sts_endpoint=s3_params.get("s3_aws_sts_endpoint"),
                s3_config=s3_params.get("s3_config"),
                s3_path=s3_params.get("s3_path"),
                s3_use_team_prefix=s3_params.get("s3_use_team_prefix", False),
            )
        return self.s3_logger

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

            # Get S3Logger instance
            s3_logger = self._get_s3_logger()

            # Create a custom batch logging element for file upload
            from datetime import datetime

            from litellm.types.integrations.s3_v2 import s3BatchLoggingElement

            # Create a custom payload for file upload
            file_upload_payload = {
                "id": f"file_upload_{int(time.time())}",
                "metadata": {
                    "file_purpose": create_file_data.get("purpose", "batch"),
                    "filename": s3_upload_params.get("key", "").split("/")[-1],
                    "content_type": s3_upload_params.get(
                        "content_type", "application/octet-stream"
                    ),
                    "file_size": len(s3_upload_params.get("content", b"")),
                    "upload_timestamp": datetime.now().isoformat(),
                },
                "file_content": s3_upload_params.get("content", b"").decode("utf-8")
                if isinstance(s3_upload_params.get("content", b""), bytes)
                else str(s3_upload_params.get("content", "")),
            }

            # Create S3 batch logging element
            s3_batch_element = s3BatchLoggingElement(
                payload=file_upload_payload,
                s3_object_key=s3_upload_params["key"],
                s3_object_download_filename=s3_upload_params["key"].split("/")[-1],
            )

            await s3_logger.async_upload_data_to_s3(s3_batch_element)

            verbose_logger.debug(
                f"File uploaded successfully to S3: {s3_upload_params['key']}"
            )

            # Create a response object that mimics httpx.Response
            class S3UploadResponse:
                def __init__(self):
                    self.status_code = 200
                    self.text = "S3 upload successful using S3Logger"

            upload_response = S3UploadResponse()

        except Exception as e:
            verbose_logger.exception(f"Error uploading file to S3: {str(e)}")
            raise BaseLLMException(
                status_code=500,
                message=str(e),
                headers={},
            )

        return bedrock_files_transformation.transform_s3_bucket_response_to_openai_file_object(
            model=None,
            raw_response=upload_response,  # type: ignore
            logging_obj=logging_obj,
            litellm_params=litellm_params,
            model_id=s3_upload_params.get("model_id", ""),
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
