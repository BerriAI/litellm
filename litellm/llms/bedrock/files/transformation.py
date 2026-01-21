import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx import Headers, Response
from openai.types.file_deleted import FileDeleted

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.files.utils import FilesAPIUtils
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.files.transformation import (
    BaseFilesConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    CreateFileRequest,
    FileTypes,
    HttpxBinaryResponseContent,
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
    PathLike,
)
from litellm.types.utils import ExtractedFileData, LlmProviders
from litellm.utils import get_llm_provider

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockError


class BedrockFilesConfig(BaseAWSLLM, BaseFilesConfig):
    """
    Config for Bedrock Files - handles S3 uploads for Bedrock batch processing
    """
    
    def __init__(self):
        self.jsonl_transformation = BedrockJsonlFilesTransformation()
        super().__init__()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK

    @property
    def file_upload_http_method(self) -> str:
        """
        Bedrock files are uploaded to S3, which requires PUT requests
        """
        return "PUT"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        # No additional headers needed for S3 uploads - AWS credentials handled by BaseAWSLLM
        return headers



    def _get_content_from_openai_file(self, openai_file_content: FileTypes) -> str:
        """
        Helper to extract content from various OpenAI file types and return as string.

        Handles:
        - Direct content (str, bytes, IO[bytes])
        - Tuple formats: (filename, content, [content_type], [headers])
        - PathLike objects
        """
        content: Union[str, bytes] = b""
        # Extract file content from tuple if necessary
        if isinstance(openai_file_content, tuple):
            # Take the second element which is always the file content
            file_content = openai_file_content[1]
        else:
            file_content = openai_file_content

        # Handle different file content types
        if isinstance(file_content, str):
            # String content can be used directly
            content = file_content
        elif isinstance(file_content, bytes):
            # Bytes content can be decoded
            content = file_content
        elif isinstance(file_content, PathLike):  # PathLike
            with open(str(file_content), "rb") as f:
                content = f.read()
        elif hasattr(file_content, "read"):  # IO[bytes]
            # File-like objects need to be read
            content = file_content.read()

        # Ensure content is string
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        return content

    def _get_s3_object_name_from_batch_jsonl(
        self,
        openai_jsonl_content: List[Dict[str, Any]],
    ) -> str:
        """
        Gets a unique S3 object name for the Bedrock batch processing job

        named as: litellm-bedrock-files/{model}/{uuid}
        """
        _model = openai_jsonl_content[0].get("body", {}).get("model", "")
        # Remove bedrock/ prefix if present
        if _model.startswith("bedrock/"):
            _model = _model[8:]
        
        # Replace colons with hyphens for Bedrock S3 URI compliance
        _model = _model.replace(":", "-")
        
        object_name = f"litellm-bedrock-files-{_model}-{uuid.uuid4()}.jsonl"
        return object_name

    def get_object_name(
        self, extracted_file_data: ExtractedFileData, purpose: str
    ) -> str:
        """
        Get the object name for the request
        """
        extracted_file_data_content = extracted_file_data.get("content")

        if extracted_file_data_content is None:
            raise ValueError("file content is required")

        if purpose == "batch":
            ## 1. If jsonl, check if there's a model name
            file_content = self._get_content_from_openai_file(
                extracted_file_data_content
            )

            # Split into lines and parse each line as JSON
            openai_jsonl_content = [
                json.loads(line) for line in file_content.splitlines() if line.strip()
            ]
            if len(openai_jsonl_content) > 0:
                return self._get_s3_object_name_from_batch_jsonl(openai_jsonl_content)

        ## 2. If not jsonl, return the filename
        filename = extracted_file_data.get("filename")
        if filename:
            return filename
        ## 3. If no file name, return timestamp
        return str(int(time.time()))

    def get_complete_file_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: Dict,
        litellm_params: Dict,
        data: CreateFileRequest,
    ) -> str:
        """
        Get the complete S3 URL for the file upload request
        """
        bucket_name = litellm_params.get("s3_bucket_name") or os.getenv("AWS_S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("S3 bucket_name is required. Set 's3_bucket_name' in litellm_params or AWS_S3_BUCKET_NAME env var")
        
        aws_region_name = self._get_aws_region_name(optional_params, model)
        
        file_data = data.get("file")
        purpose = data.get("purpose")
        if file_data is None:
            raise ValueError("file is required")
        if purpose is None:
            raise ValueError("purpose is required")
        extracted_file_data = extract_file_data(file_data)
        object_name = self.get_object_name(extracted_file_data, purpose)
        
        # S3 endpoint URL format
        s3_endpoint_url = optional_params.get("s3_endpoint_url") or f"https://s3.{aws_region_name}.amazonaws.com"
        
        return f"{s3_endpoint_url}/{bucket_name}/{object_name}"

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAICreateFileRequestOptionalParams]:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params


    def _map_openai_to_bedrock_params(
        self,
        openai_request_body: Dict[str, Any],
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transform OpenAI request body to Bedrock-compatible modelInput parameters using existing transformation logic
        """
        from litellm.types.utils import LlmProviders
        _model = openai_request_body.get("model", "")
        messages = openai_request_body.get("messages", [])
        
        # Use existing Anthropic transformation logic for Anthropic models
        if provider == LlmProviders.ANTHROPIC:
            from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
                AmazonAnthropicClaudeConfig,
            )
            
            anthropic_config = AmazonAnthropicClaudeConfig()
            
            # Extract optional params (everything except model and messages)
            optional_params = {k: v for k, v in openai_request_body.items() if k not in ["model", "messages"]}
            mapped_params = anthropic_config.map_openai_params(
                non_default_params={},
                optional_params=optional_params,
                model=_model,
                drop_params=False
            )
            
            # Transform using existing Anthropic logic
            bedrock_params = anthropic_config.transform_request(
                model=_model,
                messages=messages,
                optional_params=mapped_params,
                litellm_params={},
                headers={}
            )

            return bedrock_params
        else:
            # For other providers, use basic mapping
            bedrock_params = {
                "messages": messages,
                **{k: v for k, v in openai_request_body.items() if k not in ["model", "messages"]}
            }
            return bedrock_params

    def _transform_openai_jsonl_content_to_bedrock_jsonl_content(
        self, openai_jsonl_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Transforms OpenAI JSONL content to Bedrock batch format
        
        Bedrock batch format: { "recordId": "alphanumeric string", "modelInput": {JSON body} }
        Example:
        {
            "recordId": "CALL0000001", 
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31", 
                "max_tokens": 1024,
                "messages": [ 
                    { 
                        "role": "user", 
                        "content": [{"type": "text", "text": "Hello"}]
                    }
                ]
            }
        }
        """
        
        bedrock_jsonl_content = []
        for idx, _openai_jsonl_content in enumerate(openai_jsonl_content):
            # Extract the request body from OpenAI format
            openai_body = _openai_jsonl_content.get("body", {})
            model = openai_body.get("model", "")

            try:
                model, _, _, _ = get_llm_provider(
                            model=model, 
                            custom_llm_provider=None, 
                    )
            except Exception as e:
                verbose_logger.exception(f"litellm.llms.bedrock.files.transformation.py::_transform_openai_jsonl_content_to_bedrock_jsonl_content() - Error inferring custom_llm_provider - {str(e)}")
            
            # Determine provider from model name
            provider = self.get_bedrock_invoke_provider(model)
            
            # Transform to Bedrock modelInput format
            model_input = self._map_openai_to_bedrock_params(
                openai_request_body=openai_body,
                provider=provider
            )
            
            # Create Bedrock batch record
            record_id = _openai_jsonl_content.get("custom_id", f"CALL{str(idx).zfill(7)}")
            bedrock_record = {
                "recordId": record_id,
                "modelInput": model_input
            }
                    
            bedrock_jsonl_content.append(bedrock_record)
        return bedrock_jsonl_content

    def transform_create_file_request(
        self,
        model: str,
        create_file_data: CreateFileRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> Union[bytes, str, dict]:
        """
        Transform file request and return a pre-signed request for S3.
        This keeps the HTTP handler clean by doing all the signing here.
        """
        file_data = create_file_data.get("file")
        if file_data is None:
            raise ValueError("file is required")
        extracted_file_data = extract_file_data(file_data)
        extracted_file_data_content = extracted_file_data.get("content")
        
        if extracted_file_data_content is None:
            raise ValueError("file content is required")
            
        # Get and transform the file content
        if FilesAPIUtils.is_batch_jsonl_file(
            create_file_data=create_file_data,
            extracted_file_data=extracted_file_data,
        ):
            ## Transform JSONL content to Bedrock format
            original_file_content = self._get_content_from_openai_file(
                extracted_file_data_content
            )
            openai_jsonl_content = [
                json.loads(line) for line in original_file_content.splitlines() if line.strip()
            ]
            bedrock_jsonl_content = (
                self._transform_openai_jsonl_content_to_bedrock_jsonl_content(
                    openai_jsonl_content
                )
            )
            file_content = "\n".join(json.dumps(item) for item in bedrock_jsonl_content)
        elif isinstance(extracted_file_data_content, bytes):
            file_content = extracted_file_data_content.decode('utf-8')
        elif isinstance(extracted_file_data_content, str):
            file_content = extracted_file_data_content
        else:
            raise ValueError("Unsupported file content type")
        
        # Get the S3 URL for upload
        api_base = self.get_complete_file_url(
            api_base=None,
            api_key=None,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            data=create_file_data,
        )
        
        # Sign the request and return a pre-signed request object
        signed_headers, signed_body = self._sign_s3_request(
            content=file_content,
            api_base=api_base,
            optional_params=optional_params,
        )

        litellm_params["upload_url"] = api_base
        
        # Return a dict that tells the HTTP handler exactly what to do
        return {
            "method": "PUT",
            "url": api_base,
            "headers": signed_headers,
            "data": signed_body or file_content,
        }

    def _sign_s3_request(
        self,
        content: str,
        api_base: str,
        optional_params: dict,
    ) -> Tuple[dict, str]:
        """
        Sign S3 PUT request using the same proven logic as S3Logger.
        Reuses the exact pattern from litellm/integrations/s3_v2.py
        """
        try:
            import hashlib

            import requests
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        # Get AWS credentials using existing methods
        aws_region_name = self._get_aws_region_name(
            optional_params=optional_params, model=""
        )
        credentials = self.get_credentials(
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
        
        # Calculate SHA256 hash of the content (REQUIRED for S3)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Prepare headers with required S3 headers (same as s3_v2.py)
        request_headers = {
            "Content-Type": "application/json",  # JSONL files are JSON content
            "x-amz-content-sha256": content_hash,  # REQUIRED by S3
            "Content-Language": "en",
            "Cache-Control": "private, immutable, max-age=31536000, s-maxage=0",
        }

        # Use requests.Request to prepare the request (same pattern as s3_v2.py)
        req = requests.Request("PUT", api_base, data=content, headers=request_headers)
        prepped = req.prepare()

        # Sign the request with S3 service
        aws_request = AWSRequest(
            method=prepped.method,
            url=prepped.url,
            data=prepped.body,
            headers=prepped.headers,
        )
        
        # Get region name for non-LLM API calls (same as s3_v2.py)
        signing_region = self.get_aws_region_name_for_non_llm_api_calls(
            aws_region_name=aws_region_name
        )
        
        SigV4Auth(credentials, "s3", signing_region).add_auth(aws_request)

        # Return signed headers and body
        signed_body = aws_request.body
        if isinstance(signed_body, bytes):
            signed_body = signed_body.decode('utf-8')
        elif signed_body is None:
            signed_body = content  # Fallback to original content
        
        return dict(aws_request.headers), signed_body

    def _convert_https_url_to_s3_uri(self, https_url: str) -> tuple[str, str]:
        """
        Convert HTTPS S3 URL to s3:// URI format.
        
        Args:
            https_url: HTTPS S3 URL (e.g., "https://s3.us-west-2.amazonaws.com/bucket/key")
        
        Returns:
            Tuple of (s3_uri, filename)
        
        Example:
            Input: "https://s3.us-west-2.amazonaws.com/litellm-proxy/file.jsonl"
            Output: ("s3://litellm-proxy/file.jsonl", "file.jsonl")
        """
        import re

        # Match HTTPS S3 URL patterns
        # Pattern 1: https://s3.region.amazonaws.com/bucket/key
        # Pattern 2: https://bucket.s3.region.amazonaws.com/key
        
        pattern1 = r"https://s3\.([^.]+)\.amazonaws\.com/([^/]+)/(.+)"
        pattern2 = r"https://([^.]+)\.s3\.([^.]+)\.amazonaws\.com/(.+)"
        
        match1 = re.match(pattern1, https_url)
        match2 = re.match(pattern2, https_url)
        
        if match1:
            # Pattern: https://s3.region.amazonaws.com/bucket/key
            region, bucket, key = match1.groups()
            s3_uri = f"s3://{bucket}/{key}"
        elif match2:
            # Pattern: https://bucket.s3.region.amazonaws.com/key
            bucket, region, key = match2.groups()
            s3_uri = f"s3://{bucket}/{key}"
        else:
            # Fallback: try to extract bucket and key from URL path
            from urllib.parse import urlparse
            parsed = urlparse(https_url)
            path_parts = parsed.path.lstrip('/').split('/', 1)
            if len(path_parts) >= 2:
                bucket, key = path_parts[0], path_parts[1]
                s3_uri = f"s3://{bucket}/{key}"
            else:
                raise ValueError(f"Unable to parse S3 URL: {https_url}")
        
        # Extract filename from key
        filename = key.split("/")[-1] if "/" in key else key
        
        return s3_uri, filename

    def transform_create_file_response(
        self,
        model: Optional[str],
        raw_response: Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """
        Transform S3 File upload response into OpenAI-style FileObject
        """
        # For S3 uploads, we typically get an ETag and other metadata
        response_headers = raw_response.headers
        # Extract S3 object information from the response
        # S3 PUT object returns ETag and other metadata in headers
        content_length = response_headers.get("Content-Length", "0")
        
        # Use the actual upload URL that was used for the S3 upload
        upload_url = litellm_params.get("upload_url")
        file_id: str = ""
        filename: str = ""
        if upload_url:
            # Convert HTTPS S3 URL to s3:// URI format
            file_id, filename = self._convert_https_url_to_s3_uri(upload_url)

        return OpenAIFileObject(
            purpose="batch",  # Default purpose for Bedrock files
            id=file_id,
            filename=filename,
            created_at=int(time.time()),  # Current timestamp
            status="uploaded",
            bytes=int(content_length) if content_length.isdigit() else 0,
            object="file",
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, Headers]
    ) -> BaseLLMException:
        return BedrockError(
            status_code=status_code, message=error_message, headers=headers
        )

    def transform_retrieve_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("BedrockFilesConfig does not support file retrieval")

    def transform_retrieve_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        raise NotImplementedError("BedrockFilesConfig does not support file retrieval")

    def transform_delete_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("BedrockFilesConfig does not support file deletion")

    def transform_delete_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> FileDeleted:
        raise NotImplementedError("BedrockFilesConfig does not support file deletion")

    def transform_list_files_request(
        self,
        purpose: Optional[str],
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("BedrockFilesConfig does not support file listing")

    def transform_list_files_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> List[OpenAIFileObject]:
        raise NotImplementedError("BedrockFilesConfig does not support file listing")

    def transform_file_content_request(
        self,
        file_content_request,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("BedrockFilesConfig does not support file content retrieval")

    def transform_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> HttpxBinaryResponseContent:
        raise NotImplementedError("BedrockFilesConfig does not support file content retrieval")


class BedrockJsonlFilesTransformation:
    """
    Transforms OpenAI /v1/files/* requests to Bedrock S3 file uploads for batch processing
    """

    def transform_openai_file_content_to_bedrock_file_content(
        self, openai_file_content: Optional[FileTypes] = None
    ) -> Tuple[str, str]:
        """
        Transforms OpenAI FileContentRequest to Bedrock S3 file format
        """

        if openai_file_content is None:
            raise ValueError("contents of file are None")
        # Read the content of the file
        file_content = self._get_content_from_openai_file(openai_file_content)

        # Split into lines and parse each line as JSON
        openai_jsonl_content = [
            json.loads(line) for line in file_content.splitlines() if line.strip()
        ]
        bedrock_jsonl_content = (
            self._transform_openai_jsonl_content_to_bedrock_jsonl_content(
                openai_jsonl_content
            )
        )
        bedrock_jsonl_string = "\n".join(
            json.dumps(item) for item in bedrock_jsonl_content
        )
        object_name = self._get_s3_object_name(
            openai_jsonl_content=openai_jsonl_content
        )
        return bedrock_jsonl_string, object_name

    def _transform_openai_jsonl_content_to_bedrock_jsonl_content(
        self, openai_jsonl_content: List[Dict[str, Any]]
    ):
        """
        Delegate to the main BedrockFilesConfig transformation method
        """
        config = BedrockFilesConfig()
        return config._transform_openai_jsonl_content_to_bedrock_jsonl_content(openai_jsonl_content)

    def _get_s3_object_name(
        self,
        openai_jsonl_content: List[Dict[str, Any]],
    ) -> str:
        """
        Gets a unique S3 object name for the Bedrock batch processing job

        named as: litellm-bedrock-files-{model}-{uuid}
        """
        _model = openai_jsonl_content[0].get("body", {}).get("model", "")
        # Remove bedrock/ prefix if present
        if _model.startswith("bedrock/"):
            _model = _model[8:]
        object_name = f"litellm-bedrock-files-{_model}-{uuid.uuid4()}.jsonl"
        return object_name



    def _get_content_from_openai_file(self, openai_file_content: FileTypes) -> str:
        """
        Helper to extract content from various OpenAI file types and return as string.

        Handles:
        - Direct content (str, bytes, IO[bytes])
        - Tuple formats: (filename, content, [content_type], [headers])
        - PathLike objects
        """
        content: Union[str, bytes] = b""
        # Extract file content from tuple if necessary
        if isinstance(openai_file_content, tuple):
            # Take the second element which is always the file content
            file_content = openai_file_content[1]
        else:
            file_content = openai_file_content

        # Handle different file content types
        if isinstance(file_content, str):
            # String content can be used directly
            content = file_content
        elif isinstance(file_content, bytes):
            # Bytes content can be decoded
            content = file_content
        elif isinstance(file_content, PathLike):  # PathLike
            with open(str(file_content), "rb") as f:
                content = f.read()
        elif hasattr(file_content, "read"):  # IO[bytes]
            # File-like objects need to be read
            content = file_content.read()

        # Ensure content is string
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        return content

    def transform_s3_bucket_response_to_openai_file_object(
        self, create_file_data: CreateFileRequest, s3_upload_response: Dict[str, Any]
    ) -> OpenAIFileObject:
        """
        Transforms S3 Bucket upload file response to OpenAI FileObject
        """
        # S3 response typically contains ETag, key, etc.
        object_key = s3_upload_response.get("Key", "")
        bucket_name = s3_upload_response.get("Bucket", "")
        
        # Extract filename from object key
        filename = object_key.split("/")[-1] if "/" in object_key else object_key
        
        return OpenAIFileObject(
            purpose=create_file_data.get("purpose", "batch"),
            id=f"s3://{bucket_name}/{object_key}",
            filename=filename,
            created_at=int(time.time()),  # Current timestamp
            status="uploaded",
            bytes=s3_upload_response.get("ContentLength", 0),
            object="file",
        )
