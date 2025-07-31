import uuid
from typing import Any, Coroutine, Optional, Union

import httpx

from litellm import LlmProviders, get_secret_str
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.openai import CreateFileRequest, OpenAIFileObject

from ..base_aws_llm import BaseAWSLLM


class BedrockFilesHandler(BaseAWSLLM):
    """
    Handles Bedrock file uploads to S3 in OpenAI Files API format v1/files/*
    
    This implementation uploads files to S3 buckets for use with Bedrock batch jobs.
    """

    def __init__(self):
        super().__init__()
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=LlmProviders.BEDROCK,
        )

    def _get_s3_config(self) -> dict:
        """Get S3 configuration from environment variables."""
        bucket_name = get_secret_str("LITELLM_BEDROCK_BATCH_BUCKET")
        if not bucket_name:
            raise ValueError(
                "LITELLM_BEDROCK_BATCH_BUCKET environment variable is required for Bedrock file uploads"
            )
        
        return {
            "bucket_name": bucket_name,
            "region_name": get_secret_str("AWS_REGION") or "us-east-1",
        }

    def _generate_s3_key(self, filename: str, purpose: str) -> str:
        """Generate a unique S3 key for the uploaded file."""
        file_uuid = str(uuid.uuid4())
        # Keep original extension if present
        if "." in filename:
            ext = filename.split(".")[-1]
            return f"{purpose}/{file_uuid}.{ext}"
        return f"{purpose}/{file_uuid}"

    async def _upload_to_s3(
        self,
        file_content: bytes,
        s3_key: str,
        bucket_name: str,
        region_name: str,
        filename: str,
        **aws_params
    ) -> dict:
        """Upload file content to S3 bucket."""
        import hashlib
        
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        # Get AWS credentials
        credentials = self.get_credentials(**aws_params)
        
        # Prepare S3 URL
        url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{s3_key}"
        
        # Calculate content hash
        content_hash = hashlib.sha256(file_content).hexdigest()
        
        # Prepare headers
        headers = {
            "Content-Type": "application/octet-stream",
            "x-amz-content-sha256": content_hash,
            "Content-Length": str(len(file_content)),
        }
        
        # Create and sign AWS request
        aws_request = AWSRequest(
            method="PUT",
            url=url,
            data=file_content,
            headers=headers,
        )
        SigV4Auth(credentials, "s3", region_name).add_auth(aws_request)
        
        # Upload using httpx
        response = await self.async_httpx_client.put(
            url,
            content=file_content,
            headers=dict(aws_request.headers.items()),
        )
        response.raise_for_status()
        
        return {
            "s3_uri": f"s3://{bucket_name}/{s3_key}",
            "s3_key": s3_key,
            "bucket_name": bucket_name,
            "region_name": region_name,
        }

    async def async_create_file(
        self,
        create_file_data: CreateFileRequest,
        **aws_params
    ) -> OpenAIFileObject:
        """Async file upload to S3."""
        verbose_logger.debug("bedrock create_file_data=%s", create_file_data)
        
        s3_config = self._get_s3_config()
        file_obj = create_file_data["file"]
        purpose = create_file_data["purpose"]
        
        # Read file content
        if hasattr(file_obj, 'read'):
            file_content = file_obj.read()
            filename = getattr(file_obj, 'name', f'file.{purpose}')
        else:
            # Handle bytes directly
            file_content = file_obj
            filename = f'file.{purpose}'
        
        # Generate S3 key
        s3_key = self._generate_s3_key(filename, purpose)
        
        # Upload to S3
        upload_result = await self._upload_to_s3(
            file_content=file_content,
            s3_key=s3_key,
            bucket_name=s3_config["bucket_name"],
            region_name=s3_config["region_name"],
            filename=filename,
            **aws_params
        )
        
        # Create OpenAI-compatible response
        file_id = f"file-{str(uuid.uuid4())}"
        
        response = OpenAIFileObject(
            id=file_id,
            bytes=len(file_content),
            created_at=int(__import__('time').time()),
            filename=filename,
            object="file",
            purpose=purpose,
            status="processed",
            status_details=None,
        )
        
        # Store S3 info in hidden params for batch operations
        if not hasattr(response, '_hidden_params'):
            response._hidden_params = {}
        response._hidden_params.update(upload_result)
        
        verbose_logger.debug("bedrock create_file_response=%s", response)
        return response

    def create_file(
        self,
        _is_async: bool,
        create_file_data: CreateFileRequest,
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        **aws_params
    ) -> Union[OpenAIFileObject, Coroutine[Any, Any, OpenAIFileObject]]:
        """
        Creates a file on S3 for Bedrock batch processing.
        
        Args:
            _is_async: Whether to run async
            create_file_data: File creation request data
            timeout: Request timeout
            max_retries: Max retries for upload
            **aws_params: AWS credential parameters
        """
        if _is_async:
            return self.async_create_file(
                create_file_data=create_file_data,
                **aws_params
            )
        else:
            import asyncio
            return asyncio.run(
                self.async_create_file(
                    create_file_data=create_file_data,
                    **aws_params
                )
            )