"""
Bedrock Batches API Handler
"""

import time
from typing import Any, Dict, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
)
from litellm.types.llms.bedrock import CreateModelInvocationJobRequest

# from litellm.llms.bedrock.bedrock import AsyncBedrock, Bedrock
from litellm.types.llms.openai import (
    CreateBatchRequest,
    LiteLLMBatchCreateRequest,
)
from litellm.types.utils import LiteLLMBatch

from ..base_aws_llm import BaseAWSLLM
from .transformation import transform_openai_create_batch_to_bedrock_job_request


class BedrockBatchesAPI(BaseAWSLLM):
    """
    Bedrock methods to support for batches
    - create_batch()
    - retrieve_batch()
    - cancel_batch()
    - list_batch()
    """

    def __init__(self) -> None:
        super().__init__()

    async def acreate_batch(
        self,
        create_batch_data: CreateBatchRequest,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
        api_key: Optional[str] = None,
    ) -> LiteLLMBatch:
        import asyncio
        
        def sync_create_batch():
            return self.create_batch(
                _is_async=True,
                create_batch_data=create_batch_data,
                timeout=3600.0,  # Default timeout
                max_retries=None,
                litellm_params={},
                api_base=None,
                logging_obj=None,
                client=client,
            )
        
        # Run the sync method in an executor to make it async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_create_batch)

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        litellm_params: Optional[Dict[str, Any]] = None,
        api_base: Optional[str] = None,
        logging_obj: Optional[LiteLLMLoggingObj] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
    ) -> LiteLLMBatch:
        # Build boto3 client via existing BaseAWSLLM credential helpers
        from litellm.llms.bedrock.common_utils import init_bedrock_service_client

        # Derive input/output S3 URIs from the input_file_id returned by files.create
        file_id = create_batch_data.get("input_file_id")
        # The file_id is expected to be of the form f"s3://{bucket}/path/model_id/" or f"s3://{bucket}/{model_id}/"
        # We need to extract both the input S3 URI and the model_id
        # Example: file_id = "s3://my-bucket/path/anthropic.claude-3-sonnet-20240229-v1:0/"
        if not isinstance(file_id, str) or not file_id.startswith("s3://"):
            raise ValueError(
                "input_file_id must be an s3:// URI for Bedrock batch jobs"
            )

        # Remove the "s3://" prefix and parse the path
        s3_path = file_id[len("s3://") :].rstrip("/")
        path_parts = s3_path.split("/")

        if len(path_parts) < 1:
            raise ValueError("Invalid S3 URI format")

        bucket = path_parts[0]

        # Use the S3 URI as-is for input (points to the JSONL file)
        input_s3_uri = file_id

        # Extract model ID from the S3 path where it was embedded by the file transformation
        # Expected format: s3://bucket/model_id/filename.jsonl
        model_id = create_batch_data.get("model") or ""

        if not model_id and len(path_parts) >= 3:
            # Extract model_id from S3 path (it's the second-to-last component before filename)
            model_id = path_parts[-2]  # Get model_id from the path
            verbose_logger.debug(f"Extracted model_id from S3 path: {model_id}")

        if not isinstance(input_s3_uri, str) or not input_s3_uri.startswith("s3://"):
            raise ValueError(
                "input_file_id must be an s3:// URI for Bedrock batch jobs"
            )

        # output path: same directory as input file (without the filename)
        # s3://bucket/model_id/file.jsonl -> s3://bucket/model_id/
        try:
            without_scheme = input_s3_uri[len("s3://") :]
            bucket_and_key = without_scheme.split("/", 1)
            if len(bucket_and_key) == 1:
                # no key, just bucket; place outputs at bucket root
                output_s3_uri = f"s3://{bucket_and_key[0]}/"
            else:
                bucket, key = bucket_and_key[0], bucket_and_key[1]
                # If key contains a filename (has extension), remove it to get directory
                if "." in key and "/" in key:
                    # Remove filename to get directory path
                    prefix = key.rsplit("/", 1)[0]
                    output_s3_uri = f"s3://{bucket}/{prefix}/"
                elif "/" in key:
                    # Already a directory path
                    prefix = key.rstrip("/")
                    output_s3_uri = f"s3://{bucket}/{prefix}/"
                else:
                    # Key is just a filename or single component
                    output_s3_uri = f"s3://{bucket}/"
        except Exception:
            # Fallback to bucket root if parsing fails
            output_s3_uri = f"s3://{bucket}/"

        # Optional role from config
        s3_params = getattr(litellm, "s3_callback_params", {}) or {}
        role_arn = s3_params.get("s3_aws_role_name")
        if isinstance(role_arn, str) and not role_arn.startswith("arn:"):
            role_arn = None

        # Create a new dict with the required fields for Bedrock
        bedrock_batch_data: LiteLLMBatchCreateRequest = {
            "model": str(model_id),
            "input_file_id": create_batch_data.get("input_file_id", ""),
            "metadata": create_batch_data.get("metadata", {}),
        }

        bedrock_job_request: CreateModelInvocationJobRequest = (
            transform_openai_create_batch_to_bedrock_job_request(
                bedrock_batch_data,
                s3_input_uri=input_s3_uri,
                s3_output_uri=output_s3_uri,
                role_arn=role_arn,
            )
        )
        verbose_logger.debug(f"create_batch_data: {bedrock_job_request}")

        # Create boto3 client (respects env/profile/role)
        boto_client = init_bedrock_service_client()
        resp = boto_client.create_model_invocation_job(**bedrock_job_request)  # type: ignore
        job_arn = resp.get("jobArn")

        return LiteLLMBatch(
            id=job_arn or "",
            status="in_progress",
            object="batch",
            created_at=int(time.time()),
            endpoint="bedrock",
            input_file_id=create_batch_data.get("input_file_id", ""),
            completion_window="1h",  # Default 1 hour
        )
