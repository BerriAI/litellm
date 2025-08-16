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
from datetime import datetime


from ..base_aws_llm import BaseAWSLLM
from .transformation import BedrockBatchTransformation


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
            BedrockBatchTransformation.transform_openai_batch_request_to_bedrock_job_request(
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

        # Retrieve the full job details to transform properly
        if job_arn:
            full_resp = boto_client.get_model_invocation_job(jobIdentifier=job_arn)
            return BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
                full_resp, 
                input_file_id=create_batch_data.get("input_file_id", "")
            )
        else:
            # Fallback if job_arn is not available
            return LiteLLMBatch(
                id=job_arn or "",
                status="validating",
                object="batch",
                created_at=int(time.time()),
                endpoint="/v1/chat/completions",
                input_file_id=create_batch_data.get("input_file_id", ""),
                completion_window="24h",
            )

    def retrieve_batch(self, batch_id: str, _is_async: bool, api_base: Optional[str], litellm_params: Optional[Dict[str, Any]]) -> LiteLLMBatch:
        # Build boto3 client via existing BaseAWSLLM credential helpers
        from litellm.llms.bedrock.common_utils import init_bedrock_service_client

        boto_client = init_bedrock_service_client()
        resp = boto_client.get_model_invocation_job(jobIdentifier=batch_id)
        
        # Transform the Bedrock response to OpenAI batch format
        return BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(resp)

    def cancel_batch(self, batch_id: str, _is_async: bool, api_base: Optional[str], litellm_params: Optional[Dict[str, Any]]) -> LiteLLMBatch:
        # Build boto3 client via existing BaseAWSLLM credential helpers
        from litellm.llms.bedrock.common_utils import init_bedrock_service_client

        boto_client = init_bedrock_service_client()
        resp = boto_client.stop_model_invocation_job(jobIdentifier=batch_id)
        
        # After stopping, retrieve the job details to get the full response
        updated_resp = boto_client.get_model_invocation_job(jobIdentifier=batch_id)
        
        # Transform the Bedrock response to OpenAI batch format
        return BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(updated_resp)

    def list_batches(self, _is_async: bool, api_base: Optional[str], litellm_params: Optional[Dict[str, Any]], after: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        # Build boto3 client via existing BaseAWSLLM credential helpers
        from litellm.llms.bedrock.common_utils import init_bedrock_service_client

        boto_client = init_bedrock_service_client()
        submitTimeBefore = datetime(datetime.utcnow().year, datetime.utcnow().month, datetime.utcnow().day)
        submitTimeAfter = after        
        # Prepare list parameters
        list_params = {}
        if limit is not None:
            list_params["maxResults"] = limit
        else:
            list_params["maxResults"] = 20 # Max limit of openai
        if submitTimeBefore is not None:
            list_params["submitTimeBefore"] = submitTimeBefore
        if submitTimeAfter is not None:
            list_params["submitTimeAfter"] = submitTimeAfter
        
        list_params["sortBy"] = "CreationTime"
        list_params["sortOrder"] = "Descending"
        
        resp = boto_client.list_model_invocation_jobs(**list_params)
        
        # Transform each job summary to OpenAI batch format
        job_summaries = resp.get("invocationJobSummaries", [])
        transformed_batches = []
        
        for job_summary in job_summaries:
            # For list operations, we have limited information
            # Transform what we have available
            transformed_batch = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(job_summary)
            transformed_batches.append(transformed_batch.model_dump() if hasattr(transformed_batch, 'model_dump') else transformed_batch.dict())
        
        # Return list in OpenAI format
        return {
            "object": "list",
            "data": transformed_batches,
            "has_more": resp.get("nextToken") is not None,
            "first_id": transformed_batches[0]["id"] if transformed_batches else None,
            "last_id": transformed_batches[-1]["id"] if transformed_batches else None,
        }