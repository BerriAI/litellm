import os
import time
from typing import Any, Dict, List, Literal, Optional, Union, cast

from httpx import Headers, Response

from litellm.llms.base_llm.batches.transformation import BaseBatchesConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.bedrock import (
    BedrockCreateBatchRequest,
    BedrockCreateBatchResponse,
    BedrockInputDataConfig,
    BedrockOutputDataConfig,
    BedrockS3InputDataConfig,
    BedrockS3OutputDataConfig,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    CreateBatchRequest,
)
from litellm.types.utils import LiteLLMBatch, LlmProviders

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import CommonBatchFilesUtils


class BedrockBatchesConfig(BaseAWSLLM, BaseBatchesConfig):
    """
    Config for Bedrock Batches - handles batch job creation and management for Bedrock
    """
    
    def __init__(self):
        super().__init__()
        self.common_utils = CommonBatchFilesUtils()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK

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
        """
        Validate and prepare environment for Bedrock batch requests.
        AWS credentials are handled by BaseAWSLLM.
        """
        # Add any Bedrock-specific headers if needed
        return headers

    def get_complete_batch_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: Dict,
        litellm_params: Dict,
        data: CreateBatchRequest,
    ) -> str:
        """
        Get the complete URL for Bedrock batch creation.
        Bedrock batch jobs are created via the model invocation job API.
        """
        aws_region_name = self._get_aws_region_name(optional_params, model)
        
        # Bedrock model invocation job endpoint
        # Format: https://bedrock.{region}.amazonaws.com/model-invocation-job
        bedrock_endpoint = f"https://bedrock.{aws_region_name}.amazonaws.com/model-invocation-job"
        
        return bedrock_endpoint







    def transform_create_batch_request(
        self,
        model: str,
        create_batch_data: CreateBatchRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> Dict[str, Any]:
        """
        Transform the batch creation request to Bedrock format.
        
        Bedrock batch inference requires:
        - modelId: The Bedrock model ID
        - jobName: Unique name for the batch job
        - inputDataConfig: Configuration for input data (S3 location)
        - outputDataConfig: Configuration for output data (S3 location)
        - roleArn: IAM role ARN for the batch job
        """
        # Get required parameters
        input_file_id = create_batch_data.get("input_file_id")
        if not input_file_id:
            raise ValueError("input_file_id is required for Bedrock batch creation")
        
        # Extract S3 information from file ID using common utility
        input_bucket, input_key = self.common_utils.parse_s3_uri(input_file_id)
        
        # Get output S3 configuration
        output_bucket = litellm_params.get("s3_output_bucket_name") or os.getenv("AWS_S3_OUTPUT_BUCKET_NAME")
        if not output_bucket:
            # Use same bucket as input if no output bucket specified
            output_bucket = input_bucket
        
        # Get IAM role ARN
        role_arn = (
            litellm_params.get("aws_batch_role_arn") 
            or optional_params.get("aws_batch_role_arn")
            or os.getenv("AWS_BATCH_ROLE_ARN")
        )
        if not role_arn:
            raise ValueError(
                "AWS IAM role ARN is required for Bedrock batch jobs. "
                "Set 'aws_batch_role_arn' in litellm_params or AWS_BATCH_ROLE_ARN env var"
            )

        
        if not model:
            raise ValueError("Could not determine Bedrock model ID. Please pass `model` in your request body.")
        
        # Generate job name with the correct model ID using common utility
        job_name = self.common_utils.generate_unique_job_name(model, prefix="litellm")
        output_key = f"litellm-batch-outputs/{job_name}/"
        
        # Build input data config
        input_data_config: BedrockInputDataConfig = {
            "s3InputDataConfig": BedrockS3InputDataConfig(
                s3Uri=f"s3://{input_bucket}/{input_key}"
            )
        }
        
        # Build output data config
        output_data_config: BedrockOutputDataConfig = {
            "s3OutputDataConfig": BedrockS3OutputDataConfig(
                s3Uri=f"s3://{output_bucket}/{output_key}"
            )
        }
        
        # Create Bedrock batch request with proper typing
        bedrock_request: BedrockCreateBatchRequest = {
            "modelId": model,
            "jobName": job_name,
            "inputDataConfig": input_data_config,
            "outputDataConfig": output_data_config,
            "roleArn": role_arn
        }
        
        # Add optional parameters if provided
        completion_window = create_batch_data.get("completion_window")
        if completion_window:
            # Map OpenAI completion window to Bedrock timeout
            # OpenAI uses "24h", Bedrock expects timeout in hours
            if completion_window == "24h":
                bedrock_request["timeoutDurationInHours"] = 24

        # For Bedrock, we need to return a pre-signed request with AWS auth headers
        # Use common utility for AWS signing
        endpoint_url = f"https://bedrock.{self._get_aws_region_name(optional_params, model)}.amazonaws.com/model-invocation-job"
        signed_headers, signed_data = self.common_utils.sign_aws_request(
            service_name="bedrock",
            data=bedrock_request,
            endpoint_url=endpoint_url,
            optional_params=optional_params,
            method="POST"
        )
        
        # Return a pre-signed request format that the HTTP handler can use
        return {
            "method": "POST",
            "url": endpoint_url,
            "headers": signed_headers,
            "data": signed_data.decode('utf-8')
        }

    def transform_create_batch_response(
        self,
        model: Optional[str],
        raw_response: Response,
        logging_obj: Any,
        litellm_params: dict,
    ) -> LiteLLMBatch:
        """
        Transform Bedrock batch creation response to LiteLLM format.
        """
        try:
            response_data: BedrockCreateBatchResponse = raw_response.json()
        except Exception as e:
            raise ValueError(f"Failed to parse Bedrock batch response: {e}")
        
        # Extract information from typed Bedrock response
        job_arn = response_data.get("jobArn", "")
        status_str: str = str(response_data.get("status", "Submitted"))
        
        # Map Bedrock status to OpenAI-compatible status
        status_mapping: Dict[str, str] = {
            "Submitted": "validating",
            "Validating": "validating",
            "Scheduled": "in_progress",
            "InProgress": "in_progress", 
            "PartiallyCompleted": "completed",
            "Completed": "completed",
            "Failed": "failed",
            "Stopping": "cancelling",
            "Stopped": "cancelled",
            "Expired": "expired",
        }
        
        openai_status = cast(Literal["validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"], status_mapping.get(status_str, "validating"))
        
        # Get original request data from litellm_params if available
        original_request = litellm_params.get("original_batch_request", {})
        
        # Create LiteLLM batch object
        return LiteLLMBatch(
            id=job_arn,  # Use ARN as the batch ID
            object="batch",
            endpoint=original_request.get("endpoint", "/v1/chat/completions"),
            errors=None,
            input_file_id=original_request.get("input_file_id", ""),
            completion_window=original_request.get("completion_window", "24h"),
            status=openai_status,
            output_file_id=None,  # Will be populated when job completes
            error_file_id=None,
            created_at=int(time.time()),
            in_progress_at=int(time.time()) if status_str == "InProgress" else None,
            expires_at=None,
            finalizing_at=None,
            completed_at=None,
            failed_at=None,
            expired_at=None,
            cancelling_at=None,
            cancelled_at=None,
            request_counts=None,
            metadata=original_request.get("metadata", {}),
        )

    def transform_retrieve_batch_request(
        self,
        batch_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> Dict[str, Any]:
        """
        Transform batch retrieval request for Bedrock.
        
        Args:
            batch_id: Bedrock job ARN
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            
        Returns:
            Transformed request data for Bedrock GetModelInvocationJob API
        """
        # For Bedrock, batch_id should be the full job ARN
        # The GetModelInvocationJob API expects the full ARN as the identifier
        if not batch_id.startswith("arn:aws:bedrock:"):
            raise ValueError(f"Invalid batch_id format. Expected ARN, got: {batch_id}")
        
        # Extract the job identifier from the ARN - use the full ARN path part
        # ARN format: arn:aws:bedrock:region:account:model-invocation-job/job-name
        arn_parts = batch_id.split(":")
        if len(arn_parts) < 6:
            raise ValueError(f"Invalid ARN format: {batch_id}")
        
        region = arn_parts[3]
        # arn_parts[5] contains "model-invocation-job/{jobId}"
        
        # Build the endpoint URL for GetModelInvocationJob
        # AWS API format: GET /model-invocation-job/{jobIdentifier}
        # Use the FULL ARN as jobIdentifier and URL-encode it (includes ':' and '/')
        import urllib.parse as _ul
        encoded_arn = _ul.quote(batch_id, safe="")
        endpoint_url = f"https://bedrock.{region}.amazonaws.com/model-invocation-job/{encoded_arn}"
        
        # Use common utility for AWS signing
        signed_headers, _ = self.common_utils.sign_aws_request(
            service_name="bedrock",
            data={},  # GET request has no body
            endpoint_url=endpoint_url,
            optional_params=optional_params,
            method="GET"
        )
        
        # Return pre-signed request format
        return {
            "method": "GET",
            "url": endpoint_url,
            "headers": signed_headers,
            "data": None
        }

    def _parse_timestamps_and_status(self, response_data, status_str: str):
        """Helper to parse timestamps based on status."""
        import datetime
        def parse_timestamp(ts_str: Optional[str]) -> Optional[int]:
            if not ts_str:
                return None
            try:
                dt = datetime.datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                return int(dt.timestamp())
            except Exception:
                return None
        
        created_at = parse_timestamp(str(response_data.get("submitTime")) if response_data.get("submitTime") is not None else None)
        in_progress_states = {"InProgress", "Validating", "Scheduled"}
        in_progress_at = (
            parse_timestamp(str(response_data.get("lastModifiedTime")) if response_data.get("lastModifiedTime") is not None else None)
            if status_str in in_progress_states
            else None
        )
        completed_at = parse_timestamp(str(response_data.get("endTime")) if response_data.get("endTime") is not None else None) if status_str in {"Completed", "PartiallyCompleted"} else None
        failed_at = parse_timestamp(str(response_data.get("endTime")) if response_data.get("endTime") is not None else None) if status_str == "Failed" else None
        cancelled_at = parse_timestamp(str(response_data.get("endTime")) if response_data.get("endTime") is not None else None) if status_str == "Stopped" else None
        expires_at = parse_timestamp(str(response_data.get("jobExpirationTime")) if response_data.get("jobExpirationTime") is not None else None)
        
        return created_at, in_progress_at, completed_at, failed_at, cancelled_at, expires_at
    
    def _extract_file_configs(self, response_data):
        """Helper to extract input and output file configurations."""
        # Extract input file ID
        input_file_id = ""
        input_data_config = response_data.get("inputDataConfig", {})
        if isinstance(input_data_config, dict):
            s3_input_config = input_data_config.get("s3InputDataConfig", {})
            if isinstance(s3_input_config, dict):
                input_file_id = s3_input_config.get("s3Uri", "")
        
        # Extract output file ID
        output_file_id = None
        output_data_config = response_data.get("outputDataConfig", {})
        if isinstance(output_data_config, dict):
            s3_output_config = output_data_config.get("s3OutputDataConfig", {})
            if isinstance(s3_output_config, dict):
                output_file_id = s3_output_config.get("s3Uri", "")
        
        return input_file_id, output_file_id
    
    def _extract_errors_and_metadata(self, response_data, raw_response):
        """Helper to extract errors and enriched metadata."""
        # Extract errors
        message = response_data.get("message")
        errors = None
        if message:
            from openai.types.batch import Errors
            from openai.types.batch_error import BatchError
            errors = Errors(
                data=[BatchError(message=message, code=str(raw_response.status_code))],
                object="list"
            )
        
        # Enrich metadata with useful Bedrock fields
        enriched_metadata_raw: Dict[str, Any] = {
            "jobName": response_data.get("jobName"),
            "clientRequestToken": response_data.get("clientRequestToken"),
            "modelId": response_data.get("modelId"),
            "roleArn": response_data.get("roleArn"),
            "timeoutDurationInHours": response_data.get("timeoutDurationInHours"),
            "vpcConfig": response_data.get("vpcConfig"),
        }
        import json as _json
        enriched_metadata: Dict[str, str] = {}
        for _k, _v in enriched_metadata_raw.items():
            if _v is None:
                continue
            if isinstance(_v, (dict, list)):
                try:
                    enriched_metadata[_k] = _json.dumps(_v)
                except Exception:
                    enriched_metadata[_k] = str(_v)
            else:
                enriched_metadata[_k] = str(_v)
        
        return errors, enriched_metadata

    def transform_retrieve_batch_response(
        self,
        model: Optional[str],
        raw_response: Response,
        logging_obj: Any,
        litellm_params: dict,
    ) -> LiteLLMBatch:
        """
        Transform Bedrock batch retrieval response to LiteLLM format.
        """
        from litellm.types.llms.bedrock import BedrockGetBatchResponse
        try:
            response_data: BedrockGetBatchResponse = raw_response.json()
        except Exception as e:
            raise ValueError(f"Failed to parse Bedrock batch response: {e}")
        
        job_arn = response_data.get("jobArn", "")
        status_str: str = str(response_data.get("status", "Submitted"))
        
        # Map Bedrock status to OpenAI-compatible status
        status_mapping: Dict[str, str] = {
            "Submitted": "validating", "Validating": "validating", "Scheduled": "in_progress",
            "InProgress": "in_progress", "PartiallyCompleted": "completed", "Completed": "completed",
            "Failed": "failed", "Stopping": "cancelling", "Stopped": "cancelled", "Expired": "expired"
        }
        openai_status = cast(Literal["validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"], status_mapping.get(status_str, "validating"))
        
        # Parse timestamps
        created_at, in_progress_at, completed_at, failed_at, cancelled_at, expires_at = self._parse_timestamps_and_status(response_data, status_str)
        
        # Extract file configurations
        input_file_id, output_file_id = self._extract_file_configs(response_data)
        
        # Extract errors and metadata
        errors, enriched_metadata = self._extract_errors_and_metadata(response_data, raw_response)
                
        return LiteLLMBatch(
            id=job_arn,
            object="batch",
            endpoint="/v1/chat/completions",
            errors=errors,
            input_file_id=input_file_id,
            completion_window="24h",
            status=openai_status,
            output_file_id=output_file_id,
            error_file_id=None,
            created_at=created_at or int(time.time()),
            in_progress_at=in_progress_at,
            expires_at=expires_at,
            finalizing_at=None,
            completed_at=completed_at,
            failed_at=failed_at,
            expired_at=None,
            cancelling_at=None,
            cancelled_at=cancelled_at,
            request_counts=None,
            metadata=enriched_metadata,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, Headers]
    ) -> BaseLLMException:
        """
        Get Bedrock-specific error class using common utility.
        """
        return self.common_utils.get_error_class(error_message, status_code, headers)


