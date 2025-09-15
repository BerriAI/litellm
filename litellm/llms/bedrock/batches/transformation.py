import os
import time
from typing import Any, Dict, List, Literal, Optional, Union, cast

from httpx import Headers, Response

from litellm.llms.base_llm.batches.transformation import BaseBatchesConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.bedrock import (
    BedrockBatchJobStatus,
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
        status: BedrockBatchJobStatus = response_data.get("status", "Submitted")
        
        # Map Bedrock status to OpenAI-compatible status
        status_mapping: Dict[BedrockBatchJobStatus, str] = {
            "Submitted": "validating",
            "InProgress": "in_progress", 
            "Completed": "completed",
            "Failed": "failed",
            "Stopping": "cancelling",
            "Stopped": "cancelled"
        }
        
        openai_status = cast(Literal["validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"], status_mapping.get(status, "validating"))
        
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
            in_progress_at=int(time.time()) if status == "InProgress" else None,
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

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, Headers]
    ) -> BaseLLMException:
        """
        Get Bedrock-specific error class using common utility.
        """
        return self.common_utils.get_error_class(error_message, status_code, headers)


