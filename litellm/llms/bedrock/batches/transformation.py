from typing import Any, Dict, Optional

from litellm.llms.bedrock.common_utils import convert_bedrock_datetime_to_openai_datetime
from litellm.types.llms.bedrock import CreateModelInvocationJobRequest
from litellm.types.llms.openai import BatchJobStatus, CreateBatchRequest
from litellm.types.utils import LiteLLMBatch


class BedrockBatchTransformation:
    """
    Transforms OpenAI Batch requests to Bedrock Batch requests and vice versa

    API Ref: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/get_model_invocation_job.html
    """

    # Map Bedrock status to OpenAI status
    _status_mapping: Dict[str, BatchJobStatus] = {
        "Submitted": "validating",
        "InProgress": "in_progress", 
        "Completed": "completed",
        "Failed": "failed",
        "Stopping": "cancelling",
        "Stopped": "cancelled",
        "PartiallyCompleted": "completed",
        "Expired": "expired",
        "Validating": "validating",
        "Scheduled": "validating",
    }

    @classmethod
    def transform_openai_batch_request_to_bedrock_job_request(
        cls,
        req: CreateBatchRequest,
        *,
        s3_input_uri: str,
        s3_output_uri: str,
        role_arn: Optional[str],
    ) -> CreateModelInvocationJobRequest:
        """
        Transform OpenAI CreateBatchRequest into Bedrock CreateModelInvocationJobRequest
        using S3 input/output URIs.
        """
        model = req.get("model") or ""
        # Handle metadata safely
        metadata = req.get("metadata") or {}
        job_name = metadata.get("job_name") or req.get("custom_id") or "litellm-batch-job"

        from litellm.types.llms.bedrock import InputDataConfig, OutputDataConfig

        inputDataConfig: InputDataConfig = {"s3InputDataConfig": {"s3Uri": s3_input_uri}}
        outputDataConfig: OutputDataConfig = {
            "s3OutputDataConfig": {"s3Uri": s3_output_uri}
        }

        payload: CreateModelInvocationJobRequest = {
            "modelId": str(model),
            "jobName": str(job_name),
            "inputDataConfig": inputDataConfig,
            "outputDataConfig": outputDataConfig,
            "roleArn": str(role_arn) if role_arn else "",
        }
        return payload

    @classmethod
    def transform_bedrock_response_to_openai_batch(
        cls,
        bedrock_response: Dict[str, Any],
        input_file_id: Optional[str] = None,
    ) -> LiteLLMBatch:
        """
        Transform Bedrock batch job response to OpenAI batch format.
        
        Maps Bedrock get_model_invocation_job response to LiteLLMBatch structure.
        
        Reference: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/get_model_invocation_job.html
        """
        
        bedrock_status = bedrock_response.get("status", "Failed")
        openai_status = cls._status_mapping.get(bedrock_status, "failed")
        
        # Extract job ID from jobArn
        job_arn = bedrock_response.get("jobArn", "")
        job_id = cls._get_batch_id_from_bedrock_response(job_arn)
        
        # Convert timestamps to Unix epoch
        created_at = convert_bedrock_datetime_to_openai_datetime(bedrock_response.get("submitTime"))
        
        # Determine timestamps based on status
        in_progress_at = None
        completed_at = None
        failed_at = None
        cancelled_at = None
        cancelling_at = None

        if openai_status == "in_progress":
            in_progress_at = convert_bedrock_datetime_to_openai_datetime(bedrock_response.get("lastModifiedTime"))
        elif openai_status == "cancelling":
            cancelling_at = convert_bedrock_datetime_to_openai_datetime(bedrock_response.get("lastModifiedTime"))
        elif openai_status == "cancelled":
            cancelled_at = convert_bedrock_datetime_to_openai_datetime(bedrock_response.get("lastModifiedTime"))
        elif openai_status == "completed":
            completed_at = convert_bedrock_datetime_to_openai_datetime(bedrock_response.get("endTime"))
        elif openai_status == "failed":
            failed_at = convert_bedrock_datetime_to_openai_datetime(bedrock_response.get("endTime"))

        # Add expired_at mapped from jobExpirationTime
        expires_at = convert_bedrock_datetime_to_openai_datetime(bedrock_response.get("jobExpirationTime"))
        
        # Extract file IDs from input/output configs
        input_s3_uri = cls._get_input_file_id_from_bedrock_response(bedrock_response)
        output_s3_uri = cls._get_output_file_id_from_bedrock_response(bedrock_response)
        
        # Use provided input_file_id or fall back to S3 URI
        resolved_input_file_id = input_file_id or input_s3_uri
        
        # Handle error information
        error_file_id, errors = cls._get_error_information_from_bedrock_response(bedrock_response, openai_status)

        return LiteLLMBatch(
            id=job_id,
            object="batch",
            endpoint="/v1/chat/completions",  # Default endpoint
            status=openai_status,
            input_file_id=resolved_input_file_id,
            output_file_id=output_s3_uri or None,
            error_file_id=error_file_id,
            created_at=created_at,
            in_progress_at=in_progress_at,
            completed_at=completed_at,
            cancelled_at=cancelled_at,
            cancelling_at=cancelling_at,
            failed_at=failed_at,
            expires_at=expires_at,
            completion_window="24h",  # Default completion window
            errors=errors,
            request_counts=None,  # Bedrock doesn't provide this information
            metadata=cls._get_metadata_from_bedrock_response(bedrock_response, job_arn),
        )

    @classmethod
    def _get_batch_id_from_bedrock_response(cls, job_arn: str) -> str:
        """
        Gets the batch id from the Bedrock response safely
        
        Bedrock response: `arn:aws:bedrock:region:account-id:model-invocation-job/job-id`
        returns: `job-id`
        """
        if not job_arn:
            return ""
        
        # Split by '/' and get the last part if it exists
        parts = job_arn.split("/")
        return parts[-1] if parts else job_arn

    @classmethod
    def _get_input_file_id_from_bedrock_response(cls, response: Dict[str, Any]) -> str:
        """
        Gets the input file id from the Bedrock response
        """
        input_config = response.get("inputDataConfig", {})
        s3_input_config = input_config.get("s3InputDataConfig", {})
        return s3_input_config.get("s3Uri", "")

    @classmethod
    def _get_output_file_id_from_bedrock_response(cls, response: Dict[str, Any]) -> str:
        """
        Gets the output file id from the Bedrock response
        """
        output_config = response.get("outputDataConfig", {})
        s3_output_config = output_config.get("s3OutputDataConfig", {})
        return s3_output_config.get("s3Uri", "")

    @classmethod
    def _get_error_information_from_bedrock_response(
        cls, 
        response: Dict[str, Any], 
        openai_status: str
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Gets error information from the Bedrock response
        """
        error_file_id = None
        errors = None

        if response.get("message") and openai_status == "failed":
            # Construct errors object as per OpenAI batch error schema
            errors = {
                "object": "list",
                "data": [
                    {
                        "object": "error",
                        "code": response.get("status", "failed"),
                        "message": response.get("message"),
                        "line": None,
                        "param": None
                    }
                ]
            }
            error_file_id = None  # Bedrock doesn't provide error files

        return error_file_id, errors

    @classmethod
    def _get_metadata_from_bedrock_response(cls, response: Dict[str, Any], job_arn: str) -> Dict[str, Any]:
        """
        Gets metadata from the Bedrock response
        """
        return {
            "job_name": response.get("jobName"),
            "model_id": response.get("modelId"),
            "job_arn": job_arn,
            "failure_message": response.get("message")
        }
