from typing import Optional

from litellm.types.llms.bedrock import CreateModelInvocationJobRequest
from litellm.types.llms.openai import CreateBatchRequest


def transform_openai_create_batch_to_bedrock_job_request(
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
