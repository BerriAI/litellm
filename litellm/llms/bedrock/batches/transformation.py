from typing import Dict, Optional

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
    job_name = req.get("metadata", {}).get("job_name") or req.get("custom_id") or "litellm-batch-job"

    inputDataConfig: Dict = {
        "s3InputDataConfig": {"s3Uri": s3_input_uri}
    }
    outputDataConfig: Dict = {
        "s3OutputDataConfig": {"s3Uri": s3_output_uri}
    }

    payload: CreateModelInvocationJobRequest = {
        "modelId": model,
        "jobName": job_name,
        "inputDataConfig": inputDataConfig,
        "outputDataConfig": outputDataConfig,
    }
    if role_arn:
        payload["roleArn"] = role_arn
    return payload
