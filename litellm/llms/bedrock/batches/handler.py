from datetime import datetime
from typing import Any, Optional, cast

from openai.types.batch import BatchRequestCounts
from openai.types.batch import Metadata as OpenAIBatchMetadata

from litellm.types.utils import LiteLLMBatch

# AWS Bedrock model-invocation-job statuses -> OpenAI Batch statuses.
_BEDROCK_MIJ_STATUS_TO_OPENAI = {
    "Submitted": "validating",
    "Validating": "validating",
    "Scheduled": "validating",
    "InProgress": "in_progress",
    "Stopping": "cancelling",
    "Stopped": "cancelled",
    "Completed": "completed",
    "PartiallyCompleted": "completed",
    "Failed": "failed",
    "Expired": "expired",
}


def _extract_region_from_bedrock_arn(arn: str) -> Optional[str]:
    """ARN shape: ``arn:aws:bedrock:<region>:<account>:<type>/<id>``"""
    try:
        parts = arn.split(":")
        if len(parts) >= 4 and parts[2] == "bedrock":
            return parts[3] or None
    except Exception:
        pass
    return None


def _extract_job_id_from_arn(arn: str) -> Optional[str]:
    """``arn:aws:bedrock:<region>:<acct>:model-invocation-job/<job-id>`` -> ``<job-id>``."""
    if ":model-invocation-job/" not in arn:
        return None
    return arn.rsplit("/", 1)[-1] or None


def _predict_output_file_uri(
    output_prefix: str, input_uri: str, job_id: Optional[str]
) -> Optional[str]:
    if not output_prefix or not input_uri or not job_id:
        return None
    if not output_prefix.endswith("/"):
        output_prefix = output_prefix + "/"
    input_basename = input_uri.rsplit("/", 1)[-1]
    if not input_basename:
        return None
    return f"{output_prefix}{job_id}/{input_basename}.out"


def _to_epoch(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, datetime):
        return int(value.timestamp())
    return None


class BedrockBatchesHandler:
    """Handler for Bedrock Batches."""

    @staticmethod
    def cancel_batch(
        batch_id: str,
        aws_region_name: Optional[str] = None,
        logging_obj=None,
        **kwargs,
    ) -> "LiteLLMBatch":
        """
        Cancel an AWS Bedrock batch model invocation job using StopModelInvocationJob.
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise ImportError(
                "Missing boto3/botocore to call bedrock. Run 'pip install boto3'."
            ) from exc

        region = (
            aws_region_name or _extract_region_from_bedrock_arn(batch_id) or "us-east-1"
        )

        from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig

        creds = BedrockBatchesConfig().get_credentials(
            aws_access_key_id=kwargs.get("aws_access_key_id"),
            aws_secret_access_key=kwargs.get("aws_secret_access_key"),
            aws_session_token=kwargs.get("aws_session_token"),
            aws_region_name=region,
            aws_session_name=kwargs.get("aws_session_name"),
            aws_profile_name=kwargs.get("aws_profile_name"),
            aws_role_name=kwargs.get("aws_role_name"),
            aws_web_identity_token=kwargs.get("aws_web_identity_token"),
            aws_sts_endpoint=kwargs.get("aws_sts_endpoint"),
            aws_external_id=kwargs.get("aws_external_id"),
        )

        client = boto3.client(
            "bedrock",
            region_name=region,
            aws_access_key_id=creds.access_key,
            aws_secret_access_key=creds.secret_key,
            aws_session_token=creds.token,
        )

        try:
            client.stop_model_invocation_job(jobIdentifier=batch_id)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            error_msg = e.response.get("Error", {}).get("Message", "").lower()
            if error_code == "ValidationException" and any(
                term in error_msg for term in ["stop", "terminal", "completed", "already"]
            ):
                pass
            else:
                raise e

        return BedrockBatchesHandler._handle_model_invocation_job_status(
            batch_id=batch_id,
            aws_region_name=region,
            logging_obj=logging_obj,
            **kwargs,
        )

    @staticmethod
    def _handle_async_invoke_status(
        batch_id: str, aws_region_name: str, logging_obj=None, **kwargs
    ) -> "LiteLLMBatch":
        import asyncio
        from litellm.llms.bedrock.embed.embedding import BedrockEmbedding

        async def _async_get_status():
            embedding_handler = BedrockEmbedding()
            status_response = await embedding_handler._get_async_invoke_status(
                invocation_arn=batch_id,
                aws_region_name=aws_region_name,
                logging_obj=logging_obj,
                **kwargs,
            )

            openai_batch_metadata: OpenAIBatchMetadata = {
                "output_file_id": status_response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"],
                "failure_message": status_response.get("failureMessage") or "",
                "model_arn": status_response["modelArn"],
            }

            return LiteLLMBatch(
                id=status_response["invocationArn"],
                object="batch",
                status=status_response["status"],
                created_at=status_response["submitTime"],
                in_progress_at=status_response["lastModifiedTime"],
                completed_at=status_response.get("endTime"),
                failed_at=(
                    status_response.get("endTime")
                    if status_response["status"] == "failed"
                    else None
                ),
                request_counts=BatchRequestCounts(
                    total=1,
                    completed=1 if status_response["status"] == "completed" else 0,
                    failed=1 if status_response["status"] == "failed" else 0,
                ),
                metadata=openai_batch_metadata,
                completion_window="24h",
                endpoint="/v1/embeddings",
                input_file_id="",
            )

        import concurrent.futures

        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_async_get_status())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()

    @staticmethod
    def _handle_model_invocation_job_status(
        batch_id: str,
        aws_region_name: Optional[str] = None,
        logging_obj=None,
        **kwargs,
    ) -> "LiteLLMBatch":
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "Missing boto3 to call bedrock. Run 'pip install boto3'."
            ) from exc

        region = (
            aws_region_name or _extract_region_from_bedrock_arn(batch_id) or "us-east-1"
        )

        from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig

        creds = BedrockBatchesConfig().get_credentials(
            aws_access_key_id=kwargs.get("aws_access_key_id"),
            aws_secret_access_key=kwargs.get("aws_secret_access_key"),
            aws_session_token=kwargs.get("aws_session_token"),
            aws_region_name=region,
            aws_session_name=kwargs.get("aws_session_name"),
            aws_profile_name=kwargs.get("aws_profile_name"),
            aws_role_name=kwargs.get("aws_role_name"),
            aws_web_identity_token=kwargs.get("aws_web_identity_token"),
            aws_sts_endpoint=kwargs.get("aws_sts_endpoint"),
            aws_external_id=kwargs.get("aws_external_id"),
        )

        client = boto3.client(
            "bedrock",
            region_name=region,
            aws_access_key_id=creds.access_key,
            aws_secret_access_key=creds.secret_key,
            aws_session_token=creds.token,
        )

        if logging_obj is not None:
            url_path_id = _extract_job_id_from_arn(batch_id) or batch_id
            logging_obj.pre_call(
                input=batch_id,
                api_key="",
                additional_args={
                    "complete_input_dict": {"jobIdentifier": batch_id},
                    "api_base": (
                        f"https://bedrock.{region}.amazonaws.com/"
                        f"model-invocation-job/{url_path_id}"
                    ),
                },
            )

        response = client.get_model_invocation_job(jobIdentifier=batch_id)

        if logging_obj is not None:
            logging_obj.post_call(
                input=batch_id,
                api_key="",
                original_response=response,
                additional_args={"complete_input_dict": {"jobIdentifier": batch_id}},
            )

        bedrock_status = str(response.get("status", ""))
        openai_status = cast(
            Any,
            _BEDROCK_MIJ_STATUS_TO_OPENAI.get(bedrock_status, "in_progress"),
        )

        input_uri = (
            response.get("inputDataConfig", {})
            .get("s3InputDataConfig", {})
            .get("s3Uri", "")
        )
        output_prefix = (
            response.get("outputDataConfig", {})
            .get("s3OutputDataConfig", {})
            .get("s3Uri", "")
        )

        job_arn = response.get("jobArn", batch_id)
        job_id = _extract_job_id_from_arn(job_arn)
        output_file_uri = _predict_output_file_uri(output_prefix, input_uri, job_id)

        completed_at = _to_epoch(response.get("endTime"))

        openai_batch_metadata: OpenAIBatchMetadata = {
            "model_arn": response.get("modelId", ""),
            "job_arn": job_arn,
            "job_name": response.get("jobName", ""),
            "failure_message": response.get("message") or "",
            "input_s3_uri": input_uri,
            "output_s3_uri": output_prefix,
            "output_file_uri": output_file_uri or "",
        }

        return LiteLLMBatch(
            id=job_arn,
            object="batch",
            status=openai_status,
            created_at=_to_epoch(response.get("submitTime")) or 0,
            in_progress_at=_to_epoch(response.get("lastModifiedTime")),
            completed_at=completed_at if openai_status == "completed" else None,
            failed_at=completed_at if openai_status == "failed" else None,
            cancelled_at=completed_at if openai_status == "cancelled" else None,
            expired_at=completed_at if openai_status == "expired" else None,
            request_counts=BatchRequestCounts(total=0, completed=0, failed=0),
            metadata=openai_batch_metadata,
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=input_uri,
            output_file_id=output_file_uri if openai_status == "completed" else None,
        )
