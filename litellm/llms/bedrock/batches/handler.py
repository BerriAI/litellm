from datetime import datetime
from typing import Any, Optional, cast

from openai.types.batch import BatchRequestCounts
from openai.types.batch import Metadata as OpenAIBatchMetadata

from litellm.types.utils import LiteLLMBatch

# AWS Bedrock model-invocation-job statuses → OpenAI Batch statuses.
# Mirrors the mapping used by `BedrockBatchesConfig.transform_create_batch_response`
# so create / retrieve return consistent statuses.
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
    """
    Compute the deterministic per-job result file URI Bedrock writes to.

    Bedrock lays results out as::

        <output_prefix>/<job-id>/<basename(input_uri)>.out

    We compute it client-side so OpenAI-style ``client.files.content(output_file_id)``
    works without an extra S3 ``ListObjectsV2`` round-trip. Returns ``None`` if we
    don't have enough info; callers should fall back to the bare prefix.
    """
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
    """
    Handler for Bedrock Batches.

    Specific providers/models needed some special handling.

    E.g. Twelve Labs Embedding Async Invoke
    """

    @staticmethod
    def _handle_async_invoke_status(
        batch_id: str, aws_region_name: str, logging_obj=None, **kwargs
    ) -> "LiteLLMBatch":
        """
        Handle async invoke status check for AWS Bedrock.

        This is for Twelve Labs Embedding Async Invoke.

        Args:
            batch_id: The async invoke ARN
            aws_region_name: AWS region name
            **kwargs: Additional parameters

        Returns:
            dict: Status information including status, output_file_id (S3 URL), etc.
        """
        import asyncio

        from litellm.llms.bedrock.embed.embedding import BedrockEmbedding

        async def _async_get_status():
            # Create embedding handler instance
            embedding_handler = BedrockEmbedding()

            # Get the status of the async invoke job
            status_response = await embedding_handler._get_async_invoke_status(
                invocation_arn=batch_id,
                aws_region_name=aws_region_name,
                logging_obj=logging_obj,
                **kwargs,
            )

            # Transform response to a LiteLLMBatch object
            from litellm.types.utils import LiteLLMBatch

            openai_batch_metadata: OpenAIBatchMetadata = {
                "output_file_id": status_response["outputDataConfig"][
                    "s3OutputDataConfig"
                ]["s3Uri"],
                "failure_message": status_response.get("failureMessage") or "",
                "model_arn": status_response["modelArn"],
            }

            result = LiteLLMBatch(
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

            return result

        # Since this function is called from within an async context via run_in_executor,
        # we need to create a new event loop in a thread to avoid conflicts
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
        """
        Handle ``GetModelInvocationJob`` status check for AWS Bedrock bulk batch
        inference jobs (the ARN type returned by ``CreateModelInvocationJob``).

        ``CreateModelInvocationJob`` lives on the Bedrock **control plane**
        (``bedrock.<region>.amazonaws.com``), distinct from the data-plane
        ``bedrock-runtime`` endpoint that serves Twelve Labs async-invoke ARNs.
        The two ARN families therefore can't share a handler — see
        ``litellm/batches/main.py`` for the dispatch.

        Args:
            batch_id: A ``arn:aws:bedrock:<region>:<acct>:model-invocation-job/<id>``
                ARN (or just the trailing job id; both are accepted by
                ``GetModelInvocationJob``).
            aws_region_name: Region for the boto3 ``bedrock`` client. If omitted,
                we fall back to parsing the region out of ``batch_id`` itself.
            logging_obj: Optional litellm logging object.
            **kwargs: Optional AWS credential overrides
                (``aws_access_key_id``, ``aws_secret_access_key``,
                ``aws_session_token``, ``aws_profile_name``,
                ``aws_role_name``, ``aws_session_name``,
                ``aws_web_identity_token``, ``aws_sts_endpoint``,
                ``aws_external_id``). Unknown keys are ignored.

        Returns:
            ``LiteLLMBatch`` shaped like an OpenAI Batch resource. Note that
            ``request_counts`` is always ``(0, 0, 0)`` because
            ``GetModelInvocationJob`` does not surface per-record counts;
            callers that need accurate counts should parse
            ``manifest.json.out`` from the output S3 prefix.
        """
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "Missing boto3 to call bedrock. Run 'pip install boto3'."
            ) from exc

        # Resolve region: explicit > parsed-from-ARN > us-east-1 (boto3 default).
        region = (
            aws_region_name or _extract_region_from_bedrock_arn(batch_id) or "us-east-1"
        )

        # Resolve credentials through the same path the rest of the bedrock
        # provider uses, so model_list / env / role-assumption configs are
        # honored. We instantiate BedrockBatchesConfig (which extends
        # BaseAWSLLM) lazily to avoid a circular import at module load.
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
            # Use the bare job id in the logged URL so we don't double up the
            # `model-invocation-job/` segment when `batch_id` is a full ARN.
            # `GetModelInvocationJob` accepts either form, but only the bare id
            # produces a sensible-looking URL in logs.
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

        # Bedrock returns the output *prefix* the user supplied at job creation.
        # Actual results land at <prefix>/<job-id>/<basename(input)>.out — we
        # surface that single-file URI as `output_file_id` so the OpenAI-style
        # download flow works without an extra S3 listing call. We deliberately
        # do NOT fall back to the bare prefix when prediction fails: a prefix
        # is not a downloadable object, so handing it back as `output_file_id`
        # would reproduce the very NoSuchKey bug this handler exists to fix.
        # The bare prefix is preserved in metadata for callers that want the
        # `manifest.json.out` or want to do their own listing.
        job_arn = response.get("jobArn", batch_id)
        job_id = _extract_job_id_from_arn(job_arn)
        output_file_uri = _predict_output_file_uri(output_prefix, input_uri, job_id)

        completed_at = _to_epoch(response.get("endTime"))

        # Note: metadata uses "" (not None) for unknown URIs to satisfy the
        # OpenAI Batch metadata schema, which is `dict[str, str]`. The
        # `output_file_id` field on the LiteLLMBatch itself does carry None
        # correctly (see below), so callers should branch on that, not on
        # `metadata["output_file_uri"]`.
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
