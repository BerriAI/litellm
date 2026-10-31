"""Export request-level spend records to S3 for analysis in Hex."""

import os
from datetime import datetime
from typing import Final

from litellm.constants import DEFAULT_S3_BATCH_SIZE, DEFAULT_S3_FLUSH_INTERVAL_SECONDS
from litellm.integrations.s3_v2 import S3Logger
from litellm.types.integrations.s3_v2 import s3BatchLoggingElement
from litellm.types.utils import StandardLoggingPayload


class HexLogger(S3Logger):
    """Write spend-only JSON records using the standard AWS credential chain.

    Enable with ``callbacks: ["hex"]`` and ``HEX_S3_BUCKET_NAME``. Connect the
    bucket through Hex's Athena connection to query newly arriving records.
    Prompts, responses, API keys,
    arbitrary metadata, and error messages are never included in the export.
    """

    def __init__(
        self,
        bucket_name: str | None = None,
        path: str | None = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        flush_interval: int = DEFAULT_S3_FLUSH_INTERVAL_SECONDS,
        batch_size: int = DEFAULT_S3_BATCH_SIZE,
    ) -> None:
        resolved_bucket: Final = bucket_name or os.getenv("HEX_S3_BUCKET_NAME")
        if not resolved_bucket or not resolved_bucket.strip():
            raise ValueError("Hex requires HEX_S3_BUCKET_NAME or bucket_name.")
        if flush_interval <= 0 or batch_size <= 0:
            raise ValueError("Hex flush_interval and batch_size must be positive.")

        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # S3Logger has legacy untyped constructor options.
            s3_bucket_name=resolved_bucket,
            s3_path=path if path is not None else os.getenv("HEX_S3_PATH", "litellm-spend"),
            s3_region_name=region_name or os.getenv("HEX_S3_REGION_NAME") or os.getenv("AWS_REGION") or "us-east-1",
            s3_endpoint_url=endpoint_url or os.getenv("HEX_S3_ENDPOINT_URL"),
            s3_flush_interval=flush_interval,
            s3_batch_size=batch_size,
            s3_batch_file_upload=True,
            s3_partition_granularity="day",
            # Keep the Hex destination independent of an existing S3 callback.
            s3_callback_params_override={},
        )

    def create_s3_batch_logging_element(
        self,
        start_time: datetime,
        standard_logging_payload: StandardLoggingPayload | None,
    ) -> s3BatchLoggingElement | None:
        element: Final = super().create_s3_batch_logging_element(start_time, standard_logging_payload)
        if element is None or standard_logging_payload is None:
            return None
        metadata: Final = standard_logging_payload["metadata"]
        # An allowlist keeps arbitrary request content out of the analytics bucket.
        element.payload = {
            "id": standard_logging_payload["id"],
            "start_time": standard_logging_payload["startTime"],
            "end_time": standard_logging_payload["endTime"],
            "model": standard_logging_payload["model"],
            "model_group": standard_logging_payload["model_group"],
            "custom_llm_provider": standard_logging_payload["custom_llm_provider"],
            "call_type": standard_logging_payload["call_type"],
            "status": standard_logging_payload["status"],
            "spend": standard_logging_payload["response_cost"],
            "prompt_tokens": standard_logging_payload["prompt_tokens"],
            "completion_tokens": standard_logging_payload["completion_tokens"],
            "total_tokens": standard_logging_payload["total_tokens"],
            "response_time": standard_logging_payload["response_time"],
            "cache_hit": standard_logging_payload["cache_hit"],
            "team_id": metadata["user_api_key_team_id"],
            "organization_id": metadata["user_api_key_org_id"],
            "user_id": metadata["user_api_key_user_id"],
            "key_alias": metadata["user_api_key_alias"],
        }
        return element
