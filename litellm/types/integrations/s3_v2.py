from typing import Literal, TypedDict

from litellm.types.llms.base import LiteLLMBaseModel

S3PartitionGranularity = Literal["day", "hour"]


class s3BatchLoggingElement(LiteLLMBaseModel):
    """
    Type of element stored in self.log_queue in S3Logger
    """

    payload: dict
    s3_object_key: str
    s3_object_download_filename: str
    body: str | None = None
    content_type: str = "application/json"
    retrying_since: float | None = None


class S3LoggerSettings(TypedDict, total=False):
    s3_bucket_name: str | None
    s3_path: str | None
    s3_region_name: str | None
    s3_api_version: str | None
    s3_use_ssl: bool
    s3_verify: bool | str | None
    s3_endpoint_url: str | None
    s3_aws_access_key_id: str | None
    s3_aws_secret_access_key: str | None
    s3_aws_session_token: str | None
    s3_aws_session_name: str | None
    s3_aws_profile_name: str | None
    s3_aws_role_name: str | None
    s3_aws_web_identity_token: str | None
    s3_aws_sts_endpoint: str | None
    s3_flush_interval: int | None
    s3_batch_size: int | None
    s3_config: object
    s3_use_team_prefix: bool
    s3_strip_base64_files: bool
    s3_use_key_prefix: bool
    s3_use_virtual_hosted_style: bool
    s3_server_side_encryption: str | None
    s3_sse_kms_key_id: str | None
    s3_log_prompts_only: bool | None
    s3_partition_granularity: str | None
    s3_max_concurrent_uploads: int
    s3_max_queue_size: int | None
    s3_max_retry_age_seconds: int | None
    s3_drop_on_terminal_error: bool
    s3_adaptive_concurrency: bool
    s3_max_adaptive_concurrency: int | None
    s3_batch_file_upload: bool
