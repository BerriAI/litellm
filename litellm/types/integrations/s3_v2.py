from typing import Literal

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
