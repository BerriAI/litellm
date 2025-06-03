from typing import Dict

from pydantic import BaseModel

from litellm.types.utils import StandardLoggingPayload


class s3BatchLoggingElement(BaseModel):
    """
    Type of element stored in self.log_queue in S3Logger
    """

    payload: StandardLoggingPayload
    s3_object_key: str
    s3_object_download_filename: str
