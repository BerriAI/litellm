from pydantic import BaseModel


class s3BatchLoggingElement(BaseModel):
    """
    Type of element stored in self.log_queue in S3Logger
    """

    payload: dict
    s3_object_key: str
    s3_object_download_filename: str
