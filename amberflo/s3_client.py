import asyncio
from functools import partial
from datetime import datetime

import boto3
from botocore.client import Config

from .blob_client import BlobWriter
from .utils import get_env, make_key
from .logging import get_logger


logger = get_logger(__name__)


class S3Client(BlobWriter):
    """
    Wrapper around the AWS S3 client.
    """

    def __init__(
        self,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        region_name=None,
        bucket=None,
        path=None,
    ):
        self.region_name = get_env("AWS_REGION", region_name, required=True)
        self.bucket = get_env("AFLO_BUCKET_NAME", bucket, required=True)
        self.path = get_env("AFLO_PATH", path)

        session = boto3.Session(
            aws_access_key_id=get_env("AWS_ACCESS_KEY_ID", aws_access_key_id),
            aws_secret_access_key=get_env(
                "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
            ),
            region_name=region_name,
        )

        self.s3 = session.client(
            "s3", config=Config(retries={"max_attempts": 5, "mode": "standard"})
        )

        logger.debug(
            "Initialized S3 client: bucket: '%s', path: '%s'", self.bucket, self.path
        )

    async def put_object(self, key: str, body: bytes) -> None:
        logger.debug(f"Attempting write to S3 bucket {self.bucket}: {key}")

        # boto3 is not async and we don't want to block the event loop, so we
        # run the upload in a separate thread.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(
                self.s3.put_object,
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
                ContentEncoding="gzip",
            ),
        )

        logger.debug(f"Wrote to S3 bucket {self.bucket}: {key}")

    def make_key(self, timestamp: datetime) -> str:
        return make_key(timestamp, self.path)
