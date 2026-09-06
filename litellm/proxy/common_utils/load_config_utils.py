import asyncio
import os
import posixpath
from collections.abc import Awaitable, Mapping
from typing import TYPE_CHECKING, Final, Protocol

import yaml
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.config_includes import resolve_includes

if TYPE_CHECKING:
    from litellm.integrations.gcs_bucket.gcs_bucket_base import GCSBucketBase

_BUCKET_CONFIG_ADAPTER: Final = TypeAdapter(dict[str, object])


class BucketObjectFetcher(Protocol):
    def __call__(self, object_key: str, /) -> Awaitable[Mapping[str, object] | None]: ...


class BucketObjectReader(Protocol):
    def __call__(self, object_key: str, /) -> Awaitable[object | None]: ...


def get_file_contents_from_s3(bucket_name, object_key):
    try:
        # v0 rely on boto3 for authentication - allowing boto3 to handle IAM credentials etc
        import boto3
        from botocore.credentials import Credentials

        from litellm.main import bedrock_converse_chat_completion

        credentials: Final[Credentials] = bedrock_converse_chat_completion.get_credentials()
        s3_client: Final = boto3.client(
            "s3",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,  # Optional, if using temporary credentials
        )
        verbose_proxy_logger.debug("Retrieving %s from S3 bucket: %s", object_key, bucket_name)
        response: Final = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        verbose_proxy_logger.debug("Response: %s", response)

        # Read the file contents and directly parse YAML
        file_contents: Final = response["Body"].read().decode("utf-8")
        verbose_proxy_logger.debug("File contents retrieved from S3")

        # Parse YAML directly from string
        config: Final = yaml.safe_load(file_contents)
        return config

    except ImportError as e:
        # this is most likely if a user is not using the litellm docker container
        verbose_proxy_logger.error("ImportError: %s", e)
    except Exception as e:
        verbose_proxy_logger.error("Error retrieving file contents: %s", e)
        return None


def gcs_config_bucket(bucket_name: str) -> "GCSBucketBase | None":
    try:
        from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger

        return GCSBucketLogger(bucket_name=bucket_name)
    except Exception as e:
        verbose_proxy_logger.error("Error creating the GCS client for bucket %s: %s", bucket_name, e)
        return None


async def get_config_file_contents_from_gcs(bucket_name, object_key, gcs_bucket=None):
    try:
        bucket: Final = gcs_config_bucket(bucket_name) if gcs_bucket is None else gcs_bucket
        if bucket is None:
            return None
        file_contents = await bucket.download_gcs_object(object_key)
        if file_contents is None:
            raise Exception(f"File contents are None for {object_key}")
        # file_contentis is a bytes object, so we need to convert it to yaml
        file_contents = file_contents.decode("utf-8")
        # convert to yaml
        config: Final = yaml.safe_load(file_contents)
        return config

    except Exception as e:
        verbose_proxy_logger.error("Error retrieving file contents: %s", e)
        return None


def resolve_include_object_key(config_object_key: str, include_entry: str) -> str:
    """
    Resolve one `include` entry to the object key it names, relative to the config object's prefix.

    A leading "/" means the bucket root, mirroring how an absolute path on disk ignores the
    directory the including config sits in.
    """
    if include_entry.startswith("/"):
        return posixpath.normpath(include_entry).lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(config_object_key), include_entry))


async def resolve_bucket_includes(
    *,
    config: Mapping[str, object],
    object_key: str,
    fetch: BucketObjectFetcher,
) -> dict[str, object]:
    async def load(include_entry: str, declared_in: str) -> tuple[str, Mapping[str, object]]:
        include_key: Final = resolve_include_object_key(declared_in, include_entry)
        included: Final = await fetch(include_key)
        if included is None:
            raise FileNotFoundError(f"Included config could not be read from bucket: {include_key}")
        return include_key, included

    return await resolve_includes(config=config, location=object_key, load=load)


def bucket_object_reader(bucket_type: str | None, bucket_name: str) -> BucketObjectReader:
    """
    Build one reader for a whole config, so an `include` tree costs one bucket client rather than one per object.
    """
    if bucket_type != "gcs":

        async def read_from_s3(object_key: str) -> object | None:
            return await asyncio.to_thread(get_file_contents_from_s3, bucket_name, object_key)

        return read_from_s3

    gcs_bucket: Final = gcs_config_bucket(bucket_name)

    async def read_from_gcs(object_key: str) -> object | None:
        if gcs_bucket is None:
            return None
        return await get_config_file_contents_from_gcs(bucket_name, object_key, gcs_bucket)

    return read_from_gcs


async def get_config_from_bucket(
    *,
    bucket_type: str | None,
    bucket_name: str,
    object_key: str,
) -> dict[str, object] | None:
    read: Final = bucket_object_reader(bucket_type, bucket_name)

    async def fetch(key: str) -> Mapping[str, object] | None:
        raw: Final = await read(key)
        if raw is None:
            return None
        try:
            return _BUCKET_CONFIG_ADAPTER.validate_python(raw)
        except ValidationError as e:
            raise ValueError(f"Config object in bucket is not a YAML mapping: {key}") from e

    config: Final = await fetch(object_key)
    if config is None:
        return None

    return await resolve_bucket_includes(config=config, object_key=object_key, fetch=fetch)


def download_python_file_from_s3(
    bucket_name: str,
    object_key: str,
    local_file_path: str,
) -> bool:
    """
    Download a Python file from S3 and save it to local filesystem.

    Args:
        bucket_name (str): S3 bucket name
        object_key (str): S3 object key (file path in bucket)
        local_file_path (str): Local path where file should be saved

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import boto3
        from botocore.credentials import Credentials

        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        base_aws_llm: Final = BaseAWSLLM()

        credentials: Final[Credentials] = base_aws_llm.get_credentials()
        s3_client: Final = boto3.client(
            "s3",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
        )

        verbose_proxy_logger.debug("Downloading Python file %s from S3 bucket: %s", object_key, bucket_name)
        response: Final = s3_client.get_object(Bucket=bucket_name, Key=object_key)

        # Read the file contents
        file_contents: Final = response["Body"].read().decode("utf-8")
        verbose_proxy_logger.debug("File contents: %s", file_contents)

        # Ensure directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Write to local file
        with open(local_file_path, "w") as f:
            f.write(file_contents)

        verbose_proxy_logger.debug("Python file downloaded successfully to %s", local_file_path)
        return True

    except ImportError as e:
        verbose_proxy_logger.error("ImportError: %s", e)
        return False
    except Exception as e:
        verbose_proxy_logger.exception("Error downloading Python file: %s", e)
        return False


async def download_python_file_from_gcs(
    bucket_name: str,
    object_key: str,
    local_file_path: str,
) -> bool:
    """
    Download a Python file from GCS and save it to local filesystem.

    Args:
        bucket_name (str): GCS bucket name
        object_key (str): GCS object key (file path in bucket)
        local_file_path (str): Local path where file should be saved

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger

        gcs_bucket: Final = GCSBucketLogger(
            bucket_name=bucket_name,
        )
        file_contents = await gcs_bucket.download_gcs_object(object_key)
        if file_contents is None:
            raise Exception(f"File contents are None for {object_key}")

        # file_contents is a bytes object, decode it
        file_contents = file_contents.decode("utf-8")

        # Ensure directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Write to local file
        with open(local_file_path, "w") as f:
            f.write(file_contents)

        verbose_proxy_logger.debug("Python file downloaded successfully to %s", local_file_path)
        return True

    except Exception as e:
        verbose_proxy_logger.exception("Error downloading Python file from GCS: %s", e)
        return False


# # Example usage
# bucket_name = 'litellm-proxy'
# object_key = 'litellm_proxy_config.yaml'
