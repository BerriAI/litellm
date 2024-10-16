"""
s3 Bucket Logging Integration

async_log_success_event: Processes the event, stores it in memory for 10 seconds or until MAX_BATCH_SIZE and then flushes to s3 

NOTE 1: S3 does not provide a BATCH PUT API endpoint, so we create tasks to upload each element individually
NOTE 2: We create a httpx client with a concurrent limit of 1 to upload to s3. Files should get uploaded BUT they should not impact latency of LLM calling logic
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.llms.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.s3 import s3BatchLoggingElement
from litellm.types.utils import StandardLoggingPayload

from .custom_batch_logger import CustomBatchLogger

# Default Flush interval and batch size for s3
# Flush to s3 every 10 seconds OR every 1K requests in memory
DEFAULT_S3_FLUSH_INTERVAL_SECONDS = 10
DEFAULT_S3_BATCH_SIZE = 1000


class S3Logger(CustomBatchLogger, BaseAWSLLM):
    # Class variables or attributes
    def __init__(
        self,
        s3_bucket_name: Optional[str] = None,
        s3_path: Optional[str] = None,
        s3_region_name: Optional[str] = None,
        s3_api_version: Optional[str] = None,
        s3_use_ssl: bool = True,
        s3_verify: Optional[bool] = None,
        s3_endpoint_url: Optional[str] = None,
        s3_aws_access_key_id: Optional[str] = None,
        s3_aws_secret_access_key: Optional[str] = None,
        s3_aws_session_token: Optional[str] = None,
        s3_flush_interval: Optional[int] = DEFAULT_S3_FLUSH_INTERVAL_SECONDS,
        s3_batch_size: Optional[int] = DEFAULT_S3_BATCH_SIZE,
        s3_config=None,
        **kwargs,
    ):
        try:
            verbose_logger.debug(
                f"in init s3 logger - s3_callback_params {litellm.s3_callback_params}"
            )

            # IMPORTANT: We use a concurrent limit of 1 to upload to s3
            # Files should get uploaded BUT they should not impact latency of LLM calling logic
            self.async_httpx_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback,
                params={"concurrent_limit": 1},
            )

            if litellm.s3_callback_params is not None:
                # read in .env variables - example os.environ/AWS_BUCKET_NAME
                for key, value in litellm.s3_callback_params.items():
                    if type(value) is str and value.startswith("os.environ/"):
                        litellm.s3_callback_params[key] = litellm.get_secret(value)
                # now set s3 params from litellm.s3_logger_params
                s3_bucket_name = litellm.s3_callback_params.get("s3_bucket_name")
                s3_region_name = litellm.s3_callback_params.get("s3_region_name")
                s3_api_version = litellm.s3_callback_params.get("s3_api_version")
                s3_use_ssl = litellm.s3_callback_params.get("s3_use_ssl", True)
                s3_verify = litellm.s3_callback_params.get("s3_verify")
                s3_endpoint_url = litellm.s3_callback_params.get("s3_endpoint_url")
                s3_aws_access_key_id = litellm.s3_callback_params.get(
                    "s3_aws_access_key_id"
                )
                s3_aws_secret_access_key = litellm.s3_callback_params.get(
                    "s3_aws_secret_access_key"
                )
                s3_aws_session_token = litellm.s3_callback_params.get(
                    "s3_aws_session_token"
                )
                s3_config = litellm.s3_callback_params.get("s3_config")
                s3_path = litellm.s3_callback_params.get("s3_path")
                # done reading litellm.s3_callback_params

                s3_flush_interval = litellm.s3_callback_params.get(
                    "s3_flush_interval", DEFAULT_S3_FLUSH_INTERVAL_SECONDS
                )
                s3_batch_size = litellm.s3_callback_params.get(
                    "s3_batch_size", DEFAULT_S3_BATCH_SIZE
                )

            self.bucket_name = s3_bucket_name
            self.s3_path = s3_path
            verbose_logger.debug(f"s3 logger using endpoint url {s3_endpoint_url}")
            self.s3_bucket_name = s3_bucket_name
            self.s3_region_name = s3_region_name
            self.s3_api_version = s3_api_version
            self.s3_use_ssl = s3_use_ssl
            self.s3_verify = s3_verify
            self.s3_endpoint_url = s3_endpoint_url
            self.s3_aws_access_key_id = s3_aws_access_key_id
            self.s3_aws_secret_access_key = s3_aws_secret_access_key
            self.s3_aws_session_token = s3_aws_session_token
            self.s3_config = s3_config
            self.init_kwargs = kwargs

            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()

            verbose_logger.debug(
                f"s3 flush interval: {s3_flush_interval}, s3 batch size: {s3_batch_size}"
            )
            # Call CustomLogger's __init__
            CustomBatchLogger.__init__(
                self,
                flush_lock=self.flush_lock,
                flush_interval=s3_flush_interval,
                batch_size=s3_batch_size,
            )
            self.log_queue: List[s3BatchLoggingElement] = []

            # Call BaseAWSLLM's __init__
            BaseAWSLLM.__init__(self)

        except Exception as e:
            print_verbose(f"Got exception on init s3 client {str(e)}")
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                f"s3 Logging - Enters logging function for model {kwargs}"
            )

            s3_batch_logging_element = self.create_s3_batch_logging_element(
                start_time=start_time,
                standard_logging_payload=kwargs.get("standard_logging_object", None),
                s3_path=self.s3_path,
            )

            if s3_batch_logging_element is None:
                raise ValueError("s3_batch_logging_element is None")

            verbose_logger.debug(
                "\ns3 Logger - Logging payload = %s", s3_batch_logging_element
            )

            self.log_queue.append(s3_batch_logging_element)
            verbose_logger.debug(
                "s3 logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )
            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()
        except Exception as e:
            verbose_logger.exception(f"s3 Layer Error - {str(e)}")
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Synchronous logging function to log to s3

        Does not batch logging requests, instantly logs on s3 Bucket
        """
        try:
            s3_batch_logging_element = self.create_s3_batch_logging_element(
                start_time=start_time,
                standard_logging_payload=kwargs.get("standard_logging_object", None),
                s3_path=self.s3_path,
            )

            if s3_batch_logging_element is None:
                raise ValueError("s3_batch_logging_element is None")

            verbose_logger.debug(
                "\ns3 Logger - Logging payload = %s", s3_batch_logging_element
            )

            # log the element sync httpx client
            self.upload_data_to_s3(s3_batch_logging_element)
        except Exception as e:
            verbose_logger.exception(f"s3 Layer Error - {str(e)}")
            pass

    async def async_upload_data_to_s3(
        self, batch_logging_element: s3BatchLoggingElement
    ):
        try:
            import hashlib

            import boto3
            import requests
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        try:
            credentials: Credentials = self.get_credentials(
                aws_access_key_id=self.s3_aws_access_key_id,
                aws_secret_access_key=self.s3_aws_secret_access_key,
                aws_session_token=self.s3_aws_session_token,
                aws_region_name=self.s3_region_name,
            )

            # Prepare the URL
            url = f"https://{self.bucket_name}.s3.{self.s3_region_name}.amazonaws.com/{batch_logging_element.s3_object_key}"

            if self.s3_endpoint_url:
                url = self.s3_endpoint_url + "/" + batch_logging_element.s3_object_key

            # Convert JSON to string
            json_string = json.dumps(batch_logging_element.payload)

            # Calculate SHA256 hash of the content
            content_hash = hashlib.sha256(json_string.encode("utf-8")).hexdigest()

            # Prepare the request
            headers = {
                "Content-Type": "application/json",
                "x-amz-content-sha256": content_hash,
                "Content-Language": "en",
                "Content-Disposition": f'inline; filename="{batch_logging_element.s3_object_download_filename}"',
                "Cache-Control": "private, immutable, max-age=31536000, s-maxage=0",
            }
            req = requests.Request("PUT", url, data=json_string, headers=headers)
            prepped = req.prepare()

            # Sign the request
            aws_request = AWSRequest(
                method=prepped.method,
                url=prepped.url,
                data=prepped.body,
                headers=prepped.headers,
            )
            SigV4Auth(credentials, "s3", self.s3_region_name).add_auth(aws_request)

            # Prepare the signed headers
            signed_headers = dict(aws_request.headers.items())

            # Make the request
            response = await self.async_httpx_client.put(
                url, data=json_string, headers=signed_headers
            )
            response.raise_for_status()
        except Exception as e:
            verbose_logger.exception(f"Error uploading to s3: {str(e)}")

    async def async_send_batch(self):
        """

        Sends runs from self.log_queue

        Returns: None

        Raises: Does not raise an exception, will only verbose_logger.exception()
        """
        if not self.log_queue:
            return

        for payload in self.log_queue:
            asyncio.create_task(self.async_upload_data_to_s3(payload))

    def create_s3_batch_logging_element(
        self,
        start_time: datetime,
        standard_logging_payload: Optional[StandardLoggingPayload],
        s3_path: Optional[str],
    ) -> Optional[s3BatchLoggingElement]:
        """
        Helper function to create an s3BatchLoggingElement.

        Args:
            start_time (datetime): The start time of the logging event.
            standard_logging_payload (Optional[StandardLoggingPayload]): The payload to be logged.
            s3_path (Optional[str]): The S3 path prefix.

        Returns:
            Optional[s3BatchLoggingElement]: The created s3BatchLoggingElement, or None if payload is None.
        """
        if standard_logging_payload is None:
            return None

        s3_file_name = (
            litellm.utils.get_logging_id(start_time, standard_logging_payload) or ""
        )
        s3_object_key = (
            (s3_path.rstrip("/") + "/" if s3_path else "")
            + start_time.strftime("%Y-%m-%d")
            + "/"
            + s3_file_name
            + ".json"
        )

        s3_object_download_filename = f"time-{start_time.strftime('%Y-%m-%dT%H-%M-%S-%f')}_{standard_logging_payload['id']}.json"

        return s3BatchLoggingElement(
            payload=standard_logging_payload,  # type: ignore
            s3_object_key=s3_object_key,
            s3_object_download_filename=s3_object_download_filename,
        )

    def upload_data_to_s3(self, batch_logging_element: s3BatchLoggingElement):
        try:
            import hashlib

            import boto3
            import requests
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        try:
            credentials: Credentials = self.get_credentials(
                aws_access_key_id=self.s3_aws_access_key_id,
                aws_secret_access_key=self.s3_aws_secret_access_key,
                aws_session_token=self.s3_aws_session_token,
                aws_region_name=self.s3_region_name,
            )

            # Prepare the URL
            url = f"https://{self.bucket_name}.s3.{self.s3_region_name}.amazonaws.com/{batch_logging_element.s3_object_key}"

            if self.s3_endpoint_url:
                url = self.s3_endpoint_url + "/" + batch_logging_element.s3_object_key

            # Convert JSON to string
            json_string = json.dumps(batch_logging_element.payload)

            # Calculate SHA256 hash of the content
            content_hash = hashlib.sha256(json_string.encode("utf-8")).hexdigest()

            # Prepare the request
            headers = {
                "Content-Type": "application/json",
                "x-amz-content-sha256": content_hash,
                "Content-Language": "en",
                "Content-Disposition": f'inline; filename="{batch_logging_element.s3_object_download_filename}"',
                "Cache-Control": "private, immutable, max-age=31536000, s-maxage=0",
            }
            req = requests.Request("PUT", url, data=json_string, headers=headers)
            prepped = req.prepare()

            # Sign the request
            aws_request = AWSRequest(
                method=prepped.method,
                url=prepped.url,
                data=prepped.body,
                headers=prepped.headers,
            )
            SigV4Auth(credentials, "s3", self.s3_region_name).add_auth(aws_request)

            # Prepare the signed headers
            signed_headers = dict(aws_request.headers.items())

            httpx_client = _get_httpx_client()
            # Make the request
            response = httpx_client.put(url, data=json_string, headers=signed_headers)
            response.raise_for_status()
        except Exception as e:
            verbose_logger.exception(f"Error uploading to s3: {str(e)}")
