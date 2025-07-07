"""
s3 Bucket Logging Integration

async_log_success_event: Processes the event, stores it in memory for DEFAULT_S3_FLUSH_INTERVAL_SECONDS seconds or until DEFAULT_S3_BATCH_SIZE and then flushes to s3 
async_log_failure_event: Processes the event, stores it in memory for DEFAULT_S3_FLUSH_INTERVAL_SECONDS seconds or until DEFAULT_S3_BATCH_SIZE and then flushes to s3 
NOTE 1: S3 does not provide a BATCH PUT API endpoint, so we create tasks to upload each element individually
"""

import asyncio
import json
from datetime import datetime
from typing import List, Optional, cast

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.constants import DEFAULT_S3_BATCH_SIZE, DEFAULT_S3_FLUSH_INTERVAL_SECONDS
from litellm.integrations.s3 import get_s3_object_key
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.s3_v2 import s3BatchLoggingElement
from litellm.types.utils import StandardLoggingPayload

from .custom_batch_logger import CustomBatchLogger


class S3Logger(CustomBatchLogger, BaseAWSLLM):
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
        s3_aws_session_name: Optional[str] = None,
        s3_aws_profile_name: Optional[str] = None,
        s3_aws_role_name: Optional[str] = None,
        s3_aws_web_identity_token: Optional[str] = None,
        s3_aws_sts_endpoint: Optional[str] = None,
        s3_flush_interval: Optional[int] = DEFAULT_S3_FLUSH_INTERVAL_SECONDS,
        s3_batch_size: Optional[int] = DEFAULT_S3_BATCH_SIZE,
        s3_config=None,
        s3_use_team_prefix: bool = False,
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
            )

            self._init_s3_params(
                s3_bucket_name=s3_bucket_name,
                s3_region_name=s3_region_name,
                s3_api_version=s3_api_version,
                s3_use_ssl=s3_use_ssl,
                s3_verify=s3_verify,
                s3_endpoint_url=s3_endpoint_url,
                s3_aws_access_key_id=s3_aws_access_key_id,
                s3_aws_secret_access_key=s3_aws_secret_access_key,
                s3_aws_session_token=s3_aws_session_token,
                s3_aws_session_name=s3_aws_session_name,
                s3_aws_profile_name=s3_aws_profile_name,
                s3_aws_role_name=s3_aws_role_name,
                s3_aws_web_identity_token=s3_aws_web_identity_token,
                s3_aws_sts_endpoint=s3_aws_sts_endpoint,
                s3_config=s3_config,
                s3_path=s3_path,
                s3_use_team_prefix=s3_use_team_prefix,
            )
            verbose_logger.debug(f"s3 logger using endpoint url {s3_endpoint_url}")

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

    def _init_s3_params(
        self,
        s3_bucket_name: Optional[str] = None,
        s3_region_name: Optional[str] = None,
        s3_api_version: Optional[str] = None,
        s3_use_ssl: bool = True,
        s3_verify: Optional[bool] = None,
        s3_endpoint_url: Optional[str] = None,
        s3_aws_access_key_id: Optional[str] = None,
        s3_aws_secret_access_key: Optional[str] = None,
        s3_aws_session_token: Optional[str] = None,
        s3_aws_session_name: Optional[str] = None,
        s3_aws_profile_name: Optional[str] = None,
        s3_aws_role_name: Optional[str] = None,
        s3_aws_web_identity_token: Optional[str] = None,
        s3_aws_sts_endpoint: Optional[str] = None,
        s3_config=None,
        s3_path: Optional[str] = None,
        s3_use_team_prefix: bool = False,
    ):
        """
        Initialize the s3 params for this logging callback
        """
        litellm.s3_callback_params = litellm.s3_callback_params or {}
        # read in .env variables - example os.environ/AWS_BUCKET_NAME
        for key, value in litellm.s3_callback_params.items():
            if isinstance(value, str) and value.startswith("os.environ/"):
                litellm.s3_callback_params[key] = litellm.get_secret(value)

        self.s3_bucket_name = (
            litellm.s3_callback_params.get("s3_bucket_name") or s3_bucket_name
        )
        self.s3_region_name = (
            litellm.s3_callback_params.get("s3_region_name") or s3_region_name
        )
        self.s3_api_version = (
            litellm.s3_callback_params.get("s3_api_version") or s3_api_version
        )
        self.s3_use_ssl = (
            litellm.s3_callback_params.get("s3_use_ssl", True) or s3_use_ssl
        )
        self.s3_verify = litellm.s3_callback_params.get("s3_verify") or s3_verify
        self.s3_endpoint_url = (
            litellm.s3_callback_params.get("s3_endpoint_url") or s3_endpoint_url
        )
        self.s3_aws_access_key_id = (
            litellm.s3_callback_params.get("s3_aws_access_key_id")
            or s3_aws_access_key_id
        )

        self.s3_aws_secret_access_key = (
            litellm.s3_callback_params.get("s3_aws_secret_access_key")
            or s3_aws_secret_access_key
        )

        self.s3_aws_session_token = (
            litellm.s3_callback_params.get("s3_aws_session_token")
            or s3_aws_session_token
        )

        self.s3_aws_session_name = (
            litellm.s3_callback_params.get("s3_aws_session_name") or s3_aws_session_name
        )

        self.s3_aws_profile_name = (
            litellm.s3_callback_params.get("s3_aws_profile_name") or s3_aws_profile_name
        )

        self.s3_aws_role_name = (
            litellm.s3_callback_params.get("s3_aws_role_name") or s3_aws_role_name
        )

        self.s3_aws_web_identity_token = (
            litellm.s3_callback_params.get("s3_aws_web_identity_token")
            or s3_aws_web_identity_token
        )

        self.s3_aws_sts_endpoint = (
            litellm.s3_callback_params.get("s3_aws_sts_endpoint") or s3_aws_sts_endpoint
        )

        self.s3_config = litellm.s3_callback_params.get("s3_config") or s3_config
        self.s3_path = litellm.s3_callback_params.get("s3_path") or s3_path
        # done reading litellm.s3_callback_params
        self.s3_use_team_prefix = (
            bool(litellm.s3_callback_params.get("s3_use_team_prefix", False))
            or s3_use_team_prefix
        )

        return

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._async_log_event_base(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )
    
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await self._async_log_event_base(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )
        pass
    

    async def _async_log_event_base(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                f"s3 Logging - Enters logging function for model {kwargs}"
            )

            s3_batch_logging_element = self.create_s3_batch_logging_element(
                start_time=start_time,
                standard_logging_payload=kwargs.get("standard_logging_object", None),
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
        except Exception as e:
            verbose_logger.exception(f"s3 Layer Error - {str(e)}")
            pass


    async def async_upload_data_to_s3(
        self, batch_logging_element: s3BatchLoggingElement
    ):
        try:
            import hashlib

            import requests
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        try:
            from litellm.litellm_core_utils.asyncify import asyncify

            asyncified_get_credentials = asyncify(self.get_credentials)
            credentials = await asyncified_get_credentials(
                aws_access_key_id=self.s3_aws_access_key_id,
                aws_secret_access_key=self.s3_aws_secret_access_key,
                aws_session_token=self.s3_aws_session_token,
                aws_region_name=self.s3_region_name,
                aws_session_name=self.s3_aws_session_name,
                aws_profile_name=self.s3_aws_profile_name,
                aws_role_name=self.s3_aws_role_name,
                aws_web_identity_token=self.s3_aws_web_identity_token,
                aws_sts_endpoint=self.s3_aws_sts_endpoint,
            )

            verbose_logger.debug(
                f"s3_v2 logger - uploading data to s3 - {batch_logging_element.s3_object_key}"
            )

            # Prepare the URL
            url = f"https://{self.s3_bucket_name}.s3.{self.s3_region_name}.amazonaws.com/{batch_logging_element.s3_object_key}"

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
        verbose_logger.debug(f"s3_v2 logger - sending batch of {len(self.log_queue)}")
        if not self.log_queue:
            return

        #########################################################
        #  Flush the log queue to s3
        #  the log queue can be bounded by DEFAULT_S3_BATCH_SIZE
        #  see custom_batch_logger.py which triggers the flush
        #########################################################
        for payload in self.log_queue:
            asyncio.create_task(self.async_upload_data_to_s3(payload))

    def create_s3_batch_logging_element(
        self,
        start_time: datetime,
        standard_logging_payload: Optional[StandardLoggingPayload],
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

        team_alias = standard_logging_payload["metadata"].get("user_api_key_team_alias")

        team_alias_prefix = ""
        if (
            litellm.enable_preview_features
            and self.s3_use_team_prefix
            and team_alias is not None
        ):
            team_alias_prefix = f"{team_alias}/"

        s3_file_name = (
            litellm.utils.get_logging_id(start_time, standard_logging_payload) or ""
        )
        s3_object_key = get_s3_object_key(
            s3_path=cast(Optional[str], self.s3_path) or "",
            team_alias_prefix=team_alias_prefix,
            start_time=start_time,
            s3_file_name=s3_file_name,
        )

        s3_object_download_filename = (
            "time-"
            + start_time.strftime("%Y-%m-%dT%H-%M-%S-%f")
            + "_"
            + standard_logging_payload["id"]
            + ".json"
        )

        s3_object_download_filename = f"time-{start_time.strftime('%Y-%m-%dT%H-%M-%S-%f')}_{standard_logging_payload['id']}.json"

        return s3BatchLoggingElement(
            payload=dict(standard_logging_payload),
            s3_object_key=s3_object_key,
            s3_object_download_filename=s3_object_download_filename,
        )

    def upload_data_to_s3(self, batch_logging_element: s3BatchLoggingElement):
        try:
            import hashlib

            import requests
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        try:
            verbose_logger.debug(
                f"s3_v2 logger - uploading data to s3 - {batch_logging_element.s3_object_key}"
            )
            credentials: Credentials = self.get_credentials(
                aws_access_key_id=self.s3_aws_access_key_id,
                aws_secret_access_key=self.s3_aws_secret_access_key,
                aws_session_token=self.s3_aws_session_token,
                aws_region_name=self.s3_region_name,
            )

            # Prepare the URL
            url = f"https://{self.s3_bucket_name}.s3.{self.s3_region_name}.amazonaws.com/{batch_logging_element.s3_object_key}"

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
