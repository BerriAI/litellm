#### What this does ####
#    On success + failure, log events to Supabase

import asyncio
import datetime
import json
import os
import subprocess
import sys
import traceback
import uuid
from typing import Optional

import httpx

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.llms.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload

from .custom_batch_logger import CustomBatchLogger
from .custom_logger import CustomLogger


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
        s3_config=None,
        **kwargs,
    ):
        import boto3

        try:
            verbose_logger.debug(
                f"in init s3 logger - s3_callback_params {litellm.s3_callback_params}"
            )
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
            # Call CustomLogger's __init__
            CustomBatchLogger.__init__(self, flush_lock=self.flush_lock)

            # Call BaseAWSLLM's __init__
            BaseAWSLLM.__init__(self)

        except Exception as e:
            print_verbose(f"Got exception on init s3 client {str(e)}")
            raise e

    async def upload_data_to_s3(self, data: StandardLoggingPayload):
        try:
            import hashlib

            import boto3
            import requests
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        credentials: Credentials = self.get_credentials(
            aws_access_key_id=self.s3_aws_access_key_id,
            aws_secret_access_key=self.s3_aws_secret_access_key,
            aws_session_token=self.s3_aws_session_token,
            aws_region_name=self.s3_region_name,
        )
        object_name = uuid.uuid4().hex
        # Prepare the URL
        url = f"https://{self.bucket_name}.s3.{self.s3_region_name}.amazonaws.com/{object_name}"

        # Convert JSON to string
        json_string = json.dumps(data)

        # Calculate SHA256 hash of the content
        content_hash = hashlib.sha256(json_string.encode("utf-8")).hexdigest()

        # Prepare the request
        headers = {
            "Content-Type": "application/json",
            "x-amz-content-sha256": content_hash,
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
        asyncio.create_task(
            self.async_httpx_client.put(url, data=json_string, headers=signed_headers)
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                f"s3 Logging - Enters logging function for model {kwargs}"
            )

            # construct payload to send to s3
            # follows the same params as langfuse.py
            litellm_params = kwargs.get("litellm_params", {})
            metadata = (
                litellm_params.get("metadata", {}) or {}
            )  # if litellm_params['metadata'] == None

            # Clean Metadata before logging - never log raw metadata
            # the raw metadata can contain circular references which leads to infinite recursion
            # we clean out all extra litellm metadata params before logging
            clean_metadata = {}
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    # clean litellm metadata before logging
                    if key in [
                        "headers",
                        "endpoint",
                        "caching_groups",
                        "previous_models",
                    ]:
                        continue
                    else:
                        clean_metadata[key] = value

            # Ensure everything in the payload is converted to str
            payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )

            if payload is None:
                return

            s3_file_name = litellm.utils.get_logging_id(start_time, payload) or ""
            s3_object_key = (
                (self.s3_path.rstrip("/") + "/" if self.s3_path else "")
                + start_time.strftime("%Y-%m-%d")
                + "/"
                + s3_file_name
            )  # we need the s3 key to include the time, so we log cache hits too
            s3_object_key += ".json"

            s3_object_download_filename = (
                "time-"
                + start_time.strftime("%Y-%m-%dT%H-%M-%S-%f")
                + "_"
                + payload["id"]
                + ".json"
            )

            verbose_logger.debug("\ns3 Logger - Logging payload = %s", payload)

            self.log_queue.append(payload)
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

    async def async_send_batch(self):
        """

        Sends runs from self.log_queue

        Returns: None

        Raises: Does not raise an exception, will only verbose_logger.exception()
        """
        if not self.log_queue:
            return

        for payload in self.log_queue:
            asyncio.create_task(self.upload_data_to_s3(payload))
