import json
import os
import uuid
from datetime import datetime
from re import S
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypedDict, Union

import httpx
from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.gcs_bucket.gcs_bucket_base import GCSBucketBase
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import CommonProxyErrors, SpendLogsMetadata, SpendLogsPayload
from litellm.types.utils import (
    StandardCallbackDynamicParams,
    StandardLoggingMetadata,
    StandardLoggingPayload,
)

if TYPE_CHECKING:
    from litellm.llms.vertex_ai_and_google_ai_studio.vertex_llm_base import VertexBase
else:
    VertexBase = Any


class GCSLoggingConfig(TypedDict):
    bucket_name: str
    vertex_instance: VertexBase
    path_service_account: str


class GCSBucketLogger(GCSBucketBase):
    def __init__(self, bucket_name: Optional[str] = None) -> None:
        from litellm.proxy.proxy_server import premium_user

        super().__init__(bucket_name=bucket_name)
        self.vertex_instances: Dict[str, VertexBase] = {}
        if premium_user is not True:
            raise ValueError(
                f"GCS Bucket logging is a premium feature. Please upgrade to use it. {CommonProxyErrors.not_premium_user.value}"
            )

    #### ASYNC ####
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.proxy.proxy_server import premium_user

        if premium_user is not True:
            raise ValueError(
                f"GCS Bucket logging is a premium feature. Please upgrade to use it. {CommonProxyErrors.not_premium_user.value}"
            )
        try:
            verbose_logger.debug(
                "GCS Logger: async_log_success_event logging kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )
            gcs_logging_config: GCSLoggingConfig = await self.get_gcs_logging_config(
                kwargs
            )
            headers = await self.construct_request_headers(
                vertex_instance=gcs_logging_config["vertex_instance"],
                service_account_json=gcs_logging_config["path_service_account"],
            )
            bucket_name = gcs_logging_config["bucket_name"]

            logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )

            if logging_payload is None:
                raise ValueError("standard_logging_object not found in kwargs")

            json_logged_payload = json.dumps(logging_payload, default=str)

            # Get the current date
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Modify the object_name to include the date-based folder
            object_name = f"{current_date}/{response_obj['id']}"
            try:
                response = await self.async_httpx_client.post(
                    headers=headers,
                    url=f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={object_name}",
                    data=json_logged_payload,
                )
            except httpx.HTTPStatusError as e:
                raise Exception(f"GCS Bucket logging error: {e.response.text}")

            if response.status_code != 200:
                verbose_logger.error("GCS Bucket logging error: %s", str(response.text))

            verbose_logger.debug("GCS Bucket response %s", response)
            verbose_logger.debug("GCS Bucket status code %s", response.status_code)
            verbose_logger.debug("GCS Bucket response.text %s", response.text)
        except Exception as e:
            verbose_logger.exception(f"GCS Bucket logging error: {str(e)}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.proxy.proxy_server import premium_user

        if premium_user is not True:
            raise ValueError(
                f"GCS Bucket logging is a premium feature. Please upgrade to use it. {CommonProxyErrors.not_premium_user.value}"
            )
        try:
            verbose_logger.debug(
                "GCS Logger: async_log_failure_event logging kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )

            gcs_logging_config: GCSLoggingConfig = await self.get_gcs_logging_config(
                kwargs
            )
            headers = await self.construct_request_headers(
                vertex_instance=gcs_logging_config["vertex_instance"],
                service_account_json=gcs_logging_config["path_service_account"],
            )
            bucket_name = gcs_logging_config["bucket_name"]

            logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )

            if logging_payload is None:
                raise ValueError("standard_logging_object not found in kwargs")

            _litellm_params = kwargs.get("litellm_params") or {}
            metadata = _litellm_params.get("metadata") or {}

            json_logged_payload = json.dumps(logging_payload, default=str)

            # Get the current date
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Modify the object_name to include the date-based folder
            object_name = f"{current_date}/failure-{uuid.uuid4().hex}"

            if "gcs_log_id" in metadata:
                object_name = metadata["gcs_log_id"]

            response = await self.async_httpx_client.post(
                headers=headers,
                url=f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={object_name}",
                data=json_logged_payload,
            )

            if response.status_code != 200:
                verbose_logger.error("GCS Bucket logging error: %s", str(response.text))

            verbose_logger.debug("GCS Bucket response %s", response)
            verbose_logger.debug("GCS Bucket status code %s", response.status_code)
            verbose_logger.debug("GCS Bucket response.text %s", response.text)
        except Exception as e:
            verbose_logger.exception(f"GCS Bucket logging error: {str(e)}")

    async def get_gcs_logging_config(
        self, kwargs: Optional[Dict[str, Any]] = {}
    ) -> GCSLoggingConfig:
        """
        This function is used to get the GCS logging config for the GCS Bucket Logger.
        It checks if the dynamic parameters are provided in the kwargs and uses them to get the GCS logging config.
        If no dynamic parameters are provided, it uses the default values.
        """
        if kwargs is None:
            kwargs = {}

        standard_callback_dynamic_params: Optional[StandardCallbackDynamicParams] = (
            kwargs.get("standard_callback_dynamic_params", None)
        )

        bucket_name: str
        path_service_account: str
        if standard_callback_dynamic_params is not None:
            verbose_logger.debug("Using dynamic GCS logging")
            verbose_logger.debug(
                "standard_callback_dynamic_params: %s", standard_callback_dynamic_params
            )

            _bucket_name: Optional[str] = (
                standard_callback_dynamic_params.get("gcs_bucket_name", None)
                or self.BUCKET_NAME
            )
            _path_service_account: Optional[str] = (
                standard_callback_dynamic_params.get("gcs_path_service_account", None)
                or self.path_service_account_json
            )

            if _bucket_name is None:
                raise ValueError(
                    "GCS_BUCKET_NAME is not set in the environment, but GCS Bucket is being used as a logging callback. Please set 'GCS_BUCKET_NAME' in the environment."
                )
            if _path_service_account is None:
                raise ValueError(
                    "GCS_PATH_SERVICE_ACCOUNT is not set in the environment, but GCS Bucket is being used as a logging callback. Please set 'GCS_PATH_SERVICE_ACCOUNT' in the environment."
                )
            bucket_name = _bucket_name
            path_service_account = _path_service_account
            vertex_instance = await self.get_or_create_vertex_instance(
                credentials=path_service_account
            )
        else:
            # If no dynamic parameters, use the default instance
            if self.BUCKET_NAME is None:
                raise ValueError(
                    "GCS_BUCKET_NAME is not set in the environment, but GCS Bucket is being used as a logging callback. Please set 'GCS_BUCKET_NAME' in the environment."
                )
            if self.path_service_account_json is None:
                raise ValueError(
                    "GCS_PATH_SERVICE_ACCOUNT is not set in the environment, but GCS Bucket is being used as a logging callback. Please set 'GCS_PATH_SERVICE_ACCOUNT' in the environment."
                )
            bucket_name = self.BUCKET_NAME
            path_service_account = self.path_service_account_json
            vertex_instance = await self.get_or_create_vertex_instance(
                credentials=path_service_account
            )

        return GCSLoggingConfig(
            bucket_name=bucket_name,
            vertex_instance=vertex_instance,
            path_service_account=path_service_account,
        )

    async def get_or_create_vertex_instance(self, credentials: str) -> VertexBase:
        """
        This function is used to get the Vertex instance for the GCS Bucket Logger.
        It checks if the Vertex instance is already created and cached, if not it creates a new instance and caches it.
        """
        from litellm.llms.vertex_ai_and_google_ai_studio.vertex_llm_base import (
            VertexBase,
        )

        if credentials not in self.vertex_instances:
            vertex_instance = VertexBase()
            await vertex_instance._ensure_access_token_async(
                credentials=credentials,
                project_id=None,
                custom_llm_provider="vertex_ai",
            )
            self.vertex_instances[credentials] = vertex_instance
        return self.vertex_instances[credentials]

    async def download_gcs_object(self, object_name: str, **kwargs):
        """
        Download an object from GCS.

        https://cloud.google.com/storage/docs/downloading-objects#download-object-json
        """
        try:
            gcs_logging_config: GCSLoggingConfig = await self.get_gcs_logging_config(
                kwargs=kwargs
            )
            headers = await self.construct_request_headers(
                vertex_instance=gcs_logging_config["vertex_instance"],
                service_account_json=gcs_logging_config["path_service_account"],
            )
            bucket_name = gcs_logging_config["bucket_name"]
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_name}?alt=media"

            # Send the GET request to download the object
            response = await self.async_httpx_client.get(url=url, headers=headers)

            if response.status_code != 200:
                verbose_logger.error(
                    "GCS object download error: %s", str(response.text)
                )
                return None

            verbose_logger.debug(
                "GCS object download response status code: %s", response.status_code
            )

            # Return the content of the downloaded object
            return response.content

        except Exception as e:
            verbose_logger.error("GCS object download error: %s", str(e))
            return None

    async def delete_gcs_object(self, object_name: str, **kwargs):
        """
        Delete an object from GCS.
        """
        try:
            gcs_logging_config: GCSLoggingConfig = await self.get_gcs_logging_config(
                kwargs=kwargs
            )
            headers = await self.construct_request_headers(
                vertex_instance=gcs_logging_config["vertex_instance"],
                service_account_json=gcs_logging_config["path_service_account"],
            )
            bucket_name = gcs_logging_config["bucket_name"]
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_name}"

            # Send the DELETE request to delete the object
            response = await self.async_httpx_client.delete(url=url, headers=headers)

            if (response.status_code != 200) or (response.status_code != 204):
                verbose_logger.error(
                    "GCS object delete error: %s, status code: %s",
                    str(response.text),
                    response.status_code,
                )
                return None

            verbose_logger.debug(
                "GCS object delete response status code: %s, response: %s",
                response.status_code,
                response.text,
            )

            # Return the content of the downloaded object
            return response.text

        except Exception as e:
            verbose_logger.error("GCS object download error: %s", str(e))
            return None
