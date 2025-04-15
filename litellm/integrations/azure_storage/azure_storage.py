import asyncio
from datetime import datetime
import json
import os
import uuid
from typing import Callable, List, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.constants import AZURE_STORAGE_MSFT_VERSION
from litellm.integrations.azure_storage._azure_storage_auth import (
    AzureADTokenAuth,
    AzureAuthSharedKey,
)
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.types.utils import StandardLoggingPayload


class AzureBlobStorageLogger(CustomBatchLogger):
    def __init__(
        self,
        upload_to_daily_dir: Optional[bool] = None,
        **kwargs,
    ):
        """
        Initialize the Azure Blob Storage Logger

        Args:
            upload_to_dir - if true, uploads logs to %Y-%m-%d based directories. If not specified, defaults to
               true only if AZURE_STORAGE_ACCOUNT_KEY is set, for historical compatibility. May also be set via
               environment variable LITELLM_AZURE_STORAGE_UPLOAD_TO_DAILY_DIR.
            **kwargs: Additional keyword arguments for CustomBatchLogger, passed through
        """
        try:
            verbose_logger.debug(
                "AzureBlobStorageLogger: in init azure blob storage logger"
            )

            # Env Variables used for Azure Storage Authentication
            self.tenant_id = os.getenv("AZURE_STORAGE_TENANT_ID")
            self.client_id = os.getenv("AZURE_STORAGE_CLIENT_ID")
            self.client_secret = os.getenv("AZURE_STORAGE_CLIENT_SECRET")
            self.azure_storage_account_key: Optional[str] = os.getenv(
                "AZURE_STORAGE_ACCOUNT_KEY"
            )

            if self.azure_storage_account_key is None and (
                self.tenant_id is None
                or self.client_id is None
                or self.client_secret is None
            ):
                raise ValueError(
                    "Either Azure Storage Account Key is required (AZURE_STORAGE_ACCOUNT_KEY) or Azure AD authentication is required all of (AZURE_STORAGE_TENANT_ID, AZURE_STORAGE_CLIENT_ID, AZURE_STORAGE_CLIENT_SECRET)"
                )

            if upload_to_daily_dir is None:
                daily_dir_env = os.getenv("LITELLM_AZURE_STORAGE_UPLOAD_TO_DAILY_DIR")
                if daily_dir_env is not None:
                    upload_to_daily_dir = daily_dir_env.lower() == "true"
                else:
                    upload_to_daily_dir = self.azure_storage_account_key is not None
            self.upload_to_daily_dir = upload_to_daily_dir
            self.ensured_dir: Optional[str] = None

            # Required Env Variables for Azure Storage
            _azure_storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
            if not _azure_storage_account_name:
                raise ValueError(
                    "Missing required environment variable: AZURE_STORAGE_ACCOUNT_NAME"
                )
            self.azure_storage_account_name: str = _azure_storage_account_name
            _azure_storage_file_system = os.getenv("AZURE_STORAGE_FILE_SYSTEM")
            if not _azure_storage_file_system:
                raise ValueError(
                    "Missing required environment variable: AZURE_STORAGE_FILE_SYSTEM"
                )
            self.azure_storage_file_system: str = _azure_storage_file_system

            self.service_client = None
            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()
            self.log_queue: List[StandardLoggingPayload] = []
            self._client: Optional[httpx.AsyncClient] = None
            super().__init__(**kwargs, flush_lock=self.flush_lock)
        except Exception as e:
            verbose_logger.exception(
                f"AzureBlobStorageLogger: Got exception on init AzureBlobStorageLogger client {str(e)}"
            )
            raise e

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            auth: Callable[[httpx.Request], httpx.Request]
            if self.azure_storage_account_key:
                auth = AzureAuthSharedKey(
                    self.azure_storage_account_name, self.azure_storage_account_key
                )
            elif self.tenant_id and self.client_id and self.client_secret:
                auth = AzureADTokenAuth(
                    self.tenant_id, self.client_id, self.client_secret
                )
            else:
                raise ValueError("Missing required authentication parameters")
            self._client = httpx.AsyncClient(auth=auth)
        return self._client

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Async Log success events to Azure Blob Storage

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            self._premium_user_check()
            verbose_logger.debug(
                "AzureBlobStorageLogger: Logging - Enters logging function for model %s",
                kwargs,
            )
            standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )

            if standard_logging_payload is None:
                raise ValueError("standard_logging_payload is not set")

            self.log_queue.append(standard_logging_payload)

        except Exception as e:
            verbose_logger.exception(f"AzureBlobStorageLogger Layer Error - {str(e)}")
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Async Log failure events to Azure Blob Storage

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            self._premium_user_check()
            verbose_logger.debug(
                "AzureBlobStorageLogger: Logging - Enters logging function for model %s",
                kwargs,
            )
            standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )

            if standard_logging_payload is None:
                raise ValueError("standard_logging_payload is not set")

            self.log_queue.append(standard_logging_payload)
        except Exception as e:
            verbose_logger.exception(f"AzureBlobStorageLogger Layer Error - {str(e)}")
            pass

    async def async_send_batch(self):
        """
        Sends the in memory logs queue to Azure Blob Storage

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            if not self.log_queue:
                verbose_logger.exception("Datadog: log_queue does not exist")
                return

            verbose_logger.debug(
                "AzureBlobStorageLogger - about to flush %s events",
                len(self.log_queue),
            )

            for payload in self.log_queue:
                await self.async_upload_payload_to_azure_blob_storage(payload=payload)

        except Exception as e:
            verbose_logger.exception(
                f"AzureBlobStorageLogger Error sending batch API - {str(e)}"
            )

    async def async_upload_payload_to_azure_blob_storage(
        self, payload: StandardLoggingPayload
    ):
        """
        Uploads the payload to Azure Blob Storage using a 3-step process:
        1. Create file resource
        2. Append data
        3. Flush the data
        """
        try:
            json_payload = json.dumps(payload) + "\n"  # Add newline for each log entry
            payload_bytes = json_payload.encode("utf-8")
            filename = f"{payload.get('id') or str(uuid.uuid4())}.json"

            if self.upload_to_daily_dir:
                dir_name = datetime.now().strftime("%Y-%m-%d")
                filename = f"{dir_name}/{filename}"
            base_url = f"https://{self.azure_storage_account_name}.dfs.core.windows.net/{self.azure_storage_file_system}/{filename}"

            # Execute the 3-step upload process
            await self._create_file(base_url)
            await self._append_data(base_url, json_payload)
            await self._flush_data(base_url, len(payload_bytes))

            verbose_logger.debug(
                f"Successfully uploaded log to Azure Blob Storage: {filename}"
            )

        except Exception as e:
            verbose_logger.exception(f"Error uploading to Azure Blob Storage: {str(e)}")
            raise e

    async def _create_dir(self, base_url: str):
        """Helper method to create the directory"""
        try:
            verbose_logger.debug(f"Creating directory: {base_url}")
            headers = {
                "x-ms-version": AZURE_STORAGE_MSFT_VERSION,
                "Content-Length": "0",
            }
            request = self.client.build_request(
                "PUT", f"{base_url}?resource=directory", headers=headers
            )
            response = await self.client.send(request)
            response.raise_for_status()
            verbose_logger.debug("Successfully created directory")
        except Exception as e:
            verbose_logger.exception(f"Error creating directory: {str(e)}")
            raise

    async def _create_file(self, base_url: str):
        """Helper method to create the file resource"""
        try:
            verbose_logger.debug(f"Creating file resource at: {base_url}")
            headers = {
                "x-ms-version": AZURE_STORAGE_MSFT_VERSION,
                "Content-Length": "0",
            }
            request = self.client.build_request(
                "PUT", f"{base_url}?resource=file", headers=headers
            )
            response = await self.client.send(request)
            response.raise_for_status()
            verbose_logger.debug("Successfully created file resource")
        except Exception as e:
            verbose_logger.exception(f"Error creating file resource: {str(e)}")
            raise

    async def _append_data(self, base_url: str, json_payload: str):
        """Helper method to append data to the file"""
        try:
            verbose_logger.debug(f"Appending data to file: {base_url}")
            headers = {
                "x-ms-version": AZURE_STORAGE_MSFT_VERSION,
                "Content-Type": "application/json",
            }
            request = self.client.build_request(
                "PATCH",
                f"{base_url}?action=append&position=0",
                headers=headers,
                content=json_payload,
            )
            response = await self.client.send(request)
            response.raise_for_status()
            verbose_logger.debug("Successfully appended data")
        except Exception as e:
            verbose_logger.exception(f"Error appending data: {str(e)}")
            raise

    async def _flush_data(self, base_url: str, position: int):
        """Helper method to flush the data"""
        try:
            verbose_logger.debug(f"Flushing data at position {position}")
            headers = {
                "x-ms-version": AZURE_STORAGE_MSFT_VERSION,
                "Content-Length": "0",
            }
            request = self.client.build_request(
                "PATCH", f"{base_url}?action=flush&position={position}", headers=headers
            )
            response = await self.client.send(request)
            response.raise_for_status()
            verbose_logger.debug("Successfully flushed data")
        except Exception as e:
            verbose_logger.exception(f"Error flushing data: {str(e)}")
            raise

    def _premium_user_check(self):
        """
        Checks if the user is a premium user, raises an error if not
        """
        from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

        if premium_user is not True:
            raise ValueError(
                f"AzureBlobStorageLogger is only available for premium users. {CommonProxyErrors.not_premium_user}"
            )
