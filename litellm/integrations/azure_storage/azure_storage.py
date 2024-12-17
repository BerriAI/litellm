import asyncio
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
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload


class AzureBlobStorageLogger(CustomBatchLogger):
    def __init__(
        self,
        **kwargs,
    ):
        try:
            verbose_logger.debug(
                "AzureBlobStorageLogger: in init azure blob storage logger"
            )
            # check if the correct env variables are set

            self.azure_storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
            self.azure_storage_file_system = os.getenv("AZURE_STORAGE_FILE_SYSTEM")
            self.azure_auth_token = os.getenv("AZURE_STORAGE_AUTH_TOKEN")

            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()
            self.log_queue: List[StandardLoggingPayload] = []
            super().__init__(**kwargs, flush_lock=self.flush_lock)
        except Exception as e:
            verbose_logger.exception(
                f"AzureBlobStorageLogger: Got exception on init AzureBlobStorageLogger client {str(e)}"
            )
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Async Log success events to Azure Blob Storage

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
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
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )
            json_payload = json.dumps(payload) + "\n"  # Add newline for each log entry
            payload_bytes = json_payload.encode("utf-8")
            filename = payload.get("id") or str(uuid.uuid4())
            base_url = f"https://{self.azure_storage_account_name}.dfs.core.windows.net/{self.azure_storage_file_system}/{filename}"

            # Execute the 3-step upload process
            await self._create_file(async_client, base_url)
            await self._append_data(async_client, base_url, json_payload)
            await self._flush_data(async_client, base_url, len(payload_bytes))

            verbose_logger.debug(
                f"Successfully uploaded log to Azure Blob Storage: {filename}"
            )

        except Exception as e:
            verbose_logger.exception(f"Error uploading to Azure Blob Storage: {str(e)}")
            raise e

    async def _create_file(self, client: AsyncHTTPHandler, base_url: str):
        """Helper method to create the file resource"""
        try:
            verbose_logger.debug(f"Creating file resource at: {base_url}")
            headers = {
                "x-ms-version": "2019-07-07",
                "Content-Length": "0",
                "Authorization": f"Bearer {self.azure_auth_token}",
            }
            response = await client.put(f"{base_url}?resource=file", headers=headers)
            response.raise_for_status()
            verbose_logger.debug("Successfully created file resource")
        except Exception as e:
            verbose_logger.exception(f"Error creating file resource: {str(e)}")
            raise

    async def _append_data(
        self, client: AsyncHTTPHandler, base_url: str, json_payload: str
    ):
        """Helper method to append data to the file"""
        try:
            verbose_logger.debug(f"Appending data to file: {base_url}")
            headers = {
                "x-ms-version": "2019-07-07",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.azure_auth_token}",
            }
            response = await client.patch(
                f"{base_url}?action=append&position=0",
                headers=headers,
                data=json_payload,
            )
            response.raise_for_status()
            verbose_logger.debug("Successfully appended data")
        except Exception as e:
            verbose_logger.exception(f"Error appending data: {str(e)}")
            raise

    async def _flush_data(self, client: AsyncHTTPHandler, base_url: str, position: int):
        """Helper method to flush the data"""
        try:
            verbose_logger.debug(f"Flushing data at position {position}")
            headers = {
                "x-ms-version": "2019-07-07",
                "Content-Length": "0",
                "Authorization": f"Bearer {self.azure_auth_token}",
            }
            response = await client.patch(
                f"{base_url}?action=flush&position={position}", headers=headers
            )
            response.raise_for_status()
            verbose_logger.debug("Successfully flushed data")
        except Exception as e:
            verbose_logger.exception(f"Error flushing data: {str(e)}")
            raise
