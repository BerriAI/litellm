import asyncio
import os
import time
from litellm._uuid import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from litellm._logging import verbose_logger
from litellm.constants import _DEFAULT_TTL_FOR_HTTPX_CLIENTS, AZURE_STORAGE_MSFT_VERSION
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.azure.common_utils import get_azure_ad_token_from_entra_id
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
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

            # Env Variables used for Azure Storage Authentication
            self.tenant_id = os.getenv("AZURE_STORAGE_TENANT_ID")
            self.client_id = os.getenv("AZURE_STORAGE_CLIENT_ID")
            self.client_secret = os.getenv("AZURE_STORAGE_CLIENT_SECRET")
            self.azure_storage_account_key: Optional[str] = os.getenv(
                "AZURE_STORAGE_ACCOUNT_KEY"
            )

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
            self._service_client = None
            # Time that the azure service client expires, in order to reset the connection pool and keep it fresh
            self._service_client_timeout: Optional[float] = None

            # Internal variables used for Token based authentication
            self.azure_auth_token: Optional[str] = (
                None  # the Azure AD token to use for Azure Storage API requests
            )
            self.token_expiry: Optional[datetime] = (
                None  # the expiry time of the currentAzure AD token
            )

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
            if self.azure_storage_account_key:
                await self.upload_to_azure_data_lake_with_azure_account_key(
                    payload=payload
                )
            else:
                # Get a valid token instead of always requesting a new one
                await self.set_valid_azure_ad_token()
                async_client = get_async_httpx_client(
                    llm_provider=httpxSpecialProvider.LoggingCallback
                )
                json_payload = (
                    safe_dumps(payload) + "\n"
                )  # Add newline for each log entry
                payload_bytes = json_payload.encode("utf-8")
                filename = f"{payload.get('id') or str(uuid.uuid4())}.json"
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
                "x-ms-version": AZURE_STORAGE_MSFT_VERSION,
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
                "x-ms-version": AZURE_STORAGE_MSFT_VERSION,
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
                "x-ms-version": AZURE_STORAGE_MSFT_VERSION,
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

    ####### Helper methods to managing Authentication to Azure Storage #######
    ##########################################################################

    async def set_valid_azure_ad_token(self):
        """
        Wrapper to set self.azure_auth_token to a valid Azure AD token, refreshing if necessary

        Refreshes the token when:
        - Token is expired
        - Token is not set
        """
        # Check if token needs refresh
        if self._azure_ad_token_is_expired() or self.azure_auth_token is None:
            verbose_logger.debug("Azure AD token needs refresh")
            self.azure_auth_token = self.get_azure_ad_token_from_azure_storage(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
            # Token typically expires in 1 hour
            self.token_expiry = datetime.now() + timedelta(hours=1)
            verbose_logger.debug(f"New token will expire at {self.token_expiry}")

    def get_azure_ad_token_from_azure_storage(
        self,
        tenant_id: Optional[str],
        client_id: Optional[str],
        client_secret: Optional[str],
    ) -> str:
        """
        Gets Azure AD token to use for Azure Storage API requests
        """
        verbose_logger.debug("Getting Azure AD Token from Azure Storage")
        verbose_logger.debug(
            "tenant_id %s, client_id %s, client_secret %s",
            tenant_id,
            client_id,
            client_secret,
        )
        if tenant_id is None:
            raise ValueError(
                "Missing required environment variable: AZURE_STORAGE_TENANT_ID"
            )
        if client_id is None:
            raise ValueError(
                "Missing required environment variable: AZURE_STORAGE_CLIENT_ID"
            )
        if client_secret is None:
            raise ValueError(
                "Missing required environment variable: AZURE_STORAGE_CLIENT_SECRET"
            )

        token_provider = get_azure_ad_token_from_entra_id(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            scope="https://storage.azure.com/.default",
        )
        token = token_provider()

        verbose_logger.debug("azure auth token %s", token)

        return token

    def _azure_ad_token_is_expired(self):
        """
        Returns True if Azure AD token is expired, False otherwise
        """
        if self.azure_auth_token and self.token_expiry:
            if datetime.now() + timedelta(minutes=5) >= self.token_expiry:
                verbose_logger.debug("Azure AD token is expired. Requesting new token")
                return True
        return False

    def _premium_user_check(self):
        """
        Checks if the user is a premium user, raises an error if not
        """
        from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

        if premium_user is not True:
            raise ValueError(
                f"AzureBlobStorageLogger is only available for premium users. {CommonProxyErrors.not_premium_user}"
            )

    async def get_service_client(self):
        from azure.storage.filedatalake.aio import DataLakeServiceClient

        # expire old clients to recover from connection issues
        if (
            self._service_client_timeout
            and self._service_client
            and self._service_client_timeout > time.time()
        ):
            await self._service_client.close()
            self._service_client = None
        if not self._service_client:
            self._service_client = DataLakeServiceClient(
                account_url=f"https://{self.azure_storage_account_name}.dfs.core.windows.net",
                credential=self.azure_storage_account_key,
            )
            self._service_client_timeout = time.time() + _DEFAULT_TTL_FOR_HTTPX_CLIENTS
        return self._service_client

    async def upload_to_azure_data_lake_with_azure_account_key(
        self, payload: StandardLoggingPayload
    ):
        """
        Uploads the payload to Azure Data Lake using the Azure SDK

        This is used when Azure Storage Account Key is set - Azure Storage Account Key does not work directly with Azure Rest API
        """

        # Create an async service client

        service_client = await self.get_service_client()
        # Get file system client
        file_system_client = service_client.get_file_system_client(
            file_system=self.azure_storage_file_system
        )

        try:
            # Create directory with today's date
            from datetime import datetime

            today = datetime.now().strftime("%Y-%m-%d")
            directory_client = file_system_client.get_directory_client(today)

            # check if the directory exists
            if not await directory_client.exists():
                await directory_client.create_directory()
                verbose_logger.debug(f"Created directory: {today}")

            # Create a file client
            file_name = f"{payload.get('id') or str(uuid.uuid4())}.json"
            file_client = directory_client.get_file_client(file_name)

            # Create the file
            await file_client.create_file()

            # Content to append
            content = safe_dumps(payload).encode("utf-8")

            # Append content to the file
            await file_client.append_data(data=content, offset=0, length=len(content))

            # Flush the content to finalize the file
            await file_client.flush_data(position=len(content), offset=0)

            verbose_logger.debug(
                f"Successfully uploaded and wrote to {today}/{file_name}"
            )

        except Exception as e:
            verbose_logger.exception(f"Error occurred: {str(e)}")
