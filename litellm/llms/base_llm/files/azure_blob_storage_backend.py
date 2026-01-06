"""
Azure Blob Storage backend implementation for file storage.

This module implements the Azure Blob Storage backend for storing files
in Azure Data Lake Storage Gen2. It inherits from AzureBlobStorageLogger
to reuse all authentication and Azure Storage operations.
"""

import time
from typing import Optional
from urllib.parse import quote

from litellm._logging import verbose_logger
from litellm._uuid import uuid

from .storage_backend import BaseFileStorageBackend
from litellm.integrations.azure_storage.azure_storage import AzureBlobStorageLogger


class AzureBlobStorageBackend(BaseFileStorageBackend, AzureBlobStorageLogger):
    """
    Azure Blob Storage backend implementation.
    
    Inherits from AzureBlobStorageLogger to reuse:
    - Authentication (account key and Azure AD)
    - Service client management
    - Token management
    - All Azure Storage helper methods
    
    Reads configuration from the same environment variables as AzureBlobStorageLogger.
    """

    def __init__(self, **kwargs):
        """
        Initialize Azure Blob Storage backend.
        
        Inherits all functionality from AzureBlobStorageLogger which handles:
        - Reading environment variables
        - Authentication (account key and Azure AD)
        - Service client management
        - Token management
        
        Environment variables (same as AzureBlobStorageLogger):
        - AZURE_STORAGE_ACCOUNT_NAME (required)
        - AZURE_STORAGE_FILE_SYSTEM (required)
        - AZURE_STORAGE_ACCOUNT_KEY (optional, if using account key auth)
        - AZURE_STORAGE_TENANT_ID (optional, if using Azure AD)
        - AZURE_STORAGE_CLIENT_ID (optional, if using Azure AD)
        - AZURE_STORAGE_CLIENT_SECRET (optional, if using Azure AD)
        
        Note: We skip periodic_flush since we're not using this as a logger.
        """
        # Initialize AzureBlobStorageLogger (handles all auth and config)
        AzureBlobStorageLogger.__init__(self, **kwargs)
        
        # Disable logging functionality - we're only using this for file storage
        # The periodic_flush task will be created but will do nothing since we override it

    async def periodic_flush(self):
        """
        Override to do nothing - we're not using this as a logger.
        This prevents the periodic flush task from doing any work.
        """
        # Do nothing - this class is used for file storage, not logging
        return

    async def async_log_success_event(self, *args, **kwargs):
        """
        Override to do nothing - we're not using this as a logger.
        """
        # Do nothing - this class is used for file storage, not logging
        pass

    async def async_log_failure_event(self, *args, **kwargs):
        """
        Override to do nothing - we're not using this as a logger.
        """
        # Do nothing - this class is used for file storage, not logging
        pass

    def _generate_file_name(
        self, original_filename: str, file_naming_strategy: str
    ) -> str:
        """Generate file name based on naming strategy."""
        if file_naming_strategy == "original_filename":
            # Use original filename, but sanitize it
            return quote(original_filename, safe="")
        elif file_naming_strategy == "timestamp":
            # Use timestamp
            extension = original_filename.split(".")[-1] if "." in original_filename else ""
            timestamp = int(time.time() * 1000)  # milliseconds
            return f"{timestamp}.{extension}" if extension else str(timestamp)
        else:  # default to "uuid"
            # Use UUID
            extension = original_filename.split(".")[-1] if "." in original_filename else ""
            file_uuid = str(uuid.uuid4())
            return f"{file_uuid}.{extension}" if extension else file_uuid

    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        path_prefix: Optional[str] = None,
        file_naming_strategy: str = "uuid",
    ) -> str:
        """
        Upload a file to Azure Blob Storage.
        
        Returns the blob URL in format: https://{account}.blob.core.windows.net/{container}/{path}
        """
        try:
            # Generate file name
            file_name = self._generate_file_name(filename, file_naming_strategy)
            
            # Build full path
            if path_prefix:
                # Remove leading/trailing slashes and normalize
                prefix = path_prefix.strip("/")
                full_path = f"{prefix}/{file_name}"
            else:
                full_path = file_name

            if self.azure_storage_account_key:
                # Use Azure SDK with account key (reuse logger's method)
                storage_url = await self._upload_file_with_account_key(
                    file_content=file_content,
                    full_path=full_path,
                )
            else:
                # Use REST API with Azure AD token (reuse logger's methods)
                storage_url = await self._upload_file_with_azure_ad(
                    file_content=file_content,
                    full_path=full_path,
                )

            verbose_logger.debug(
                f"Successfully uploaded file to Azure Blob Storage: {storage_url}"
            )
            return storage_url

        except Exception as e:
            verbose_logger.exception(f"Error uploading file to Azure Blob Storage: {str(e)}")
            raise

    async def _upload_file_with_account_key(
        self, file_content: bytes, full_path: str
    ) -> str:
        """Upload file using Azure SDK with account key authentication."""
        # Reuse the logger's service client method
        service_client = await self.get_service_client()
        file_system_client = service_client.get_file_system_client(
            file_system=self.azure_storage_file_system
        )

        # Create filesystem (container) if it doesn't exist
        if not await file_system_client.exists():
            await file_system_client.create_file_system()
            verbose_logger.debug(f"Created filesystem: {self.azure_storage_file_system}")

        # Extract directory and filename (similar to logger's pattern)
        path_parts = full_path.split("/")
        if len(path_parts) > 1:
            directory_path = "/".join(path_parts[:-1])
            file_name = path_parts[-1]
            
            # Create directory if needed (like logger does)
            directory_client = file_system_client.get_directory_client(directory_path)
            if not await directory_client.exists():
                await directory_client.create_directory()
                verbose_logger.debug(f"Created directory: {directory_path}")
            
            # Get file client from directory (same pattern as logger)
            file_client = directory_client.get_file_client(file_name)
        else:
            # No directory, create file directly in root
            file_client = file_system_client.get_file_client(full_path)

        # Create, append, and flush (same pattern as logger's upload_to_azure_data_lake_with_azure_account_key)
        await file_client.create_file()
        await file_client.append_data(data=file_content, offset=0, length=len(file_content))
        await file_client.flush_data(position=len(file_content), offset=0)

        # Return blob URL (not DFS URL)
        blob_url = f"https://{self.azure_storage_account_name}.blob.core.windows.net/{self.azure_storage_file_system}/{full_path}"
        return blob_url

    async def _upload_file_with_azure_ad(
        self, file_content: bytes, full_path: str
    ) -> str:
        """Upload file using REST API with Azure AD authentication."""
        # Reuse the logger's token management
        await self.set_valid_azure_ad_token()
        
        from litellm.llms.custom_httpx.http_handler import (
            get_async_httpx_client,
            httpxSpecialProvider,
        )
        
        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        # Use DFS endpoint for upload
        base_url = f"https://{self.azure_storage_account_name}.dfs.core.windows.net/{self.azure_storage_file_system}/{full_path}"

        # Execute 3-step upload process: create, append, flush
        # Reuse the logger's helper methods
        await self._create_file(async_client, base_url)
        # Append data - logger's _append_data expects string, so we create our own for bytes
        await self._append_data_bytes(async_client, base_url, file_content)
        await self._flush_data(async_client, base_url, len(file_content))

        # Return blob URL (not DFS URL)
        blob_url = f"https://{self.azure_storage_account_name}.blob.core.windows.net/{self.azure_storage_file_system}/{full_path}"
        return blob_url

    async def _append_data_bytes(
        self, client, base_url: str, file_content: bytes
    ):
        """Append binary data to file using REST API."""
        from litellm.constants import AZURE_STORAGE_MSFT_VERSION
        
        headers = {
            "x-ms-version": AZURE_STORAGE_MSFT_VERSION,
            "Content-Type": "application/octet-stream",
            "Authorization": f"Bearer {self.azure_auth_token}",
        }
        response = await client.patch(
            f"{base_url}?action=append&position=0",
            headers=headers,
            content=file_content,
        )
        response.raise_for_status()

    async def download_file(self, storage_url: str) -> bytes:
        """
        Download a file from Azure Blob Storage.
        
        Args:
            storage_url: Blob URL in format: https://{account}.blob.core.windows.net/{container}/{path}
        
        Returns:
            bytes: File content
        """
        try:
            # Parse blob URL to extract path
            # URL format: https://{account}.blob.core.windows.net/{container}/{path}
            if ".blob.core.windows.net/" not in storage_url:
                raise ValueError(f"Invalid Azure Blob Storage URL: {storage_url}")

            # Extract path after container name
            container_and_path = storage_url.split(".blob.core.windows.net/", 1)[1]
            path_parts = container_and_path.split("/", 1)
            if len(path_parts) < 2:
                raise ValueError(f"Invalid Azure Blob Storage URL format: {storage_url}")
            file_path = path_parts[1]  # Path after container name

            if self.azure_storage_account_key:
                # Use Azure SDK (reuse logger's service client)
                return await self._download_file_with_account_key(file_path)
            else:
                # Use REST API (reuse logger's token management)
                return await self._download_file_with_azure_ad(file_path)

        except Exception as e:
            verbose_logger.exception(f"Error downloading file from Azure Blob Storage: {str(e)}")
            raise

    async def _download_file_with_account_key(self, file_path: str) -> bytes:
        """Download file using Azure SDK with account key."""
        # Reuse the logger's service client method
        service_client = await self.get_service_client()
        file_system_client = service_client.get_file_system_client(
            file_system=self.azure_storage_file_system
        )
        # Ensure filesystem exists (should already exist, but check for safety)
        if not await file_system_client.exists():
            raise ValueError(f"Filesystem {self.azure_storage_file_system} does not exist")
        file_client = file_system_client.get_file_client(file_path)
        # Download file
        download_response = await file_client.download_file()
        file_content = await download_response.readall()
        return file_content

    async def _download_file_with_azure_ad(self, file_path: str) -> bytes:
        """Download file using REST API with Azure AD token."""
        # Reuse the logger's token management
        await self.set_valid_azure_ad_token()
        
        from litellm.llms.custom_httpx.http_handler import (
            get_async_httpx_client,
            httpxSpecialProvider,
        )
        from litellm.constants import AZURE_STORAGE_MSFT_VERSION

        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        # Use blob endpoint for download (simpler than DFS)
        blob_url = f"https://{self.azure_storage_account_name}.blob.core.windows.net/{self.azure_storage_file_system}/{file_path}"
        
        headers = {
            "x-ms-version": AZURE_STORAGE_MSFT_VERSION,
            "Authorization": f"Bearer {self.azure_auth_token}",
        }
        
        response = await async_client.get(blob_url, headers=headers)
        response.raise_for_status()
        return response.content

