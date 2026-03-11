"""
Base storage backend interface for file storage backends.

This module defines the abstract base class that all file storage backends
(e.g., Azure Blob Storage, S3, GCS) must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseFileStorageBackend(ABC):
    """
    Abstract base class for file storage backends.
    
    All storage backends (Azure Blob Storage, S3, GCS, etc.) must implement
    these methods to provide a consistent interface for file operations.
    """

    @abstractmethod
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        path_prefix: Optional[str] = None,
        file_naming_strategy: str = "uuid",
    ) -> str:
        """
        Upload a file to the storage backend.
        
        Args:
            file_content: The file content as bytes
            filename: Original filename (may be used for naming strategy)
            content_type: MIME type of the file
            path_prefix: Optional path prefix for organizing files
            file_naming_strategy: Strategy for naming files ("uuid", "timestamp", "original_filename")
        
        Returns:
            str: The storage URL where the file can be accessed/downloaded
        
        Raises:
            Exception: If upload fails
        """
        pass

    @abstractmethod
    async def download_file(self, storage_url: str) -> bytes:
        """
        Download a file from the storage backend.
        
        Args:
            storage_url: The storage URL returned from upload_file
        
        Returns:
            bytes: The file content
        
        Raises:
            Exception: If download fails
        """
        pass

    async def delete_file(self, storage_url: str) -> None:
        """
        Delete a file from the storage backend.
        
        This is optional and can be overridden by backends that support deletion.
        Default implementation does nothing.
        
        Args:
            storage_url: The storage URL of the file to delete
        
        Raises:
            Exception: If deletion fails
        """
        # Default implementation: no-op
        # Backends can override if they support deletion
        pass

