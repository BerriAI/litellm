"""
Factory for creating storage backend instances.

This module provides a factory function to instantiate the correct storage backend
based on the backend type. Backends use the same configuration as their corresponding
callbacks (e.g., azure_storage uses the same env vars as AzureBlobStorageLogger).
"""

from litellm._logging import verbose_logger

from .azure_blob_storage_backend import AzureBlobStorageBackend
from .storage_backend import BaseFileStorageBackend


def get_storage_backend(backend_type: str) -> BaseFileStorageBackend:
    """
    Factory function to create a storage backend instance.
    
    Backends are configured using the same environment variables as their
    corresponding callbacks. For example, "azure_storage" uses the same
    env vars as AzureBlobStorageLogger.
    
    Args:
        backend_type: Backend type identifier (e.g., "azure_storage")
    
    Returns:
        BaseFileStorageBackend: Instance of the appropriate storage backend
    
    Raises:
        ValueError: If backend_type is not supported
    """
    verbose_logger.debug(f"Creating storage backend: type={backend_type}")
    
    if backend_type == "azure_storage":
        return AzureBlobStorageBackend()
    else:
        raise ValueError(
            f"Unsupported storage backend type: {backend_type}. "
            f"Supported types: azure_storage"
        )

