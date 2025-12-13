"""
Storage backend service for file upload operations.

This module provides a service class for handling file uploads to custom
storage backends (e.g., Azure Blob Storage) and managing associated metadata.
"""

import base64
import time
from typing import Any, List, Mapping, cast

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid as uuid_module
from litellm.llms.base_llm.files.storage_backend_factory import get_storage_backend
from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.llms.openai import OpenAIFileObject, OpenAIFilesPurpose
from litellm.types.utils import SpecialEnums


class StorageBackendFileService:
    """
    Service for handling file uploads to storage backends.
    
    This service encapsulates the logic for:
    - Uploading files to storage backends
    - Creating file objects with storage metadata
    - Generating unified file IDs for managed files
    - Storing files in the managed files system
    """
    
    @staticmethod
    async def upload_file_to_storage_backend(
        file_data: Mapping[str, Any],
        target_storage: str,
        target_model_names: List[str],
        purpose: OpenAIFilesPurpose,
        proxy_logging_obj: ProxyLogging,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> OpenAIFileObject:
        """
        Upload a file to a storage backend and create a file object.
        
        Args:
            file_data: File data dictionary from extract_file_data()
            target_storage: Storage backend name (e.g., "azure_storage")
            target_model_names: List of model names for managed files
            purpose: File purpose (e.g., "user_data", "batch")
            proxy_logging_obj: Proxy logging object for accessing hooks
            user_api_key_dict: User API key authentication data
            
        Returns:
            OpenAIFileObject: Created file object with storage metadata
            
        Raises:
            ProxyException: If storage backend is invalid or upload fails
        """
        # Get storage backend instance
        try:
            storage_backend = get_storage_backend(target_storage)
        except ValueError as e:
            raise ProxyException(
                message=str(e),
                type="invalid_request_error",
                param="target_storage",
                code=400,
            )
        
        # Extract file information
        file_content = file_data["content"]
        filename = file_data.get("filename", "file")
        content_type = file_data.get("content_type", "application/octet-stream")
        
        # Upload to storage backend
        storage_url = await storage_backend.upload_file(
            file_content=file_content,
            filename=filename,
            content_type=content_type,
            path_prefix="",
            file_naming_strategy="uuid",
        )
        
        verbose_proxy_logger.debug(
            f"Storage backend upload complete: backend={target_storage}, url={storage_url}"
        )
        
        # Create file object with storage metadata
        file_object = StorageBackendFileService._create_file_object_with_storage_metadata(
            file_content=file_content,
            filename=filename,
            purpose=purpose,
            target_storage=target_storage,
            storage_url=storage_url,
        )
        
        # Store in managed files if target_model_names provided
        if target_model_names:
            await StorageBackendFileService._store_in_managed_files(
                file_object=file_object,
                file_data=file_data,
                target_model_names=target_model_names,
                target_storage=target_storage,
                storage_url=storage_url,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=user_api_key_dict,
            )
        
        return file_object
    
    @staticmethod
    def _create_file_object_with_storage_metadata(
        file_content: bytes,
        filename: str,
        purpose: OpenAIFilesPurpose,
        target_storage: str,
        storage_url: str,
    ) -> OpenAIFileObject:
        """
        Create an OpenAIFileObject with storage backend metadata.
        
        Args:
            file_content: File content bytes
            filename: Original filename
            purpose: File purpose
            target_storage: Storage backend name
            storage_url: URL where file is stored
            
        Returns:
            OpenAIFileObject: File object with storage metadata in _hidden_params
        """
        file_id = f"file-{uuid_module.uuid4().hex[:24]}"
        file_object = OpenAIFileObject(
            id=file_id,
            object="file",
            purpose=purpose,
            created_at=int(time.time()),
            bytes=len(file_content),
            filename=filename,
            status="uploaded",
        )
        
        # Store storage metadata in hidden params
        if not hasattr(file_object, "_hidden_params") or file_object._hidden_params is None:
            file_object._hidden_params = {}
        file_object._hidden_params.update({
            "storage_backend": target_storage,
            "storage_url": storage_url,
        })
        
        return file_object
    
    @staticmethod
    def _create_unified_file_id(
        file_type: str,
        target_model_names: List[str],
        file_id: str,
    ) -> str:
        """
        Create a base64-encoded unified file ID for managed files.
        
        Args:
            file_type: MIME type of the file
            target_model_names: List of model names
            file_id: Original file ID
            
        Returns:
            str: Base64-encoded unified file ID
        """
        unified_file_id_str = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
            file_type,
            str(uuid_module.uuid4()),
            ",".join(target_model_names),
            file_id,
            None,
        )
        
        base64_unified_file_id = (
            base64.urlsafe_b64encode(unified_file_id_str.encode()).decode().rstrip("=")
        )
        
        return base64_unified_file_id
    
    @staticmethod
    async def _store_in_managed_files(
        file_object: OpenAIFileObject,
        file_data: Mapping[str, Any],
        target_model_names: List[str],
        target_storage: str,
        storage_url: str,
        proxy_logging_obj: ProxyLogging,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        """
        Store file in managed files system with unified file ID.
        
        Args:
            file_object: File object to store
            file_data: File data dictionary
            target_model_names: List of model names
            target_storage: Storage backend name
            storage_url: URL where file is stored
            proxy_logging_obj: Proxy logging object
            user_api_key_dict: User API key authentication data
        """
        managed_files_obj = proxy_logging_obj.get_proxy_hook("managed_files")
        if not managed_files_obj or not isinstance(managed_files_obj, BaseFileEndpoints):
            verbose_proxy_logger.warning(
                "Managed files hook not available, skipping managed files storage"
            )
            return
        managed_files_obj = cast(Any, managed_files_obj)
        
        # Create model mappings using storage URL
        model_mappings = {
            model_name: storage_url
            for model_name in target_model_names
        }
        
        # Create unified file ID
        file_type = file_data.get("content_type", "application/octet-stream")
        base64_unified_file_id = StorageBackendFileService._create_unified_file_id(
            file_type=file_type,
            target_model_names=target_model_names,
            file_id=file_object.id,
        )
        
        # Update file object ID to unified ID
        file_object.id = base64_unified_file_id
        
        verbose_proxy_logger.debug(
            f"Storing file in managed files: unified_id={base64_unified_file_id}, "
            f"storage_backend={target_storage}, storage_url={storage_url}"
        )
        
        # Store in managed files
        await managed_files_obj.store_unified_file_id(
            file_id=base64_unified_file_id,
            file_object=file_object,
            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            model_mappings=model_mappings,
            user_api_key_dict=user_api_key_dict,
        )

