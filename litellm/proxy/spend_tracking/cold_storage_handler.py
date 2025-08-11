"""
This module is responsible for handling Getting/Setting the proxy server request from cold storage.

It allows fetching a dict of the proxy server request from s3 or GCS bucket.
"""
from typing import Optional, cast

import litellm
from litellm import _custom_logger_compatible_callbacks_literal
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger


class ColdStorageHandler:
    """
    This class is responsible for handling Getting/Setting the proxy server request from cold storage.

    It allows fetching a dict of the proxy server request from s3 or GCS bucket.
    """
    
    async def get_proxy_server_request_from_cold_storage_with_object_key(
        self,
        object_key: str,
    ) -> Optional[dict]:
        """
        Get the proxy server request from cold storage using the object key directly.
        
        Args:
            object_key: The S3/GCS object key to retrieve
            
        Returns:
            Optional[dict]: The proxy server request dict or None if not found
        """
        
        # select the custom logger to use for cold storage
        custom_logger_name: Optional[_custom_logger_compatible_callbacks_literal] = self._select_custom_logger_for_cold_storage()

        # if no custom logger name is configured, return None
        if custom_logger_name is None:
            return None

        # get the active/initialized custom logger
        custom_logger: Optional[CustomLogger] = litellm.logging_callback_manager.get_active_custom_logger_for_callback_name(custom_logger_name)

        # if no custom logger is found, return None
        if custom_logger is None:
            return None 
        
        proxy_server_request = await custom_logger.get_proxy_server_request_from_cold_storage_with_object_key(
            object_key=object_key,
        )

        return proxy_server_request
        


    def _select_custom_logger_for_cold_storage(
        self,
    ) -> Optional[_custom_logger_compatible_callbacks_literal]:
        cold_storage_custom_logger: Optional[_custom_logger_compatible_callbacks_literal] = ColdStorageHandler._get_configured_cold_storage_custom_logger()

        return cold_storage_custom_logger
    

    @staticmethod
    def _get_configured_cold_storage_custom_logger() -> Optional[_custom_logger_compatible_callbacks_literal]:
        """Return the configured cold storage custom logger.

        During interpreter shutdown importing ``proxy_server`` can raise a
        ``RuntimeError`` (e.g. "can't register atexit after shutdown").
        In these scenarios we gracefully return ``None`` instead of bubbling
        the exception up the call stack.
        """

        try:
            from litellm.proxy.proxy_server import general_settings
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Unable to import proxy_server for cold storage logging: {e}"
            )
            return None

        cold_storage_custom_logger: Optional[str] = general_settings.get(
            "cold_storage_custom_logger"
        )
        if not cold_storage_custom_logger:
            verbose_proxy_logger.debug(
                "No cold storage custom logger found in general settings"
            )
            return None

        return cast(
            _custom_logger_compatible_callbacks_literal, cold_storage_custom_logger
        )