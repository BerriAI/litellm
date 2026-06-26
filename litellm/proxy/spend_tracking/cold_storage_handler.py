"""
This module is responsible for handling Getting/Setting the proxy server request from cold storage.

It allows fetching a dict of the proxy server request from s3 or GCS bucket.
"""

import litellm
from litellm import _custom_logger_compatible_callbacks_literal
from litellm.integrations.custom_logger import CustomLogger


class ColdStorageHandler:
    """
    This class is responsible for handling Getting/Setting the proxy server request from cold storage.

    It allows fetching a dict of the proxy server request from s3 or GCS bucket.

    The cold storage logger can be injected for testing; when omitted it is
    resolved from the configured ``litellm.cold_storage_custom_logger``.
    """

    def __init__(self, cold_storage_logger: CustomLogger | None = None):
        self._injected_cold_storage_logger = cold_storage_logger

    async def get_proxy_server_request_from_cold_storage_with_object_key(
        self,
        object_key: str,
    ) -> dict | None:
        """
        Get the proxy server request from cold storage using the object key directly.

        Args:
            object_key: The S3/GCS object key to retrieve

        Returns:
            Optional[dict]: The proxy server request dict or None if not found
        """
        custom_logger = self._injected_cold_storage_logger or self._resolve_cold_storage_logger()
        if custom_logger is None:
            return None

        return await custom_logger.get_proxy_server_request_from_cold_storage_with_object_key(
            object_key=object_key,
        )

    def _resolve_cold_storage_logger(self) -> CustomLogger | None:
        custom_logger_name = self._select_custom_logger_for_cold_storage()
        if custom_logger_name is None:
            return None
        return litellm.logging_callback_manager.get_active_custom_logger_for_callback_name(custom_logger_name)

    def _select_custom_logger_for_cold_storage(
        self,
    ) -> _custom_logger_compatible_callbacks_literal | None:
        cold_storage_custom_logger: _custom_logger_compatible_callbacks_literal | None = (
            litellm.cold_storage_custom_logger
        )

        return cold_storage_custom_logger
