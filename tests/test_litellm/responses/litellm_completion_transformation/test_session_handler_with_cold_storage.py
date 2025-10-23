"""
Unit tests for cold storage object key integration.

Tests for the changes to integrate cold storage handling across different components:
1. Add cold_storage_object_key field to StandardLoggingMetadata and SpendLogsMetadata
2. S3Logger generates object key when cold storage is enabled
3. Store object key in SpendLogsMetadata via spend_tracking_utils
4. Session handler uses object key from spend logs metadata
5. S3Logger supports retrieval using provided object key
"""

import json
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.s3_v2 import S3Logger
from litellm.proxy._types import SpendLogsMetadata, SpendLogsPayload
from litellm.proxy.spend_tracking.cold_storage_handler import ColdStorageHandler
from litellm.proxy.spend_tracking.spend_tracking_utils import _get_spend_logs_metadata
from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesSessionHandler,
)
from litellm.types.utils import StandardLoggingMetadata, StandardLoggingPayload


class TestColdStorageObjectKeyIntegration:
    """Test suite for cold storage object key integration."""

    def test_standard_logging_metadata_has_cold_storage_object_key_field(self):
        """
        Test: Add cold_storage_object_key field to StandardLoggingMetadata.
        
        This test verifies that the StandardLoggingMetadata TypedDict has the 
        cold_storage_object_key field for storing S3/GCS object keys.
        """
        from litellm.types.utils import StandardLoggingMetadata

        # Create a StandardLoggingMetadata instance with cold_storage_object_key
        metadata = StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            cold_storage_object_key="test/path/to/object.json"
        )
        
        # Verify the field can be set and accessed
        assert metadata.get("cold_storage_object_key") == "test/path/to/object.json"
        
        assert "cold_storage_object_key" in StandardLoggingMetadata.__annotations__

    def test_spend_logs_metadata_has_cold_storage_object_key_field(self):
        """
        Test: Add cold_storage_object_key field to SpendLogsMetadata.
        
        This test verifies that the SpendLogsMetadata TypedDict has the 
        cold_storage_object_key field for storing S3/GCS object keys.
        """
        # Create a SpendLogsMetadata instance with cold_storage_object_key
        metadata = SpendLogsMetadata(
            user_api_key="test_key",
            cold_storage_object_key="test/path/to/object.json"
        )
        
        # Verify the field can be set and accessed
        assert metadata.get("cold_storage_object_key") == "test/path/to/object.json"
        
        # Verify it's part of the SpendLogsMetadata annotations
        assert "cold_storage_object_key" in SpendLogsMetadata.__annotations__


    def test_spend_tracking_utils_stores_object_key_in_metadata(self):
        """
        Test: Store object key in SpendLogsMetadata via spend_tracking_utils.
        
        This test verifies that the _get_spend_logs_metadata function extracts
        the cold_storage_object_key from StandardLoggingPayload and stores it
        in SpendLogsMetadata.
        """
        # Create test data
        metadata = {
            "user_api_key": "test_key",
            "user_api_key_team_id": "test_team"
        }
        
        
        # Call the function
        result = _get_spend_logs_metadata(
            metadata=metadata,
            cold_storage_object_key="test/path/to/object.json"
        )
        
        # Verify the object key is stored in the result
        assert result.get("cold_storage_object_key") == "test/path/to/object.json"


    def test_session_handler_extracts_object_key_from_spend_log(self):
        """
        Test: Session handler extracts object key from spend logs metadata.
        
        This test verifies that the ResponsesSessionHandler can extract the
        cold_storage_object_key from spend log metadata.
        """
        # Create test spend log
        spend_log = {
            "request_id": "test_request_id",
            "metadata": json.dumps({
                "cold_storage_object_key": "test/path/to/object.json",
                "user_api_key": "test_key"
            })
        }
        
        # Test the extraction method
        object_key = ResponsesSessionHandler._get_cold_storage_object_key_from_spend_log(spend_log)
        
        assert object_key == "test/path/to/object.json"

    def test_session_handler_handles_dict_metadata_in_spend_log(self):
        """
        Test: Session handler handles dict metadata in spend log.
        
        This test verifies that the method works when metadata is already a dict.
        """
        # Create test spend log with dict metadata
        spend_log = {
            "request_id": "test_request_id",
            "metadata": {
                "cold_storage_object_key": "test/path/to/object.json",
                "user_api_key": "test_key"
            }
        }
        
        # Test the extraction method
        object_key = ResponsesSessionHandler._get_cold_storage_object_key_from_spend_log(spend_log)
        
        assert object_key == "test/path/to/object.json"


    @pytest.mark.asyncio
    async def test_cold_storage_handler_supports_object_key_retrieval(self):
        """
        Test: ColdStorageHandler supports object key retrieval.
        
        This test verifies that the ColdStorageHandler has the new method
        for retrieving objects using object keys directly.
        """
        handler = ColdStorageHandler()
        
        # Mock the custom logger
        mock_logger = AsyncMock()
        mock_logger.get_proxy_server_request_from_cold_storage_with_object_key = AsyncMock(
            return_value={"test": "data"}
        )
        
        with patch.object(handler, '_select_custom_logger_for_cold_storage', return_value="s3_v2"), \
             patch('litellm.logging_callback_manager.get_active_custom_logger_for_callback_name', return_value=mock_logger):
            
            result = await handler.get_proxy_server_request_from_cold_storage_with_object_key(
                object_key="test/path/to/object.json"
            )
            
            assert result == {"test": "data"}
            mock_logger.get_proxy_server_request_from_cold_storage_with_object_key.assert_called_once_with(
                object_key="test/path/to/object.json"
            )

    @pytest.mark.asyncio
    @patch('asyncio.create_task')  # Mock asyncio.create_task to avoid event loop issues
    async def test_s3_logger_supports_object_key_retrieval(self, mock_create_task):
        """
        Test: S3Logger supports retrieval using provided object key.
        
        This test verifies that the S3Logger can retrieve objects using
        the object key directly without generating it from request_id and start_time.
        """
        # Create S3Logger instance
        s3_logger = S3Logger(s3_bucket_name="test-bucket")
        
        # Mock the _download_object_from_s3 method
        with patch.object(s3_logger, '_download_object_from_s3', return_value={"test": "data"}) as mock_download:
            result = await s3_logger.get_proxy_server_request_from_cold_storage_with_object_key(
                object_key="test/path/to/object.json"
            )
            
            assert result == {"test": "data"}
            mock_download.assert_called_once_with("test/path/to/object.json")