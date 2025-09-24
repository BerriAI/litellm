import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.s3_v2 import S3Logger
from litellm.types.utils import StandardLoggingPayload


class TestS3V2UnitTests:
    """Test that S3 v2 integration only uses safe_dumps and not json.dumps"""

    def test_s3_v2_source_code_analysis(self):
        """Test that S3 v2 source code only imports and uses safe_dumps"""
        import inspect

        from litellm.integrations import s3_v2

        # Get the source code of the s3_v2 module
        source_code = inspect.getsource(s3_v2)

        # Verify that json.dumps is not used directly in the code
        assert (
            "json.dumps(" not in source_code
        ), "S3 v2 should not use json.dumps directly"

    @patch('asyncio.create_task')
    @patch('litellm.integrations.s3_v2.CustomBatchLogger.periodic_flush')
    def test_s3_v2_endpoint_url(self, mock_periodic_flush, mock_create_task):
        """testing s3 endpoint url"""
        from unittest.mock import AsyncMock, MagicMock
        from litellm.types.integrations.s3_v2 import s3BatchLoggingElement

        # Mock periodic_flush and create_task to prevent async task creation during init
        mock_periodic_flush.return_value = None
        mock_create_task.return_value = None

        # Mock response for all tests
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        # Create a test batch logging element
        test_element = s3BatchLoggingElement(
            s3_object_key="2025-09-14/test-key.json",
            payload={"test": "data"},
            s3_object_download_filename="test-file.json"
        )

        # Test 1: Custom endpoint URL with bucket name
        s3_logger = S3Logger(
            s3_bucket_name="test-bucket",
            s3_endpoint_url="https://s3.amazonaws.com",
            s3_aws_access_key_id="test-key",
            s3_aws_secret_access_key="test-secret",
            s3_region_name="us-east-1"
        )

        s3_logger.async_httpx_client = AsyncMock()
        s3_logger.async_httpx_client.put.return_value = mock_response

        asyncio.run(s3_logger.async_upload_data_to_s3(test_element))

        call_args = s3_logger.async_httpx_client.put.call_args
        assert call_args is not None
        url = call_args[0][0]
        expected_url = "https://s3.amazonaws.com/test-bucket/2025-09-14/test-key.json"
        assert url == expected_url, f"Expected URL {expected_url}, got {url}"

        # Test 2: MinIO-compatible endpoint
        s3_logger_minio = S3Logger(
            s3_bucket_name="litellm-logs",
            s3_endpoint_url="https://minio.example.com:9000",
            s3_aws_access_key_id="minio-key",
            s3_aws_secret_access_key="minio-secret",
            s3_region_name="us-east-1"
        )

        s3_logger_minio.async_httpx_client = AsyncMock()
        s3_logger_minio.async_httpx_client.put.return_value = mock_response

        asyncio.run(s3_logger_minio.async_upload_data_to_s3(test_element))

        call_args_minio = s3_logger_minio.async_httpx_client.put.call_args
        assert call_args_minio is not None
        url_minio = call_args_minio[0][0]
        expected_minio_url = "https://minio.example.com:9000/litellm-logs/2025-09-14/test-key.json"
        assert url_minio == expected_minio_url, f"Expected MinIO URL {expected_minio_url}, got {url_minio}"

        # Test 3: Custom endpoint without bucket name (should fall back to default)
        s3_logger_no_bucket = S3Logger(
            s3_endpoint_url="https://s3.amazonaws.com",
            s3_aws_access_key_id="test-key",
            s3_aws_secret_access_key="test-secret",
            s3_region_name="us-east-1"
        )

        s3_logger_no_bucket.async_httpx_client = AsyncMock()
        s3_logger_no_bucket.async_httpx_client.put.return_value = mock_response

        asyncio.run(s3_logger_no_bucket.async_upload_data_to_s3(test_element))

        call_args_no_bucket = s3_logger_no_bucket.async_httpx_client.put.call_args
        assert call_args_no_bucket is not None
        url_no_bucket = call_args_no_bucket[0][0]
        # Should use default S3 URL format when bucket is missing (bucket becomes None in URL)
        assert "s3.us-east-1.amazonaws.com" in url_no_bucket
        assert "https://" in url_no_bucket
        # Should not include the custom endpoint since bucket is missing
        assert "https://s3.amazonaws.com/" not in url_no_bucket

        # Test 4: Sync upload method with custom endpoint
        s3_logger_sync = S3Logger(
            s3_bucket_name="sync-bucket",
            s3_endpoint_url="https://custom.s3.endpoint.com",
            s3_aws_access_key_id="sync-key",
            s3_aws_secret_access_key="sync-secret",
            s3_region_name="us-east-1"
        )

        mock_sync_client = MagicMock()
        mock_sync_client.put.return_value = mock_response

        with patch('litellm.integrations.s3_v2._get_httpx_client', return_value=mock_sync_client):
            s3_logger_sync.upload_data_to_s3(test_element)

            call_args_sync = mock_sync_client.put.call_args
            assert call_args_sync is not None
            url_sync = call_args_sync[0][0]
            expected_sync_url = "https://custom.s3.endpoint.com/sync-bucket/2025-09-14/test-key.json"
            assert url_sync == expected_sync_url, f"Expected sync URL {expected_sync_url}, got {url_sync}"

        # Test 5: Download method with custom endpoint
        s3_logger_download = S3Logger(
            s3_bucket_name="download-bucket",
            s3_endpoint_url="https://download.s3.endpoint.com",
            s3_aws_access_key_id="download-key",
            s3_aws_secret_access_key="download-secret",
            s3_region_name="us-east-1"
        )

        mock_download_response = MagicMock()
        mock_download_response.status_code = 200
        mock_download_response.json = MagicMock(return_value={"downloaded": "data"})
        s3_logger_download.async_httpx_client = AsyncMock()
        s3_logger_download.async_httpx_client.get.return_value = mock_download_response

        result = asyncio.run(s3_logger_download._download_object_from_s3("2025-09-14/download-test-key.json"))

        call_args_download = s3_logger_download.async_httpx_client.get.call_args
        assert call_args_download is not None
        url_download = call_args_download[0][0]
        expected_download_url = "https://download.s3.endpoint.com/download-bucket/2025-09-14/download-test-key.json"
        assert url_download == expected_download_url, f"Expected download URL {expected_download_url}, got {url_download}"

        assert result == {"downloaded": "data"}