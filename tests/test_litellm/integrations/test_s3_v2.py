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

@pytest.mark.asyncio
async def test_strip_base64_removes_file_and_nontext_entries():
    logger = S3Logger(s3_strip_base64_files=True)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {"type": "image", "file": {"file_data": "data:image/png;base64,AAAA"}},
                    {"type": "file", "file": {"file_data": "data:application/pdf;base64,BBBB"}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Response"},
                    {"type": "audio", "file": {"file_data": "data:audio/wav;base64,CCCC"}},
                ],
            },
        ]
    }

    stripped = await logger._strip_base64_from_messages(payload)

    # 1️⃣ File/image/audio entries are removed
    assert len(stripped["messages"][0]["content"]) == 1
    assert stripped["messages"][0]["content"][0]["text"] == "Hello world"

    assert len(stripped["messages"][1]["content"]) == 1
    assert stripped["messages"][1]["content"][0]["text"] == "Response"

    # 2️⃣ No 'file' keys remain
    for msg in stripped["messages"]:
        for content in msg["content"]:
            assert "file" not in content
            assert content.get("type") == "text"


@pytest.mark.asyncio
async def test_strip_base64_keeps_non_file_content():
    logger = S3Logger(s3_strip_base64_files=True)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Just text"},
                    {"type": "text", "text": "Another message"},
                ],
            }
        ]
    }

    stripped = await logger._strip_base64_from_messages(payload)

    # Should not modify pure text messages
    assert stripped["messages"][0]["content"] == payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_strip_base64_handles_empty_or_missing_messages():
    logger = S3Logger(s3_strip_base64_files=True)

    # Missing messages key
    payload_no_messages = {}
    stripped1 = await logger._strip_base64_from_messages(payload_no_messages)
    assert stripped1 == payload_no_messages

    # Empty messages list
    payload_empty = {"messages": []}
    stripped2 = await logger._strip_base64_from_messages(payload_empty)
    assert stripped2 == payload_empty


@pytest.mark.asyncio
async def test_strip_base64_mixed_nested_objects():
    """
    Handles weird/nested content structures gracefully.
    """
    logger = S3Logger(s3_strip_base64_files=True)

    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Keep me"},
                    {"type": "custom", "metadata": "ignore but non-text"},
                    {"foo": "bar"},
                    {"file": {"file_data": "data:application/pdf;base64,XXX"}},
                ],
                "extra": {"trace_id": "123"},
            }
        ]
    }

    stripped = await logger._strip_base64_from_messages(payload)

    # Custom/non-text and file entries removed
    content = stripped["messages"][0]["content"]
    assert len(content) == 2
    assert {"type": "text", "text": "Keep me"} in content
    assert {"foo": "bar"} in content
    # Extra metadata preserved
    assert stripped["messages"][0]["extra"]["trace_id"] == "123"


@pytest.mark.asyncio
async def test_s3_verify_false_handling():
    """
    Test that s3_verify=False is properly handled and not treated as None.
    
    This is a regression test for the bug where s3_verify=False was being
    ignored because 'False or s3_verify' would evaluate to s3_verify (None).
    """
    from unittest.mock import AsyncMock, patch

    import litellm

    # Set up s3_callback_params with s3_verify=False
    litellm.s3_callback_params = {
        "s3_bucket_name": "test-bucket",
        "s3_endpoint_url": "https://localhost:443",
        "s3_aws_access_key_id": "minioadmin",
        "s3_aws_secret_access_key": "minioadmin",
        "s3_region_name": "us-east-1",
        "s3_verify": False,  # This should NOT be ignored
        "s3_use_ssl": False,  # This should also NOT be ignored
    }
    
    with patch('asyncio.create_task'):
        with patch('litellm.integrations.s3_v2.get_async_httpx_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            
            # Create logger
            logger = S3Logger()
            
            # Verify s3_verify is False, not None
            assert logger.s3_verify is False, f"Expected s3_verify=False, got {logger.s3_verify}"
            assert logger.s3_use_ssl is False, f"Expected s3_use_ssl=False, got {logger.s3_use_ssl}"
            
            # Verify that get_async_httpx_client was called with ssl_verify=False
            mock_get_client.assert_called_once()
            call_kwargs = mock_get_client.call_args.kwargs
            assert 'params' in call_kwargs, "params should be passed to get_async_httpx_client"
            assert call_kwargs['params'] == {'ssl_verify': False}, f"Expected ssl_verify=False in params, got {call_kwargs.get('params')}"
    
    # Clean up
    litellm.s3_callback_params = None


@pytest.mark.asyncio
async def test_s3_verify_none_handling():
    """
    Test that s3_verify=None uses default behavior.
    """
    from unittest.mock import AsyncMock, patch

    import litellm

    # Set up s3_callback_params without s3_verify
    litellm.s3_callback_params = {
        "s3_bucket_name": "test-bucket",
        "s3_aws_access_key_id": "test-key",
        "s3_aws_secret_access_key": "test-secret",
        "s3_region_name": "us-east-1",
    }
    
    with patch('asyncio.create_task'):
        with patch('litellm.integrations.s3_v2.get_async_httpx_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            
            # Create logger without explicit s3_verify
            logger = S3Logger()
            
            # Verify s3_verify is None (default)
            assert logger.s3_verify is None, f"Expected s3_verify=None, got {logger.s3_verify}"
            
            # Verify that get_async_httpx_client was called
            mock_get_client.assert_called_once()
            call_kwargs = mock_get_client.call_args.kwargs
            # When s3_verify is None, params={'ssl_verify': None} which is fine - uses default behavior
            # The important thing is it's not False
            if 'params' in call_kwargs and call_kwargs['params'] is not None:
                assert call_kwargs['params'].get('ssl_verify') is None
            # Either params is None or params={'ssl_verify': None} is acceptable
    
    # Clean up
    litellm.s3_callback_params = None


@pytest.mark.asyncio
async def test_s3_verify_false_creates_httpx_client_with_verify_false():
    """
    Test that when s3_verify=False, the actual httpx client has verify=False.
    
    This validates that ssl_verify=False flows through to the httpx.AsyncClient.
    """
    from unittest.mock import patch

    import litellm

    # Set up s3_callback_params with s3_verify=False
    litellm.s3_callback_params = {
        "s3_bucket_name": "test-bucket",
        "s3_endpoint_url": "https://localhost:443",
        "s3_aws_access_key_id": "minioadmin",
        "s3_aws_secret_access_key": "minioadmin",
        "s3_region_name": "us-east-1",
        "s3_verify": False,
    }
    
    with patch('asyncio.create_task'):
        # Create logger - this creates the httpx client
        logger = S3Logger()
        
        # Verify the logger has s3_verify=False
        assert logger.s3_verify is False
        
        # Check the actual httpx client has verify=False
        # The async_httpx_client.client is the actual httpx.AsyncClient
        if hasattr(logger.async_httpx_client, 'client'):
            httpx_client = logger.async_httpx_client.client
            # Check the _verify attribute (httpx internal)
            if hasattr(httpx_client, '_verify'):
                assert httpx_client._verify is False, f"Expected httpx client _verify=False, got {httpx_client._verify}"
    
    # Clean up
    litellm.s3_callback_params = None


@pytest.mark.asyncio
async def test_s3_verify_false_async_client():
    """
    Test that the async httpx client respects s3_verify=False.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    import litellm
    from litellm.types.integrations.s3_v2 import s3BatchLoggingElement

    # Set up s3_callback_params with s3_verify=False
    litellm.s3_callback_params = {
        "s3_bucket_name": "test-bucket",
        "s3_endpoint_url": "https://localhost:443",
        "s3_aws_access_key_id": "minioadmin",
        "s3_aws_secret_access_key": "minioadmin",
        "s3_region_name": "us-east-1",
        "s3_verify": False,
    }
    
    with patch('asyncio.create_task'):
        logger = S3Logger()
        
        # Verify s3_verify is False
        assert logger.s3_verify is False
        
        # Create test element
        test_element = s3BatchLoggingElement(
            s3_object_key="2025-11-03/test-key.json",
            payload={"test": "data"},
            s3_object_download_filename="test-file.json"
        )
        
        # Mock the async httpx client's put method
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        logger.async_httpx_client.put = AsyncMock(return_value=mock_response)
        
        # Call async upload
        await logger.async_upload_data_to_s3(test_element)
        
        # Verify put was called
        assert logger.async_httpx_client.put.called
        
        # Check that the async httpx client was created with verify=False
        if hasattr(logger.async_httpx_client, 'client'):
            httpx_client = logger.async_httpx_client.client
            if hasattr(httpx_client, '_verify'):
                assert httpx_client._verify is False, f"Expected async httpx client _verify=False, got {httpx_client._verify}"
    
    # Clean up
    litellm.s3_callback_params = None


@pytest.mark.asyncio
async def test_strip_base64_recursive_redaction():
    logger = S3Logger(s3_strip_base64_files=True)
    payload = {
        "messages": [
            {
                "content": [
                    {"type": "text", "text": "normal text"},
                    {"type": "text", "text": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"},
                    {"type": "text", "text": "Nested: {'data': 'data:application/pdf;base64,AAA...'}"},
                    {"file": {"file_data": "data:application/pdf;base64,AAAA"}},
                    {"metadata": {"preview": "data:audio/mp3;base64,AAAAA=="}},
                ]
            }
        ]
    }

    result = await logger._strip_base64_from_messages(payload)
    content = result["messages"][0]["content"]

    # Dropped file-type entries
    assert not any("file" in c for c in content)

    # Base64 redacted globally
    import json
    for c in content:
        if isinstance(c, dict):
            s = json.dumps(c).lower()
            # "[base64_redacted]" is fine, but raw base64 is not
            assert "base64," not in s, f"Found real base64 blob in: {s}"



# --------------------------------------------------------------
# Shared fixture that silences asyncio.create_task during tests
# --------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_asyncio_create_task():
    """Prevent 'no running event loop' errors when S3Logger calls asyncio.create_task()."""
    with patch("asyncio.create_task"):
        yield


# --------------------------------------------------------------
# Parametrized prefix combination test
# --------------------------------------------------------------
@pytest.mark.parametrize(
    "use_team_prefix,use_key_prefix,team_alias,key_alias,expected_prefix",
    [
        (False, False, "teamA", "keyA", ""),
        (True, False, "teamA", "keyA", "teamA/"),
        (False, True, "teamA", "keyA", "keyA/"),
        (True, True, "teamA", "keyA", "teamA/keyA/"),
        (True, True, None, "keyA", "keyA/"),
        (True, True, "teamA", None, "teamA/"),
        (True, True, None, None, ""),
    ],
)
def test_s3_object_key_prefix_combinations(
        use_team_prefix, use_key_prefix, team_alias, key_alias, expected_prefix
):
    """
    Validate correct S3 prefix composition for team alias + key alias combinations.
    """
    with patch("litellm.integrations.s3_v2.get_s3_object_key") as mock_get_key:
        mock_get_key.return_value = "mocked/s3/object/key.json"

        logger = S3Logger(
            s3_bucket_name="test-bucket",
            s3_region_name="us-east-1",
            s3_use_team_prefix=use_team_prefix,
            s3_use_key_prefix=use_key_prefix,
        )

        payload = StandardLoggingPayload(
            id="abc123",
            metadata={
                "user_api_key_team_alias": team_alias,
                "user_api_key_alias": key_alias,
            },
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        )

        result = logger.create_s3_batch_logging_element(datetime.utcnow(), payload)
        assert result is not None
        mock_get_key.assert_called_once()

        prefix_arg = mock_get_key.call_args.kwargs.get("prefix")
        assert prefix_arg == expected_prefix, (
            f"Expected prefix '{expected_prefix}', got '{prefix_arg}' "
            f"for team={team_alias}, key={key_alias}, "
            f"use_team_prefix={use_team_prefix}, use_key_prefix={use_key_prefix}"
        )


# --------------------------------------------------------------
# Test prefix priority and concatenation
# --------------------------------------------------------------
def test_prefix_priority_and_path_construction():
    """
    Validate that prefix components are ordered and joined with '/' only once.
    """
    with patch("litellm.integrations.s3_v2.get_s3_object_key") as mock_get_key:
        mock_get_key.return_value = "mocked/key"

        logger = S3Logger(s3_use_team_prefix=True, s3_use_key_prefix=True)
        payload = StandardLoggingPayload(
            id="xyz999",
            metadata={
                "user_api_key_team_alias": "Team-Alpha",
                "user_api_key_alias": "API-12345",
            },
            messages=[],
        )

        logger.create_s3_batch_logging_element(datetime.utcnow(), payload)
        prefix_arg = mock_get_key.call_args.kwargs.get("prefix", "")

        assert prefix_arg == "Team-Alpha/API-12345/"
        assert "//" not in prefix_arg


# --------------------------------------------------------------
# Test when prefixes are disabled
# --------------------------------------------------------------
def test_prefix_absent_when_flags_disabled():
    """
    Verify prefix is omitted entirely when prefix flags are False.
    """
    with patch("litellm.integrations.s3_v2.get_s3_object_key") as mock_get_key:
        mock_get_key.return_value = "mocked/key"

        logger = S3Logger(s3_use_team_prefix=False, s3_use_key_prefix=False)
        payload = StandardLoggingPayload(
            id="no-prefix",
            metadata={
                "user_api_key_team_alias": "team-x",
                "user_api_key_alias": "key-x",
            },
            messages=[],
        )

        logger.create_s3_batch_logging_element(datetime.utcnow(), payload)
        prefix_arg = mock_get_key.call_args.kwargs.get("prefix", None)
        assert prefix_arg == "", f"Expected empty prefix, got {prefix_arg}"


# --------------------------------------------------------------
# Integration-style test (asyncio fixture will patch create_task)
# --------------------------------------------------------------
@pytest.mark.asyncio
async def test_combined_prefix_reflects_in_s3_object_key():
    """
    Integration-style test ensuring final s3_object_key includes both prefixes correctly.
    """
    logger = S3Logger(s3_use_team_prefix=True, s3_use_key_prefix=True)
    payload = StandardLoggingPayload(
        id="int-test",
        metadata={
            "user_api_key_team_alias": "myteam",
            "user_api_key_alias": "apikey",
        },
        messages=[],
    )

    result = logger.create_s3_batch_logging_element(datetime.utcnow(), payload)
    key = result.s3_object_key
    assert "myteam/apikey/" in key, f"Expected both prefixes in key: {key}"
