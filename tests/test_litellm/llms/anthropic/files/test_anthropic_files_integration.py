"""
Test Anthropic Files API integration

This test suite validates the end-to-end integration of Anthropic Files API
with LiteLLM, testing all operations: upload, retrieve, list, delete, and content.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import httpx
import io

import litellm
from litellm.llms.openai.openai import FileDeleted, FileObject
from litellm.types.llms.openai import OpenAIFileObject


class TestAnthropicFilesIntegration:
    """Test Anthropic Files API integration with LiteLLM"""

    @pytest.mark.asyncio
    async def test_acreate_file_success(self):
        """
        Test async file upload to Anthropic.

        Validates that file upload correctly transforms request and response.
        """
        # Setup mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "file_test123",
            "filename": "test.txt",
            "size_bytes": 100,
            "created_at": "2024-01-15T10:30:00Z",
        }
        mock_response.status_code = 200

        # Mock the HTTP handler
        with patch("litellm.llms.custom_httpx.llm_http_handler.AsyncHTTPHandler.post") as mock_post:
            mock_post.return_value = mock_response

            # Create a test file
            test_file = io.BytesIO(b"Test content for Anthropic")
            test_file.name = "test.txt"

            # Upload file
            result = await litellm.acreate_file(
                file=test_file,
                purpose="assistants",
                custom_llm_provider="anthropic",
                api_key="test_api_key"
            )

            # Verify result
            assert isinstance(result, (FileObject, OpenAIFileObject))
            assert result.id == "file_test123"
            assert result.filename == "test.txt"
            assert result.bytes == 100
            assert result.purpose == "assistants"

    @pytest.mark.asyncio
    async def test_afile_retrieve_success(self):
        """
        Test async file retrieval from Anthropic.

        Validates that file retrieval correctly fetches and transforms file metadata.
        """
        # Setup mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "file_test456",
            "filename": "retrieved.txt",
            "size_bytes": 200,
            "created_at": "2024-01-15T11:00:00Z",
        }
        mock_response.status_code = 200

        # Mock the AsyncHTTPHandler.get method
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get") as mock_get:
            mock_get.return_value = mock_response

            # Retrieve file
            result = await litellm.afile_retrieve(
                file_id="file_test456",
                custom_llm_provider="anthropic",
                api_key="test_api_key"
            )

            # Verify result
            assert isinstance(result, (FileObject, OpenAIFileObject))
            assert result.id == "file_test456"
            assert result.filename == "retrieved.txt"
            assert result.bytes == 200

            # Verify the GET request was made to correct URL
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "file_test456" in call_args.kwargs["url"]

    @pytest.mark.asyncio
    async def test_afile_list_success(self):
        """
        Test async file listing from Anthropic.

        Validates that file list correctly fetches and transforms multiple files.
        """
        # Setup mock response with multiple files
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "file_1",
                    "filename": "file1.txt",
                    "size_bytes": 100,
                    "created_at": "2024-01-15T10:00:00Z",
                },
                {
                    "id": "file_2",
                    "filename": "file2.txt",
                    "size_bytes": 200,
                    "created_at": "2024-01-15T11:00:00Z",
                },
            ]
        }
        mock_response.status_code = 200

        # Mock the AsyncHTTPHandler.get method
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get") as mock_get:
            mock_get.return_value = mock_response

            # List files
            result = await litellm.afile_list(
                custom_llm_provider="anthropic",
                api_key="test_api_key"
            )

            # Verify result
            assert isinstance(result, list)
            assert len(result) == 2

            assert result[0].id == "file_1"
            assert result[0].filename == "file1.txt"
            assert result[0].bytes == 100

            assert result[1].id == "file_2"
            assert result[1].filename == "file2.txt"
            assert result[1].bytes == 200

    @pytest.mark.asyncio
    async def test_afile_delete_success(self):
        """
        Test async file deletion from Anthropic.

        Validates that file deletion correctly sends DELETE request.
        """
        # Setup mock response (Anthropic returns empty response on delete)
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200

        # Mock the AsyncHTTPHandler.delete method
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.delete") as mock_delete:
            mock_delete.return_value = mock_response

            # Delete file
            result = await litellm.afile_delete(
                file_id="file_test789",
                custom_llm_provider="anthropic",
                api_key="test_api_key"
            )

            # Verify result
            assert isinstance(result, FileDeleted)
            assert result.id == "file_test789"
            assert result.deleted is True

            # Verify the DELETE request was made to correct URL
            mock_delete.assert_called_once()
            call_args = mock_delete.call_args
            assert "file_test789" in call_args.kwargs["url"]

    def test_create_file_sync_success(self):
        """
        Test synchronous file upload to Anthropic.

        Validates that sync file upload works correctly.
        """
        # Setup mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "file_sync123",
            "filename": "sync_test.txt",
            "size_bytes": 150,
            "created_at": "2024-01-15T12:00:00Z",
        }
        mock_response.status_code = 200

        # Mock the HTTP handler
        with patch("litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post") as mock_post:
            mock_post.return_value = mock_response

            # Create a test file
            test_file = io.BytesIO(b"Sync test content")
            test_file.name = "sync_test.txt"

            # Upload file
            result = litellm.create_file(
                file=test_file,
                purpose="assistants",
                custom_llm_provider="anthropic",
                api_key="test_api_key"
            )

            # Verify result
            assert isinstance(result, (FileObject, OpenAIFileObject))
            assert result.id == "file_sync123"
            assert result.filename == "sync_test.txt"

    def test_file_retrieve_sync_success(self):
        """
        Test synchronous file retrieval from Anthropic.
        """
        # Setup mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "file_sync456",
            "filename": "sync_retrieved.txt",
            "size_bytes": 250,
            "created_at": "2024-01-15T13:00:00Z",
        }
        mock_response.status_code = 200

        # Mock the HTTPHandler.get method
        with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get") as mock_get:
            mock_get.return_value = mock_response

            # Retrieve file
            result = litellm.file_retrieve(
                file_id="file_sync456",
                custom_llm_provider="anthropic",
                api_key="test_api_key"
            )

            # Verify result
            assert isinstance(result, (FileObject, OpenAIFileObject))
            assert result.id == "file_sync456"
            assert result.filename == "sync_retrieved.txt"

    def test_file_list_sync_success(self):
        """
        Test synchronous file listing from Anthropic.
        """
        # Setup mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "file_sync1",
                    "filename": "sync_file1.txt",
                    "size_bytes": 300,
                    "created_at": "2024-01-15T14:00:00Z",
                },
            ]
        }
        mock_response.status_code = 200

        # Mock the HTTPHandler.get method
        with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get") as mock_get:
            mock_get.return_value = mock_response

            # List files
            result = litellm.file_list(
                custom_llm_provider="anthropic",
                api_key="test_api_key"
            )

            # Verify result
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0].id == "file_sync1"

    def test_file_delete_sync_success(self):
        """
        Test synchronous file deletion from Anthropic.
        """
        # Setup mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200

        # Mock the HTTPHandler.delete method
        with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.delete") as mock_delete:
            mock_delete.return_value = mock_response

            # Delete file
            result = litellm.file_delete(
                file_id="file_sync_del",
                custom_llm_provider="anthropic",
                api_key="test_api_key"
            )

            # Verify result
            assert isinstance(result, FileDeleted)
            assert result.id == "file_sync_del"
            assert result.deleted is True

