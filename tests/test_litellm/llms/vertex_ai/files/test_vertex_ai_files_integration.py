"""
Test Vertex AI files integration with main files API
"""

import pytest
from unittest.mock import AsyncMock, patch

import litellm
from litellm.types.llms.openai import HttpxBinaryResponseContent


class TestVertexAIFilesIntegration:
    """Test integration of Vertex AI files with main litellm API"""

    @pytest.mark.asyncio
    async def test_litellm_afile_content_vertex_ai_provider(self):
        """Test litellm.afile_content with vertex_ai provider"""
        file_id = "gs%3A%2F%2Ftest-bucket%2Ftest-file.txt"
        expected_content = b"test file content"

        # Mock the vertex_ai_files_instance.file_content method
        with patch(
            "litellm.files.main.vertex_ai_files_instance.file_content",
            new_callable=AsyncMock,
        ) as mock_file_content:
            # Create a mock HttpxBinaryResponseContent response
            import httpx

            mock_response = httpx.Response(
                status_code=200,
                content=expected_content,
                headers={"content-type": "application/octet-stream"},
                request=httpx.Request(
                    method="GET", url="gs://test-bucket/test-file.txt"
                ),
            )
            mock_file_content.return_value = HttpxBinaryResponseContent(
                response=mock_response
            )

            # Call litellm.afile_content
            result = await litellm.afile_content(
                file_id=file_id,
                custom_llm_provider="vertex_ai",
                vertex_project="test-project",
                vertex_location="us-central1",
                vertex_credentials=None,
            )

            # Verify the result
            assert isinstance(result, HttpxBinaryResponseContent)
            assert result.response.content == expected_content
            assert result.response.status_code == 200

            # Verify the mock was called with correct parameters
            mock_file_content.assert_called_once()
            call_kwargs = mock_file_content.call_args.kwargs
            assert call_kwargs["_is_async"] is True
            assert call_kwargs["file_content_request"]["file_id"] == file_id
            assert call_kwargs["vertex_project"] == "test-project"
            assert call_kwargs["vertex_location"] == "us-central1"

    def test_litellm_file_content_vertex_ai_provider(self):
        """Test litellm.file_content with vertex_ai provider (sync)"""
        file_id = "gs%3A%2F%2Ftest-bucket%2Ftest-file.txt"
        expected_content = b"test file content"

        # Mock the vertex_ai_files_instance.file_content method
        with patch(
            "litellm.files.main.vertex_ai_files_instance.file_content"
        ) as mock_file_content:
            # Create a mock HttpxBinaryResponseContent response
            import httpx

            mock_response = httpx.Response(
                status_code=200,
                content=expected_content,
                headers={"content-type": "application/octet-stream"},
                request=httpx.Request(
                    method="GET", url="gs://test-bucket/test-file.txt"
                ),
            )
            mock_file_content.return_value = HttpxBinaryResponseContent(
                response=mock_response
            )

            # Call litellm.file_content
            result = litellm.file_content(
                file_id=file_id,
                custom_llm_provider="vertex_ai",
                vertex_project="test-project",
                vertex_location="us-central1",
                vertex_credentials=None,
            )

            # Verify the result
            assert isinstance(result, HttpxBinaryResponseContent)
            assert result.response.content == expected_content
            assert result.response.status_code == 200

            # Verify the mock was called with correct parameters
            mock_file_content.assert_called_once()
            call_kwargs = mock_file_content.call_args.kwargs
            assert call_kwargs["_is_async"] is False
            assert call_kwargs["file_content_request"]["file_id"] == file_id
            assert call_kwargs["vertex_project"] == "test-project"
            assert call_kwargs["vertex_location"] == "us-central1"

    def test_litellm_file_content_vertex_ai_with_model_provider_detection(self):
        """Test litellm.file_content with model parameter for provider detection"""
        file_id = "gs%3A%2F%2Ftest-bucket%2Ftest-file.txt"
        expected_content = b"test file content"

        # Mock the vertex_ai_files_instance.file_content method
        with patch(
            "litellm.files.main.vertex_ai_files_instance.file_content"
        ) as mock_file_content:
            # Mock get_llm_provider to return vertex_ai
            with patch("litellm.files.main.get_llm_provider") as mock_get_provider:
                mock_get_provider.return_value = (
                    "vertex_ai/gemini-pro",
                    "vertex_ai",
                    None,
                    None,
                )

                # Create a mock HttpxBinaryResponseContent response
                import httpx

                mock_response = httpx.Response(
                    status_code=200,
                    content=expected_content,
                    headers={"content-type": "application/octet-stream"},
                    request=httpx.Request(
                        method="GET", url="gs://test-bucket/test-file.txt"
                    ),
                )
                mock_file_content.return_value = HttpxBinaryResponseContent(
                    response=mock_response
                )

                # Call litellm.file_content with model to trigger provider detection
                result = litellm.file_content(
                    file_id=file_id,
                    model="vertex_ai/gemini-pro",  # This should trigger provider detection
                    vertex_project="test-project",
                    vertex_location="us-central1",
                )

                # Verify the result
                assert isinstance(result, HttpxBinaryResponseContent)
                assert result.response.content == expected_content

                # Verify provider detection was called
                mock_get_provider.assert_called_once()

    def test_litellm_file_content_vertex_ai_error_cases(self):
        """Test error handling in vertex_ai file_content"""
        # Test missing file_id
        with pytest.raises(ValueError, match="file_id is required"):
            litellm.file_content(
                file_id="",  # Empty file_id should cause error
                custom_llm_provider="vertex_ai",
                vertex_project="test-project",
            )

    def test_vertex_ai_provider_in_supported_providers_list(self):
        """Test that vertex_ai is included in supported providers for file_content"""
        # This test ensures the type annotations and error messages include vertex_ai

        # Test that calling with unsupported provider raises appropriate error
        with pytest.raises(Exception) as exc_info:
            litellm.file_content(
                file_id="test-file-id",
                custom_llm_provider="unsupported_provider",  # This should fail
            )

        # The error message should mention supported providers including vertex_ai
        error_message = str(exc_info.value)
        assert "vertex_ai" in error_message or "supported" in error_message.lower()

    @pytest.mark.asyncio
    async def test_vertex_ai_file_content_with_timeout_and_retries(self):
        """Test vertex_ai file_content with timeout and retry configuration"""
        file_id = "gs%3A%2F%2Ftest-bucket%2Ftest-file.txt"
        expected_content = b"test file content"

        # Mock the vertex_ai_files_instance.file_content method
        with patch(
            "litellm.files.main.vertex_ai_files_instance.file_content",
            new_callable=AsyncMock,
        ) as mock_file_content:
            # Create a mock HttpxBinaryResponseContent response
            import httpx

            mock_response = httpx.Response(
                status_code=200,
                content=expected_content,
                headers={"content-type": "application/octet-stream"},
                request=httpx.Request(
                    method="GET", url="gs://test-bucket/test-file.txt"
                ),
            )
            mock_file_content.return_value = HttpxBinaryResponseContent(
                response=mock_response
            )

            # Call with custom timeout and max_retries
            result = await litellm.afile_content(
                file_id=file_id,
                custom_llm_provider="vertex_ai",
                vertex_project="test-project",
                vertex_location="us-central1",
                timeout=120,
                max_retries=5,
            )

            # Verify the result
            assert isinstance(result, HttpxBinaryResponseContent)
            assert result.response.content == expected_content

            # Verify the timeout and max_retries were passed through
            call_kwargs = mock_file_content.call_args.kwargs
            assert call_kwargs["timeout"] == 120
            assert call_kwargs["max_retries"] == 5
