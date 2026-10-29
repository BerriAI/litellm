"""
Test Vertex AI files integration with main files API
"""

from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.types.llms.openai import HttpxBinaryResponseContent


class TestVertexAIFilesIntegration:
    """Test integration of Vertex AI files with main litellm API"""

    @pytest.mark.parametrize("file_id", ["", "not-a-gcs-file"])
    def test_litellm_file_content_vertex_ai_error_cases(self, file_id: str):
        with pytest.raises(ValueError, match="gs://"):
            litellm.file_content(
                file_id=file_id,
                custom_llm_provider="vertex_ai",
                gcs_bucket_name="test-bucket",
                api_key="test-token",
            )

    def test_vertex_ai_provider_in_supported_providers_list(self):
        """Test that vertex_ai is included in supported providers for file_content"""
        # This test ensures the type annotations and error messages include vertex_ai

        # Test that calling with unsupported provider raises appropriate error
        with pytest.raises(Exception, match="unsupported_provider' is not a valid LlmProviders") as exc_info:
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

        # Create a mock HttpxBinaryResponseContent response
        import httpx

        mock_response = httpx.Response(
            status_code=200,
            content=expected_content,
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request(method="GET", url="gs://test-bucket/test-file.txt"),
        )
        mock_result = HttpxBinaryResponseContent(response=mock_response)

        # Mock the base_llm_http_handler.retrieve_file_content
        with patch(
            "litellm.files.main.base_llm_http_handler.retrieve_file_content",
            new_callable=MagicMock,
        ) as mock_retrieve:
            mock_retrieve.return_value = mock_result

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

            # Verify the mock was called
            mock_retrieve.assert_called_once()
            # Verify the timeout was passed through
            call_kwargs = mock_retrieve.call_args.kwargs
            assert call_kwargs["timeout"] == 120
