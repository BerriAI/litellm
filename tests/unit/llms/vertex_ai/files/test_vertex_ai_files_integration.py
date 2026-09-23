"""
Test Vertex AI files integration with main files API
"""

from typing import Final
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.llms.openai import HttpxBinaryResponseContent


class TestVertexAIFilesIntegration:
    """Test integration of Vertex AI files with main litellm API"""




    def test_litellm_file_content_vertex_ai_error_cases(self):
        """Test error handling in vertex_ai file_content"""
        # Test missing file_id - the VertexAI provider config's
        # transform_file_content_request should handle empty file_id.
        # Since the code now goes through base_llm_http_handler, we mock
        # ProviderConfigManager to return None so it falls through to the
        # old vertex_ai code path that validates file_id.
        with patch(
            "litellm.files.main.ProviderConfigManager.get_provider_files_config",
            return_value=None,
        ):
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


NATIVE_VERTEX_ROWS: Final = (
    b'{"request": {"contents": [{"role": "user", "parts": [{"text": "Who won the 2024 Tour de France?"}]}],'
    b' "tools": [{"googleSearch": {"excludeDomains": ["example.com"]}}]}}\n'
    b'{"request": {"contents": [{"role": "user", "parts": [{"text": "What is the tallest building in Tokyo?"}]}],'
    b' "tools": [{"googleSearch": {}}]}}\n'
)


def _gcs_upload_transport(uploads: list[httpx.Request]) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        uploads.append(request)
        object_name: Final = parse_qs(urlparse(str(request.url)).query)["name"][0]
        return httpx.Response(
            200,
            json={
                "id": f"my-bucket/{object_name}/1758585600000000",
                "name": object_name,
                "size": str(len(request.read())),
                "timeCreated": "2026-09-23T00:00:00.000Z",
            },
        )

    return httpx.MockTransport(respond)


def test_create_file_passthrough_kwarg_ships_native_rows_byte_for_byte_under_the_passthrough_prefix():
    uploads: Final[list[httpx.Request]] = []
    file_object = litellm.create_file(
        file=("batch.jsonl", NATIVE_VERTEX_ROWS, "application/jsonl"),
        purpose="batch",
        custom_llm_provider="vertex_ai",
        passthrough=True,
        model="vertex_ai/gemini-2.5-flash",
        gcs_bucket_name="my-bucket",
        api_key="test-token",
        client=HTTPHandler(client=httpx.Client(transport=_gcs_upload_transport(uploads))),
    )
    (upload,) = uploads
    object_name: Final = parse_qs(urlparse(str(upload.url)).query)["name"][0]
    assert upload.read() == NATIVE_VERTEX_ROWS
    assert object_name.startswith("litellm-vertex-files/passthrough/publishers/google/models/gemini-2.5-flash/")
    assert file_object.id == f"gs://my-bucket/{object_name}"
