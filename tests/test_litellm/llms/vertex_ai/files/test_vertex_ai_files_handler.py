"""
Test Vertex AI files handler functionality
"""

import asyncio
from types import MappingProxyType
import pytest
from unittest.mock import AsyncMock, patch

import httpx

from litellm.llms.vertex_ai.files.handler import VertexAIFilesHandler
from litellm.types.llms.openai import FileContentRequest, HttpxBinaryResponseContent


def _mock_gcs_logging_config(bucket_name: str = "test-bucket"):
    return {
        "bucket_name": bucket_name,
        "path_service_account": None,
        "vertex_instance": None,
    }


class TestVertexAIFilesHandler:
    """Test Vertex AI files handler"""

    def setup_method(self):
        """Setup test method"""
        self.handler = VertexAIFilesHandler()

    def test_extract_bucket_and_object_from_file_id_standard_path(self):
        """Test extraction of bucket and object from URL-encoded file_id with standard path"""
        # Sample file_id with nested folder structure
        file_id = (
            "gs%3A%2F%2Ftest-bucket%2Flitellm-vertex-files"
            "%2Ftest-folder%2Fsub-folder%2Ftest-file.txt"
        )

        bucket_name, object_path = self.handler._extract_bucket_and_object_from_file_id(
            file_id=file_id,
            configured_bucket_name="test-bucket",
        )

        # Verify bucket name extraction
        assert bucket_name == "test-bucket"

        expected_object = "litellm-vertex-files/test-folder/sub-folder/test-file.txt"
        assert object_path == expected_object

    def test_extract_bucket_and_object_from_file_id_rejects_bucket_only(self):
        """Test extraction when only bucket name is provided"""
        file_id = "gs%3A%2F%2Ftest-bucket"

        with pytest.raises(ValueError, match="object name"):
            self.handler._extract_bucket_and_object_from_file_id(
                file_id=file_id,
                configured_bucket_name="test-bucket",
            )

    def test_extract_bucket_and_object_from_file_id_rejects_unmanaged_path(self):
        """Test extraction with simple path"""
        file_id = "gs%3A%2F%2Ftest-bucket%2Ftest-file.txt"

        with pytest.raises(ValueError, match="LiteLLM-managed"):
            self.handler._extract_bucket_and_object_from_file_id(
                file_id=file_id,
                configured_bucket_name="test-bucket",
            )

    def test_extract_bucket_and_object_from_file_id_allows_trusted_legacy_flag(self):
        file_id = "gs%3A%2F%2Ftest-bucket%2Ftest-file.txt"
        trusted_credentials = MappingProxyType({"allow_legacy_cloud_file_ids": True})

        bucket_name, object_path = self.handler._extract_bucket_and_object_from_file_id(
            file_id=file_id,
            configured_bucket_name="test-bucket",
            litellm_params={
                "_litellm_internal_model_credentials": trusted_credentials,
            },
        )

        assert bucket_name == "test-bucket"
        assert object_path == "test-file.txt"

    def test_extract_bucket_and_object_from_file_id_rejects_no_gs_prefix(self):
        """Test extraction when gs:// prefix is missing"""
        file_id = "test-bucket%2Flitellm-vertex-files%2Ftest-file.txt"

        with pytest.raises(ValueError, match="gs://"):
            self.handler._extract_bucket_and_object_from_file_id(
                file_id=file_id,
                configured_bucket_name="test-bucket",
            )

    def test_extract_bucket_and_object_from_file_id_rejects_wrong_bucket(self):
        file_id = "gs%3A%2F%2Fother-bucket%2Flitellm-vertex-files%2Ftest-file.txt"

        with pytest.raises(ValueError, match="configured storage bucket"):
            self.handler._extract_bucket_and_object_from_file_id(
                file_id=file_id,
                configured_bucket_name="test-bucket",
            )

    @pytest.mark.asyncio
    async def test_afile_content_success(self):
        """Test successful async file content retrieval"""
        # Setup test data
        file_id = (
            "gs%3A%2F%2Ftest-bucket%2Flitellm-vertex-files"
            "%2Fuploads%2Fabc-test-file.txt"
        )
        expected_content = b"test file content"

        file_content_request = FileContentRequest(
            file_id=file_id, extra_headers=None, extra_body=None
        )

        # Mock the download_gcs_object method
        with (
            patch.object(
                self.handler, "download_gcs_object", new_callable=AsyncMock
            ) as mock_download,
            patch.object(
                self.handler,
                "get_gcs_logging_config",
                new_callable=AsyncMock,
                return_value=_mock_gcs_logging_config(),
            ),
        ):
            mock_download.return_value = expected_content

            # Call the method
            result = await self.handler.afile_content(
                file_content_request=file_content_request,
                vertex_credentials=None,
                vertex_project="test-project",
                vertex_location="us-central1",
                timeout=60.0,
                max_retries=3,
            )

            # Verify the result
            assert isinstance(result, HttpxBinaryResponseContent)
            assert hasattr(result, "response")
            assert result.response.content == expected_content
            assert result.response.status_code == 200

            # Verify the download was called with correct parameters
            mock_download.assert_called_once()
            call_args = mock_download.call_args
            assert (
                call_args.kwargs["object_name"]
                == "litellm-vertex-files/uploads/abc-test-file.txt"
            )
            assert "standard_callback_dynamic_params" in call_args.kwargs
            assert (
                call_args.kwargs["standard_callback_dynamic_params"]["gcs_bucket_name"]
                == "test-bucket"
            )

    @pytest.mark.asyncio
    async def test_afile_content_missing_file_id(self):
        """Test async file content retrieval with missing file_id"""
        file_content_request = FileContentRequest(extra_headers=None, extra_body=None)

        # Should raise ValueError for missing file_id
        with pytest.raises(
            ValueError, match="file_id is required in file_content_request"
        ):
            await self.handler.afile_content(
                file_content_request=file_content_request,
                vertex_credentials=None,
                vertex_project="test-project",
                vertex_location="us-central1",
                timeout=60.0,
                max_retries=3,
            )

    @pytest.mark.asyncio
    async def test_afile_content_download_failure(self):
        """Test async file content retrieval when download fails"""
        file_id = (
            "gs%3A%2F%2Ftest-bucket%2Flitellm-vertex-files"
            "%2Fuploads%2Fabc-test-file.txt"
        )

        file_content_request = FileContentRequest(
            file_id=file_id, extra_headers=None, extra_body=None
        )

        # Mock download to return None (failure)
        with (
            patch.object(
                self.handler, "download_gcs_object", new_callable=AsyncMock
            ) as mock_download,
            patch.object(
                self.handler,
                "get_gcs_logging_config",
                new_callable=AsyncMock,
                return_value=_mock_gcs_logging_config(),
            ),
        ):
            mock_download.return_value = None

            # Should raise ValueError for failed download
            with pytest.raises(
                ValueError,
                match="Failed to download file from GCS: gs://test-bucket/litellm-vertex-files/uploads/abc-test-file.txt",
            ):
                await self.handler.afile_content(
                    file_content_request=file_content_request,
                    vertex_credentials=None,
                    vertex_project="test-project",
                    vertex_location="us-central1",
                    timeout=60.0,
                    max_retries=3,
                )

    def test_file_content_sync_success(self):
        """Test successful sync file content retrieval"""
        file_id = "gs%3A%2F%2Ftest-bucket%2Ftest-file.txt"
        expected_content = b"test file content"

        file_content_request = FileContentRequest(
            file_id=file_id, extra_headers=None, extra_body=None
        )

        # Create expected response
        mock_response = httpx.Response(
            status_code=200,
            content=expected_content,
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request(method="GET", url="gs://test-bucket/test-file.txt"),
        )
        expected_result = HttpxBinaryResponseContent(response=mock_response)

        # Mock asyncio.run to return our expected result
        with patch("asyncio.run") as mock_run:
            mock_run.return_value = expected_result

            result = self.handler.file_content(
                _is_async=False,
                file_content_request=file_content_request,
                api_base="",
                vertex_credentials=None,
                vertex_project="test-project",
                vertex_location="us-central1",
                timeout=60.0,
                max_retries=3,
            )

            # Verify the result
            assert result == expected_result

            # Verify asyncio.run was called (indicating sync execution)
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_file_content_async_mode(self):
        """Test async file content retrieval when _is_async=True"""
        file_id = "gs%3A%2F%2Ftest-bucket%2Ftest-file.txt"
        expected_content = b"test file content"

        file_content_request = FileContentRequest(
            file_id=file_id, extra_headers=None, extra_body=None
        )

        # Mock the afile_content method
        with patch.object(
            self.handler, "afile_content", new_callable=AsyncMock
        ) as mock_afile_content:
            mock_response = httpx.Response(
                status_code=200,
                content=expected_content,
                headers={"content-type": "application/octet-stream"},
                request=httpx.Request(
                    method="GET", url="gs://test-bucket/test-file.txt"
                ),
            )
            mock_afile_content.return_value = HttpxBinaryResponseContent(
                response=mock_response
            )

            # Call the method with _is_async=True
            result = self.handler.file_content(
                _is_async=True,
                file_content_request=file_content_request,
                api_base="",
                vertex_credentials=None,
                vertex_project="test-project",
                vertex_location="us-central1",
                timeout=60.0,
                max_retries=3,
            )

            # Should return a coroutine since _is_async=True
            assert asyncio.iscoroutine(result)

            # Await the result
            final_result = await result
            assert isinstance(final_result, HttpxBinaryResponseContent)
            assert final_result.response.content == expected_content

    def test_httpx_response_compatibility(self):
        """Test that the created HttpxBinaryResponseContent is compatible with expected interface"""
        # Test the mock response creation logic
        expected_content = b"test file content"
        decoded_path = "gs://test-bucket/test-file.txt"

        mock_response = httpx.Response(
            status_code=200,
            content=expected_content,
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request(method="GET", url=decoded_path),
        )

        result = HttpxBinaryResponseContent(response=mock_response)

        # Verify the response properties
        assert result.response.status_code == 200
        assert result.response.content == expected_content
        assert result.response.headers["content-type"] == "application/octet-stream"

        # Verify it has the expected interface (matching OpenAI file content response)
        assert hasattr(result, "response")
        assert hasattr(result.response, "content")
        assert hasattr(result.response, "status_code")
        assert hasattr(result.response, "headers")
