import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from litellm.integrations.gcs_bucket.gcs_bucket_base import GCSBucketBase


class TestGCSBucketBase:
    def test_construct_request_headers_with_project_id(self):
        """Test that construct_request_headers correctly uses project_id if passed from env"""
        test_project_id = "test-project"
        os.environ["GOOGLE_SECRET_MANAGER_PROJECT_ID"] = test_project_id

        try:
            # Create handler
            handler = GCSBucketBase(bucket_name="test-bucket")

            # Mock the Vertex AI auth calls
            mock_auth_header = "mock-auth-header"
            mock_token = "mock-token"

            with (
                patch(
                    "litellm.vertex_chat_completion._ensure_access_token"
                ) as mock_ensure_token,
                patch(
                    "litellm.vertex_chat_completion._get_token_and_url"
                ) as mock_get_token,
            ):
                mock_ensure_token.return_value = (mock_auth_header, test_project_id)
                mock_get_token.return_value = (mock_token, "mock-url")

                # Call construct_request_headers
                headers = handler.sync_construct_request_headers()

                # Verify headers
                assert headers == {
                    "Authorization": f"Bearer {mock_token}",
                    "Content-Type": "application/json",
                }

                # Verify _ensure_access_token was called with correct project_id
                mock_ensure_token.assert_called_once_with(
                    credentials=None,  # No service account in this test
                    project_id=test_project_id,  # Should use project_id from env
                    custom_llm_provider="vertex_ai",
                )
        finally:
            # Clean up environment variable
            del os.environ["GOOGLE_SECRET_MANAGER_PROJECT_ID"]

    def test_construct_request_headers_without_project_id(self):
        """Test that construct_request_headers works when no project_id is in env"""
        # Ensure environment variable is not set
        if "GOOGLE_SECRET_MANAGER_PROJECT_ID" in os.environ:
            del os.environ["GOOGLE_SECRET_MANAGER_PROJECT_ID"]

        # Create handler
        handler = GCSBucketBase(bucket_name="test-bucket")

        # Mock the Vertex AI auth calls
        mock_auth_header = "mock-auth-header"
        mock_token = "mock-token"

        with (
            patch(
                "litellm.vertex_chat_completion._ensure_access_token"
            ) as mock_ensure_token,
            patch(
                "litellm.vertex_chat_completion._get_token_and_url"
            ) as mock_get_token,
        ):
            mock_ensure_token.return_value = (mock_auth_header, None)
            mock_get_token.return_value = (mock_token, "mock-url")

            # Call construct_request_headers
            headers = handler.sync_construct_request_headers()

            # Verify headers
            assert headers == {
                "Authorization": f"Bearer {mock_token}",
                "Content-Type": "application/json",
            }

            # Verify _ensure_access_token was called with project_id as None
            mock_ensure_token.assert_called_once_with(
                credentials=None,  # No service account in this test
                project_id=None,  # Should be None when no env var is set
                custom_llm_provider="vertex_ai",
            )

    @pytest.mark.asyncio
    async def test_log_json_data_on_gcs_url_encodes_object_name(self):
        handler = GCSBucketBase(bucket_name="test-bucket")
        handler.async_httpx_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "logs/object"}
        handler.async_httpx_client.post.return_value = mock_response

        await handler._log_json_data_on_gcs(
            headers={"Authorization": "Bearer token"},
            bucket_name="test-bucket",
            object_name="logs/object?uploadType=media&name=evil",
            logging_payload={"ok": True},
        )

        post_url = handler.async_httpx_client.post.call_args.kwargs["url"]
        assert "name=logs%2Fobject%3FuploadType%3Dmedia%26name%3Devil" in post_url
        assert "name=logs/object?" not in post_url

    def test_gcs_log_id_is_only_used_as_sanitized_hint(self):
        logger = GCSBucketLogger.__new__(GCSBucketLogger)

        object_name = logger._get_object_name(
            kwargs={
                "litellm_params": {
                    "metadata": {"gcs_log_id": "../../target?uploadType=media"}
                }
            },
            logging_payload={"id": "payload"},
            response_obj={"id": "response-id"},
        )

        assert "/custom-" in object_name
        assert object_name.endswith("-target_uploadType_media")
        assert ".." not in object_name
        assert "?" not in object_name
