import os
from unittest.mock import patch
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

            with patch(
                "litellm.vertex_chat_completion._ensure_access_token"
            ) as mock_ensure_token, patch(
                "litellm.vertex_chat_completion._get_token_and_url"
            ) as mock_get_token:
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

        with patch(
            "litellm.vertex_chat_completion._ensure_access_token"
        ) as mock_ensure_token, patch(
            "litellm.vertex_chat_completion._get_token_and_url"
        ) as mock_get_token:
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
