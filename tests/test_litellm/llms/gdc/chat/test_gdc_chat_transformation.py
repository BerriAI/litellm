import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Adds the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.gdc.chat.transformation import GDCGeminiConfig

TEST_API_KEY = '{"type": "service_account", "project_id": "test-project"}'
TEST_MODEL = "gdc/gemini-2.5-flash"
TEST_API_BASE = "https://gdc-endpoint.com"
TEST_PROJECT = "test-project"
TEST_LOCATION = "test-location"

class TestGDCGeminiConfig:
    def test_get_complete_url(self):
        config = GDCGeminiConfig()
        
        # Test basic URL formatting
        url = config.get_complete_url(
            api_base=TEST_API_BASE,
            api_key=None,
            model=TEST_MODEL,
            optional_params={"vertex_project": TEST_PROJECT,  "vertex_location":TEST_LOCATION},
            litellm_params={},
        )
        assert url == f"{TEST_API_BASE}/v1/projects/{TEST_PROJECT}/locations/{TEST_LOCATION}/chat/completions"

    def test_get_complete_url_missing_api_base(self):
        config = GDCGeminiConfig()
        with pytest.raises(Exception, match="api_base/host is required for GDC Gemini"):
            config.get_complete_url(
                api_base=None,
                api_key=None,
                model=TEST_MODEL,
                optional_params={"vertex_project": TEST_PROJECT,  "vertex_location":TEST_LOCATION},
                litellm_params={},
            )

    def test_get_complete_url_missing_project(self):
        config = GDCGeminiConfig()
        with pytest.raises(Exception, match="project is required for GDC Gemini"):
            config.get_complete_url(
                api_base=TEST_API_BASE,
                api_key=None,
                model=TEST_MODEL,
                optional_params={},
                litellm_params={},
            )

    @patch("google.auth.load_credentials_from_dict")
    @patch("requests.Session")
    def test_validate_environment(self, mock_session, mock_load_creds):
        # Setup mocks for google.auth and requests.Session
        mock_creds = MagicMock()
        mock_creds.token = "mock-token"
        mock_creds.with_gdch_audience.return_value = mock_creds
        mock_load_creds.return_value = (mock_creds, None)
        
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        config = GDCGeminiConfig()
        headers = {}
        
        result = config.validate_environment(
            headers=headers,
            model=TEST_MODEL,
            messages=[],
            optional_params={"vertex_project": TEST_PROJECT,  "vertex_location":TEST_LOCATION},
            litellm_params={},
            api_key=TEST_API_KEY,
            api_base=TEST_API_BASE,
        )

        assert result["Authorization"] == "Bearer mock-token"
        assert result["Content-Type"] == "application/json"
        assert result["x-goog-user-project"] == f"projects/{TEST_PROJECT}"
        
        mock_creds.with_gdch_audience.assert_called_once_with(TEST_API_BASE)
        mock_creds.refresh.assert_called_once()
        assert mock_session_instance.verify is True

    def test_transform_request(self):
        config = GDCGeminiConfig()
        
        messages = [{"role": "user", "content": "Hello"}]
        headers = {}
        
        data = config.transform_request(
            model=TEST_MODEL,
            messages=messages,
            optional_params={"vertex_project": TEST_PROJECT,  "vertex_location":TEST_LOCATION},
            litellm_params={"ssl_verify": True},
            headers=headers,
        )
        
        # Verify provider prefix is stripped
        assert data["model"] == "gemini-2.5-flash"
        # Verify routing/auth fields are popped
        assert "vertex_project" not in data
        assert "ssl_verify" not in data
