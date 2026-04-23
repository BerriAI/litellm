"""
Test Anthropic Files API transformation functionality.

Tests the AnthropicFilesConfig class which transforms between
OpenAI-compatible file operations and Anthropic's Files API format.
"""

import io
import time

import httpx
import pytest
from unittest.mock import Mock, patch

from litellm.llms.anthropic.files.transformation import (
    AnthropicFilesConfig,
    ANTHROPIC_FILES_API_BASE,
    ANTHROPIC_FILES_BETA_HEADER,
)
from litellm.types.llms.openai import OpenAIFileObject
from litellm.types.utils import LlmProviders


class TestAnthropicFilesConfig:
    """Test AnthropicFilesConfig transformation methods."""

    def setup_method(self):
        self.config = AnthropicFilesConfig()

    def test_custom_llm_provider(self):
        assert self.config.custom_llm_provider == LlmProviders.ANTHROPIC

    def test_get_complete_url_default(self):
        url = self.config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{ANTHROPIC_FILES_API_BASE}/v1/files"

    def test_get_complete_url_custom_base(self):
        url = self.config.get_complete_url(
            api_base="https://custom.api.com",
            api_key="test-key",
            model="",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/files"

    def test_get_complete_url_strips_trailing_slash(self):
        url = self.config.get_complete_url(
            api_base="https://custom.api.com/",
            api_key="test-key",
            model="",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/files"

    def test_validate_environment_sets_headers(self):
        headers = {}
        result = self.config.validate_environment(
            headers=headers,
            model="",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-ant-test-key",
        )
        assert result["x-api-key"] == "sk-ant-test-key"
        assert result["anthropic-version"] == "2023-06-01"
        assert result["anthropic-beta"] == ANTHROPIC_FILES_BETA_HEADER

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_key",
        return_value=None,
    )
    def test_validate_environment_missing_api_key(self, mock_get_key):
        with pytest.raises(ValueError, match="Anthropic API key is required"):
            self.config.validate_environment(
                headers={},
                model="",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_get_supported_openai_params(self):
        params = self.config.get_supported_openai_params(model="")
        assert "purpose" in params

    def test_transform_create_file_request(self):
        file_content = b"test file content"
        file_tuple = ("test.txt", file_content, "text/plain")

        result = self.config.transform_create_file_request(
            model="",
            create_file_data={
                "file": file_tuple,
                "purpose": "messages",
            },
            optional_params={},
            litellm_params={},
        )

        assert "file" in result
        assert "purpose" in result
        # file should be a tuple (filename, content, content_type)
        assert result["file"][0] == "test.txt"
        assert result["file"][1] == file_content
        assert result["file"][2] == "text/plain"
        # purpose should be (None, value) for multipart form field
        assert result["purpose"] == (None, "messages")

    def test_transform_create_file_request_missing_file(self):
        with pytest.raises(ValueError, match="File data is required"):
            self.config.transform_create_file_request(
                model="",
                create_file_data={"purpose": "messages"},
                optional_params={},
                litellm_params={},
            )

    def test_transform_create_file_request_default_purpose(self):
        file_tuple = ("test.txt", b"content", "text/plain")
        result = self.config.transform_create_file_request(
            model="",
            create_file_data={"file": file_tuple},
            optional_params={},
            litellm_params={},
        )
        assert result["purpose"] == (None, "messages")

    def test_transform_create_file_response(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "file-abc123",
            "type": "file",
            "filename": "document.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 12345,
            "created_at": "2025-01-15T10:30:00Z",
        }

        result = self.config.transform_create_file_response(
            model=None,
            raw_response=mock_response,
            logging_obj=Mock(),
            litellm_params={},
        )

        assert isinstance(result, OpenAIFileObject)
        assert result.id == "file-abc123"
        assert result.filename == "document.pdf"
        assert result.bytes == 12345
        assert result.object == "file"
        assert result.purpose == "messages"
        assert result.status == "uploaded"

    def test_transform_retrieve_file_request(self):
        url, params = self.config.transform_retrieve_file_request(
            file_id="file-abc123",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{ANTHROPIC_FILES_API_BASE}/v1/files/file-abc123"
        assert params == {}

    def test_transform_retrieve_file_request_custom_base(self):
        url, params = self.config.transform_retrieve_file_request(
            file_id="file-abc123",
            optional_params={},
            litellm_params={"api_base": "https://custom.api.com"},
        )
        assert url == "https://custom.api.com/v1/files/file-abc123"
        assert params == {}

    def test_transform_retrieve_file_response(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "file-abc123",
            "type": "file",
            "filename": "document.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 5000,
            "created_at": "2025-06-01T12:00:00Z",
        }

        result = self.config.transform_retrieve_file_response(
            raw_response=mock_response,
            logging_obj=Mock(),
            litellm_params={},
        )

        assert isinstance(result, OpenAIFileObject)
        assert result.id == "file-abc123"
        assert result.bytes == 5000

    def test_transform_delete_file_request(self):
        url, params = self.config.transform_delete_file_request(
            file_id="file-abc123",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{ANTHROPIC_FILES_API_BASE}/v1/files/file-abc123"
        assert params == {}

    def test_transform_delete_file_response(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "file-abc123",
            "type": "file_deleted",
        }

        result = self.config.transform_delete_file_response(
            raw_response=mock_response,
            logging_obj=Mock(),
            litellm_params={},
        )

        assert result.id == "file-abc123"
        assert result.deleted is True
        assert result.object == "file"

    def test_transform_list_files_request(self):
        url, params = self.config.transform_list_files_request(
            purpose=None,
            optional_params={},
            litellm_params={},
        )
        assert url == f"{ANTHROPIC_FILES_API_BASE}/v1/files"
        assert params == {}

    def test_transform_list_files_request_with_purpose(self):
        url, params = self.config.transform_list_files_request(
            purpose="messages",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{ANTHROPIC_FILES_API_BASE}/v1/files"
        assert params == {"purpose": "messages"}

    def test_transform_list_files_response(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "file-1",
                    "filename": "a.txt",
                    "size_bytes": 100,
                    "created_at": "2025-01-01T00:00:00Z",
                },
                {
                    "id": "file-2",
                    "filename": "b.txt",
                    "size_bytes": 200,
                    "created_at": "2025-01-02T00:00:00Z",
                },
            ],
            "has_more": False,
        }

        result = self.config.transform_list_files_response(
            raw_response=mock_response,
            logging_obj=Mock(),
            litellm_params={},
        )

        assert len(result) == 2
        assert result[0].id == "file-1"
        assert result[0].filename == "a.txt"
        assert result[1].id == "file-2"

    def test_transform_list_files_response_empty(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"data": [], "has_more": False}

        result = self.config.transform_list_files_response(
            raw_response=mock_response,
            logging_obj=Mock(),
            litellm_params={},
        )
        assert result == []

    def test_transform_file_content_request(self):
        url, params = self.config.transform_file_content_request(
            file_content_request={"file_id": "file-abc123"},
            optional_params={},
            litellm_params={},
        )
        assert url == f"{ANTHROPIC_FILES_API_BASE}/v1/files/file-abc123/content"
        assert params == {}

    def test_transform_file_content_response(self):
        mock_response = Mock(spec=httpx.Response)
        result = self.config.transform_file_content_response(
            raw_response=mock_response,
            logging_obj=Mock(),
            litellm_params={},
        )
        assert result.response == mock_response

    def test_parse_anthropic_file_with_size_bytes(self):
        """Test that size_bytes is correctly mapped to bytes field."""
        result = AnthropicFilesConfig._parse_anthropic_file(
            {
                "id": "file-test",
                "filename": "test.pdf",
                "size_bytes": 9999,
                "created_at": "2025-03-01T00:00:00Z",
            }
        )
        assert result.bytes == 9999

    def test_parse_anthropic_file_fallback_bytes_field(self):
        """Test fallback to 'bytes' field when 'size_bytes' is missing."""
        result = AnthropicFilesConfig._parse_anthropic_file(
            {
                "id": "file-test",
                "filename": "test.pdf",
                "bytes": 7777,
                "created_at": "2025-03-01T00:00:00Z",
            }
        )
        assert result.bytes == 7777

    def test_parse_anthropic_file_invalid_timestamp(self):
        """Test that invalid timestamps fall back to current time."""
        result = AnthropicFilesConfig._parse_anthropic_file(
            {
                "id": "file-test",
                "filename": "test.pdf",
                "size_bytes": 100,
                "created_at": "not-a-date",
            }
        )
        # Should not raise, should use current time
        assert isinstance(result.created_at, int)
        assert result.created_at > 0

    def test_parse_anthropic_file_missing_timestamp(self):
        """Test that missing timestamps fall back to current time."""
        result = AnthropicFilesConfig._parse_anthropic_file(
            {
                "id": "file-test",
                "filename": "test.pdf",
                "size_bytes": 100,
            }
        )
        assert isinstance(result.created_at, int)
        assert result.created_at > 0

    def test_get_error_class(self):
        error = self.config.get_error_class(
            error_message="Not found",
            status_code=404,
            headers={},
        )
        assert error.status_code == 404
        assert error.message == "Not found"


class TestProviderConfigRegistration:
    """Test that AnthropicFilesConfig is properly registered."""

    def test_provider_config_returns_anthropic_files_config(self):
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_files_config(
            model="",
            provider=LlmProviders.ANTHROPIC,
        )
        assert config is not None
        assert isinstance(config, AnthropicFilesConfig)
