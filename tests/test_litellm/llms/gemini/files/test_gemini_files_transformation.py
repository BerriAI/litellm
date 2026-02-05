"""
Test Google AI Studio (Gemini) files transformation functionality
"""

import os
import pytest
from unittest.mock import Mock, patch

import httpx

from litellm.llms.gemini.files.transformation import GoogleAIStudioFilesHandler
from litellm.types.llms.openai import OpenAIFileObject


class TestGoogleAIStudioFilesTransformation:
    """Test Google AI Studio files transformation"""

    def setup_method(self):
        """Setup test method"""
        self.handler = GoogleAIStudioFilesHandler()

    def test_transform_retrieve_file_request_with_full_uri(self):
        """
        Test that transform_retrieve_file_request returns empty params dict
        to avoid 'Content-Type' query parameter error
        
        Regression test for: https://github.com/BerriAI/litellm/issues/XXX
        When retrieving a file, the API was incorrectly trying to pass Content-Type
        as a query parameter, which Gemini API rejected.
        """
        file_id = "https://generativelanguage.googleapis.com/v1beta/files/test123"
        litellm_params = {"api_key": "test-api-key"}

        url, params = self.handler.transform_retrieve_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params=litellm_params,
        )

        # Verify URL is constructed correctly with API key
        assert "key=test-api-key" in url
        assert file_id in url

        # CRITICAL: params should be empty dict, not contain Content-Type or any other params
        # These would be incorrectly interpreted as query parameters
        assert params == {}, f"Expected empty params dict, got: {params}"
        assert "Content-Type" not in params, "Content-Type should not be in query params"

    def test_transform_retrieve_file_request_with_file_name_only(self):
        """
        Test that transform_retrieve_file_request handles file_id without full URI
        """
        file_id = "files/test123"
        litellm_params = {"api_key": "test-api-key"}

        url, params = self.handler.transform_retrieve_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params=litellm_params,
        )

        # Verify URL is constructed correctly
        assert "generativelanguage.googleapis.com" in url
        assert file_id in url
        assert "key=test-api-key" in url

        # CRITICAL: params should be empty dict
        assert params == {}, f"Expected empty params dict, got: {params}"
        assert "Content-Type" not in params, "Content-Type should not be in query params"

    @patch.dict('os.environ', {}, clear=True)
    @patch('litellm.llms.gemini.common_utils.get_secret_str', return_value=None)
    def test_transform_retrieve_file_request_missing_api_key(self, mock_get_secret):
        """Test that transform_retrieve_file_request raises error when API key is missing"""
        file_id = "files/test123"
        litellm_params = {}

        with pytest.raises(ValueError, match="api_key is required"):
            self.handler.transform_retrieve_file_request(
                file_id=file_id,
                optional_params={},
                litellm_params=litellm_params,
            )

    def test_transform_retrieve_file_response_success(self):
        """Test successful transformation of Gemini file retrieval response"""
        # Mock response data from Gemini API
        mock_response_data = {
            "name": "files/test123",
            "displayName": "test_file.pdf",
            "mimeType": "application/pdf",
            "sizeBytes": "1024",
            "createTime": "2024-01-15T10:30:00.123456Z",
            "updateTime": "2024-01-15T10:30:00.123456Z",
            "expirationTime": "2024-01-17T10:30:00.123456Z",
            "sha256Hash": "abcd1234",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/test123",
            "state": "ACTIVE",
        }

        # Create mock httpx response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data

        # Create mock logging object
        mock_logging_obj = Mock()

        # Transform response
        result = self.handler.transform_retrieve_file_response(
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            litellm_params={},
        )

        # Verify transformation
        assert isinstance(result, OpenAIFileObject)
        assert result.id == mock_response_data["uri"]
        assert result.filename == mock_response_data["displayName"]
        assert result.bytes == int(mock_response_data["sizeBytes"])
        assert result.object == "file"
        assert result.purpose == "user_data"
        assert result.status == "processed"  # ACTIVE state maps to processed
        assert result.status_details is None

    def test_transform_retrieve_file_response_failed_state(self):
        """Test transformation of Gemini file retrieval response with FAILED state"""
        mock_response_data = {
            "name": "files/test123",
            "displayName": "test_file.pdf",
            "mimeType": "application/pdf",
            "sizeBytes": "1024",
            "createTime": "2024-01-15T10:30:00.123456Z",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/test123",
            "state": "FAILED",
            "error": {"message": "Upload failed", "code": "INTERNAL"},
        }

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_logging_obj = Mock()

        result = self.handler.transform_retrieve_file_response(
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            litellm_params={},
        )

        # Verify error state handling
        assert result.status == "error"
        assert result.status_details is not None
        assert "message" in result.status_details

    def test_transform_retrieve_file_response_processing_state(self):
        """Test transformation of Gemini file retrieval response with PROCESSING state"""
        mock_response_data = {
            "name": "files/test123",
            "displayName": "test_file.pdf",
            "mimeType": "application/pdf",
            "sizeBytes": "1024",
            "createTime": "2024-01-15T10:30:00.123456Z",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/test123",
            "state": "PROCESSING",
        }

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_logging_obj = Mock()

        result = self.handler.transform_retrieve_file_response(
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            litellm_params={},
        )

        # PROCESSING state should map to "uploaded" status
        assert result.status == "uploaded"

    def test_transform_retrieve_file_response_missing_createTime(self):
        """
        Test that transform_retrieve_file_response raises proper error when createTime is missing
        
        This tests the error scenario that occurs when API returns an error response
        without the expected file metadata fields.
        """
        # Mock error response from Gemini API (missing createTime)
        mock_response_data = {
            "error": {
                "code": 400,
                "message": "Invalid request",
                "status": "INVALID_ARGUMENT",
            }
        }

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_logging_obj = Mock()

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError, match="Error parsing file retrieve response"):
            self.handler.transform_retrieve_file_response(
                raw_response=mock_response,
                logging_obj=mock_logging_obj,
                litellm_params={},
            )

    def test_validate_environment(self):
        """Test that validate_environment properly adds API key to headers"""
        headers = {}
        api_key = "test-gemini-api-key"

        result_headers = self.handler.validate_environment(
            headers=headers,
            model="gemini-pro",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=api_key,
        )

        # Verify API key is added to headers
        assert "x-goog-api-key" in result_headers
        assert result_headers["x-goog-api-key"] == api_key

    @patch.dict('os.environ', {}, clear=True)
    @patch('litellm.llms.gemini.common_utils.get_secret_str', return_value=None)
    def test_validate_environment_missing_api_key(self, mock_get_secret):
        """Test that validate_environment raises error when API key is missing"""
        headers = {}

        with pytest.raises(
            ValueError, match="GEMINI_API_KEY is required for Google AI Studio file operations"
        ):
            self.handler.validate_environment(
                headers=headers,
                model="gemini-pro",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_get_complete_url(self):
        """Test that get_complete_url constructs proper upload URL"""
        api_base = "https://generativelanguage.googleapis.com"
        api_key = "test-api-key"
        
        url = self.handler.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model="gemini-pro",
            optional_params={},
            litellm_params={},
        )

        # Verify URL structure
        assert api_base in url
        assert "upload/v1beta/files" in url
        assert f"key={api_key}" in url

    def test_transform_delete_file_request_with_full_uri(self):
        """Test delete file request transformation with full URI"""
        file_id = "https://generativelanguage.googleapis.com/v1beta/files/test123"
        litellm_params = {
            "api_key": "test-api-key",
            "api_base": "https://generativelanguage.googleapis.com",
        }

        url, params = self.handler.transform_delete_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params=litellm_params,
        )

        # Verify URL extraction
        assert "files/test123" in url
        assert "generativelanguage.googleapis.com" in url
        
        # Params should be empty (API key goes in header via validate_environment)
        assert params == {}

    def test_transform_delete_file_request_with_file_name_only(self):
        """Test delete file request transformation with file name only"""
        file_id = "files/test123"
        litellm_params = {
            "api_key": "test-api-key",
            "api_base": "https://generativelanguage.googleapis.com",
        }

        url, params = self.handler.transform_delete_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params=litellm_params,
        )

        # Verify URL construction
        assert file_id in url
        assert "generativelanguage.googleapis.com" in url
        assert params == {}
