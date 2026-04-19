from unittest.mock import MagicMock, patch

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class TestGetModelsUrl:
    """Test that get_models() constructs the correct /v1/models URL from api_base."""

    def _get_url_for_api_base(self, api_base: str) -> str:
        """Call get_models() with a mocked HTTP client and return the URL it used."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        with patch("litellm.module_level_client.get", return_value=mock_response) as mock_get:
            OpenAIGPTConfig().get_models(api_key="fake-key", api_base=api_base)
            return mock_get.call_args.kwargs["url"]

    def test_plain_base_url(self):
        """Standard OpenAI base URL without path should get /v1/models appended."""
        assert self._get_url_for_api_base("https://api.openai.com") == "https://api.openai.com/v1/models"

    def test_trailing_slash(self):
        """Trailing slash should be stripped before appending /v1/models."""
        assert self._get_url_for_api_base("https://api.openai.com/") == "https://api.openai.com/v1/models"

    def test_v1_already_present(self):
        """api_base ending in /v1 should only get /models appended, not /v1/models."""
        assert self._get_url_for_api_base("https://api.openai.com/v1") == "https://api.openai.com/v1/models"

    def test_v1_with_trailing_slash(self):
        """api_base ending in /v1/ should be normalized and get /models appended."""
        assert self._get_url_for_api_base("https://api.openai.com/v1/") == "https://api.openai.com/v1/models"

    def test_subpath_with_v1(self):
        """api_base with a sub-path ending in /v1 should preserve the full path."""
        assert self._get_url_for_api_base("https://opencode.ai/zen/v1") == "https://opencode.ai/zen/v1/models"

    def test_subpath_with_v1_trailing_slash(self):
        """Sub-path with /v1/ should be normalized and get /models appended."""
        assert self._get_url_for_api_base("https://opencode.ai/zen/v1/") == "https://opencode.ai/zen/v1/models"

    def test_subpath_without_v1(self):
        """Sub-path without /v1 should get /v1/models appended."""
        assert self._get_url_for_api_base("https://opencode.ai/zen") == "https://opencode.ai/zen/v1/models"

    def test_localhost_with_port_and_v1(self):
        """Localhost with port and /v1 should only get /models appended."""
        assert self._get_url_for_api_base("http://localhost:11434/v1") == "http://localhost:11434/v1/models"

    def test_localhost_without_v1(self):
        """Localhost with port but no /v1 should get /v1/models appended."""
        assert self._get_url_for_api_base("http://localhost:11434") == "http://localhost:11434/v1/models"

    def test_default_api_base(self):
        """If api_base is None, it should default to OpenAI and append /v1/models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        with patch("litellm.module_level_client.get", return_value=mock_response) as mock_get:
            OpenAIGPTConfig().get_models(api_key="fake-key", api_base=None)
            assert mock_get.call_args.kwargs["url"] == "https://api.openai.com/v1/models"

    @patch("litellm.llms.openai.chat.gpt_transformation.get_secret_str", return_value="default-key")
    def test_default_api_key(self, mock_get_secret):
        """If api_key is None, it should fetch from secrets."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        with patch("litellm.module_level_client.get", return_value=mock_response) as mock_get:
            OpenAIGPTConfig().get_models(api_key=None, api_base="https://example.com")
            assert mock_get.call_args.kwargs["url"] == "https://example.com/v1/models"
            assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer default-key"

    def test_get_models_error(self):
        """If the API returns an error, get_models should raise an exception."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Error message"

        import pytest
        with patch("litellm.module_level_client.get", return_value=mock_response):
            with pytest.raises(Exception) as exc:
                OpenAIGPTConfig().get_models(api_key="fake-key", api_base="https://example.com")
            assert "Failed to get models: Error message" in str(exc.value)
