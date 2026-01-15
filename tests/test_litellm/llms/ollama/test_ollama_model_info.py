import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

"""
Unit tests for OllamaModelInfo.get_models functionality.
"""
# Ensure a dummy httpx module is available for import in tests
import sys
import types

# Provide a dummy httpx module for import in get_models
if "httpx" not in sys.modules:
    # Create a minimal module with HTTPStatusError
    httpx_mod = types.ModuleType("httpx")
    httpx_mod.HTTPStatusError = Exception
    sys.modules["httpx"] = httpx_mod

import httpx

from litellm.llms.ollama.common_utils import OllamaModelInfo


class DummyResponse:
    """
    A dummy response object to simulate httpx responses.
    """

    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            # Simulate an HTTP status error
            raise httpx.HTTPStatusError(
                "Error status code", request=None, response=None
            )

    def json(self):
        return self._json


class TestOllamaModelInfo:
    def test_get_models_from_dict_response(self, monkeypatch):
        """
        When the /api/tags endpoint returns a dict with a 'models' list,
        get_models should extract and return sorted unique model names.
        """
        calls = []
        call_headers = []
        sample = {
            "models": [
                {"name": "zeta"},
                {"model": "alpha"},
                {"name": 123},  # non-str should be ignored
                "invalid",  # non-dict should be ignored
            ]
        }

        def mock_get(url, headers):
            calls.append(url)
            call_headers.append(headers)
            return DummyResponse(sample, status_code=200)

        monkeypatch.setattr(httpx, "get", mock_get)
        info = OllamaModelInfo()
        models = info.get_models()
        # Only 'alpha' and 'zeta' should be returned, sorted alphabetically
        assert models == ["alpha", "zeta"]
        # Ensure correct endpoint was called
        assert calls and calls[0].endswith("/api/tags")
        assert call_headers and call_headers[0] == {}

    def test_get_models_from_dict_response_api_key(self, monkeypatch):
        """
        When the /api/tags endpoint returns a dict with a 'models' list,
        get_models should extract and return sorted unique model names.
        """
        calls = []
        call_headers = []

        def mock_get(url, headers):
            calls.append(url)
            call_headers.append(headers)
            return DummyResponse({}, status_code=200)

        old_environ = dict(os.environ)
        os.environ.update({"OLLAMA_API_KEY": "test_api_key"})
        monkeypatch.setattr(httpx, "get", mock_get)
        info = OllamaModelInfo()
        models = info.get_models()
        os.environ.clear()
        os.environ.update(old_environ)
        assert models == []
        # Ensure correct endpoint was called
        assert calls and calls[0].endswith("/api/tags")
        assert call_headers and call_headers[0] == {'Authorization': 'Bearer test_api_key'}

    def test_get_models_from_list_response(self, monkeypatch):
        """
        When the /api/tags endpoint returns a list of dicts,
        get_models should extract and return sorted unique model names.
        """
        sample = [
            {"name": "m1"},
            {"model": "m2"},
            {},  # no name/model key should be ignored
        ]

        def mock_get(url, headers):
            return DummyResponse(sample, status_code=200)

        monkeypatch.setattr(httpx, "get", mock_get)
        info = OllamaModelInfo()
        models = info.get_models()
        assert models == ["m1", "m2"]

    def test_get_models_fallback_on_error(self, monkeypatch):
        """
        If the httpx.get call raises an exception, get_models should
        fall back to the static models_by_provider list prefixed by 'ollama/'.
        """

        def mock_get(url, headers):
            raise Exception("connection failure")

        monkeypatch.setattr(httpx, "get", mock_get)
        info = OllamaModelInfo()
        models = info.get_models()
        # Default static ollama_models is ['llama2'], so expect ['ollama/llama2']
        assert models == ["ollama/llama2"]


class TestOllamaAuthHeaders:
    """Tests for Ollama authentication header handling in completion calls."""

    def test_ollama_completion_with_api_key_adds_auth_header(self, monkeypatch):
        """
        Test that when an api_key is provided to ollama completion,
        the Authorization header is added with Bearer token format.

        This tests the bug fix where Ollama requests with API keys
        were not including the Authorization header.
        """
        import litellm
        from unittest.mock import MagicMock, patch

        # Track the headers that were passed to the completion call
        captured_headers = {}

        def mock_completion(*args, **kwargs):
            # Capture the headers that were passed
            if 'headers' in kwargs:
                captured_headers.update(kwargs['headers'])
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch('litellm.main.base_llm_http_handler.completion', side_effect=mock_completion):
            try:
                # Call completion with ollama provider and api_key
                litellm.completion(
                    model="ollama/llama2",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key="test-api-key-12345",
                    api_base="http://localhost:11434"
                )

                # Verify that Authorization header was added
                assert "Authorization" in captured_headers, \
                    "Authorization header should be present when api_key is provided"
                assert captured_headers["Authorization"] == "Bearer test-api-key-12345", \
                    f"Authorization header should be 'Bearer test-api-key-12345', got {captured_headers.get('Authorization')}"

            except Exception as e:
                pytest.fail(f"Ollama completion with api_key failed: {e}")

    def test_ollama_chat_completion_with_api_key_adds_auth_header(self, monkeypatch):
        """
        Test that when an api_key is provided to ollama_chat completion,
        the Authorization header is added with Bearer token format.

        This tests the bug fix for the ollama_chat provider variant.
        """
        import litellm
        from unittest.mock import MagicMock, patch

        # Track the headers that were passed to the completion call
        captured_headers = {}

        def mock_completion(*args, **kwargs):
            # Capture the headers that were passed
            if 'headers' in kwargs:
                captured_headers.update(kwargs['headers'])
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch('litellm.main.base_llm_http_handler.completion', side_effect=mock_completion):
            try:
                # Call completion with ollama_chat provider and api_key
                litellm.completion(
                    model="ollama_chat/llama2",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key="test-api-key-67890",
                    api_base="http://localhost:11434"
                )

                # Verify that Authorization header was added
                assert "Authorization" in captured_headers, \
                    "Authorization header should be present when api_key is provided"
                assert captured_headers["Authorization"] == "Bearer test-api-key-67890", \
                    f"Authorization header should be 'Bearer test-api-key-67890', got {captured_headers.get('Authorization')}"

            except Exception as e:
                pytest.fail(f"Ollama_chat completion with api_key failed: {e}")

    def test_ollama_completion_without_api_key_no_auth_header(self, monkeypatch):
        """
        Test that when no api_key is provided to ollama completion,
        no Authorization header is added.
        """
        import litellm
        from unittest.mock import MagicMock, patch

        # Track the headers that were passed to the completion call
        captured_headers = {}

        def mock_completion(*args, **kwargs):
            # Capture the headers that were passed
            if 'headers' in kwargs:
                captured_headers.update(kwargs['headers'])
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch('litellm.main.base_llm_http_handler.completion', side_effect=mock_completion):
            try:
                # Call completion without api_key
                litellm.completion(
                    model="ollama/llama2",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_base="http://localhost:11434"
                )

                # Verify that Authorization header was NOT added
                assert "Authorization" not in captured_headers, \
                    "Authorization header should not be present when api_key is not provided"

            except Exception as e:
                pytest.fail(f"Ollama completion without api_key failed: {e}")

    def test_ollama_completion_preserves_existing_auth_header(self, monkeypatch):
        """
        Test that when an Authorization header is already present in headers,
        it is not overwritten even if api_key is provided.

        This ensures the fix respects existing Authorization headers.
        """
        import litellm
        from unittest.mock import MagicMock, patch

        # Track the headers that were passed to the completion call
        captured_headers = {}

        def mock_completion(*args, **kwargs):
            # Capture the headers that were passed
            if 'headers' in kwargs:
                captured_headers.update(kwargs['headers'])
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch('litellm.main.base_llm_http_handler.completion', side_effect=mock_completion):
            try:
                # Call completion with both api_key and existing Authorization header
                existing_auth = "Bearer existing-token"
                litellm.completion(
                    model="ollama/llama2",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key="test-api-key-should-not-be-used",
                    api_base="http://localhost:11434",
                    headers={"Authorization": existing_auth}
                )

                # Verify that existing Authorization header was preserved
                assert "Authorization" in captured_headers, \
                    "Authorization header should be present"
                assert captured_headers["Authorization"] == existing_auth, \
                    f"Existing Authorization header should be preserved, got {captured_headers.get('Authorization')}"

            except Exception as e:
                pytest.fail(f"Ollama completion with existing auth header failed: {e}")

    def test_ollama_completion_with_ollama_com_api_base(self, monkeypatch):
        """
        Test that when using https://ollama.com as api_base with an api_key,
        the Authorization header is correctly added.

        This tests the real-world use case of using Ollama's hosted service.
        """
        import litellm
        from unittest.mock import MagicMock, patch

        # Track the headers and api_base that were passed to the completion call
        captured_headers = {}
        captured_api_base = None

        def mock_completion(*args, **kwargs):
            nonlocal captured_api_base
            # Capture the headers and api_base that were passed
            if 'headers' in kwargs:
                captured_headers.update(kwargs['headers'])
            if 'api_base' in kwargs:
                captured_api_base = kwargs['api_base']
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch('litellm.main.base_llm_http_handler.completion', side_effect=mock_completion):
            try:
                # Call completion with ollama.com as api_base and api_key
                litellm.completion(
                    model="ollama/qwen3-vl:235b-cloud",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key="test-ollama-com-api-key",
                    api_base="https://ollama.com"
                )

                # Verify that Authorization header was added
                assert "Authorization" in captured_headers, \
                    "Authorization header should be present when using ollama.com with api_key"
                assert captured_headers["Authorization"] == "Bearer test-ollama-com-api-key", \
                    f"Authorization header should be 'Bearer test-ollama-com-api-key', got {captured_headers.get('Authorization')}"

                # Verify the api_base was passed correctly
                assert captured_api_base == "https://ollama.com", \
                    f"API base should be 'https://ollama.com', got {captured_api_base}"

            except Exception as e:
                pytest.fail(f"Ollama completion with ollama.com api_base failed: {e}")

    def test_ollama_chat_completion_with_ollama_com_api_base(self, monkeypatch):
        """
        Test that when using https://ollama.com as api_base with an api_key
        for ollama_chat provider, the Authorization header is correctly added.

        This tests the real-world use case for the ollama_chat variant.
        """
        import litellm
        from unittest.mock import MagicMock, patch

        # Track the headers and api_base that were passed to the completion call
        captured_headers = {}
        captured_api_base = None

        def mock_completion(*args, **kwargs):
            nonlocal captured_api_base
            # Capture the headers and api_base that were passed
            if 'headers' in kwargs:
                captured_headers.update(kwargs['headers'])
            if 'api_base' in kwargs:
                captured_api_base = kwargs['api_base']
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch('litellm.main.base_llm_http_handler.completion', side_effect=mock_completion):
            try:
                # Call completion with ollama.com as api_base and api_key
                litellm.completion(
                    model="ollama_chat/qwen3-vl:235b-cloud",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key="test-ollama-com-chat-key",
                    api_base="https://ollama.com"
                )

                # Verify that Authorization header was added
                assert "Authorization" in captured_headers, \
                    "Authorization header should be present when using ollama.com with api_key"
                assert captured_headers["Authorization"] == "Bearer test-ollama-com-chat-key", \
                    f"Authorization header should be 'Bearer test-ollama-com-chat-key', got {captured_headers.get('Authorization')}"

                # Verify the api_base was passed correctly
                assert captured_api_base == "https://ollama.com", \
                    f"API base should be 'https://ollama.com', got {captured_api_base}"

            except Exception as e:
                pytest.fail(f"Ollama_chat completion with ollama.com api_base failed: {e}")

    def test_ollama_completion_with_ollama_com_without_api_key_fails_gracefully(self, monkeypatch):
        """
        Test that when using https://ollama.com as api_base without an api_key,
        no Authorization header is added (which would likely fail on the server side,
        but we're testing the client behavior).

        This ensures we don't add empty or None Authorization headers.
        """
        import litellm
        from unittest.mock import MagicMock, patch

        # Track the headers that were passed to the completion call
        captured_headers = {}

        def mock_completion(*args, **kwargs):
            # Capture the headers that were passed
            if 'headers' in kwargs:
                captured_headers.update(kwargs['headers'])
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch('litellm.main.base_llm_http_handler.completion', side_effect=mock_completion):
            try:
                # Call completion with ollama.com but no api_key
                litellm.completion(
                    model="ollama/llama2",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_base="https://ollama.com"
                )

                # Verify that Authorization header was NOT added
                assert "Authorization" not in captured_headers, \
                    "Authorization header should not be present when api_key is not provided, even with ollama.com"

            except Exception as e:
                pytest.fail(f"Ollama completion with ollama.com without api_key failed: {e}")
