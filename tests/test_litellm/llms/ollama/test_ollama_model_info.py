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
