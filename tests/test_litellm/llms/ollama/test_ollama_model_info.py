import os
import sys

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
import litellm

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
        assert models == ["ollama/alpha", "ollama/zeta"]
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
        assert call_headers and call_headers[0] == {
            "Authorization": "Bearer test_api_key"
        }

    def test_get_models_does_not_leak_server_key_to_provided_api_base(
        self, monkeypatch
    ):
        """Model discovery should not send server-side keys to caller-supplied bases."""
        call_headers = []

        def mock_get(url, headers):
            call_headers.append(headers)
            return DummyResponse({"models": []}, status_code=200)

        monkeypatch.setenv("OLLAMA_API_KEY", "server-side-ollama-key")
        monkeypatch.setattr(litellm, "api_key", "global-provider-key")
        monkeypatch.setattr(litellm, "openai_key", "global-openai-key")
        monkeypatch.setattr(httpx, "get", mock_get)

        info = OllamaModelInfo()
        models = info.get_models(api_base="https://attacker.example")

        assert models == []
        assert call_headers[0] == {}

    def test_get_models_uses_explicit_api_key_for_provided_api_base(self, monkeypatch):
        """Model discovery should send an explicitly supplied key to the provided base."""
        call_headers = []

        def mock_get(url, headers):
            call_headers.append(headers)
            return DummyResponse({"models": []}, status_code=200)

        monkeypatch.setenv("OLLAMA_API_KEY", "server-side-ollama-key")
        monkeypatch.setattr(httpx, "get", mock_get)

        info = OllamaModelInfo()
        models = info.get_models(
            api_base="https://ollama.example",
            api_key="explicit-api-key",
        )

        assert models == []
        assert call_headers[0] == {"Authorization": "Bearer explicit-api-key"}

    def test_get_models_empty_key_does_not_leak_to_provided_api_base(
        self, monkeypatch
    ):
        """An empty explicit key must not fall back to server-side creds for a custom base."""
        call_headers = []

        def mock_get(url, headers):
            call_headers.append(headers)
            return DummyResponse({"models": []}, status_code=200)

        monkeypatch.setenv("OLLAMA_API_KEY", "server-side-ollama-key")
        monkeypatch.setattr(litellm, "api_key", "global-provider-key")
        monkeypatch.setattr(litellm, "openai_key", "global-openai-key")
        monkeypatch.setattr(httpx, "get", mock_get)

        info = OllamaModelInfo()
        models = info.get_models(api_base="https://attacker.example", api_key="")

        assert models == []
        assert call_headers[0] == {}

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
        assert models == ["ollama/m1", "ollama/m2"]

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

    def test_get_models_no_double_prefix(self, monkeypatch):
        """
        Names that already carry the 'ollama/' prefix (or are returned by an
        Ollama server that's been configured to emit them) should not be
        prefixed a second time.
        """
        sample = {
            "models": [
                {"name": "ollama/already-prefixed"},
                {"name": "fresh"},
                {"name": "hf.co/Qwen/Qwen3-14B:latest"},
            ]
        }

        def mock_get(url, headers):
            return DummyResponse(sample, status_code=200)

        monkeypatch.setattr(httpx, "get", mock_get)
        info = OllamaModelInfo()
        models = info.get_models()
        assert models == [
            "ollama/already-prefixed",
            "ollama/fresh",
            "ollama/hf.co/Qwen/Qwen3-14B:latest",
        ]


class TestOllamaGetModelInfo:
    """Tests for OllamaConfig.get_model_info() api_base threading and graceful fallback."""

    def test_get_model_info_uses_provided_api_base(self, monkeypatch):
        """When api_base is passed, get_model_info should use it instead of env var or default."""
        from litellm.llms.ollama.completion.transformation import OllamaConfig

        captured_urls = []

        def mock_post(url, json, headers=None):
            captured_urls.append(url)
            resp = DummyResponse(
                {
                    "template": "{{ .System }} tools {{ .Prompt }}",
                    "model_info": {"context_length": 4096},
                },
                status_code=200,
            )
            return resp

        monkeypatch.setattr("litellm.module_level_client.post", mock_post)

        config = OllamaConfig()
        result = config.get_model_info(
            "my-custom-model", api_base="http://my-remote-server:11434"
        )

        assert captured_urls[0] == "http://my-remote-server:11434/api/show"
        assert result["max_tokens"] == 4096

    def test_get_model_info_falls_back_to_env_var(self, monkeypatch):
        """When no api_base is passed, should fall back to OLLAMA_API_BASE env var."""
        from litellm.llms.ollama.completion.transformation import OllamaConfig

        captured_urls = []
        captured_headers = []

        def mock_post(url, json, headers=None):
            captured_urls.append(url)
            captured_headers.append(headers)
            return DummyResponse({"template": "", "model_info": {}}, status_code=200)

        monkeypatch.setattr("litellm.module_level_client.post", mock_post)
        monkeypatch.setenv("OLLAMA_API_BASE", "http://env-server:11434")
        monkeypatch.setenv("OLLAMA_API_KEY", "env-api-key")

        config = OllamaConfig()
        config.get_model_info("my-custom-model")

        assert captured_urls[0] == "http://env-server:11434/api/show"
        assert captured_headers[0] == {"Authorization": "Bearer env-api-key"}

    def test_get_model_info_uses_explicit_api_key_for_provided_api_base(
        self, monkeypatch
    ):
        """When api_key is explicit, model info should send it to the provided api_base."""
        from litellm.llms.ollama.completion.transformation import OllamaConfig

        captured_headers = []

        def mock_post(url, json, headers=None):
            captured_headers.append(headers)
            return DummyResponse({"template": "", "model_info": {}}, status_code=200)

        monkeypatch.setattr("litellm.module_level_client.post", mock_post)

        config = OllamaConfig()
        config.get_model_info(
            "my-custom-model",
            api_base="http://my-remote-server:11434",
            api_key="explicit-api-key",
        )

        assert captured_headers[0] == {"Authorization": "Bearer explicit-api-key"}

    def test_get_model_info_empty_key_does_not_leak_to_provided_api_base(
        self, monkeypatch
    ):
        """An empty explicit key must not fall back to server-side creds for a custom base."""
        from litellm.llms.ollama.completion.transformation import OllamaConfig

        captured_headers = []

        def mock_post(url, json, headers=None):
            captured_headers.append(headers)
            return DummyResponse({"template": "", "model_info": {}}, status_code=200)

        monkeypatch.setattr("litellm.module_level_client.post", mock_post)
        monkeypatch.setenv("OLLAMA_API_KEY", "server-side-ollama-key")
        monkeypatch.setattr(litellm, "api_key", "global-provider-key")
        monkeypatch.setattr(litellm, "openai_key", "global-openai-key")

        config = OllamaConfig()
        config.get_model_info(
            "my-custom-model",
            api_base="https://attacker.example",
            api_key="",
        )

        assert captured_headers[0] == {}

    def test_litellm_get_model_info_does_not_leak_server_key_to_provided_api_base(
        self, monkeypatch
    ):
        """Global model info should not send server-side keys to caller-supplied bases."""
        captured_headers = []

        def mock_post(url, json, headers=None):
            captured_headers.append(headers)
            return DummyResponse(
                {
                    "template": "{{ .System }} tools {{ .Prompt }}",
                    "model_info": {"llama.context_length": 32768},
                },
                status_code=200,
            )

        litellm.get_model_info.cache_clear()
        monkeypatch.setattr("litellm.module_level_client.post", mock_post)
        monkeypatch.setenv("OLLAMA_API_KEY", "server-side-ollama-key")
        monkeypatch.setattr(litellm, "api_key", "global-provider-key")
        monkeypatch.setattr(litellm, "openai_key", "global-openai-key")
        try:
            model_info = litellm.get_model_info(
                "ollama/unknown-model",
                api_base="https://attacker.example",
            )
        finally:
            litellm.get_model_info.cache_clear()

        assert model_info["max_input_tokens"] == 32768
        assert captured_headers[0] == {}

    def test_litellm_get_model_info_forwards_explicit_api_key_to_provided_base(
        self, monkeypatch
    ):
        """An explicit api_key passed to litellm.get_model_info must reach the provided base."""
        captured_headers = []

        def mock_post(url, json, headers=None):
            captured_headers.append(headers)
            return DummyResponse(
                {
                    "template": "{{ .System }} tools {{ .Prompt }}",
                    "model_info": {"llama.context_length": 32768},
                },
                status_code=200,
            )

        litellm.get_model_info.cache_clear()
        monkeypatch.setattr("litellm.module_level_client.post", mock_post)
        monkeypatch.setenv("OLLAMA_API_KEY", "server-side-ollama-key")
        try:
            model_info = litellm.get_model_info(
                "ollama/unknown-model",
                api_base="https://ollama.example",
                api_key="explicit-api-key",
            )
        finally:
            litellm.get_model_info.cache_clear()

        assert model_info["max_input_tokens"] == 32768
        assert captured_headers[0] == {"Authorization": "Bearer explicit-api-key"}

    def test_litellm_get_model_info_does_not_cache_on_api_key(self, monkeypatch):
        """Regression: api_key must not be part of the get_model_info cache key.

        Distinct api_keys for the same (model, api_base) must not each create their
        own cache entry (which would churn the shared LRU cache), and every explicit
        key must still reach the backend rather than be served from a result cached
        with a different key.
        """
        from litellm.utils import _cached_get_model_info

        captured_headers = []

        def mock_post(url, json, headers=None):
            captured_headers.append(headers)
            return DummyResponse(
                {
                    "template": "{{ .System }} tools {{ .Prompt }}",
                    "model_info": {"llama.context_length": 32768},
                },
                status_code=200,
            )

        monkeypatch.setattr("litellm.module_level_client.post", mock_post)
        litellm.get_model_info.cache_clear()
        try:
            for api_key in ("key-one", "key-two", "key-three"):
                litellm.get_model_info(
                    "ollama/unknown-model",
                    api_base="https://ollama.example",
                    api_key=api_key,
                )

            assert _cached_get_model_info.cache_info().currsize <= 1
            assert captured_headers == [
                {"Authorization": "Bearer key-one"},
                {"Authorization": "Bearer key-two"},
                {"Authorization": "Bearer key-three"},
            ]
        finally:
            litellm.get_model_info.cache_clear()

    def test_get_model_info_normalizes_generate_api_base(self, monkeypatch):
        """When completion passes the final generate URL, model info should use the server base."""
        from litellm.llms.ollama.completion.transformation import OllamaConfig

        captured_urls = []

        def mock_post(url, json, headers=None):
            captured_urls.append(url)
            return DummyResponse({"template": "", "model_info": {}}, status_code=200)

        monkeypatch.setattr("litellm.module_level_client.post", mock_post)

        config = OllamaConfig()
        config.get_model_info(
            "my-custom-model", api_base="http://localhost:11434/api/generate"
        )

        assert captured_urls[0] == "http://localhost:11434/api/show"

    def test_get_model_info_graceful_fallback_on_connection_error(self, monkeypatch):
        """When the Ollama server is unreachable, should return defaults instead of raising."""
        from litellm.llms.ollama.completion.transformation import OllamaConfig

        def mock_post(url, json, headers=None):
            raise ConnectionError("Connection refused")

        monkeypatch.setattr("litellm.module_level_client.post", mock_post)
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

        config = OllamaConfig()
        result = config.get_model_info(
            "my-custom-model", api_base="http://unreachable:11434"
        )

        assert result["key"] == "my-custom-model"
        assert result["litellm_provider"] == "ollama"
        assert result["input_cost_per_token"] == 0.0
        assert result["output_cost_per_token"] == 0.0
        assert result["max_tokens"] is None

    def test_get_model_info_graceful_fallback_on_http_error_status(self, monkeypatch):
        """A non-2xx /api/show response must fall back to defaults, not parse the error body."""
        from litellm.llms.ollama.completion.transformation import OllamaConfig

        def mock_post(url, json, headers=None):
            return DummyResponse(
                {
                    "template": "{{ .System }} tools {{ .Prompt }}",
                    "model_info": {"llama.context_length": 8192},
                },
                status_code=404,
            )

        monkeypatch.setattr("litellm.module_level_client.post", mock_post)

        config = OllamaConfig()
        result = config.get_model_info(
            "my-custom-model", api_base="http://localhost:11434"
        )

        assert result["key"] == "my-custom-model"
        assert result["litellm_provider"] == "ollama"
        assert result["max_tokens"] is None
        assert result["max_input_tokens"] is None
        assert "supports_function_calling" not in result

    def test_get_model_info_strips_ollama_prefix(self, monkeypatch):
        """Should strip 'ollama/' or 'ollama_chat/' prefix from model name."""
        from litellm.llms.ollama.completion.transformation import OllamaConfig

        captured_json = []

        def mock_post(url, json, headers=None):
            captured_json.append(json)
            return DummyResponse({"template": "", "model_info": {}}, status_code=200)

        monkeypatch.setattr("litellm.module_level_client.post", mock_post)

        config = OllamaConfig()
        config.get_model_info(
            "ollama/my-custom-model", api_base="http://localhost:11434"
        )
        assert captured_json[0]["name"] == "my-custom-model"

        config.get_model_info(
            "ollama_chat/my-custom-model", api_base="http://localhost:11434"
        )
        assert captured_json[1]["name"] == "my-custom-model"

    def test_get_model_info_skips_network_for_static_model(self, monkeypatch):
        """Statically-priced models must not trigger an /api/show network call."""
        from litellm.llms.ollama.completion.transformation import OllamaConfig

        def mock_post(url, json, headers=None):
            raise AssertionError("Static Ollama model should not query /api/show")

        monkeypatch.setattr("litellm.module_level_client.post", mock_post)

        config = OllamaConfig()
        assert config.get_model_info("ollama/llama2") is None

    def test_litellm_get_model_info_uses_provider_hook_for_unknown_model(
        self, monkeypatch
    ):
        """Unmapped Ollama models should use the provider-level dynamic hook."""
        captured_json = []

        def mock_post(url, json, headers=None):
            captured_json.append(json)
            return DummyResponse(
                {
                    "template": "{{ .System }} tools {{ .Prompt }}",
                    "model_info": {"llama.context_length": 32768},
                },
                status_code=200,
            )

        litellm.get_model_info.cache_clear()
        monkeypatch.setattr("litellm.module_level_client.post", mock_post)
        try:
            model_info = litellm.get_model_info(
                "ollama/unknown-model", api_base="http://localhost:11434"
            )
        finally:
            litellm.get_model_info.cache_clear()

        assert model_info["max_input_tokens"] == 32768
        assert model_info["supports_function_calling"] is True
        assert captured_json[0]["name"] == "unknown-model"

    def test_litellm_get_model_info_keeps_static_map_for_known_model(self, monkeypatch):
        """Mapped Ollama models should keep using the static model map."""

        def mock_post(url, json, headers=None):
            raise AssertionError("Static Ollama model should not query /api/show")

        litellm.get_model_info.cache_clear()
        monkeypatch.setattr("litellm.module_level_client.post", mock_post)
        try:
            model_info = litellm.get_model_info("ollama/llama2")
        finally:
            litellm.get_model_info.cache_clear()

        assert model_info["key"] == "ollama/llama2"
        assert model_info["litellm_provider"] == "ollama"


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
            if "headers" in kwargs:
                captured_headers.update(kwargs["headers"])
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch(
            "litellm.main.base_llm_http_handler.completion", side_effect=mock_completion
        ):
            try:
                # Call completion with ollama provider and api_key
                litellm.completion(
                    model="ollama/llama2",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key="test-api-key-12345",
                    api_base="http://localhost:11434",
                )

                # Verify that Authorization header was added
                assert (
                    "Authorization" in captured_headers
                ), "Authorization header should be present when api_key is provided"
                assert (
                    captured_headers["Authorization"] == "Bearer test-api-key-12345"
                ), f"Authorization header should be 'Bearer test-api-key-12345', got {captured_headers.get('Authorization')}"

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
            if "headers" in kwargs:
                captured_headers.update(kwargs["headers"])
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch(
            "litellm.main.base_llm_http_handler.completion", side_effect=mock_completion
        ):
            try:
                # Call completion with ollama_chat provider and api_key
                litellm.completion(
                    model="ollama_chat/llama2",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key="test-api-key-67890",
                    api_base="http://localhost:11434",
                )

                # Verify that Authorization header was added
                assert (
                    "Authorization" in captured_headers
                ), "Authorization header should be present when api_key is provided"
                assert (
                    captured_headers["Authorization"] == "Bearer test-api-key-67890"
                ), f"Authorization header should be 'Bearer test-api-key-67890', got {captured_headers.get('Authorization')}"

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
            if "headers" in kwargs:
                captured_headers.update(kwargs["headers"])
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch(
            "litellm.main.base_llm_http_handler.completion", side_effect=mock_completion
        ):
            try:
                # Call completion without api_key
                litellm.completion(
                    model="ollama/llama2",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_base="http://localhost:11434",
                )

                # Verify that Authorization header was NOT added
                assert (
                    "Authorization" not in captured_headers
                ), "Authorization header should not be present when api_key is not provided"

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
            if "headers" in kwargs:
                captured_headers.update(kwargs["headers"])
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch(
            "litellm.main.base_llm_http_handler.completion", side_effect=mock_completion
        ):
            try:
                # Call completion with both api_key and existing Authorization header
                existing_auth = "Bearer existing-token"
                litellm.completion(
                    model="ollama/llama2",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key="test-api-key-should-not-be-used",
                    api_base="http://localhost:11434",
                    headers={"Authorization": existing_auth},
                )

                # Verify that existing Authorization header was preserved
                assert (
                    "Authorization" in captured_headers
                ), "Authorization header should be present"
                assert (
                    captured_headers["Authorization"] == existing_auth
                ), f"Existing Authorization header should be preserved, got {captured_headers.get('Authorization')}"

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
            if "headers" in kwargs:
                captured_headers.update(kwargs["headers"])
            if "api_base" in kwargs:
                captured_api_base = kwargs["api_base"]
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch(
            "litellm.main.base_llm_http_handler.completion", side_effect=mock_completion
        ):
            try:
                # Call completion with ollama.com as api_base and api_key
                litellm.completion(
                    model="ollama/qwen3-vl:235b-cloud",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key="test-ollama-com-api-key",
                    api_base="https://ollama.com",
                )

                # Verify that Authorization header was added
                assert (
                    "Authorization" in captured_headers
                ), "Authorization header should be present when using ollama.com with api_key"
                assert (
                    captured_headers["Authorization"]
                    == "Bearer test-ollama-com-api-key"
                ), f"Authorization header should be 'Bearer test-ollama-com-api-key', got {captured_headers.get('Authorization')}"

                # Verify the api_base was passed correctly
                assert (
                    captured_api_base == "https://ollama.com"
                ), f"API base should be 'https://ollama.com', got {captured_api_base}"

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
            if "headers" in kwargs:
                captured_headers.update(kwargs["headers"])
            if "api_base" in kwargs:
                captured_api_base = kwargs["api_base"]
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch(
            "litellm.main.base_llm_http_handler.completion", side_effect=mock_completion
        ):
            try:
                # Call completion with ollama.com as api_base and api_key
                litellm.completion(
                    model="ollama_chat/qwen3-vl:235b-cloud",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key="test-ollama-com-chat-key",
                    api_base="https://ollama.com",
                )

                # Verify that Authorization header was added
                assert (
                    "Authorization" in captured_headers
                ), "Authorization header should be present when using ollama.com with api_key"
                assert (
                    captured_headers["Authorization"]
                    == "Bearer test-ollama-com-chat-key"
                ), f"Authorization header should be 'Bearer test-ollama-com-chat-key', got {captured_headers.get('Authorization')}"

                # Verify the api_base was passed correctly
                assert (
                    captured_api_base == "https://ollama.com"
                ), f"API base should be 'https://ollama.com', got {captured_api_base}"

            except Exception as e:
                pytest.fail(
                    f"Ollama_chat completion with ollama.com api_base failed: {e}"
                )

    def test_ollama_completion_with_ollama_com_without_api_key_fails_gracefully(
        self, monkeypatch
    ):
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
            if "headers" in kwargs:
                captured_headers.update(kwargs["headers"])
            # Return a mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            return mock_response

        # Mock the base_llm_http_handler.completion method at the module level
        with patch(
            "litellm.main.base_llm_http_handler.completion", side_effect=mock_completion
        ):
            try:
                # Call completion with ollama.com but no api_key
                litellm.completion(
                    model="ollama/llama2",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_base="https://ollama.com",
                )

                # Verify that Authorization header was NOT added
                assert (
                    "Authorization" not in captured_headers
                ), "Authorization header should not be present when api_key is not provided, even with ollama.com"

            except Exception as e:
                pytest.fail(
                    f"Ollama completion with ollama.com without api_key failed: {e}"
                )
