"""
Tests for OpenCode provider registration (litellm/llms/opencode/).

These tests fail before the feature exists and fail if the dispatch
mapping, auth header selection, or URL construction are mutated.
"""

import json


import respx  # noqa: F401  # required for pytest-respx fixture
from httpx import Response

import litellm
import pytest

from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
)
from litellm.llms.opencode.chat.transformation import OpenCodeConfig
from litellm.llms.opencode.common_utils import OpenCodeException

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_response(model: str, content: str, **usage_kwargs) -> dict:
    """Build a standard chat-completion response body."""
    prompt = usage_kwargs.get("prompt_tokens", 1)
    completion = usage_kwargs.get("completion_tokens", 1)
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


class TestOpenCodeConfig:
    """Tests for the OpenCodeConfig class itself."""

    def test_zen_surface_custom_llm_provider(self):
        cfg = OpenCodeConfig(surface="zen")
        assert cfg.custom_llm_provider == "opencode_zen"

    def test_go_surface_custom_llm_provider(self):
        cfg = OpenCodeConfig(surface="go")
        assert cfg.custom_llm_provider == "opencode_go"

    def test_zen_base_url(self):
        cfg = OpenCodeConfig(surface="zen")
        assert cfg._base_url() == "https://opencode.ai/zen/v1"

    def test_go_base_url(self):
        cfg = OpenCodeConfig(surface="go")
        assert cfg._base_url() == "https://opencode.ai/zen/go/v1"

    def test_get_complete_url_zen_default(self):
        cfg = OpenCodeConfig(surface="zen")
        url = cfg.get_complete_url(None, None, "gpt-5.1", {}, {})
        assert url == "https://opencode.ai/zen/v1/chat/completions"

    def test_get_complete_url_go_default(self):
        cfg = OpenCodeConfig(surface="go")
        url = cfg.get_complete_url(None, None, "gpt-5.1", {}, {})
        assert url == "https://opencode.ai/zen/go/v1/chat/completions"

    def test_get_complete_url_api_base_override(self):
        cfg = OpenCodeConfig(surface="zen")
        url = cfg.get_complete_url("http://localhost:4000", None, "gpt-5.1", {}, {})
        assert url == "http://localhost:4000/chat/completions"

    def test_get_complete_url_api_base_trailing_slash(self):
        cfg = OpenCodeConfig(surface="zen")
        url = cfg.get_complete_url("http://localhost:4000/", None, "gpt-5.1", {}, {})
        assert url == "http://localhost:4000/chat/completions"

    def test_error_class(self):
        cfg = OpenCodeConfig(surface="zen")
        err = cfg.get_error_class("bad request", 400, {})
        assert isinstance(err, OpenCodeException)
        assert err.status_code == 400
        assert err.message == "bad request"


# ---------------------------------------------------------------------------
# validate_environment — Bearer header injection
# ---------------------------------------------------------------------------


class TestValidateEnvironment:
    """Tests for header injection in validate_environment."""

    def setup_method(self):
        self.cfg = OpenCodeConfig(surface="zen")

    def test_bearer_header_with_explicit_key(self):
        headers: dict = {}
        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-test-123",
        )
        assert result["Authorization"] == "Bearer sk-test-123"

    def test_content_type_default(self):
        headers: dict = {}
        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-test",
        )
        assert "Content-Type" in result
        assert result["Content-Type"] == "application/json"

    def test_no_key_no_auth_header(self, monkeypatch):
        # Isolate from any OPENCODE_*_API_KEY present in the shell env so the
        # shared fallback cannot inject an Authorization header.
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
        monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
        headers: dict = {}
        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert "Authorization" not in result

    def test_env_var_key_resolution(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "sk-env-123")
        headers: dict = {}
        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert result["Authorization"] == "Bearer sk-env-123"
        monkeypatch.delenv("OPENCODE_ZEN_API_KEY")

    def test_shared_fallback_key(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "sk-shared-456")
        headers: dict = {}
        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert result["Authorization"] == "Bearer sk-shared-456"
        monkeypatch.delenv("OPENCODE_API_KEY")

    def test_module_var_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "sk-env")
        monkeypatch.setattr(litellm, "opencode_zen_api_key", "sk-module")
        headers: dict = {}
        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert result["Authorization"] == "Bearer sk-module"

    def test_explicit_key_takes_precedence_over_module_var(self, monkeypatch):
        monkeypatch.setattr(litellm, "opencode_zen_api_key", "sk-module")
        headers: dict = {}
        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-explicit",
        )
        assert result["Authorization"] == "Bearer sk-explicit"

    def test_global_api_key_does_not_override_opencode_key(self, monkeypatch):
        """A process-wide litellm.api_key must not win over an OpenCode key.

        Regression guard for the cross-provider credential-disclosure claim: a
        mixed-provider process sets litellm.api_key for some other provider, and
        that unrelated credential must never be sent to opencode.ai when an
        OpenCode-specific key is configured.
        """
        monkeypatch.setattr(litellm, "api_key", "sk-global-other-provider")
        monkeypatch.setattr(litellm, "opencode_zen_api_key", "sk-opencode")
        headers: dict = {}
        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert result["Authorization"] == "Bearer sk-opencode"

    def test_go_surface_uses_go_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-go-789")
        go_cfg = OpenCodeConfig(surface="go")
        headers: dict = {}
        result = go_cfg.validate_environment(
            headers=headers,
            model="gpt-5.1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert result["Authorization"] == "Bearer sk-go-789"
        monkeypatch.delenv("OPENCODE_GO_API_KEY")


# ---------------------------------------------------------------------------
# Integration — mocked completion call
# ---------------------------------------------------------------------------


class TestMockedCompletion:
    """End-to-end tests using mocked HTTP transport."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, monkeypatch):
        """Ensure module-level keys and flags are clean after each test."""
        monkeypatch.setattr(litellm, "opencode_zen_api_key", None)
        monkeypatch.setattr(litellm, "opencode_go_api_key", None)
        monkeypatch.setattr(litellm, "opencode_api_key", None)
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "api_base", None)
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
        litellm.in_memory_llm_clients_cache.flush_cache()

    def test_dispatch_sends_to_chat_completions_url(self, respx_mock, monkeypatch):
        """Model opencode_zen/<model> reaches /chat/completions endpoint."""
        respx_mock.post("https://opencode.ai/zen/v1/chat/completions").mock(
            return_value=Response(200, json=_make_response("grok-4.5", "Hello"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-fake")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_zen/grok-4.5",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
        )

        assert result is not None
        assert result.model == "grok-4.5"
        assert result.choices[0].message.content == "Hello"
        assert len(respx_mock.calls) > 0
        request = respx_mock.calls[0].request
        assert request.headers["Authorization"] == "Bearer sk-fake"
        body = json.loads(request.read())
        assert body["messages"] == [{"role": "user", "content": "hi"}]

    def test_go_dispatch_custom_llm_provider(self, respx_mock, monkeypatch):
        """opencode_go models use the opencode_go custom_llm_provider."""
        respx_mock.post("https://opencode.ai/zen/go/v1/chat/completions").mock(
            return_value=Response(200, json=_make_response("grok-4.5", "Go works"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-fake")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_go/grok-4.5",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_go",
        )

        assert result is not None
        assert result.choices[0].message.content == "Go works"
        assert len(respx_mock.calls) > 0
        request = respx_mock.calls[0].request
        assert "/zen/go/" in request.url.path

    def test_unknown_model_routes_to_chat_arm(self, respx_mock, monkeypatch):
        """Unknown models still route to the chat arm."""
        respx_mock.post("https://opencode.ai/zen/v1/chat/completions").mock(
            return_value=Response(200, json=_make_response("brand-new-model", "I am new"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-fake")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_zen/brand-new-model",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
        )

        assert result is not None
        assert result.model == "brand-new-model"
        assert len(respx_mock.calls) > 0

    def test_upstream_error_surfaces_as_connection_error(self, respx_mock, monkeypatch):
        """A non-2xx upstream response surfaces as litellm's public
        APIConnectionError, not a TypeError from mis-constructing the error
        class. Regression: get_error_class returned the class instead of an
        instance, so raising it crashed with
        ``BaseLLMException.__init__() missing 2 required positional arguments``."""
        respx_mock.post("https://opencode.ai/zen/v1/chat/completions").mock(
            return_value=Response(503, json={"error": {"message": "Service Unavailable"}})
        )

        monkeypatch.setattr(litellm, "api_key", "sk-fake")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        with pytest.raises(
            (litellm.exceptions.APIConnectionError, litellm.exceptions.ServiceUnavailableError)
        ) as excinfo:
            litellm.completion(
                model="opencode_zen/grok-4.5",
                messages=[{"role": "user", "content": "hi"}],
                custom_llm_provider="opencode_zen",
            )

        assert "503" in str(excinfo.value) or "Service Unavailable" in str(excinfo.value)

    def test_api_base_override(self, respx_mock, monkeypatch):
        """Explicit api_base overrides the default gateway URL."""
        respx_mock.post("http://localhost:4000/chat/completions").mock(
            return_value=Response(200, json=_make_response("grok-4.5", "local"))
        )

        monkeypatch.setattr(litellm, "api_base", "http://localhost:4000")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_zen/grok-4.5",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
        )

        assert result is not None
        assert result.choices[0].message.content == "local"
        assert len(respx_mock.calls) > 0

    def test_bearer_auth_from_surface_key(self, respx_mock, monkeypatch):
        """Surface-specific env var provides the Bearer token."""
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "sk-surface-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        respx_mock.post("https://opencode.ai/zen/v1/chat/completions").mock(
            return_value=Response(200, json=_make_response("grok-4.5", "ok"))
        )

        result = litellm.completion(
            model="opencode_zen/grok-4.5",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
        )

        assert result is not None
        auth = respx_mock.calls[0].request.headers["Authorization"]
        assert auth == "Bearer sk-surface-key"
        monkeypatch.delenv("OPENCODE_ZEN_API_KEY")

    def test_global_api_key_not_sent_on_dispatch(self, respx_mock, monkeypatch):
        """main.py builds the Bearer header from the OpenCode key, not the global.

        Regression guard for the credential-disclosure claim: this header is
        constructed in the completion dispatcher rather than in
        validate_environment, so the precedence ordering is asserted end-to-end
        with a process-wide litellm.api_key also configured.
        """
        respx_mock.post("https://opencode.ai/zen/v1/chat/completions").mock(
            return_value=Response(200, json=_make_response("grok-4.5", "ok"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-global-other-provider")
        monkeypatch.setattr(litellm, "opencode_zen_api_key", "sk-opencode")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_zen/grok-4.5",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
        )

        assert result is not None
        auth = respx_mock.calls[0].request.headers["Authorization"]
        assert auth == "Bearer sk-opencode"
        assert "sk-global-other-provider" not in auth


# ---------------------------------------------------------------------------
# Cost map
# ---------------------------------------------------------------------------


class TestCostMap:
    """Cost-map entries for OpenCode models."""

    @pytest.fixture(autouse=True)
    def _load_cost_map(self, monkeypatch):
        """Ensure litellm.model_cost is populated before each test."""
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    def test_cost_map_entry_exists(self):
        """A cost-map entry exists for opencode_zen/grok-4.5."""
        assert "opencode_zen/grok-4.5" in litellm.model_cost
        entry = litellm.model_cost["opencode_zen/grok-4.5"]
        assert entry["litellm_provider"] == "opencode_zen"
        assert entry["mode"] == "chat"
        assert entry["max_input_tokens"] == 500000
        assert entry["max_output_tokens"] == 500000

    def test_cost_map_entry_has_pricing(self):
        """Cost-map entry carries the correct pricing."""
        entry = litellm.model_cost["opencode_zen/grok-4.5"]
        assert entry["input_cost_per_token"] == 2e-06
        assert entry["output_cost_per_token"] == 6e-06

    def test_cost_map_entry_for_free_model(self):
        """Free models have zero pricing but still have a cost-map entry."""
        entry = litellm.model_cost["opencode_zen/big-pickle"]
        assert entry["input_cost_per_token"] == 0.0
        assert entry["output_cost_per_token"] == 0.0


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    """Tests for streaming response handler."""

    def test_streaming_handler_available(self):
        """OpenCodeConfig returns a streaming handler."""
        cfg = OpenCodeConfig(surface="zen")
        handler = cfg.get_model_response_iterator(
            streaming_response=None,
            sync_stream=False,
        )
        assert handler is not None


# ---------------------------------------------------------------------------
# Wildcard model-list registration
# ---------------------------------------------------------------------------


class TestWildcardModelRegistration:
    """OpenCode providers are registered in models_by_provider so wildcard
    routes (e.g. ``opencode_go/*``) expand to the cost-map models in the
    playground and /model_group/info."""

    @pytest.fixture(autouse=True)
    def _load_cost_map(self, monkeypatch):
        """Ensure litellm.model_cost is populated before each test.

        The module-level ``opencode_go_models`` / ``opencode_zen_models`` sets
        are filled at import time from the remote cost map, which predates the
        un-merged opencode feature. Re-run ``add_known_models`` against the
        local backup so the wildcard expansion sees the reconciled roster.
        """
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        litellm.add_known_models()

    def test_opencode_providers_registered_in_models_by_provider(self):
        """Both surfaces are keys in models_by_provider."""
        assert "opencode_go" in litellm.models_by_provider
        assert "opencode_zen" in litellm.models_by_provider

    def test_opencode_models_populated_from_cost_map(self):
        """The model sets are populated from the cost map, not empty."""
        assert len(litellm.opencode_go_models) > 0
        assert len(litellm.opencode_zen_models) > 0
        assert "opencode_go/deepseek-v4-flash" in litellm.opencode_go_models
        assert "opencode_zen/claude-sonnet-5" in litellm.opencode_zen_models

    def test_get_provider_models_expands_wildcard(self):
        """get_provider_models returns the cost-map models for both surfaces."""
        from litellm.proxy.auth.model_checks import get_provider_models

        go_models = get_provider_models(provider="opencode_go")
        zen_models = get_provider_models(provider="opencode_zen")

        assert go_models is not None
        assert zen_models is not None
        assert any(m.startswith("opencode_go/") for m in go_models)
        assert any(m.startswith("opencode_zen/") for m in zen_models)

    def test_get_known_models_from_wildcard_expands(self):
        """A wildcard route expands to the full model list for the surface."""
        from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

        go_models = get_known_models_from_wildcard("opencode_go/*")
        zen_models = get_known_models_from_wildcard("opencode_zen/*")

        assert len(go_models) > 0
        assert len(zen_models) > 0
        assert all(m.startswith("opencode_go/") for m in go_models)
        assert all(m.startswith("opencode_zen/") for m in zen_models)
        assert "opencode_go/*" not in go_models
        assert "opencode_zen/*" not in zen_models
