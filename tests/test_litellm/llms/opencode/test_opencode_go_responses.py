"""
Tests for the OpenCode Go responses arm.

These tests verify the Go responses config class, validate_environment,
resolver, mocked completion, and cost-map entry for gpt-5.6-luna.

Acceptance criteria from Issue 04:
- opencode_go/gpt-5.6-luna is taken over by the responses bridge
- Takes over before any chat or messages dispatch
- responses-config resolver returns Go Responses config
- Cost-map entry carries "mode": "responses"
- Bearer auth works from explicit key, module var, env var, shared fallback
"""

import json


import respx  # noqa: F401  # required for pytest-respx fixture
from httpx import Response

import litellm
import pytest

from litellm.llms.opencode.go.responses.transformation import (
    OpenCodeGoResponsesAPIConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GO_RESPONSE_ENDPOINT = "https://opencode.ai/zen/go/v1/responses"


def _make_responses_response(model: str, content: str, **usage_kwargs) -> dict:
    """Build a standard Responses API response body."""
    prompt = usage_kwargs.get("prompt_tokens", 1)
    completion = usage_kwargs.get("completion_tokens", 1)
    return {
        "id": "resp-123",
        "object": "response",
        "created_at": 1700000000,
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        ],
        "usage": {
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


# ---------------------------------------------------------------------------
# Responses config class
# ---------------------------------------------------------------------------


class TestOpenCodeGoResponsesAPIConfig:
    """Tests for the OpenCodeGoResponsesAPIConfig class itself."""

    def test_custom_llm_provider(self):
        """custom_llm_provider should return OPENCODE_GO."""
        cfg = OpenCodeGoResponsesAPIConfig()
        assert cfg.custom_llm_provider == LlmProviders.OPENCODE_GO

    def test_get_complete_url_default(self):
        """Default URL should point to Go Responses API endpoint."""
        cfg = OpenCodeGoResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == GO_RESPONSE_ENDPOINT

    def test_get_complete_url_custom_base_with_v1(self):
        """Custom api_base ending in /v1 should append /responses."""
        cfg = OpenCodeGoResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="http://localhost:4000/v1",
            litellm_params={},
        )
        assert url == "http://localhost:4000/v1/responses"

    def test_get_complete_url_custom_base_with_v1_trailing_slash(self):
        """Trailing slash on /v1 should be stripped before appending /responses."""
        cfg = OpenCodeGoResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="http://localhost:4000/v1/",
            litellm_params={},
        )
        assert url == "http://localhost:4000/v1/responses"

    def test_get_complete_url_custom_base_with_responses(self):
        """api_base already ending in /responses should pass through."""
        cfg = OpenCodeGoResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://my-gateway.example.com/v1/responses",
            litellm_params={},
        )
        assert url == "https://my-gateway.example.com/v1/responses"

    def test_get_complete_url_custom_base_no_suffix(self):
        """Base without /v1 should get /v1/responses appended."""
        cfg = OpenCodeGoResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="http://localhost:4000",
            litellm_params={},
        )
        assert url == "http://localhost:4000/v1/responses"

    def test_get_complete_url_custom_base_trailing_slash(self):
        """Trailing slash on bare base should be stripped."""
        cfg = OpenCodeGoResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="http://localhost:4000/",
            litellm_params={},
        )
        assert url == "http://localhost:4000/v1/responses"

    def test_get_complete_url_env_var_base(self, monkeypatch):
        """OPENCODE_GO_API_BASE env var should be used as fallback base."""
        monkeypatch.setenv("OPENCODE_GO_API_BASE", "http://env-gateway.example.com")
        cfg = OpenCodeGoResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "http://env-gateway.example.com/v1/responses"

    def test_get_complete_url_module_var_base(self, monkeypatch):
        """Module-level opencode_go_api_base should override env var."""
        monkeypatch.setenv("OPENCODE_GO_API_BASE", "http://env-gateway.example.com")
        monkeypatch.setattr(litellm, "opencode_go_api_base", "http://module-gateway.example.com")
        cfg = OpenCodeGoResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "http://module-gateway.example.com/v1/responses"

    def test_no_native_websocket(self):
        """OpenCode Go does not support native WebSocket for Responses API."""
        cfg = OpenCodeGoResponsesAPIConfig()
        assert cfg.supports_native_websocket() is False


# ---------------------------------------------------------------------------
# validate_environment — Bearer header injection
# ---------------------------------------------------------------------------


class TestGoValidateEnvironment:
    """Tests for header injection in OpenCodeGoResponsesAPIConfig.validate_environment."""

    def setup_method(self):
        self.cfg = OpenCodeGoResponsesAPIConfig()

    def test_bearer_header_with_explicit_key(self):
        headers: dict = {}
        from litellm.types.router import GenericLiteLLMParams

        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.6-luna",
            litellm_params=GenericLiteLLMParams(api_key="sk-test-123"),
        )
        assert result["Authorization"] == "Bearer sk-test-123"
        assert result["Content-Type"] == "application/json"

    def test_bearer_header_from_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-env-123")
        headers: dict = {}
        from litellm.types.router import GenericLiteLLMParams

        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.6-luna",
            litellm_params=GenericLiteLLMParams(),
        )
        assert result["Authorization"] == "Bearer sk-env-123"
        monkeypatch.delenv("OPENCODE_GO_API_KEY")

    def test_bearer_header_from_module_key(self, monkeypatch):
        """Module-level opencode_go_api_key should be used."""
        monkeypatch.setattr(litellm, "opencode_go_api_key", "sk-module-456")
        headers: dict = {}
        from litellm.types.router import GenericLiteLLMParams

        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.6-luna",
            litellm_params=GenericLiteLLMParams(),
        )
        assert result["Authorization"] == "Bearer sk-module-456"
        monkeypatch.setattr(litellm, "opencode_go_api_key", None)

    def test_shared_fallback_key(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "sk-shared-789")
        headers: dict = {}
        from litellm.types.router import GenericLiteLLMParams

        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.6-luna",
            litellm_params=GenericLiteLLMParams(),
        )
        assert result["Authorization"] == "Bearer sk-shared-789"
        monkeypatch.delenv("OPENCODE_API_KEY")

    def test_raises_without_any_key(self, monkeypatch):
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)

        from litellm.types.router import GenericLiteLLMParams

        with pytest.raises(ValueError, match="OpenCode Go API key is required"):
            self.cfg.validate_environment(
                headers={},
                model="gpt-5.6-luna",
                litellm_params=GenericLiteLLMParams(),
            )


# ---------------------------------------------------------------------------
# Responses-config resolver
# ---------------------------------------------------------------------------


class TestGoResponsesConfigResolver:
    """Test that the responses-config resolver returns the correct config for Go."""

    def test_provider_config_manager_returns_go_config(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENCODE_GO,
        )
        assert config is not None, "OpenCode Go must be registered in the responses-config resolver"
        assert isinstance(config, OpenCodeGoResponsesAPIConfig)

    def test_resolver_returns_different_config_than_zen(self):
        """Go resolver should not return Zen config."""
        zen_config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENCODE_ZEN,
        )
        go_config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENCODE_GO,
        )

        assert zen_config is not None
        assert go_config is not None
        assert type(zen_config) is not type(go_config)
        assert isinstance(go_config, OpenCodeGoResponsesAPIConfig)

    def test_go_config_url_is_go(self):
        """The resolver config URL should point to Go, not Zen."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENCODE_GO,
        )
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == GO_RESPONSE_ENDPOINT
        assert "opencode.ai/zen/go" in url

    def test_go_config_not_chat_config(self):
        """Verify the responses config is distinct from the chat config."""
        from litellm.llms.opencode.chat.transformation import OpenCodeConfig

        responses_cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENCODE_GO,
        )
        assert responses_cfg is not None
        assert type(responses_cfg) is not OpenCodeConfig
        url = responses_cfg.get_complete_url(api_base=None, litellm_params={})
        assert "/responses" in url
        assert "/chat/completions" not in url


# ---------------------------------------------------------------------------
# Integration — mocked completion call hits /v1/responses
# ---------------------------------------------------------------------------


class TestGoMockedCompletion:
    """Tests using mocked HTTP transport to verify the Go responses bridge."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        """Clean state for every test."""
        monkeypatch.setattr(litellm, "opencode_zen_api_key", None)
        monkeypatch.setattr(litellm, "opencode_go_api_key", None)
        monkeypatch.setattr(litellm, "opencode_api_key", None)
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "api_base", None)
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
        litellm.in_memory_llm_clients_cache.flush_cache()

    def test_responses_bridge_hits_responses_endpoint(self, respx_mock, monkeypatch):
        """opencode_go/gpt-5.6-luna is routed to /v1/responses, not /v1/chat/completions."""
        respx_mock.post(GO_RESPONSE_ENDPOINT).mock(
            return_value=Response(200, json=_make_responses_response("gpt-5.6-luna", "go responses work"))
        )

        monkeypatch.setattr(litellm, "api_key", "sk-fake")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        result = litellm.completion(
            model="opencode_go/gpt-5.6-luna",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is not None
        assert result.model == "gpt-5.6-luna"
        output = result.choices[0].message.content
        assert output is not None
        assert len(respx_mock.calls) > 0
        request = respx_mock.calls[0].request
        assert "/v1/responses" in str(request.url)
        assert request.headers["Authorization"] == "Bearer sk-fake"

    def test_bearer_auth_from_module_key(self, respx_mock, monkeypatch):
        """Module-level opencode_go_api_key provides the Bearer token."""
        respx_mock.post(GO_RESPONSE_ENDPOINT).mock(
            return_value=Response(200, json=_make_responses_response("gpt-5.6-luna", "auth ok"))
        )

        monkeypatch.setattr(litellm, "opencode_go_api_key", "sk-module-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

        result = litellm.completion(
            model="opencode_go/gpt-5.6-luna",
            messages=[{"role": "user", "content": "test"}],
        )

        assert result is not None
        auth = respx_mock.calls[0].request.headers["Authorization"]
        assert auth == "Bearer sk-module-key"
        monkeypatch.setattr(litellm, "opencode_go_api_key", None)

    def test_responses_bridge_sends_correct_body(self, respx_mock, monkeypatch):
        """The request body should use the responses API format."""

        def capture_request(request):
            body = json.loads(request.read())
            assert "model" in body
            return Response(200, json=_make_responses_response(body.get("model", "gpt-5.6-luna"), "ok"))

        respx_mock.post(GO_RESPONSE_ENDPOINT).mock(side_effect=capture_request)

        monkeypatch.setattr(litellm, "api_key", "sk-fake")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

        result = litellm.completion(
            model="opencode_go/gpt-5.6-luna",
            messages=[{"role": "user", "content": "check body shape"}],
        )

        assert result is not None

        respx_mock.post(GO_RESPONSE_ENDPOINT).mock(side_effect=capture_request)

        monkeypatch.setattr(litellm, "api_key", "sk-fake")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

        result = litellm.completion(
            model="opencode_go/gpt-5.6-luna",
            messages=[{"role": "user", "content": "check body shape"}],
        )

        assert result is not None

    def test_non_responses_model_does_not_use_responses_endpoint(self, respx_mock, monkeypatch):
        """A Go chat model must hit /v1/chat/completions, not /v1/responses."""
        chat_url = "https://opencode.ai/zen/go/v1/chat/completions"
        respx_mock.post(chat_url).mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "chat ok"}}],
                    "model": "deepseek-v4-pro",
                },
            )
        )

        monkeypatch.setattr(litellm, "api_key", "sk-fake")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

        result = litellm.completion(
            model="opencode_go/deepseek-v4-pro",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is not None
        assert len(respx_mock.calls) > 0
        call_path = respx_mock.calls[0].request.url.path
        assert "/chat/completions" in call_path
        assert "/responses" not in call_path


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _load_cost_map(monkeypatch):
    """Ensure the local model_cost map is loaded for every test."""
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


# ---------------------------------------------------------------------------
# Cost-map entries
# ---------------------------------------------------------------------------


class TestGoCostMap:
    """Cost-map entry for opencode_go gpt-5.6-luna."""

    def _check_base_entry(self, model_key):
        """Return the cost-map entry for a model, asserting it exists."""
        assert model_key in litellm.model_cost, f"{model_key} must be in cost-map"
        entry = litellm.model_cost[model_key]
        return entry

    def test_gpt_5_6_luna_responses(self):
        entry = self._check_base_entry("opencode_go/gpt-5.6-luna")
        assert entry["mode"] == "responses"
        assert entry["litellm_provider"] == "opencode_go"
        assert entry["input_cost_per_token"] == 1e-07
        assert entry["output_cost_per_token"] == 6e-07

    def test_go_messages_models_stay_on_messages(self):
        """Go messages models must remain on messages mode."""
        for model in [
            "opencode_go/qwen3.8-max",
            "opencode_go/minimax-m3",
            "opencode_go/qwen3.5-plus",
        ]:
            entry = self._check_base_entry(model)
            assert entry["mode"] == "messages", f"{model} mode must be messages, got {entry['mode']}"

    def test_go_chat_models_stay_on_chat(self):
        """Go chat models must remain on chat mode."""
        for model in [
            "opencode_go/deepseek-v4-pro",
            "opencode_go/grok-4.5",
            "opencode_go/mimo-v2.5",
        ]:
            entry = self._check_base_entry(model)
            assert entry["mode"] == "chat", f"{model} mode must be chat, got {entry['mode']}"

    def test_go_gpt_5_6_luna_not_on_zen_responses(self):
        """The Go gpt-5.6-luna entry must exist as its own opencode_go key."""
        go_entry = self._check_base_entry("opencode_go/gpt-5.6-luna")
        assert go_entry["mode"] == "responses"
        assert go_entry["litellm_provider"] == "opencode_go"
        # Zen gpt-5.6-luna is a separate entry
        zen_entry = self._check_base_entry("opencode_zen/gpt-5.6-luna")
        assert zen_entry["mode"] == "responses"
        assert zen_entry["litellm_provider"] == "opencode_zen"
        # They should have different pricing
        assert (
            go_entry["input_cost_per_token"] != zen_entry["input_cost_per_token"]
            or go_entry["output_cost_per_token"] != zen_entry["output_cost_per_token"]
        )

    def test_go_cost_map_matches_live_roster(self):
        """The Go cost map must exactly match the live /v1/models roster.

        Regression guard: Zen-only crossover models (gpt-5.x family, the
        *-free models, grok-build-0.1, big-pickle) were leaking into the Go
        cost map and showing up in the wildcard model list even though they
        are not callable on Go. The cost map must contain exactly the models
        served by https://opencode.ai/zen/go/v1/models.
        """
        live_go_models = {
            "opencode_go/deepseek-v4-flash",
            "opencode_go/deepseek-v4-pro",
            "opencode_go/glm-5",
            "opencode_go/glm-5.1",
            "opencode_go/glm-5.2",
            "opencode_go/gpt-5.6-luna",
            "opencode_go/grok-4.5",
            "opencode_go/hy3",
            "opencode_go/kimi-k2.5",
            "opencode_go/kimi-k2.6",
            "opencode_go/kimi-k2.7-code",
            "opencode_go/kimi-k3",
            "opencode_go/mimo-v2-omni",
            "opencode_go/mimo-v2-pro",
            "opencode_go/mimo-v2.5",
            "opencode_go/mimo-v2.5-pro",
            "opencode_go/minimax-m2.5",
            "opencode_go/minimax-m2.7",
            "opencode_go/minimax-m3",
            "opencode_go/qwen3.5-plus",
            "opencode_go/qwen3.6-plus",
            "opencode_go/qwen3.7-max",
            "opencode_go/qwen3.7-plus",
            "opencode_go/qwen3.8-flash",
            "opencode_go/qwen3.8-max",
        }
        actual_go = {k for k in litellm.model_cost if k.startswith("opencode_go/")}
        assert actual_go == live_go_models, (
            f"Go cost map diverged from live roster. "
            f"extra={sorted(actual_go - live_go_models)} "
            f"missing={sorted(live_go_models - actual_go)}"
        )

    def test_go_crossover_models_removed(self):
        """Zen-only crossover models must not appear under the Go provider."""
        for model in [
            "opencode_go/gpt-5.6-sol",
            "opencode_go/gpt-5",
            "opencode_go/grok-build-0.1",
            "opencode_go/big-pickle",
            "opencode_go/mimo-v2.5-free",
        ]:
            assert model not in litellm.model_cost, f"{model} must not be in Go cost map"

    def test_new_go_models_present(self):
        """The 5 live Go models added to the gateway must have cost-map entries."""
        for model in [
            "opencode_go/hy3",
            "opencode_go/mimo-v2.5",
            "opencode_go/mimo-v2.5-pro",
            "opencode_go/mimo-v2-omni",
            "opencode_go/mimo-v2-pro",
        ]:
            entry = self._check_base_entry(model)
            assert entry["mode"] == "chat", f"{model} mode must be chat, got {entry['mode']}"
