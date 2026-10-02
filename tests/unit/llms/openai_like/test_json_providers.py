"""
Tests for JSON-based provider configuration system.
"""

import json
import os
import sys
from unittest.mock import patch

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

try:
    import pytest
except ImportError:
    # pytest not available, will run as standalone script
    pytest = None

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)


class TestJSONProviderLoader:
    """Test JSON provider loading and configuration"""

    def test_load_json_providers(self):
        """Test that JSON providers load correctly"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        # Verify publicai is loaded
        assert JSONProviderRegistry.exists("publicai")

        # Get publicai config
        publicai = JSONProviderRegistry.get("publicai")
        assert publicai is not None
        assert publicai.base_url == "https://api.publicai.co/v1"
        assert publicai.api_key_env == "PUBLICAI_API_KEY"
        assert publicai.api_base_env == "PUBLICAI_API_BASE"
        assert publicai.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_dynamic_config_generation(self):
        """Test dynamic config class creation"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Test API info resolution
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.publicai.co/v1"

        # Test with custom base
        api_base, api_key = config._get_openai_compatible_provider_info("https://custom.api.com", "test-key")
        assert api_base == "https://custom.api.com"
        assert api_key == "test-key"

    def test_parameter_mapping(self):
        """Test parameter mapping works"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Test parameter mapping
        optional_params = {}
        non_default_params = {"max_completion_tokens": 100, "temperature": 0.7}
        result = config.map_openai_params(non_default_params, optional_params, "gpt-4", False)

        # max_completion_tokens should be mapped to max_tokens
        assert "max_tokens" in result
        assert result["max_tokens"] == 100
        assert "max_completion_tokens" not in result

        # temperature should be passed through
        assert result["temperature"] == 0.7

    def test_supported_params(self):
        """Test that config returns supported params"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Get supported params
        supported = config.get_supported_openai_params("gpt-4")

        # Should have standard OpenAI params
        assert isinstance(supported, list)
        assert len(supported) > 0

    def test_tool_params_excluded_when_function_calling_not_supported(self):
        """Test that tool-related params are excluded for models that don't support
        function calling. Regression test for https://github.com/BerriAI/litellm/issues/21125
        """
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Mock supports_function_calling to return False
        with patch("litellm.utils.supports_function_calling", return_value=False):
            supported = config.get_supported_openai_params("some-model-without-fc")

        tool_params = [
            "tools",
            "tool_choice",
            "function_call",
            "functions",
            "parallel_tool_calls",
        ]
        for param in tool_params:
            assert param not in supported, (
                f"'{param}' should not be in supported params when function calling is not supported"
            )

        # Non-tool params should still be present
        assert "temperature" in supported
        assert "max_tokens" in supported
        assert "stop" in supported

    def test_tool_params_included_when_function_calling_supported(self):
        """Test that tool-related params are included for models that support function calling."""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Mock supports_function_calling to return True
        with patch("litellm.utils.supports_function_calling", return_value=True):
            supported = config.get_supported_openai_params("some-model-with-fc")

        assert "tools" in supported
        assert "tool_choice" in supported

    def test_provider_resolution(self):
        """Test that provider resolution finds JSON providers"""
        from litellm.litellm_core_utils.get_llm_provider_logic import (
            get_llm_provider,
        )

        model, provider, api_key, api_base = get_llm_provider(
            model="publicai/gpt-4",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "gpt-4"
        assert provider == "publicai"
        assert api_base == "https://api.publicai.co/v1"

    def test_provider_config_manager(self):
        """Test that ProviderConfigManager returns JSON-based configs"""
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(model="gpt-4", provider=LlmProviders.PUBLICAI)

        assert config is not None
        assert config.custom_llm_provider == "publicai"


class TestPinstripes:
    """Tests for Pinstripes JSON-configured provider"""

    def test_pinstripes_json_config_exists(self):
        """Test that pinstripes is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("pinstripes")

        pinstripes = JSONProviderRegistry.get("pinstripes")
        assert pinstripes is not None
        assert pinstripes.base_url == "https://pinstripes.io/v1"
        assert pinstripes.api_key_env == "PINSTRIPES_API_KEY"
        assert pinstripes.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_pinstripes_provider_resolution(self):
        """Test that provider resolution finds pinstripes and returns the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="pinstripes/ps/glm-4.5-air",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "ps/glm-4.5-air"
        assert provider == "pinstripes"
        assert api_base == "https://pinstripes.io/v1"

    def test_pinstripes_dynamic_config(self):
        """Test dynamic config class creation for pinstripes"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("pinstripes")
        config_class = create_config_class(provider)
        config = config_class()

        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://pinstripes.io/v1"

        api_base, api_key = config._get_openai_compatible_provider_info("https://custom.pinstripes.io/v1", "test-key")
        assert api_base == "https://custom.pinstripes.io/v1"
        assert api_key == "test-key"

    def test_pinstripes_parameter_mapping(self):
        """Test that max_completion_tokens is mapped to max_tokens for pinstripes"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("pinstripes")
        config_class = create_config_class(provider)
        config = config_class()

        optional_params = {}
        non_default_params = {"max_completion_tokens": 100, "temperature": 0.7}
        result = config.map_openai_params(non_default_params, optional_params, "ps/glm-4.5-air", False)

        assert "max_tokens" in result
        assert result["max_tokens"] == 100
        assert "max_completion_tokens" not in result
        assert result["temperature"] == 0.7


class TestDarkbloom:
    def test_darkbloom_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        darkbloom = JSONProviderRegistry.get("darkbloom")
        assert darkbloom is not None
        assert darkbloom.base_url == "https://api.darkbloom.dev/v1"
        assert darkbloom.api_key_env == "DARKBLOOM_API_KEY"
        assert darkbloom.api_base_env == "DARKBLOOM_API_BASE"
        assert darkbloom.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_darkbloom_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="darkbloom/gemma-4-26b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "gemma-4-26b"
        assert provider == "darkbloom"
        assert api_key is None
        assert api_base == "https://api.darkbloom.dev/v1"

    def test_darkbloom_dynamic_config(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("darkbloom")
        config_class = create_config_class(provider)
        config = config_class()

        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.darkbloom.dev/v1"

        api_base, api_key = config._get_openai_compatible_provider_info("https://custom.darkbloom.dev/v1", "test-key")
        assert api_base == "https://custom.darkbloom.dev/v1"
        assert api_key == "test-key"

    def test_darkbloom_complete_url_appends_endpoint(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("darkbloom")
        config_class = create_config_class(provider)
        config = config_class()

        url = config.get_complete_url(
            api_base="https://api.darkbloom.dev/v1",
            api_key="test-key",
            model="darkbloom/gemma-4-26b",
            optional_params={},
            litellm_params={},
            stream=True,
        )

        assert url == "https://api.darkbloom.dev/v1/chat/completions"

    def test_darkbloom_provider_config_manager(self):
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(model="gemma-4-26b", provider=LlmProviders.DARKBLOOM)

        assert config is not None
        assert config.custom_llm_provider == "darkbloom"


def test_api_route_provider_routing():
    model, provider, api_key, api_base = get_llm_provider(
        model="api_route/model-name",
        custom_llm_provider=None,
        api_key="test-key",
        api_base="https://example.com/v1",
    )

    assert model == "model-name"
    assert provider == "api_route"
    assert api_key == "test-key"
    assert api_base == "https://example.com/v1"


def test_api_route_credentials_from_environment(monkeypatch):
    monkeypatch.setenv("API_ROUTE_API_KEY", "env-key")
    monkeypatch.setenv("API_ROUTE_BASE_URL", "https://env.example.com/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="api_route/model-name",
        custom_llm_provider=None,
        api_key=None,
        api_base=None,
    )

    assert provider == "api_route"
    assert api_key == "env-key"
    assert api_base == "https://env.example.com/v1"


def test_api_route_uses_default_base_url(monkeypatch):
    monkeypatch.delenv("API_ROUTE_BASE_URL", raising=False)

    _, _, api_key, api_base = get_llm_provider(
        model="api_route/model-name",
        custom_llm_provider=None,
        api_key="test-key",
        api_base=None,
    )

    assert api_base == "https://global.api-route.com/v1"
    assert api_key == "test-key"


def test_api_route_explicit_api_base_overrides_default(monkeypatch):
    monkeypatch.delenv("API_ROUTE_BASE_URL", raising=False)

    _, _, _, api_base = get_llm_provider(
        model="api_route/model-name",
        custom_llm_provider=None,
        api_key="test-key",
        api_base="https://custom.example.com/v1",
    )

    assert api_base == "https://custom.example.com/v1"


@pytest.mark.parametrize("api_base", [None, "https://env.example.com/v1"])
def test_api_route_chat_completion_request(monkeypatch, respx_mock, api_base):
    if api_base is None:
        monkeypatch.delenv("API_ROUTE_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("API_ROUTE_BASE_URL", api_base)
    request_url = f"{api_base or 'https://global.api-route.com/v1'}/chat/completions"
    route = respx_mock.post(request_url).respond(
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "model-name",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    response = litellm.completion(
        model="api_route/deepseek-v4.1-flash",
        messages=[{"role": "user", "content": "Hello!"}],
        api_key="test-key",
        tools=tools,
    )

    request = route.calls[0].request
    body = json.loads(request.content)
    assert str(request.url) == request_url
    assert request.headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Hello!"}]
    assert body["tools"] == tools
    assert response.choices[0].message.content == "Hello!"


def test_api_route_streaming_request(respx_mock):
    route = respx_mock.post("https://stream.example.com/v1/chat/completions").respond(
        headers={"content-type": "text/event-stream"},
        text='data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\ndata: [DONE]\n\n',
    )

    chunks = list(
        litellm.completion(
            model="api_route/model-name",
            messages=[{"role": "user", "content": "Hello!"}],
            api_key="stream-key",
            api_base="https://stream.example.com/v1",
            stream=True,
        )
    )

    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "model-name"
    assert body["stream"] is True
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"


def test_api_route_pricing_supports_spend_accounting():
    model = "api_route/deepseek-v4.1-flash"
    pricing = litellm.model_cost[model]

    input_cost, output_cost = litellm.cost_per_token(
        model=model,
        prompt_tokens=100,
        completion_tokens=50,
        custom_llm_provider="api_route",
    )

    assert input_cost == pytest.approx(100 * pricing["input_cost_per_token"])
    assert output_cost == pytest.approx(50 * pricing["output_cost_per_token"])
    assert input_cost > 0
    assert output_cost > 0
