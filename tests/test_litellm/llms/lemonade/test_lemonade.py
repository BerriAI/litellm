import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

import litellm
from litellm.llms.lemonade.chat.transformation import LemonadeChatConfig
from litellm.types.utils import ModelResponse


def test_lemonade_config_initialization():
    """Test that LemonadeChatConfig can be initialized with various parameters"""
    config = LemonadeChatConfig(
        temperature=0.7, max_tokens=100, top_p=0.9, top_k=50, repeat_penalty=1.1
    )

    assert config.custom_llm_provider == "lemonade"
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.top_p == 0.9
    assert config.top_k == 50
    assert config.repeat_penalty == 1.1


def test_get_openai_compatible_provider_info(monkeypatch):
    """Test the provider info method returns correct API base and key"""
    monkeypatch.delenv("LEMONADE_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "lemonade_key", None)
    monkeypatch.setattr(litellm, "api_key", None)
    config = LemonadeChatConfig()

    api_base, key = config._get_openai_compatible_provider_info(
        api_base=None, api_key=None
    )

    assert api_base == "http://localhost:8000/api/v1"
    assert key == "lemonade"


def test_get_openai_compatible_provider_info_with_custom_base(monkeypatch):
    """Test the provider info method with custom API base"""
    monkeypatch.delenv("LEMONADE_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "lemonade_key", None)
    monkeypatch.setattr(litellm, "api_key", None)
    config = LemonadeChatConfig()

    custom_api_base = "https://custom.lemonade.ai/v1"
    api_base, key = config._get_openai_compatible_provider_info(
        api_base=custom_api_base, api_key=None
    )

    assert api_base == custom_api_base
    assert key == "lemonade"


def test_get_openai_compatible_provider_info_with_api_key_env(monkeypatch):
    """Test the provider info method reads Lemonade's API key from the environment."""
    monkeypatch.setenv("LEMONADE_API_KEY", "test-key")
    monkeypatch.setattr(litellm, "lemonade_key", None)
    monkeypatch.setattr(litellm, "api_key", None)
    config = LemonadeChatConfig()

    api_base, key = config._get_openai_compatible_provider_info(
        api_base=None, api_key=None
    )

    assert api_base == "http://localhost:8000/api/v1"
    assert key == "test-key"


def test_get_openai_compatible_provider_info_skips_env_key_for_custom_base(
    monkeypatch,
):
    """Test that caller-supplied bases do not receive server-side Lemonade keys."""
    monkeypatch.setenv("LEMONADE_API_KEY", "server-side-lemonade-key")
    monkeypatch.setattr(litellm, "lemonade_key", "configured-lemonade-key")
    monkeypatch.setattr(litellm, "api_key", None)
    config = LemonadeChatConfig()

    api_base, key = config._get_openai_compatible_provider_info(
        api_base="https://attacker.example/v1", api_key=None
    )

    assert api_base == "https://attacker.example/v1"
    assert key == "lemonade"
    assert config._get_auth_headers(key) == {}


def test_get_openai_compatible_provider_info_uses_explicit_key_for_custom_base(
    monkeypatch,
):
    """Test that explicitly supplied Lemonade keys are sent to supplied bases."""
    monkeypatch.setenv("LEMONADE_API_KEY", "server-side-lemonade-key")
    monkeypatch.setattr(litellm, "lemonade_key", "configured-lemonade-key")
    monkeypatch.setattr(litellm, "api_key", None)
    config = LemonadeChatConfig()

    api_base, key = config._get_openai_compatible_provider_info(
        api_base="https://lemonade.example/v1", api_key="explicit-lemonade-key"
    )

    assert api_base == "https://lemonade.example/v1"
    assert key == "explicit-lemonade-key"
    assert config._get_auth_headers(key) == {
        "Authorization": "Bearer explicit-lemonade-key"
    }


def test_get_openai_compatible_provider_info_empty_key_does_not_leak_to_custom_base(
    monkeypatch,
):
    """An empty explicit key must not fall back to server-side Lemonade creds for a custom base."""
    monkeypatch.setenv("LEMONADE_API_KEY", "server-side-lemonade-key")
    monkeypatch.setattr(litellm, "lemonade_key", "configured-lemonade-key")
    monkeypatch.setattr(litellm, "api_key", None)
    config = LemonadeChatConfig()

    api_base, key = config._get_openai_compatible_provider_info(
        api_base="https://attacker.example/v1", api_key=""
    )

    assert api_base == "https://attacker.example/v1"
    assert key == "lemonade"
    assert config._get_auth_headers(key) == {}


def test_get_openai_compatible_provider_info_ignores_global_api_key(monkeypatch):
    """Test that Lemonade discovery does not send unrelated global API keys."""
    monkeypatch.delenv("LEMONADE_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "lemonade_key", None)
    monkeypatch.setattr(litellm, "api_key", "global-openai-key")
    config = LemonadeChatConfig()

    api_base, key = config._get_openai_compatible_provider_info(
        api_base="http://lemonade.test/v1", api_key=None
    )

    assert api_base == "http://lemonade.test/v1"
    assert key == "lemonade"
    assert config._get_auth_headers(key) == {}


def test_get_models_does_not_leak_lemonade_key_to_custom_base(monkeypatch):
    """Test Lemonade discovery does not send server-side keys to supplied bases."""
    monkeypatch.setenv("LEMONADE_API_KEY", "server-side-lemonade-key")
    monkeypatch.setattr(litellm, "lemonade_key", "configured-lemonade-key")
    monkeypatch.setattr(litellm, "api_key", "global-provider-key")
    config = LemonadeChatConfig()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": []}

    with patch.object(
        litellm.module_level_client, "get", return_value=response
    ) as mock_get:
        models = config.get_models(api_base="https://attacker.example/v1")

    assert models == []
    assert mock_get.call_args.kwargs["headers"] == {}


def test_get_model_info_uses_loaded_context_size():
    """Test that Lemonade model info prefers the effective loaded ctx_size."""
    config = LemonadeChatConfig()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "Qwen3.6-35B-A3B-GGUF",
        "recipe_options": {"ctx_size": 65536},
        "max_context_window": 262144,
    }

    with patch.object(
        litellm.module_level_client, "get", return_value=response
    ) as mock_get:
        model_info = config.get_model_info(
            model="lemonade/Qwen3.6-35B-A3B-GGUF",
            api_base="http://lemonade.test/v1",
        )

    assert model_info["key"] == "lemonade/Qwen3.6-35B-A3B-GGUF"
    assert model_info["litellm_provider"] == "lemonade"
    assert model_info["max_input_tokens"] == 65536
    assert model_info["provider_specific_entry"] == {
        "recipe_options": {"ctx_size": 65536},
        "max_context_window": 262144,
    }
    assert "supports_function_calling" not in model_info
    assert "supports_response_schema" not in model_info
    assert "supports_tool_choice" not in model_info
    assert mock_get.call_args.kwargs["headers"] == {}


def test_get_model_info_falls_back_when_server_unavailable():
    """Test that Lemonade metadata lookup failures return safe defaults."""
    config = LemonadeChatConfig()

    with patch.object(
        litellm.module_level_client, "get", side_effect=Exception("boom")
    ):
        model_info = config.get_model_info(
            model="lemonade/Qwen3.6-35B-A3B-GGUF",
            api_base="http://lemonade.test/v1",
        )

    assert model_info["key"] == "lemonade/Qwen3.6-35B-A3B-GGUF"
    assert model_info["litellm_provider"] == "lemonade"
    assert model_info["mode"] == "chat"
    assert model_info["input_cost_per_token"] == 0.0
    assert model_info["output_cost_per_token"] == 0.0
    assert model_info["max_tokens"] is None
    assert model_info["max_input_tokens"] is None
    assert model_info["max_output_tokens"] is None
    assert "supports_function_calling" not in model_info
    assert "supports_response_schema" not in model_info
    assert "supports_tool_choice" not in model_info


def test_get_model_info_reads_context_from_provider_specific_entry():
    """Test that Lemonade model info uses provider-specific runtime metadata."""
    config = LemonadeChatConfig()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "Qwen3.6-35B-A3B-GGUF",
        "provider_specific_entry": {
            "recipe_options": {"ctx_size": "32768"},
            "max_context_window": 262144,
        },
    }

    with patch.object(litellm.module_level_client, "get", return_value=response):
        model_info = config.get_model_info(
            model="lemonade/Qwen3.6-35B-A3B-GGUF",
            api_base="http://lemonade.test/v1",
        )

    assert model_info["max_input_tokens"] == 32768
    assert model_info["provider_specific_entry"] == {
        "recipe_options": {"ctx_size": "32768"},
        "max_context_window": 262144,
    }


def test_get_model_info_sends_lemonade_api_key_for_configured_base(monkeypatch):
    """Test that Lemonade model info uses auth for configured servers."""
    monkeypatch.setenv("LEMONADE_API_KEY", "test-key")
    monkeypatch.setenv("LEMONADE_API_BASE", "http://lemonade.test/v1")
    monkeypatch.setattr(litellm, "lemonade_key", None)
    monkeypatch.setattr(litellm, "api_key", None)
    config = LemonadeChatConfig()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "Qwen3.6-35B-A3B-GGUF",
        "recipe_options": {"ctx_size": 65536},
    }

    with patch.object(
        litellm.module_level_client, "get", return_value=response
    ) as mock_get:
        config.get_model_info(
            model="lemonade/Qwen3.6-35B-A3B-GGUF",
        )

    assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer test-key"}


def test_get_model_info_sends_explicit_lemonade_api_key_for_custom_base(monkeypatch):
    """Test that Lemonade model info sends explicitly supplied auth to supplied bases."""
    monkeypatch.setenv("LEMONADE_API_KEY", "server-side-key")
    monkeypatch.setattr(litellm, "lemonade_key", None)
    monkeypatch.setattr(litellm, "api_key", None)
    config = LemonadeChatConfig()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "Qwen3.6-35B-A3B-GGUF",
        "recipe_options": {"ctx_size": 65536},
    }

    with patch.object(
        litellm.module_level_client, "get", return_value=response
    ) as mock_get:
        config.get_model_info(
            model="lemonade/Qwen3.6-35B-A3B-GGUF",
            api_base="http://lemonade.test/v1",
            api_key="explicit-test-key",
        )

    assert mock_get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer explicit-test-key"
    }


def test_litellm_get_model_info_does_not_leak_lemonade_key_to_custom_base(
    monkeypatch,
):
    """Test top-level model info does not send server-side keys to supplied bases."""
    monkeypatch.setenv("LEMONADE_API_KEY", "server-side-lemonade-key")
    monkeypatch.setattr(litellm, "lemonade_key", "configured-lemonade-key")
    monkeypatch.setattr(litellm, "api_key", "global-provider-key")
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "Qwen3.6-35B-A3B-GGUF",
        "max_input_tokens": 65536,
        "max_context_window": 262144,
    }

    litellm.get_model_info.cache_clear()
    with patch.object(
        litellm.module_level_client, "get", return_value=response
    ) as mock_get:
        try:
            model_info = litellm.get_model_info(
                model="lemonade/Qwen3.6-35B-A3B-GGUF",
                api_base="https://attacker.example/v1",
            )
        finally:
            litellm.get_model_info.cache_clear()

    assert model_info["max_input_tokens"] == 65536
    assert mock_get.call_args.kwargs["headers"] == {}


def test_litellm_get_model_info_forwards_explicit_lemonade_key_to_custom_base(
    monkeypatch,
):
    """Top-level model info must forward an explicit api_key to the supplied base."""
    monkeypatch.setenv("LEMONADE_API_KEY", "server-side-lemonade-key")
    monkeypatch.setattr(litellm, "lemonade_key", "configured-lemonade-key")
    monkeypatch.setattr(litellm, "api_key", "global-provider-key")
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "Qwen3.6-35B-A3B-GGUF",
        "max_input_tokens": 65536,
    }

    litellm.get_model_info.cache_clear()
    with patch.object(
        litellm.module_level_client, "get", return_value=response
    ) as mock_get:
        try:
            model_info = litellm.get_model_info(
                model="lemonade/Qwen3.6-35B-A3B-GGUF",
                api_base="https://lemonade.example/v1",
                api_key="explicit-lemonade-key",
            )
        finally:
            litellm.get_model_info.cache_clear()

    assert model_info["max_input_tokens"] == 65536
    assert mock_get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer explicit-lemonade-key"
    }


def test_litellm_get_model_info_uses_lemonade_api_base():
    """Test that LiteLLM model info is wired to Lemonade's model metadata API."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "Qwen3.6-35B-A3B-GGUF",
        "max_input_tokens": 65536,
        "max_context_window": 262144,
    }

    litellm.get_model_info.cache_clear()
    with patch.object(litellm.module_level_client, "get", return_value=response):
        try:
            model_info = litellm.get_model_info(
                model="lemonade/Qwen3.6-35B-A3B-GGUF",
                api_base="http://lemonade.test/v1",
            )
        finally:
            litellm.get_model_info.cache_clear()

    assert model_info["max_input_tokens"] == 65536
    assert response.raise_for_status.called
    assert response.json.called


def test_transform_response():
    """Test the response transformation adds lemonade prefix to model name"""
    config = LemonadeChatConfig()

    # Mock raw response
    raw_response = MagicMock()
    raw_response.status_code = 200
    raw_response.headers = {}

    # Create a model response
    model_response = ModelResponse()

    # Mock the parent class transform_response method
    with patch.object(
        config.__class__.__bases__[0], "transform_response"
    ) as mock_parent:
        mock_parent.return_value = model_response

        result = config.transform_response(
            model="test-model",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key="test-key",
            json_mode=False,
        )

        # Check that the model name is prefixed with "lemonade/"
        assert hasattr(result, "model")
        assert result.model == "lemonade/test-model"


def test_config_get_config():
    """Test that get_config method returns the configuration"""
    config_dict = LemonadeChatConfig.get_config()
    assert isinstance(config_dict, dict)


def test_response_format_support():
    """Test that response_format parameter is supported"""
    response_format = {"type": "json_object"}

    config = LemonadeChatConfig(response_format=response_format)
    assert config.response_format == response_format


def test_tools_support():
    """Test that tools parameter is supported"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
            },
        }
    ]

    config = LemonadeChatConfig(tools=tools)
    assert config.tools == tools


def test_functions_support():
    """Test that functions parameter is supported"""
    functions = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    config = LemonadeChatConfig(functions=functions)
    assert config.functions == functions


def test_stop_parameter_support():
    """Test that stop parameter supports both string and list"""
    # Test with string
    config1 = LemonadeChatConfig(stop="STOP")
    assert config1.stop == "STOP"

    # Test with list
    config2 = LemonadeChatConfig(stop=["STOP", "END"])
    assert config2.stop == ["STOP", "END"]


def test_logit_bias_support():
    """Test that logit_bias parameter is supported"""
    logit_bias = {"50256": -100}

    config = LemonadeChatConfig(logit_bias=logit_bias)
    assert config.logit_bias == logit_bias


def test_presence_penalty_support():
    """Test that presence_penalty parameter is supported"""
    config = LemonadeChatConfig(presence_penalty=0.5)
    assert config.presence_penalty == 0.5


def test_n_parameter_support():
    """Test that n parameter (number of completions) is supported"""
    config = LemonadeChatConfig(n=3)
    assert config.n == 3


def test_max_completion_tokens_support():
    """Test that max_completion_tokens parameter is supported"""
    config = LemonadeChatConfig(max_completion_tokens=150)
    assert config.max_completion_tokens == 150
