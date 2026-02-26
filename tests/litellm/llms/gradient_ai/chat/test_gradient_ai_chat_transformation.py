import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.gradient_ai.chat.transformation import GradientAIConfig, GRADIENT_AI_SERVERLESS_ENDPOINT

DO_ENDPOINT_PATH = "/api/v1/chat/completions"
DO_BASE_URL = "https://api.gradient_ai.com"

@pytest.fixture
def config():
    return GradientAIConfig()

def test_validate_environment_sets_headers(monkeypatch, config):
    monkeypatch.setenv("GRADIENT_AI_API_KEY", "test-key")
    headers = {}
    result = config.validate_environment(
        headers=headers,
        model="gradient_ai/test-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    assert result["Authorization"] == "Bearer test-key"
    assert result["Content-Type"] == "application/json"

def test_get_complete_url_custom_base(config):
    url = config.get_complete_url(
        api_base=DO_BASE_URL,
        api_key="test-key",
        model="gradient_ai/test-model",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{DO_BASE_URL}{DO_ENDPOINT_PATH}"

def test_get_complete_url_default_serverless(monkeypatch, config):
    monkeypatch.delenv("GRADIENT_AI_AGENT_ENDPOINT", raising=False)
    url = config.get_complete_url(
        api_base=None,
        api_key="test-key",
        model="gradient_ai/test-model",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{GRADIENT_AI_SERVERLESS_ENDPOINT}/v1/chat/completions"

def test_get_complete_url_with_env_endpoint(monkeypatch, config):
    monkeypatch.setenv("GRADIENT_AI_AGENT_ENDPOINT", DO_BASE_URL)
    url = config.get_complete_url(
        api_base=None,
        api_key="test-key",
        model="gradient_ai/test-model",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{DO_BASE_URL}{DO_ENDPOINT_PATH}"

def test_transform_messages_handles_dicts_only(config):
    messages = [
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Hi!"},
    ]
    out = config._transform_messages(messages, model="gradient_ai/test-model")
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] == "Hello!"
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "Hi!"

def test_get_openai_compatible_provider_info_env(monkeypatch, config):
    monkeypatch.setenv("GRADIENT_AI_AGENT_ENDPOINT", DO_BASE_URL)
    monkeypatch.setenv("GRADIENT_AI_API_KEY", "env-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == DO_BASE_URL
    assert api_key == "env-key"

def test_get_openai_compatible_provider_info_default(monkeypatch, config):
    monkeypatch.delenv("GRADIENT_AI_AGENT_ENDPOINT", raising=False)
    monkeypatch.setenv("GRADIENT_AI_API_KEY", "env-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == GRADIENT_AI_SERVERLESS_ENDPOINT
    assert api_key == "env-key"



def test_map_openai_params_allows_tools_explicitly(config):
    """
    Verifies that 'tools', 'tool_choice', and 'parallel_tool_calls' are included and accepted
    """
    non_default_params = {
        "temperature": 0.7,
        "max_tokens": 512,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "unknown_param": "this should be ignored or raise depending on drop_params"
    }

    optional_params = {}

    # drop_params=True → should pass without raising and preserve tools
    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params.copy(),
        model="gradient_ai/test-model",
        drop_params=True,
    )

    assert result["temperature"] == 0.7
    assert result["max_tokens"] == 512
    assert "tools" in result
    assert len(result["tools"]) == 1
    assert result["tools"][0]["type"] == "function"
    assert result["tool_choice"] == "auto"
    assert result["parallel_tool_calls"] is True
    # unknown_param should be dropped
    assert "unknown_param" not in result


def test_map_openai_params_no_tools_if_not_provided(config):
    """Ensures tools-related keys are not added if not present in non_default_params"""
    non_default_params = {
        "temperature": 0.9,
        "top_p": 0.95,
    }

    optional_params = {}
    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="gradient_ai/test-model",
        drop_params=False,
    )

    assert "tools" not in result
    assert "tool_choice" not in result
    assert "parallel_tool_calls" not in result
    assert result["temperature"] == 0.9


def test_map_openai_params_tool_choice_as_dict(config):
    """Tests tool_choice in object format (forcing a specific tool)"""
    non_default_params = {
        "tool_choice": {
            "type": "function",
            "function": {"name": "search_database"}
        },
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_database",
                    "description": "Search the internal database",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
                }
            }
        ]
    }

    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model="gradient_ai/test-model",
        drop_params=True,
    )

    assert "tool_choice" in result
    assert result["tool_choice"]["type"] == "function"
    assert result["tool_choice"]["function"]["name"] == "search_database"
    assert "tools" in result
