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