import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.digitalocean.chat.transformation import DigitalOceanConfig

DO_ENDPOINT_PATH = "/api/v1/chat/completions"
DO_BASE_URL = "https://api.digitalocean.com"

@pytest.fixture
def config():
    return DigitalOceanConfig()

def test_validate_environment_sets_headers(monkeypatch, config):
    monkeypatch.setenv("DIGITALOCEAN_API_KEY", "test-key")
    headers = {}
    result = config.validate_environment(
        headers=headers,
        model="digitalocean/test-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    assert result["Authorization"] == "Bearer test-key"
    assert result["Content-Type"] == "application/json"

def test_get_complete_url(config):
    url = config.get_complete_url(
        api_base=DO_BASE_URL,
        api_key="test-key",
        model="digitalocean/test-model",
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
    out = config._transform_messages(messages, model="digitalocean/test-model")
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] == "Hello!"
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "Hi!"

def test_get_openai_compatible_provider_info_env(monkeypatch, config):
    monkeypatch.setenv("DIGITALOCEAN_AGENT_ENDPOINT", DO_BASE_URL)
    monkeypatch.setenv("DIGITALOCEAN_API_KEY", "env-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == DO_BASE_URL
    assert api_key == "env-key"