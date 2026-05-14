"""
Unit tests for the Avian OpenAI-like provider.
"""

import os
import sys

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

AVIAN_BASE_URL = "https://api.avian.io/v1"


def _get_config():
    provider = JSONProviderRegistry.get("avian")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_avian_provider_registered():
    provider = JSONProviderRegistry.get("avian")
    assert provider is not None
    assert provider.base_url == AVIAN_BASE_URL
    assert provider.api_key_env == "AVIAN_API_KEY"


def test_avian_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("AVIAN_API_KEY", "test-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == AVIAN_BASE_URL
    assert api_key == "test-key"


def test_avian_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=AVIAN_BASE_URL,
        api_key="test-key",
        model="avian/deepseek/deepseek-v3.2",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{AVIAN_BASE_URL}/chat/completions"


def test_avian_in_provider_list():
    from litellm.types.utils import LlmProviders

    assert LlmProviders.AVIAN.value == "avian"


def test_avian_retry_on_422_with_drop_params():
    config = _get_config()
    mock_request = httpx.Request(method="POST", url=AVIAN_BASE_URL)
    mock_response = httpx.Response(
        status_code=422,
        text='{"detail": "max_completion_tokens: Extra inputs are not permitted"}',
        request=mock_request,
    )
    error = httpx.HTTPStatusError(
        message="422", request=mock_request, response=mock_response
    )
    assert config.should_retry_llm_api_inside_llm_translation_on_http_error(
        e=error, litellm_params={"drop_params": True}
    )


def test_avian_no_retry_without_drop_params():
    config = _get_config()
    mock_request = httpx.Request(method="POST", url=AVIAN_BASE_URL)
    mock_response = httpx.Response(
        status_code=422,
        text='{"detail": "Extra inputs are not permitted"}',
        request=mock_request,
    )
    error = httpx.HTTPStatusError(
        message="422", request=mock_request, response=mock_response
    )
    assert not config.should_retry_llm_api_inside_llm_translation_on_http_error(
        e=error, litellm_params={}
    )


def test_avian_max_retry_count():
    config = _get_config()
    assert config.max_retry_on_unprocessable_entity_error == 2


@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.skipif(
    not os.environ.get("AVIAN_API_KEY"),
    reason="AVIAN_API_KEY not set",
)
def test_avian_chat_completion():
    """Live API test — requires AVIAN_API_KEY."""
    import litellm

    response = litellm.completion(
        model="avian/deepseek/deepseek-v3.2",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )
    assert response.choices[0].message.content is not None
