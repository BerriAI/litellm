"""
Unit tests for the Obscurify.ai OpenAI-like provider.
"""

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

OBSCURIFY_BASE_URL = "https://obscurify.ai/v1"


def _get_config():
    provider = JSONProviderRegistry.get("obscurify")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_obscurify_provider_registered():
    provider = JSONProviderRegistry.get("obscurify")
    assert provider is not None
    assert provider.base_url == OBSCURIFY_BASE_URL
    assert provider.api_key_env == "OBSCURIFY_API_KEY"
    assert provider.api_base_env == "OBSCURIFY_API_BASE"
    assert "obscurify" in litellm.provider_list


def test_obscurify_keeps_tools_and_reasoning_effort():
    config = _get_config()
    supported = config.get_supported_openai_params(model="z-ai/glm-5.3")
    assert "tools" in supported
    assert "tool_choice" in supported
    assert "reasoning_effort" in supported


def test_obscurify_resolves_model_with_slash():
    model, provider, api_key, api_base = get_llm_provider(
        model="obscurify/deepseek/deepseek-v4-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )
    assert model == "deepseek/deepseek-v4-flash"
    assert provider == "obscurify"
    assert api_base == OBSCURIFY_BASE_URL


def test_obscurify_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("OBSCURIFY_API_KEY", "sk-obs1_test")
    api_base, api_key = config.get_openai_compatible_provider_info(None, None)
    assert api_base == OBSCURIFY_BASE_URL
    assert api_key == "sk-obs1_test"


def test_obscurify_maps_max_completion_tokens():
    config = _get_config()
    params = config.map_openai_params(
        non_default_params={"max_completion_tokens": 256},
        optional_params={},
        model="obscurify/deepseek/deepseek-v4-flash",
        drop_params=False,
    )
    assert params.get("max_tokens") == 256
    assert "max_completion_tokens" not in params


def test_obscurify_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=OBSCURIFY_BASE_URL,
        api_key="sk-obs1_test",
        model="obscurify/deepseek/deepseek-v4-flash",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{OBSCURIFY_BASE_URL}/chat/completions"


def test_obscurify_responses_api_config():
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.OBSCURIFY,
        model="deepseek/deepseek-v4-flash",
    )
    assert config is not None


OPENAI_KEY = "sk-openai-must-not-reach-obscurify"

CHAT_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "deepseek/deepseek-v4-flash",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture
def openai_key_only(monkeypatch: pytest.MonkeyPatch):
    """OPENAI_API_KEY is configured, Obscurify's own key is not."""
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.delenv("OBSCURIFY_API_KEY", raising=False)
    monkeypatch.delenv("OBSCURIFY_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", None)


def test_obscurify_chat_without_own_key_never_sends_openai_key(openai_key_only):
    with respx.mock(assert_all_called=False) as router:
        route = router.route(host="obscurify.ai").mock(return_value=httpx.Response(200, json=CHAT_RESPONSE))
        with pytest.raises(litellm.AuthenticationError, match="OBSCURIFY_API_KEY"):
            litellm.completion(
                model="obscurify/deepseek/deepseek-v4-flash",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert not route.called


def test_obscurify_responses_without_own_key_never_sends_openai_key(openai_key_only):
    with respx.mock(assert_all_called=False) as router:
        route = router.route(host="obscurify.ai").mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(litellm.AuthenticationError, match="OBSCURIFY_API_KEY"):
            litellm.responses(model="obscurify/deepseek/deepseek-v4-flash", input="hi")
        assert not route.called


def test_obscurify_chat_sends_obscurify_key_when_both_are_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OBSCURIFY_API_KEY", "sk-obs1_test")
    monkeypatch.delenv("OBSCURIFY_API_BASE", raising=False)
    with respx.mock() as router:
        route = router.post(f"{OBSCURIFY_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=CHAT_RESPONSE)
        )
        litellm.completion(
            model="obscurify/deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert route.call_count == 1
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-obs1_test"
