import platform

import httpx
import respx

import litellm
from litellm._version import version as LITELLM_VERSION
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

SARVAM_API_BASE = "https://api.sarvam.ai/v1"
SARVAM_USER_AGENT = f"litellm/{LITELLM_VERSION} Python/{platform.python_version()}"


def _chat_response() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "sarvam-105b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_sarvam_config_declares_a_user_agent_template():
    config = JSONProviderRegistry.get("sarvam")
    assert config is not None
    assert config.base_url == SARVAM_API_BASE
    assert config.user_agent == "litellm/{version} Python/{python}"


def test_sarvam_attribution_headers_report_running_versions():
    config = JSONProviderRegistry.get("sarvam")
    assert config is not None
    assert dict(config.get_attribution_headers()) == {"User-Agent": SARVAM_USER_AGENT}


def test_generated_config_class_forwards_attribution_headers():
    """Providers that are also in LlmProviders get a generated config class, so
    the class itself must forward the JSON config's attribution headers."""
    from litellm.llms.openai_like.dynamic_config import create_config_class

    config = JSONProviderRegistry.get("sarvam")
    assert config is not None
    instance = create_config_class(config)()
    assert dict(instance.get_attribution_headers()) == {"User-Agent": SARVAM_USER_AGENT}


def test_json_provider_without_user_agent_sends_no_attribution_headers():
    config = JSONProviderRegistry.get("aiand")
    assert config is not None
    assert dict(config.get_attribution_headers()) == {}


def test_sarvam_chat_completion_sends_the_litellm_user_agent(respx_mock: respx.MockRouter):
    route = respx_mock.post(f"{SARVAM_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response())
    )

    litellm.completion(
        model="sarvam/sarvam-105b",
        api_key="test-key",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert route.called
    assert route.calls.last.request.headers["User-Agent"] == SARVAM_USER_AGENT


def test_sarvam_caller_user_agent_wins(respx_mock: respx.MockRouter):
    route = respx_mock.post(f"{SARVAM_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response())
    )

    litellm.completion(
        model="sarvam/sarvam-105b",
        api_key="test-key",
        messages=[{"role": "user", "content": "Hello"}],
        headers={"User-Agent": "caller-app/1.0"},
    )

    assert route.called
    assert route.calls.last.request.headers["User-Agent"] == "caller-app/1.0"
