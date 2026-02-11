from typing import Optional
from unittest.mock import patch

import pytest

import litellm
from litellm.llms.litellm_proxy.chat.transformation import LiteLLMProxyChatConfig


def test_litellm_proxy_chat_transformation():
    """
    Assert messages are not transformed when calling litellm proxy
    """
    config = LiteLLMProxyChatConfig()
    file_content = [
        {"type": "text", "text": "What is this document about?"},
        {
            "type": "file",
            "file": {
                "file_id": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "format": "application/pdf",
            },
        },
    ]
    messages = [{"role": "user", "content": file_content}]
    assert config.transform_request(
        model="model",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    ) == {"model": "model", "messages": messages}


def test_litellm_gateway_from_sdk_with_user_param():
    from litellm.llms.litellm_proxy.chat.transformation import LiteLLMProxyChatConfig

    supported_params = LiteLLMProxyChatConfig().get_supported_openai_params(
        "openai/gpt-4o"
    )
    print(f"supported_params: {supported_params}")
    assert "user" in supported_params


def test_litellm_proxy_api_key_fallback_when_unset():
    """
    Verify that _get_openai_compatible_provider_info returns a non-None
    api_key when LITELLM_PROXY_API_KEY is not set.

    Regression test for https://github.com/BerriAI/litellm/issues/20925.
    Without this fallback, the OpenAI SDK raises AuthenticationError because
    it requires a non-None api_key value.
    """
    import os

    config = LiteLLMProxyChatConfig()

    with patch.dict(os.environ, {}, clear=False):
        # Ensure env var is NOT set
        os.environ.pop("LITELLM_PROXY_API_KEY", None)

        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base="http://localhost:4000",
            api_key=None,
        )

        assert api_key is not None, "api_key should never be None"
        assert isinstance(api_key, str)
        assert len(api_key) > 0


def test_litellm_proxy_api_key_uses_provided_key():
    """
    Verify that an explicitly provided api_key takes precedence.
    """
    config = LiteLLMProxyChatConfig()
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base="http://localhost:4000",
        api_key="sk-my-real-key",
    )
    assert api_key == "sk-my-real-key"


def test_litellm_proxy_get_api_key_fallback():
    """
    Verify that the static get_api_key method returns a non-None value
    when no key is configured.
    """
    import os

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LITELLM_PROXY_API_KEY", None)
        key = LiteLLMProxyChatConfig.get_api_key(api_key=None)
        assert key is not None
        assert isinstance(key, str)
