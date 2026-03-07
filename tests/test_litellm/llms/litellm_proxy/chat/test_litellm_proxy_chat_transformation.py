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


@pytest.mark.parametrize(
    "input_api_key, env_api_key, expected_api_key",
    [
        ("user-provided-key", "secret-key", "user-provided-key"),
        (None, "secret-key", "secret-key"),
        (None, None, "fake-api-key"),
        ("", "secret-key", "secret-key"),
        ("", None, "fake-api-key"),
    ],
)
def test_get_openai_compatible_provider_info_api_key(
    input_api_key, env_api_key, expected_api_key
):
    config = LiteLLMProxyChatConfig()
    env = {}
    if env_api_key is not None:
        env["LITELLM_PROXY_API_KEY"] = env_api_key

    with patch.dict("os.environ", env, clear=True):
        _, result_key = config._get_openai_compatible_provider_info(
            api_base=None, api_key=input_api_key
        )
        assert result_key == expected_api_key


@pytest.mark.parametrize(
    "input_api_key, env_api_key, expected_api_key",
    [
        ("user-provided-key", "secret-key", "user-provided-key"),
        (None, "secret-key", "secret-key"),
        (None, None, "fake-api-key"),
        ("", "secret-key", "secret-key"),
        ("", None, "fake-api-key"),
    ],
)
def test_get_api_key(input_api_key, env_api_key, expected_api_key):
    env = {}
    if env_api_key is not None:
        env["LITELLM_PROXY_API_KEY"] = env_api_key

    with patch.dict("os.environ", env, clear=True):
        result = LiteLLMProxyChatConfig.get_api_key(input_api_key)
        assert result == expected_api_key


def test_completion_with_litellm_proxy_no_api_key():
    """
    E2E mock test: USE_LITELLM_PROXY=true with no LITELLM_PROXY_API_KEY
    should use "fake-api-key" as fallback.
    """
    with patch(
        "litellm.main.openai_chat_completions.completion"
    ) as mock_completion_func:
        mock_completion_func.return_value = {}

        env = {
            "USE_LITELLM_PROXY": "true",
            "LITELLM_PROXY_API_BASE": "http://localhost:4000",
        }
        with patch.dict("os.environ", env, clear=True):
            _ = litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

            mock_completion_func.assert_called_once()
            _, call_kwargs = mock_completion_func.call_args
            assert call_kwargs.get("api_key") == "fake-api-key"
            assert call_kwargs.get("custom_llm_provider") == "litellm_proxy"


def test_completion_with_litellm_proxy_does_not_use_openai_key():
    """
    OPENAI_API_KEY should NOT be sent to litellm_proxy.
    Even when OPENAI_API_KEY is in the environment, the proxy should use
    "fake-api-key" (truthy value stops the or-chain in main.py:2366-2371).
    """
    with patch(
        "litellm.main.openai_chat_completions.completion"
    ) as mock_completion_func:
        mock_completion_func.return_value = {}

        env = {
            "USE_LITELLM_PROXY": "true",
            "LITELLM_PROXY_API_BASE": "http://localhost:4000",
            "OPENAI_API_KEY": "sk-real-openai-key",
        }
        with patch.dict("os.environ", env, clear=True):
            _ = litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

            _, call_kwargs = mock_completion_func.call_args
            assert call_kwargs.get("api_key") == "fake-api-key"
            assert call_kwargs.get("api_key") != "sk-real-openai-key"
