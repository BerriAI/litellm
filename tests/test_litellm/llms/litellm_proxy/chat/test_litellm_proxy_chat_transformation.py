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


def test_translate_developer_role_hoists_a_later_developer_message_before_the_downstream_proxy_sees_it():
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi there"},
        {"role": "developer", "content": "Answer with exactly one word."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    translated = LiteLLMProxyChatConfig().translate_developer_role_to_system_role(
        messages=messages,
        custom_llm_provider="litellm_proxy",
        api_base="http://inner-proxy:4000",
    )

    assert list(translated) == [
        {"role": "system", "content": "You are terse.\n\nAnswer with exactly one word."},
        {"role": "user", "content": "Hi there"},
        {"role": "user", "content": "What is the capital of France?"},
    ]
