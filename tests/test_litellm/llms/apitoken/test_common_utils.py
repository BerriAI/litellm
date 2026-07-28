"""
Test apiToken.sale api_base normalization
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path

from litellm.llms.apitoken.common_utils import build_messages_url


@pytest.mark.parametrize(
    "api_base, expected",
    [
        # bare host
        ("https://api.apitoken.sale", "https://api.apitoken.sale/v1/messages"),
        # trailing slash must not produce a double slash
        ("https://api.apitoken.sale/", "https://api.apitoken.sale/v1/messages"),
        # versioned base must not produce /v1/v1/messages
        ("https://api.apitoken.sale/v1", "https://api.apitoken.sale/v1/messages"),
        ("https://api.apitoken.sale/v1/", "https://api.apitoken.sale/v1/messages"),
        # already complete
        (
            "https://api.apitoken.sale/v1/messages",
            "https://api.apitoken.sale/v1/messages",
        ),
        (
            "https://api.apitoken.sale/v1/messages/",
            "https://api.apitoken.sale/v1/messages",
        ),
        # self-hosted gateway on a sub-path
        ("https://gateway.example.com/proxy", "https://gateway.example.com/proxy/v1/messages"),
    ],
)
def test_build_messages_url(api_base, expected):
    assert build_messages_url(api_base) == expected


@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.apitoken.sale",
        "https://api.apitoken.sale/",
        "https://api.apitoken.sale/v1",
        "https://api.apitoken.sale/v1/messages",
    ],
)
def test_configs_normalize_api_base(api_base):
    """Both the chat and messages configs must agree on the normalized URL."""
    from litellm.llms.apitoken.chat.transformation import ApiTokenChatConfig
    from litellm.llms.apitoken.messages.transformation import ApiTokenMessagesConfig

    expected = "https://api.apitoken.sale/v1/messages"
    for config in (ApiTokenChatConfig(), ApiTokenMessagesConfig()):
        assert (
            config.get_complete_url(
                api_base=api_base,
                api_key="sk-pool-test",
                model="claude-opus-4-8",
                optional_params={},
                litellm_params={},
            )
            == expected
        )


def test_endpoint_only_detection():
    """
    Passing only api_base (no apitoken/ prefix) must resolve to the apitoken
    provider — the detection branch is reachable via openai_compatible_endpoints.
    """
    import litellm
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    assert "api.apitoken.sale" in litellm.openai_compatible_endpoints

    _model, provider, _key, _base = get_llm_provider(
        model="claude-opus-4-8",
        api_base="https://api.apitoken.sale",
    )
    assert provider == "apitoken"
