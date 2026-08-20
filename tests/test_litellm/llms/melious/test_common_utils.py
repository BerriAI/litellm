import pytest

from litellm.llms.melious.common_utils import (
    MELIOUS_API_BASE,
    MELIOUS_OPENAI_API_BASE,
    anthropic_messages_url,
    openai_api_base,
)


def test_defaults():
    assert MELIOUS_API_BASE == "https://api.melious.ai"
    assert MELIOUS_OPENAI_API_BASE == "https://api.melious.ai/v1"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://api.melious.ai", "https://api.melious.ai/v1"),
        ("https://api.melious.ai/", "https://api.melious.ai/v1"),
        ("https://api.melious.ai/v1", "https://api.melious.ai/v1"),
        ("https://api.melious.ai/v1/", "https://api.melious.ai/v1"),
        ("https://api.melious.ai/v1/chat/completions", "https://api.melious.ai/v1"),
        ("https://gateway.example/melious", "https://gateway.example/melious/v1"),
    ],
)
def test_openai_api_base(given, expected):
    assert openai_api_base(given) == expected


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://api.melious.ai", "https://api.melious.ai/v1/messages"),
        ("https://api.melious.ai/", "https://api.melious.ai/v1/messages"),
        ("https://api.melious.ai/v1", "https://api.melious.ai/v1/messages"),
        ("https://api.melious.ai/v1/", "https://api.melious.ai/v1/messages"),
        ("https://api.melious.ai/v1/messages", "https://api.melious.ai/v1/messages"),
        ("https://api.melious.ai/v1/messages/", "https://api.melious.ai/v1/messages"),
        ("https://api.melious.ai/v1/chat/completions", "https://api.melious.ai/v1/messages"),
        ("https://gateway.example/melious", "https://gateway.example/melious/v1/messages"),
    ],
)
def test_anthropic_messages_url(given, expected):
    assert anthropic_messages_url(given) == expected


def test_both_helpers_are_idempotent():
    assert openai_api_base(openai_api_base(MELIOUS_API_BASE)) == MELIOUS_OPENAI_API_BASE
    once = anthropic_messages_url(MELIOUS_API_BASE)
    assert anthropic_messages_url(once) == once
