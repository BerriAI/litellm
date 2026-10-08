from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.rust_bridge.chat_completions.route_host import arguments, connection_defaults, response
from litellm.types.utils import ModelResponse


def test_response_builds_the_public_model_response() -> None:
    built: Final = response(
        MappingProxyType(
            {
                "id": "chatcmpl-native",
                "object": "chat.completion",
                "created": 1,
                "model": "claude-sonnet-4-5",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "native"},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
        )
    )

    assert isinstance(built, ModelResponse)
    assert built.id == "chatcmpl-native"
    assert built.choices[0].message.content == "native"
    assert built.usage is not None
    assert built.usage.total_tokens == 5


def test_arguments_are_the_public_kwargs_view() -> None:
    kwargs: Final = MappingProxyType({"metadata": {"user_id": "u"}})

    assert arguments(kwargs) is kwargs


@pytest.mark.parametrize(
    ("provider", "global_key", "provider_key", "expected_key", "expected_base"),
    (
        ("anthropic", "global", "provider", "provider", "https://configured.invalid"),
        ("anthropic", "global", None, "global", "https://configured.invalid"),
        ("anthropic", "global", "", "global", "https://configured.invalid"),
        ("anthropic", None, None, None, "https://configured.invalid"),
        ("bedrock", "global", "provider", None, None),
    ),
)
def test_connection_defaults_preserve_provider_precedence(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    global_key: str | None,
    provider_key: str | None,
    expected_key: str | None,
    expected_base: str | None,
) -> None:
    monkeypatch.setattr(litellm, "api_key", global_key)
    monkeypatch.setattr(litellm, "anthropic_key", provider_key)
    monkeypatch.setattr(litellm, "api_base", "https://configured.invalid")
    assert connection_defaults(provider) == (expected_key, expected_base)
