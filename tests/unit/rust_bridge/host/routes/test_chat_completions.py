from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.rust_bridge.host.routes.chat_completions import body_params, connection_defaults, response
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


def test_body_params_keep_request_params_and_leave_out_transport_and_unset_ones() -> None:
    body: Final = body_params(
        MappingProxyType(
            {
                "model": "anthropic/claude-sonnet-5-5",
                "temperature": 0.2,
                "max_tokens": None,
                "api_key": "sk-secret",
                "api_base": "https://example.invalid",
                "timeout": 30,
            }
        )
    )

    assert body == {"temperature": 0.2}
