import litellm
import pytest
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.anthropic.experimental_pass_through.messages.utils import (
    AnthropicMessagesRequestUtils,
)


def test_messages_drop_params_strips_speed_for_unsupported_models():
    original = litellm.drop_params
    litellm.drop_params = True
    try:
        optional_params = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            params={
                "max_tokens": 1024,
                "speed": "fast",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            model="claude-sonnet-4-6",
            drop_params=False,
        )
        config = AnthropicMessagesConfig()
        headers, _ = config.validate_anthropic_messages_environment(
            headers={},
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params=dict(optional_params),
            litellm_params={},
        )
        result = config.transform_anthropic_messages_request(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_optional_request_params=dict(optional_params),
            litellm_params={},
            headers=headers,
        )
    finally:
        litellm.drop_params = original

    assert "speed" not in optional_params
    assert "speed" not in result
    assert "fast-mode-2026-02-01" not in headers.get("anthropic-beta", "")


def test_messages_drop_params_keeps_speed_for_supporting_models():
    original = litellm.drop_params
    litellm.drop_params = True
    try:
        optional_params = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            params={"max_tokens": 1024, "speed": "fast"},
            model="claude-opus-4-6",
            drop_params=False,
        )
        config = AnthropicMessagesConfig()
        headers, _ = config.validate_anthropic_messages_environment(
            headers={},
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params=dict(optional_params),
            litellm_params={},
        )
        result = config.transform_anthropic_messages_request(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_optional_request_params=dict(optional_params),
            litellm_params={},
            headers=headers,
        )
    finally:
        litellm.drop_params = original

    assert optional_params.get("speed") == "fast"
    assert result.get("speed") == "fast"
    assert "fast-mode-2026-02-01" in headers.get("anthropic-beta", "")


def test_messages_raises_when_speed_unsupported_and_drop_params_false(monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)

    with pytest.raises(litellm.utils.UnsupportedParamsError, match="drop_params"):
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            params={"max_tokens": 1024, "speed": "fast"},
            model="claude-sonnet-4-6",
            drop_params=False,
        )


def test_messages_drops_speed_for_vertex_opus_with_drop_params(monkeypatch):
    """Regression: a vertex_ai Opus passthrough must drop ``speed`` even though the
    prefix-stripped model id maps to a fast-mode-capable direct-Anthropic entry."""
    monkeypatch.setattr(litellm, "drop_params", True)
    optional_params = (
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            params={"max_tokens": 1024, "speed": "fast"},
            model="claude-opus-4-8",
            drop_params=False,
            custom_llm_provider="vertex_ai",
        )
    )

    assert "speed" not in optional_params


def test_messages_raises_for_vertex_opus_without_drop_params(monkeypatch):
    """Regression: vertex_ai Opus passthrough raises rather than forwarding an
    unsupported ``speed`` when drop_params is unset."""
    monkeypatch.setattr(litellm, "drop_params", False)

    with pytest.raises(litellm.utils.UnsupportedParamsError, match="drop_params"):
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            params={"max_tokens": 1024, "speed": "fast"},
            model="claude-opus-4-8",
            drop_params=False,
            custom_llm_provider="vertex_ai",
        )
