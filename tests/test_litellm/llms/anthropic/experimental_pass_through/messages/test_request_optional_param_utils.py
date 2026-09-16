"""
Regression tests for the /v1/messages request-parse fast paths:

- get_requested_anthropic_messages_optional_param must still filter to the
  valid AnthropicMessagesRequestOptionalParams keys and drop None values,
  while resolving the (static) type hints only once per process.
"""

import pytest

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.utils import (
    AnthropicMessagesRequestUtils,
    _anthropic_messages_optional_param_keys,
)


def test_optional_param_filtering_unchanged():
    params = {
        "temperature": 0.5,
        "top_p": None,  # None dropped
        "tools": [{"name": "x"}],
        "not_a_real_param": "drop me",  # invalid key dropped
        "stream": True,
    }
    result = (
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            params
        )
    )
    assert result == {"temperature": 0.5, "tools": [{"name": "x"}], "stream": True}
    assert "top_p" not in result
    assert "not_a_real_param" not in result


def test_valid_keys_are_memoized():
    _anthropic_messages_optional_param_keys.cache_clear()
    first = _anthropic_messages_optional_param_keys()
    for _ in range(50):
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            {"temperature": 0.1}
        )
    info = _anthropic_messages_optional_param_keys.cache_info()
    # Resolved exactly once despite many calls.
    assert info.misses == 1
    assert info.hits >= 50
    # Stable identity (frozenset) returned each call.
    assert _anthropic_messages_optional_param_keys() is first
    assert isinstance(first, frozenset)
    assert "temperature" in first and "tools" in first


def test_empty_params():
    assert (
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            {}
        )
        == {}
    )


def test_drop_params_strips_speed_for_unsupported_model():
    original = litellm.drop_params
    litellm.drop_params = True
    try:
        result = (
            AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
                params={"speed": "fast", "temperature": 0.5},
                model="claude-sonnet-4-6",
            )
        )
    finally:
        litellm.drop_params = original

    assert result == {"temperature": 0.5}
    assert "speed" not in result


def test_drop_params_keeps_speed_for_supporting_model():
    original = litellm.drop_params
    litellm.drop_params = True
    try:
        result = (
            AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
                params={"speed": "fast"},
                model="claude-opus-4-6",
            )
        )
    finally:
        litellm.drop_params = original

    assert result == {"speed": "fast"}


def test_drop_params_strips_sampling_params_for_unsupported_model(monkeypatch):
    # claude-opus-4-7 has supports_sampling_params: false in the model map; the
    # API 400s on these rather than ignoring them.
    monkeypatch.setattr(litellm, "drop_params", False)
    result = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
        params={"temperature": 0.3, "top_p": 0.9, "top_k": 40, "stream": True},
        model="claude-opus-4-7",
        drop_params=True,
    )

    assert result == {"stream": True}


def test_drop_params_strips_sampling_params_for_provider_prefixed_model(monkeypatch):
    # Vertex-routed ids must resolve the same capability flag.
    monkeypatch.setattr(litellm, "drop_params", False)
    result = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
        params={"temperature": 0.3, "top_p": 0.9, "top_k": 40},
        model="vertex_ai/claude-opus-4-7",
        drop_params=True,
    )

    assert result == {}


def test_sampling_params_kept_for_supporting_model(monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)
    result = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
        params={"temperature": 0.3, "top_p": 0.9, "top_k": 40},
        model="claude-sonnet-4-6",
        drop_params=True,
    )

    assert result == {"temperature": 0.3, "top_p": 0.9, "top_k": 40}


def test_temperature_1_kept_for_unsupported_model(monkeypatch):
    # temperature=1 is the one value these models still accept.
    monkeypatch.setattr(litellm, "drop_params", False)
    result = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
        params={"temperature": 1},
        model="claude-opus-4-7",
        drop_params=True,
    )

    assert result == {"temperature": 1}


def test_sampling_param_raises_clean_400_without_drop_params(monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)
    with pytest.raises(litellm.utils.UnsupportedParamsError, match="does not support temperature"):
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            params={"temperature": 0.3},
            model="claude-opus-4-7",
            drop_params=False,
        )
