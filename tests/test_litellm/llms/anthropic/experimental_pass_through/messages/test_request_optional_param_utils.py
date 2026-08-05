"""
Regression tests for the /v1/messages request-parse fast paths:

- get_requested_anthropic_messages_optional_param must still filter to the
  valid AnthropicMessagesRequestOptionalParams keys and drop None values,
  while resolving the (static) type hints only once per process.
"""

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
