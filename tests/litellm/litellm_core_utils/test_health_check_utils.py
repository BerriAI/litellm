"""
Tests for ``litellm.litellm_core_utils.health_check_utils._filter_model_params``.

Regression coverage for #26406 / #26604: chat-only params (``messages``,
``max_tokens``) must be stripped before non-chat health check handlers, but
``max_tokens`` must be preserved for the ``completion`` mode handler
(``litellm.atext_completion``) so the ``BACKGROUND_HEALTH_CHECK_MAX_TOKENS``
cost cap still applies.
"""

from litellm.litellm_core_utils.health_check_utils import (
    _COMPLETION_HEALTH_CHECK_STRIP_KEYS,
    _NON_CHAT_HEALTH_CHECK_STRIP_KEYS,
    _filter_model_params,
)


def _sample_params() -> dict:
    return {
        "model": "openai/dall-e-3",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 5,
        "api_key": "sk-fake",
    }


def test_filter_strips_messages_and_max_tokens_for_non_chat_handlers():
    """Default behavior: strip both ``messages`` and ``max_tokens``.

    Reproduces the original failure (OpenAI image-generation 400 on
    ``max_tokens``) — these keys must not reach non-chat handlers.
    """
    filtered = _filter_model_params(model_params=_sample_params())

    assert "messages" not in filtered
    assert "max_tokens" not in filtered
    assert filtered == {"model": "openai/dall-e-3", "api_key": "sk-fake"}


def test_filter_keeps_max_tokens_for_completion_mode():
    """``completion`` mode passes ``keep_max_tokens=True``.

    ``litellm.atext_completion`` accepts ``max_tokens``; stripping it would
    silently remove the ``BACKGROUND_HEALTH_CHECK_MAX_TOKENS`` cost cap.
    """
    filtered = _filter_model_params(
        model_params=_sample_params(), keep_max_tokens=True
    )

    assert "messages" not in filtered
    assert filtered.get("max_tokens") == 5
    assert filtered["model"] == "openai/dall-e-3"
    assert filtered["api_key"] == "sk-fake"


def test_filter_does_not_mutate_input():
    params = _sample_params()
    snapshot = dict(params)

    _filter_model_params(model_params=params)
    _filter_model_params(model_params=params, keep_max_tokens=True)

    assert params == snapshot


def test_strip_key_sets_are_consistent():
    """``_NON_CHAT`` is a strict superset of ``_COMPLETION`` — only
    ``max_tokens`` differs between the two modes."""
    assert _COMPLETION_HEALTH_CHECK_STRIP_KEYS == {"messages"}
    assert _NON_CHAT_HEALTH_CHECK_STRIP_KEYS == {"messages", "max_tokens"}
    assert _COMPLETION_HEALTH_CHECK_STRIP_KEYS < _NON_CHAT_HEALTH_CHECK_STRIP_KEYS
