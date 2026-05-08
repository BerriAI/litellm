"""
Unit tests for litellm issue #26241:
Anthropic -> Responses adapter must respect litellm.drop_params when
mapping metadata.user_id to the Responses API `user` field.
"""
import pytest
import litellm
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)

_ADAPTER = LiteLLMAnthropicToResponsesAPIAdapter()

_BASE_REQUEST = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    "metadata": {"user_id": "test-user-123"},
}


def test_user_field_set_when_drop_params_false():
    """user field IS forwarded when drop_params is False (default)."""
    litellm.drop_params = False
    try:
        kwargs = _ADAPTER.translate_request(_BASE_REQUEST)
        assert kwargs.get("user") == "test-user-123", (
            "Expected user field to be set when drop_params=False"
        )
    finally:
        litellm.drop_params = False


def test_user_field_omitted_when_drop_params_true():
    """user field is NOT forwarded when drop_params is True (fixes #26241)."""
    litellm.drop_params = True
    try:
        kwargs = _ADAPTER.translate_request(_BASE_REQUEST)
        assert "user" not in kwargs, (
            f"Expected user field to be absent when drop_params=True, got {kwargs.get('user')}"
        )
    finally:
        litellm.drop_params = False


def test_user_field_absent_when_no_metadata():
    """No user field if metadata is missing entirely."""
    litellm.drop_params = False
    req = {**_BASE_REQUEST, "metadata": None}
    try:
        kwargs = _ADAPTER.translate_request(req)
        assert "user" not in kwargs
    finally:
        litellm.drop_params = False


def test_user_field_absent_when_user_id_missing_from_metadata():
    """No user field if metadata dict has no user_id key."""
    litellm.drop_params = False
    req = {**_BASE_REQUEST, "metadata": {"other_key": "value"}}
    try:
        kwargs = _ADAPTER.translate_request(req)
        assert "user" not in kwargs
    finally:
        litellm.drop_params = False


def test_user_field_truncated_to_64_chars():
    """user_id longer than 64 chars is truncated (existing behavior, not regressed)."""
    litellm.drop_params = False
    long_id = "a" * 100
    req = {**_BASE_REQUEST, "metadata": {"user_id": long_id}}
    try:
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs.get("user") == "a" * 64
    finally:
        litellm.drop_params = False
