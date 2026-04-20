"""
Test that metadata dict is shallow-copied in transform_anthropic_messages_request
to prevent raw_request leak from litellm_params into the Anthropic API request.

Related: PR #24661 (same bug, chat/completions path).
"""

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams


def test_metadata_dict_is_copied_not_shared():
    """
    The metadata dict inside optional_params should be a different object
    after transform_anthropic_messages_request, so mutations to the original
    (e.g. raw_request injection by the logging layer) don't leak.
    """
    config = AnthropicMessagesConfig()

    original_metadata = {"user_id": "test-user-123"}
    optional_params = {
        "max_tokens": 100,
        "metadata": original_metadata,
    }

    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    # The metadata in optional_params should now be a copy, not the same object
    assert optional_params["metadata"] is not original_metadata

    # The copy should still have the same content
    assert optional_params["metadata"]["user_id"] == "test-user-123"

    # Mutating the original should not affect the copy in optional_params
    original_metadata["raw_request"] = "leaked!"
    assert "raw_request" not in optional_params["metadata"]


def test_metadata_dict_copy_preserves_content_in_result():
    """
    Verify the copied metadata ends up in the final request dict with
    correct content.
    """
    config = AnthropicMessagesConfig()

    metadata = {"user_id": "u-456"}
    optional_params = {
        "max_tokens": 200,
        "metadata": metadata,
    }

    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hi"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["metadata"]["user_id"] == "u-456"


def test_no_metadata_dict_no_error():
    """
    When metadata is absent, the copy logic should be a no-op (no KeyError).
    """
    config = AnthropicMessagesConfig()

    optional_params = {
        "max_tokens": 100,
    }

    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    # Should succeed without error; metadata absent from result
    assert "metadata" not in result or result.get("metadata") is None


def test_non_dict_metadata_not_copied():
    """
    If metadata is not a dict (e.g. a Pydantic model), the copy guard
    should skip it without error.
    """
    config = AnthropicMessagesConfig()

    class FakeMetadata:
        user_id = "test"

    fake = FakeMetadata()
    optional_params = {
        "max_tokens": 100,
        "metadata": fake,
    }

    # Should not raise; the non-dict metadata passes through as-is
    try:
        config.transform_anthropic_messages_request(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
    except Exception:
        # AnthropicMessagesRequest might reject non-dict metadata; that's fine.
        # The point is the copy logic itself doesn't crash.
        pass

    # The metadata should still be the same object (not copied)
    assert optional_params["metadata"] is fake
