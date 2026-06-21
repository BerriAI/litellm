"""Tests for `metadata` param support on the Anthropic /v1/messages pass-through route."""

import pytest
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)

@pytest.fixture
def config():
    return AnthropicMessagesConfig()

def test_metadata_in_supported_params(config):
    supported = config.get_supported_anthropic_messages_params(model="claude-3-5-sonnet-20241022")
    assert "metadata" in supported

def test_metadata_passes_through_transform_request(config):
    from litellm.types.router import GenericLiteLLMParams
    optional_params = {
        "max_tokens": 100,
        "metadata": {"user_id": "user-abc-123"},
    }
    result = config.transform_anthropic_messages_request(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert result.get("metadata") == {"user_id": "user-abc-123"}

def test_metadata_without_user_id_passes_through(config):
    from litellm.types.router import GenericLiteLLMParams
    optional_params = {
        "max_tokens": 50,
        "metadata": {"user_id": "test-user"},
    }
    result = config.transform_anthropic_messages_request(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "Hi"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "metadata" in result
