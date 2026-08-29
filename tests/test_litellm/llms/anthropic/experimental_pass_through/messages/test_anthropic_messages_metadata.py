"""
Tests for Anthropic Messages passthrough metadata support.

Related issue: https://github.com/BerriAI/litellm/issues/30663
"""

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams


class TestAnthropicMessagesMetadataSupport:
    """Test that metadata is properly supported in Anthropic Messages passthrough."""

    def setup_method(self):
        self.config = AnthropicMessagesConfig()

    def test_metadata_in_supported_params(self):
        """Verify 'metadata' is listed in supported Anthropic Messages params."""
        supported_params = self.config.get_supported_anthropic_messages_params(
            model="claude-sonnet-4-20250514"
        )
        assert (
            "metadata" in supported_params
        ), "'metadata' should be in supported params for Anthropic Messages passthrough"

    def test_metadata_appears_exactly_once(self):
        """Verify 'metadata' is not duplicated in the supported params list."""
        supported_params = self.config.get_supported_anthropic_messages_params(
            model="claude-sonnet-4-20250514"
        )
        assert supported_params.count("metadata") == 1

    def test_core_params_still_present(self):
        """Regression: ensure adding metadata did not remove existing params."""
        supported_params = self.config.get_supported_anthropic_messages_params(
            model="claude-sonnet-4-20250514"
        )
        expected_core_params = [
            "messages",
            "model",
            "system",
            "max_tokens",
            "temperature",
        ]
        for param in expected_core_params:
            assert param in supported_params

    def test_metadata_forwarded_in_transformed_request(self):
        """Verify 'metadata' is actually forwarded in the final transformed Anthropic Messages request body."""
        result = self.config.transform_anthropic_messages_request(
            model="claude-sonnet-4-20250514",
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            anthropic_messages_optional_request_params={
                "max_tokens": 10,
                "metadata": {
                    "user_id": "test-user-123",
                },
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["metadata"] == {
            "user_id": "test-user-123",
        }