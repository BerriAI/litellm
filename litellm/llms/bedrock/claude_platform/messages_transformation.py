"""
Claude Platform on AWS - /messages endpoint

Rust sends /v1/messages calls on this route. Python keeps the config for provider lookups, the error class
and the URL the Rust host names in errors.
"""

from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig

from .common_utils import BedrockClaudePlatformMixin


class BedrockClaudePlatformMessagesConfig(BedrockClaudePlatformMixin, AnthropicMessagesConfig):
    def should_filter_anthropic_beta_headers(self) -> bool:
        return False
