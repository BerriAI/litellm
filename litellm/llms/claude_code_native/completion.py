"""
Claude Code Native Provider Completion Handler

This module provides the completion handler for the Claude Code Native provider,
reusing all Anthropic's chat completion logic.
"""

from litellm.llms.anthropic.chat.handler import AnthropicChatCompletion


class ClaudeCodeNativeChatCompletion(AnthropicChatCompletion):
    """
    Completion handler for Claude Code Native Provider.

    This class reuses all of Anthropic's chat completion logic without
    modifications, as the only differences are in the headers and system
    message handling which are implemented in ClaudeCodeNativeConfig.
    """

    def __init__(self) -> None:
        super().__init__()


def completion():
    """
    Factory function to create a ClaudeCodeNativeChatCompletion instance.
    """
    return ClaudeCodeNativeChatCompletion()
