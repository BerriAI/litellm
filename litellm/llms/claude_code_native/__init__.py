"""
Claude Code Native Provider for LiteLLM

This module provides the Claude Code Native provider, a variant of Anthropic
that uses Bearer authentication and a specific system prompt for Claude Code.
"""
from .completion import (
    ClaudeCodeNativeChatCompletion,
    completion,
)
from .transformation import ClaudeCodeNativeConfig

__all__ = [
    "ClaudeCodeNativeConfig",
    "ClaudeCodeNativeChatCompletion",
    "completion",
]
