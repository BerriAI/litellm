"""
Type definitions for callback hooks.

This module provides centralized type aliases for call_type parameters
used across CustomLogger subclasses to maintain consistency and avoid
code duplication.
"""

from typing import Literal

# Type alias for async_pre_call_hook call_type parameter
PreCallHookCallType = Literal[
    "completion",
    "text_completion",
    "embeddings",
    "aembedding",  # async embeddings
    "image_generation",
    "moderation",
    "audio_transcription",
    "pass_through_endpoint",
    "rerank",
    "mcp_call",
    "anthropic_messages",
]

# Type alias for async_moderation_hook call_type parameter
ModerationHookCallType = Literal[
    "completion",
    "embeddings",
    "aembedding",  # async embeddings
    "image_generation",
    "moderation",
    "audio_transcription",
    "responses",
    "mcp_call",
    "anthropic_messages",
]
