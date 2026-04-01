"""
A2A to LiteLLM Completion Bridge.

This module provides transformation between A2A protocol messages and
LiteLLM completion API, enabling any LiteLLM-supported provider to be
invoked via the A2A protocol.
"""

from litellm.a2a_protocol.litellm_completion_bridge.handler import (
    A2ACompletionBridgeHandler,
    handle_a2a_completion,
    handle_a2a_completion_streaming,
)
from litellm.a2a_protocol.litellm_completion_bridge.transformation import (
    A2ACompletionBridgeTransformation,
)

__all__ = [
    "A2ACompletionBridgeTransformation",
    "A2ACompletionBridgeHandler",
    "handle_a2a_completion",
    "handle_a2a_completion_streaming",
]
