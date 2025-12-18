"""
LiteLLM Interactions to Responses API Bridge

This module provides a bridge from the Google Interactions API to the OpenAI Responses API.
It enables using any provider supported by the Responses API (including the Chat Completion
bridge) with the Interactions API interface.

The bridge follows the same pattern as the Responses to Completions bridge:
- transformation.py: Request/response transformation logic
- handler.py: Main handler that routes to Responses API
- streaming_iterator.py: Streaming response transformation
"""

from litellm.interactions.litellm_responses_transformation.handler import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.interactions.litellm_responses_transformation.streaming_iterator import (
    LiteLLMResponsesStreamingIterator,
)
from litellm.interactions.litellm_responses_transformation.transformation import (
    LiteLLMResponsesInteractionsConfig,
)

__all__ = [
    "LiteLLMResponsesInteractionsConfig",
    "LiteLLMResponsesTransformationHandler",
    "LiteLLMResponsesStreamingIterator",
]
