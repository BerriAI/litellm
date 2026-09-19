"""
Bridge module for connecting Interactions API to Responses API via litellm.responses().
"""

from litellm.interactions.litellm_responses_transformation.handler import (
    LiteLLMResponsesInteractionsHandler,
)
from litellm.interactions.litellm_responses_transformation.transformation import (
    LiteLLMResponsesInteractionsConfig,
)

__all__ = [
    "LiteLLMResponsesInteractionsConfig",  # Transformation config class (not BaseInteractionsAPIConfig)
    "LiteLLMResponsesInteractionsHandler",
]
