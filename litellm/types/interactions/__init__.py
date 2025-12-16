"""
Type definitions for Google Interactions API

Based on https://ai.google.dev/api/interactions-api
"""

from litellm.types.interactions.main import (
    # Request types
    InteractionsAPIRequestParams,
    InteractionsAPIOptionalRequestParams,
    InteractionInputContent,
    InteractionInputPart,
    InteractionTool,
    InteractionToolConfig,
    InteractionGenerationConfig,
    InteractionSafetySettings,
    # Response types
    InteractionsAPIResponse,
    InteractionCandidate,
    InteractionContent,
    InteractionPart,
    InteractionUsageMetadata,
    InteractionPromptFeedback,
    InteractionGroundingMetadata,
    InteractionSafetyRating,
    InteractionCitationMetadata,
    InteractionCitationSource,
    # Streaming types
    InteractionsAPIStreamingResponse,
    # Result types
    DeleteInteractionResult,
    CancelInteractionResult,
)

__all__ = [
    # Request types
    "InteractionsAPIRequestParams",
    "InteractionsAPIOptionalRequestParams",
    "InteractionInputContent",
    "InteractionInputPart",
    "InteractionTool",
    "InteractionToolConfig",
    "InteractionGenerationConfig",
    "InteractionSafetySettings",
    # Response types
    "InteractionsAPIResponse",
    "InteractionCandidate",
    "InteractionContent",
    "InteractionPart",
    "InteractionUsageMetadata",
    "InteractionPromptFeedback",
    "InteractionGroundingMetadata",
    "InteractionSafetyRating",
    "InteractionCitationMetadata",
    "InteractionCitationSource",
    # Streaming types
    "InteractionsAPIStreamingResponse",
    # Result types
    "DeleteInteractionResult",
    "CancelInteractionResult",
]
