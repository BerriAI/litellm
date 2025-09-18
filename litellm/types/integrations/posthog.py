from typing import Any, Dict, TypedDict

POSTHOG_MAX_BATCH_SIZE = 100


class PostHogEventPayload(TypedDict):
    """PostHog event payload structure"""

    event: str  # "$ai_generation" or "$ai_embedding"
    properties: Dict[str, Any]
    distinct_id: str


class PostHogCredentialsObject(TypedDict):
    """PostHog credentials configuration"""

    POSTHOG_API_KEY: str
    POSTHOG_HOST: str
