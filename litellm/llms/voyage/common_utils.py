"""
Shared helpers for the Voyage (VoyageAI by MongoDB) provider.
"""

from typing import Final

from litellm.secret_managers.main import get_secret_str

VOYAGE_API_BASE: Final = "https://api.voyageai.com/v1"
MONGODB_API_BASE: Final = "https://ai.mongodb.com/v1"
MONGODB_API_KEY_PREFIX: Final = "al-"


def get_voyage_api_key(api_key: str | None = None) -> str | None:
    """Resolve the key a Voyage request will authenticate with, explicit value first."""
    return (
        api_key
        or get_secret_str("VOYAGE_API_KEY")
        or get_secret_str("VOYAGE_AI_API_KEY")
        or get_secret_str("VOYAGE_AI_TOKEN")
    )


def get_default_base_url(api_key: str | None = None) -> str:
    """
    Pick the host that issued the key: MongoDB-issued keys (``al-`` prefix) are only
    valid on ai.mongodb.com, every other key on api.voyageai.com.

    Mirrors ``voyageai.util.get_default_base_url`` in the official SDK:
    https://github.com/voyage-ai/voyageai-python/blob/main/voyageai/util.py
    """
    resolved: Final = get_voyage_api_key(api_key)
    if resolved is not None and resolved.startswith(MONGODB_API_KEY_PREFIX):
        return MONGODB_API_BASE
    return VOYAGE_API_BASE
