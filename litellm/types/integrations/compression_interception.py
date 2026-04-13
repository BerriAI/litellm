"""
Type definitions for Compression Interception integration.
"""

from typing import Any, Dict, List, Optional, TypedDict


class CompressionInterceptionConfig(TypedDict, total=False):
    """
    Configuration parameters for CompressionInterceptionLogger.

    Used in proxy_config.yaml under litellm_settings:
        litellm_settings:
          compression_interception_params:
            enabled_providers: ["openai", "anthropic"]
            compression_trigger: 12000
            compression_target: 8000
            embedding_model: "text-embedding-3-small"
            embedding_model_params:
              dimensions: 512
    """

    enabled_providers: List[str]
    """Optional provider allowlist. If omitted, applies to all providers."""

    compression_trigger: int
    """Only compress requests above this prompt token threshold."""

    compression_target: Optional[int]
    """Target token count after compression. If None, uses compressor default."""

    embedding_model: Optional[str]
    """Optional embedding model used for hybrid ranking."""

    embedding_model_params: Optional[Dict[str, Any]]
    """Optional params forwarded to litellm.embedding() scorer."""
