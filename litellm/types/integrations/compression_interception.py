"""
Type definitions for Compression Interception integration.
"""

from typing import Any, Dict, Literal, Optional, TypedDict


class CompressionInterceptionConfig(TypedDict, total=False):
    """
    Configuration parameters for CompressionInterceptionLogger.

    Used in proxy_config.yaml under litellm_settings:
        litellm_settings:
          compression_interception_params:
            enabled: true
            compression_trigger: 100000
            compression_target: 70000
            embedding_model: "text-embedding-3-small"
            embedding_model_params:
              dimensions: 512
    """

    enabled: bool
    compression_trigger: int
    compression_target: Optional[int]
    embedding_model: Optional[str]
    embedding_model_params: Optional[Dict[str, Any]]


class CompressionSavingsMetadata(TypedDict):
    """
    Per-request prompt-compression savings recorded into the spend-log metadata
    JSON so daily spend aggregates can track tokens saved by compression.
    """

    tokens_before: int
    tokens_after: int
    tokens_saved: int
    source: Literal["compression_interception"]
