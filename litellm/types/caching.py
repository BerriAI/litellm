from enum import Enum
from typing import Literal, Optional, TypedDict


class LiteLLMCacheType(str, Enum):
    LOCAL = "local"
    REDIS = "redis"
    REDIS_SEMANTIC = "redis-semantic"
    S3 = "s3"
    DISK = "disk"
    QDRANT_SEMANTIC = "qdrant-semantic"


CachingSupportedCallTypes = Literal[
    "completion",
    "acompletion",
    "embedding",
    "aembedding",
    "atranscription",
    "transcription",
    "atext_completion",
    "text_completion",
    "arerank",
    "rerank",
]


class RedisPipelineIncrementOperation(TypedDict):
    """
    TypeDict for 1 Redis Pipeline Increment Operation
    """

    key: str
    increment_value: float
    ttl: Optional[int]


DynamicCacheControl = TypedDict(
    "DynamicCacheControl",
    {
        # Will cache the response for the user-defined amount of time (in seconds).
        "ttl": Optional[int],
        # Namespace to use for caching
        "namespace": Optional[str],
        # Max Age to use for caching
        "s-maxage": Optional[int],
        "s-max-age": Optional[int],
        # Will not return a cached response, but instead call the actual endpoint.
        "no-cache": Optional[bool],
        # Will not store the response in the cache.
        "no-store": Optional[bool],
    },
)
