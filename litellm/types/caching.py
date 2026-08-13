from enum import Enum
from typing import Any, Final, Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import TypedDict


class LiteLLMCacheType(str, Enum):
    LOCAL = "local"
    REDIS = "redis"
    REDIS_SEMANTIC = "redis-semantic"
    VALKEY_SEMANTIC = "valkey-semantic"
    S3 = "s3"
    DISK = "disk"
    QDRANT_SEMANTIC = "qdrant-semantic"
    AZURE_BLOB = "azure-blob"
    GCS = "gcs"


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
    "responses",
    "aresponses",
]


class RedisPipelineIncrementOperation(TypedDict):
    """
    TypeDict for 1 Redis Pipeline Increment Operation
    """

    key: str
    increment_value: float
    ttl: int | None


class RedisPipelineSetOperation(TypedDict):
    """
    TypeDict for 1 Redis Pipeline Set Operation
    """

    key: str
    value: Any
    ttl: int | None


class RedisPipelineRpushOperation(TypedDict):
    """
    TypedDict for 1 Redis Pipeline RPUSH Operation
    """

    key: str
    values: list[Any]


class RedisPipelineLpopOperation(TypedDict):
    """
    TypedDict for 1 Redis Pipeline LPOP Operation
    """

    key: str
    count: int | None


DynamicCacheControl = TypedDict(
    "DynamicCacheControl",
    {
        # Will cache the response for the user-defined amount of time (in seconds).
        "ttl": int | None,
        # Namespace to use for caching
        "namespace": str | None,
        # Max Age to use for caching
        "s-maxage": int | None,
        "s-max-age": int | None,
        # Will not return a cached response, but instead call the actual endpoint.
        "no-cache": bool | None,
        # Will not store the response in the cache.
        "no-store": bool | None,
    },
)


class CachePingResponse(BaseModel):
    status: str
    cache_type: str
    ping_response: bool | None = None
    set_cache_response: str | None = None
    litellm_cache_params: str | None = None

    # intentionally a dict, since we run masker.mask_dict() on HealthCheckCacheParams
    health_check_cache_params: dict | None = None


class HealthCheckCacheParams(BaseModel):
    """
    Cache Params returned on /cache/ping call
    """

    host: str | None = None
    port: str | int | None = None
    redis_kwargs: dict[str, Any] | None = None
    namespace: str | None = None
    redis_version: str | int | float | None = None


class CachedEmbedding(TypedDict):
    """Type definition for cached embedding objects"""

    embedding: list[float] | None
    index: int | None
    object: str | None
    model: str | None
    prompt_tokens: int | None
    prompt_tokens_details: dict | None
