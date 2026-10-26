import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Literal, Protocol, TypeVar

from pydantic import ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm.types.llms.base import LiteLLMBaseModel


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


class SemanticCacheScope(str, Enum):
    KEY = "key"
    END_USER = "end_user"


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
    "anthropic_messages",
    "aanthropic_messages",
]

KeyPresence = Literal["present", "missing", "unavailable"]


@dataclass(frozen=True, slots=True)
class CacheFailure:
    kind: Literal["unavailable", "invalid_response"]
    message: str
    reason: Literal["circuit_open", "timeout", "other"] = "other"
    validation_error: ValidationError | None = field(default=None, compare=False, repr=False)


CacheValue: Final = TypeVar("CacheValue")


class CacheOperation(Awaitable[CacheValue | CacheFailure], Protocol[CacheValue]):
    @property
    def done(self) -> bool: ...

    def on_settled(
        self, hook: Callable[[asyncio.Future[CacheValue | CacheFailure]], Awaitable[None] | None]
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class WindowCounter:
    window_key: str
    counter_key: str


@dataclass(frozen=True, slots=True)
class WindowCount:
    window_start: str
    count: int


@dataclass(frozen=True, slots=True)
class WindowReservation:
    window_key: str
    counter_key: str
    limit: int
    increment: int
    ttl_seconds: int
    window_seconds: int


@dataclass(frozen=True, slots=True)
class CapacityExceeded:
    index: int
    count: int
    limit: int


@dataclass(frozen=True, slots=True)
class ConcurrencySlot:
    key: str
    limit: int
    ttl_seconds: int
    slot_id: str


@dataclass(frozen=True, slots=True)
class ConcurrencyRelease:
    key: str
    slot_id: str


@dataclass(frozen=True, slots=True)
class ConcurrencyRead:
    key: str
    ttl_seconds: int


@dataclass(frozen=True, slots=True)
class CounterIncrement:
    key: str
    increment: float
    ttl_seconds: int


@dataclass(frozen=True, slots=True)
class GuardedIncrement:
    window_key: str
    counter_key: str
    expected_window_start: str
    increment: int
    ttl_seconds: int


@dataclass(frozen=True, slots=True)
class CounterUpdate:
    applied: bool
    count: int


@dataclass(frozen=True, slots=True)
class LimitedIncrement:
    accepted: bool
    count: int


@dataclass(frozen=True, slots=True)
class ThrottleKeys:
    pair_counter: str
    pair_block: str
    source_counter: str
    source_block: str


class CacheSubscription(Protocol):
    async def receive(self, *, timeout_seconds: float) -> str | bytes | None | CacheFailure: ...

    async def close(self) -> None | CacheFailure: ...


@dataclass(frozen=True, slots=True)
class CachePoolInfo:
    available_connections: int | None
    in_use_connections: int | None
    max_connections: int | None
    connection_class: str | None


@dataclass(frozen=True, slots=True)
class CachePublish:
    channel: str
    message: str


@dataclass(frozen=True, slots=True)
class CacheQueueUpdate:
    key: str
    values: tuple[str, ...]


DEFAULT_CACHING_SUPPORTED_CALL_TYPES: tuple[CachingSupportedCallTypes, ...] = (
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
    "anthropic_messages",
    "aanthropic_messages",
)


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
    value: object
    ttl: int | None


class RedisPipelineRpushOperation(TypedDict):
    """
    TypedDict for 1 Redis Pipeline RPUSH Operation
    """

    key: str
    values: Sequence[Any]


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


class CachePingResponse(LiteLLMBaseModel):
    status: str
    cache_type: str
    ping_response: bool | None = None
    set_cache_response: str | None = None
    litellm_cache_params: str | None = None

    # intentionally a dict, since we run masker.mask_dict() on HealthCheckCacheParams
    health_check_cache_params: dict | None = None


class HealthCheckCacheParams(LiteLLMBaseModel):
    """
    Cache Params returned on /cache/ping call
    """

    host: str | None = None
    port: str | int | None = None
    redis_kwargs: dict[str, object] | None = None
    namespace: str | None = None
    redis_version: str | int | float | None = None


EMBEDDING_CACHE_FORMAT_VERSION: Final = 2

CACHED_STREAM_EVENTS_KEY: Final = "litellm_cached_anthropic_sse_events"


class CachedEmbedding(TypedDict):
    """Type definition for cached embedding objects"""

    embedding: ReadOnly[list[float] | str | None]
    index: ReadOnly[int | None]
    object: ReadOnly[str | None]
    model: ReadOnly[str | None]
    prompt_tokens: ReadOnly[int | None]
    prompt_tokens_details: ReadOnly[dict | None]
    format_version: ReadOnly[int]
