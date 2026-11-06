from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RedisStoreSettings:
    """What `connect` takes from a `RedisCache`, captured once when the cache is built."""

    redis_kwargs: Mapping[str, object]
    namespace: str | None
    default_ttl: int
