from typing import Final, TypeVar

from pydantic import BaseModel

from litellm.caching.in_memory_cache import InMemoryCache

T = TypeVar("T", bound=BaseModel)  # noqa: LIT010  # type checkers require a bare TypeVar binding


class FrozenModelCache:
    def __init__(self, memory: InMemoryCache) -> None:
        self._memory: Final = memory

    def get(self, key: str, model_type: type[T]) -> T | None:
        cached: Final = self._memory.get_cache(key)
        return cached if isinstance(cached, model_type) and cached.model_config.get("frozen") else None

    def remember(self, key: str, value: BaseModel, ttl: float) -> None:
        if value.model_config.get("frozen"):
            self._memory.set_cache(key, value, ttl=ttl)
