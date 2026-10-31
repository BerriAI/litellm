import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from litellm._internal_context import service_target
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any

TARGET: Final = "router_cooldowns"


class CooldownCacheValue(TypedDict):
    exception_received: str
    status_code: str
    timestamp: float
    cooldown_time: float


_ENTRY: Final = TypeAdapter(CooldownCacheValue)


def cooldown_key(model_id: str) -> str:
    return f"deployment:{model_id}:cooldown"


def decode(raw: object) -> CooldownCacheValue | None:
    try:
        if isinstance(raw, str):
            return _ENTRY.validate_json(raw)
        return _ENTRY.validate_python(raw) if isinstance(raw, dict) else None
    except ValidationError:
        return None


class RouterCooldownStore(Protocol):
    def put(self, model_id: str, entry: CooldownCacheValue, *, ttl: float) -> None: ...

    def get_many(
        self, model_ids: Sequence[str], parent_otel_span: Span | None
    ) -> tuple[CooldownCacheValue | None, ...]: ...

    async def async_get_many(
        self, model_ids: Sequence[str], parent_otel_span: Span | None
    ) -> tuple[CooldownCacheValue | None, ...]: ...

    def reset_local_ttl(self, model_id: str, entry: CooldownCacheValue, *, ttl: float) -> None: ...

    def drop_local(self, model_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class DualCacheRouterCooldownStore:
    memory: InMemoryCache
    tiers: Callable[[], DualCache]

    def put(self, model_id: str, entry: CooldownCacheValue, *, ttl: float) -> None:
        with service_target(TARGET):
            self.tiers().set_cache(key=cooldown_key(model_id), value=json.dumps(entry), ttl=ttl)

    def get_many(
        self, model_ids: Sequence[str], parent_otel_span: Span | None
    ) -> tuple[CooldownCacheValue | None, ...]:
        with service_target(TARGET):
            raw: Final = self.tiers().batch_get_cache(
                keys=[cooldown_key(model_id) for model_id in model_ids], parent_otel_span=parent_otel_span
            )
        return tuple(decode(value) for value in raw or ())

    async def async_get_many(
        self, model_ids: Sequence[str], parent_otel_span: Span | None
    ) -> tuple[CooldownCacheValue | None, ...]:
        with service_target(TARGET):
            raw: Final = await self.tiers().async_batch_get_cache(
                keys=[cooldown_key(model_id) for model_id in model_ids], parent_otel_span=parent_otel_span
            )
        return tuple(decode(value) for value in raw or ())

    def reset_local_ttl(self, model_id: str, entry: CooldownCacheValue, *, ttl: float) -> None:
        self.memory.delete_cache(cooldown_key(model_id))
        self.memory.set_cache(cooldown_key(model_id), json.dumps(entry), ttl=ttl)

    def drop_local(self, model_id: str) -> None:
        self.memory.delete_cache(cooldown_key(model_id))
