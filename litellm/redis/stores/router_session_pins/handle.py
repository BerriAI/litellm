from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Protocol

from litellm.caching.redis_batch import NativePending
from litellm.redis.stores.router_session_pins.keys import PinKey
from litellm.rust_bridge._native import NativeRedisClient, NativeSend


class RouterSessionPinsHandle(Protocol):
    @property
    def has_redis(self) -> bool: ...
    def attach(self, client: NativeRedisClient, /) -> bool: ...
    def get(self, key: PinKey) -> Awaitable[object]: ...
    def get_local(self, key: PinKey) -> object: ...
    def set_local(self, key: PinKey, value: object, ttl_seconds: float) -> None: ...
    def delete_local(self, key: PinKey) -> None: ...
    def set(self, key: PinKey, value: object, ttl_seconds: float) -> Awaitable[None]: ...
    def get_shared(self, key: PinKey) -> Awaitable[object]: ...
    def read(self, key: PinKey, ttl_seconds: float, send: NativeSend) -> NativePending[object]: ...
    def claim(
        self,
        key: PinKey,
        value: object,
        ttl_seconds: float,
        eligible_values: Sequence[object] | None,
        preserve_legacy_model_ids: bool,
    ) -> Awaitable[object]: ...
    def claim_local(
        self,
        key: PinKey,
        value: object,
        ttl_seconds: float,
        eligible_values: Sequence[object] | None,
        preserve_legacy_model_ids: bool,
    ) -> object: ...
