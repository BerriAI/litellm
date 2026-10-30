import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.redis.transport.contract import RedisFailure, RedisSubscription, RedisTransport

AUTH_CACHE_INVALIDATION_CHANNEL: Final = "litellm_proxy.auth_cache_invalidation"
_DECODED_JSON: Final = TypeAdapter(object)


def auth_invalidation_channel(namespace: str | None = None) -> str:
    return AUTH_CACHE_INVALIDATION_CHANNEL if namespace is None else f"{namespace}:{AUTH_CACHE_INVALIDATION_CHANNEL}"


@dataclass(frozen=True, slots=True)
class AuthInvalidation:
    cache_key: str
    new_value: float | None = None
    ttl: float | None = None


def encode_auth_invalidation(cache_key: str, new_value: float | None = None, ttl: float | None = None) -> str:
    message: Final[Mapping[str, object]] = MappingProxyType(
        asdict(AuthInvalidation(cache_key=cache_key, new_value=new_value, ttl=ttl))
    )
    return json.dumps({field: value for field, value in message.items() if value is not None})


def finite_number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def decode_auth_invalidation(data: object) -> AuthInvalidation | RedisFailure:
    payload: Final = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    if not isinstance(payload, str):
        return RedisFailure(kind="decode", message="invalidation payload must be a string")
    try:
        parsed: Final = as_str_mapping(_DECODED_JSON.validate_python(json.loads(payload)))
    except json.JSONDecodeError as error:
        return RedisFailure(kind="decode", message=str(error))
    if parsed is None:
        return RedisFailure(kind="decode", message="invalidation payload must be an object")
    cache_key: Final = parsed.get("cache_key")
    if not isinstance(cache_key, str):
        return RedisFailure(kind="decode", message="invalidation cache_key must be a string")
    return AuthInvalidation(
        cache_key=cache_key,
        new_value=finite_number_or_none(parsed.get("new_value")),
        ttl=finite_number_or_none(parsed.get("ttl")),
    )


def decode_auth_message(message: object) -> AuthInvalidation | RedisFailure:
    if isinstance(message, AuthInvalidation):
        return message
    envelope: Final = as_str_mapping(message)
    return decode_auth_invalidation(
        envelope.get("data") if isinstance(message, dict) and envelope is not None else None
    )


class AuthInvalidationSubscription(Protocol):
    async def receive(self, *, timeout: float) -> AuthInvalidation | None | RedisFailure: ...

    async def close(self) -> None | RedisFailure: ...


class AuthInvalidationStore(Protocol):
    async def publish_change(
        self, cache_key: str, new_value: float | None = None, ttl: float | None = None
    ) -> None | RedisFailure: ...

    async def publish_evictions(self, cache_keys: Sequence[str]) -> None | RedisFailure: ...

    async def subscribe(self) -> AuthInvalidationSubscription | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportAuthInvalidationSubscription:
    subscription: RedisSubscription

    async def receive(self, *, timeout: float) -> AuthInvalidation | None | RedisFailure:
        message: Final = await self.subscription.get_message(timeout=timeout)
        if message is None or isinstance(message, RedisFailure):
            return message
        return decode_auth_invalidation(message.data)

    async def close(self) -> None | RedisFailure:
        return await self.subscription.close()


@dataclass(frozen=True, slots=True)
class TransportAuthInvalidationStore:
    transport: RedisTransport
    empty_namespace: bool = False

    async def publish_change(
        self, cache_key: str, new_value: float | None = None, ttl: float | None = None
    ) -> None | RedisFailure:
        result: Final = await self.transport.publish(
            auth_invalidation_channel("" if self.empty_namespace else None),
            encode_auth_invalidation(cache_key, new_value, ttl),
        )
        return result if isinstance(result, RedisFailure) else None

    async def publish_evictions(self, cache_keys: Sequence[str]) -> None | RedisFailure:
        result: Final = await self.transport.publish_all(
            auth_invalidation_channel("" if self.empty_namespace else None),
            tuple(encode_auth_invalidation(cache_key) for cache_key in cache_keys),
        )
        return result if isinstance(result, RedisFailure) else None

    async def subscribe(self) -> AuthInvalidationSubscription | RedisFailure:
        subscription: Final = await self.transport.subscribe(
            auth_invalidation_channel("" if self.empty_namespace else None)
        )
        if isinstance(subscription, RedisFailure):
            return subscription
        return TransportAuthInvalidationSubscription(subscription)
