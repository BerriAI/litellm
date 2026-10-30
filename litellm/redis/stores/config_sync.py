import json
from dataclasses import asdict, dataclass
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisSubscription, RedisTransport

CONFIG_SYNC_CHANNEL: Final = "litellm_proxy.config_change"


def config_sync_channel(namespace: str | None = None) -> str:
    return CONFIG_SYNC_CHANNEL if namespace is None else f"{namespace}:{CONFIG_SYNC_CHANNEL}"


@dataclass(frozen=True, slots=True)
class ConfigChange:
    object_type: str
    model_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _PubSubEnvelope:
    data: str | bytes


@dataclass(frozen=True, slots=True)
class ReceivedConfigChange:
    change: ConfigChange | RedisFailure


_CONFIG_CHANGE: Final = TypeAdapter(ConfigChange)
_ENVELOPE: Final = TypeAdapter(_PubSubEnvelope)


def encode_config_change(object_type: str, model_ids: tuple[str, ...] = ()) -> str:
    return json.dumps(asdict(ConfigChange(object_type=object_type, model_ids=model_ids)))


def decode_config_change(message: object) -> ConfigChange | RedisFailure:
    if isinstance(message, ReceivedConfigChange):
        return message.change
    try:
        payload: Final = message if isinstance(message, (str, bytes)) else _ENVELOPE.validate_python(message).data
        return _CONFIG_CHANGE.validate_json(payload)
    except ValidationError as error:
        return RedisFailure(kind="decode", message=str(error))


class ConfigSyncSubscription(Protocol):
    async def receive(self, *, timeout: float) -> ReceivedConfigChange | None | RedisFailure: ...

    async def close(self) -> None | RedisFailure: ...


class ConfigSyncStore(Protocol):
    async def publish_change(self, object_type: str, model_ids: tuple[str, ...] = ()) -> None | RedisFailure: ...

    async def subscribe(self) -> ConfigSyncSubscription | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportConfigSyncSubscription:
    subscription: RedisSubscription

    async def receive(self, *, timeout: float) -> ReceivedConfigChange | None | RedisFailure:
        message: Final = await self.subscription.get_message(timeout=timeout)
        if isinstance(message, RedisFailure) and message.kind == "decode":
            return ReceivedConfigChange(message)
        if message is None or isinstance(message, RedisFailure):
            return message
        return ReceivedConfigChange(decode_config_change(message.data))

    async def close(self) -> None | RedisFailure:
        return await self.subscription.close()


@dataclass(frozen=True, slots=True)
class TransportConfigSyncStore:
    transport: RedisTransport
    empty_namespace: bool = False

    async def publish_change(self, object_type: str, model_ids: tuple[str, ...] = ()) -> None | RedisFailure:
        result: Final = await self.transport.publish(
            config_sync_channel("" if self.empty_namespace else None), encode_config_change(object_type, model_ids)
        )
        return result if isinstance(result, RedisFailure) else None

    async def subscribe(self) -> ConfigSyncSubscription | RedisFailure:
        subscription: Final = await self.transport.subscribe(config_sync_channel("" if self.empty_namespace else None))
        if isinstance(subscription, RedisFailure):
            return subscription
        return TransportConfigSyncSubscription(subscription)
