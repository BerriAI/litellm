import json
from dataclasses import asdict, dataclass
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisSubscription, RedisTransport

CONFIG_SYNC_CHANNEL: Final = "litellm_proxy.config_change"


@dataclass(frozen=True, slots=True)
class ConfigChange:
    object_type: str
    model_ids: tuple[str, ...] = ()


_CONFIG_CHANGE: Final = TypeAdapter(ConfigChange)


def encode_config_change(change: ConfigChange) -> str:
    return json.dumps(asdict(change))


def decode_config_change(payload: str | bytes) -> ConfigChange | None:
    try:
        return _CONFIG_CHANGE.validate_json(payload)
    except ValidationError:
        return None


class ConfigSyncStore(Protocol):
    async def publish_change(self, change: ConfigChange) -> int | RedisFailure: ...

    async def subscribe(self) -> RedisSubscription | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportConfigSyncStore:
    transport: RedisTransport

    async def publish_change(self, change: ConfigChange) -> int | RedisFailure:
        return await self.transport.publish(CONFIG_SYNC_CHANNEL, encode_config_change(change))

    async def subscribe(self) -> RedisSubscription | RedisFailure:
        return await self.transport.subscribe(CONFIG_SYNC_CHANNEL)
