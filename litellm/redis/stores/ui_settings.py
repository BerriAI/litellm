import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_SETTINGS: Final = TypeAdapter(dict[str, JsonValue])


def settings_key() -> str:
    return "ui_settings:settings_dict"


class UISettingsStore(Protocol):
    async def get(self) -> Mapping[str, JsonValue] | None | RedisFailure: ...

    async def put(self, settings: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportUISettingsStore:
    transport: RedisTransport

    async def get(self) -> Mapping[str, JsonValue] | None | RedisFailure:
        raw: Final = await self.transport.get(settings_key())
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            decoded: Final = raw.decode("utf-8")
            try:
                return _SETTINGS.validate_python(json.loads(decoded))
            except json.JSONDecodeError:
                return _SETTINGS.validate_python(ast.literal_eval(decoded))
        except (ValueError, SyntaxError, ValidationError) as error:
            return RedisFailure(kind="decode", message=f"UI settings are not an object: {error}")

    async def put(self, settings: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure:
        try:
            payload: Final = json.dumps(settings)
        except (TypeError, ValueError) as error:
            return RedisFailure(kind="decode", message=f"UI settings are not JSON: {error}")
        result: Final = await self.transport.set(settings_key(), payload, ttl=ttl)
        return result if isinstance(result, RedisFailure) else None
