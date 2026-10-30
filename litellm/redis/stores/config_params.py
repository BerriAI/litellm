import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm.redis.transport.contract import RedisFailure, RedisTransport

CONFIG_PARAM_MISS: Final = "__litellm_config_param_miss__"
_DOCUMENT: Final = TypeAdapter(object)
_ROW: Final = TypeAdapter(dict[str, object])


def config_param_key(param_name: str) -> str:
    return f"litellm_config:param:{param_name}"


def pack_config_row(row: object) -> Mapping[str, object]:
    return _ROW.validate_python(
        {
            **{field: getattr(row, field) for field in ("param_name", "param_value")},
            "reload_revision": getattr(row, "reload_revision", None),
        }
    )


def decode_config_param(raw: bytes) -> "CachedConfigParam | RedisFailure":
    try:
        decoded: Final = raw.decode("utf-8")
        try:
            return CachedConfigParam(_DOCUMENT.validate_python(json.loads(decoded)))
        except ValueError:
            return CachedConfigParam(_DOCUMENT.validate_python(ast.literal_eval(decoded)))
    except (ValueError, SyntaxError, UnicodeError) as error:
        return RedisFailure(kind="decode", message=str(error))


@dataclass(frozen=True, slots=True)
class CachedConfigParam:
    value: object


class ConfigParamsStore(Protocol):
    async def load(self, param_name: str) -> CachedConfigParam | None | RedisFailure: ...

    async def fill(
        self, param_name: str, row: Mapping[str, object] | None, *, ttl: timedelta
    ) -> None | RedisFailure: ...

    async def evict(self, param_name: str) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportConfigParamsStore:
    transport: RedisTransport

    async def load(self, param_name: str) -> CachedConfigParam | None | RedisFailure:
        raw: Final = await self.transport.get(config_param_key(param_name))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        value: Final = decode_config_param(raw)
        if isinstance(value, RedisFailure):
            return value
        return None if value.value is None else value

    async def fill(self, param_name: str, row: Mapping[str, object] | None, *, ttl: timedelta) -> None | RedisFailure:
        try:
            value: Final = json.dumps(row if row is not None else CONFIG_PARAM_MISS)
        except (TypeError, ValueError) as error:
            return RedisFailure(kind="decode", message=str(error))
        if ttl <= timedelta(0):
            return RedisFailure(kind="server", message="invalid expire time in set")
        result: Final = await self.transport.set(config_param_key(param_name), value, ttl=ttl)
        return result if isinstance(result, RedisFailure) else None

    async def evict(self, param_name: str) -> None | RedisFailure:
        result: Final = await self.transport.delete(config_param_key(param_name))
        return result if isinstance(result, RedisFailure) else None
