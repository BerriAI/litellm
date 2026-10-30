import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.redis.transport.contract import RedisFailure, RedisTransport


@with_config(ConfigDict(extra="allow", strict=True))
class CachedHealthCheckResults(TypedDict):
    healthy_endpoints: NotRequired[ReadOnly[list[Mapping[str, object]]]]
    unhealthy_endpoints: NotRequired[ReadOnly[list[Mapping[str, object]]]]
    timestamp: NotRequired[ReadOnly[int | float]]
    checked_by: NotRequired[ReadOnly[object]]


_RESULTS: Final = TypeAdapter(CachedHealthCheckResults)
_VALUE: Final = TypeAdapter(object)


@dataclass(frozen=True, slots=True)
class SharedHealthLockOwner:
    value: object


def lock_key() -> str:
    return "health_check_lock"


def results_key() -> str:
    return "health_check_results"


def model_lock_key(model_name: str) -> str:
    return f"{lock_key()}:{model_name}"


def model_results_key(model_name: str) -> str:
    return f"{results_key()}:{model_name}"


def _decode_cache_value(raw: bytes) -> object:
    decoded: Final = raw.decode("utf-8")
    try:
        return _VALUE.validate_python(json.loads(decoded))
    except ValueError:
        return _VALUE.validate_python(ast.literal_eval(decoded))


class SharedHealthStore(Protocol):
    async def acquire(self, owner: str, *, ttl: timedelta) -> bool | RedisFailure: ...

    async def owner(self) -> SharedHealthLockOwner | None | RedisFailure: ...

    async def release(self, owner: str) -> bool | RedisFailure: ...

    async def put(self, snapshot: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure: ...

    async def get(self) -> CachedHealthCheckResults | None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportSharedHealthStore:
    transport: RedisTransport

    async def acquire(self, owner: str, *, ttl: timedelta) -> bool | RedisFailure:
        if ttl <= timedelta(0):
            return RedisFailure(kind="server", message="invalid expire time in 'set' command")
        return await self.transport.set(lock_key(), json.dumps(owner), ttl=ttl, nx=True)

    async def owner(self) -> SharedHealthLockOwner | None | RedisFailure:
        raw: Final = await self.transport.get(lock_key())
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            value: Final = _decode_cache_value(raw)
            return None if value is None else SharedHealthLockOwner(value)
        except Exception as exc:
            return RedisFailure(kind="decode", message=f"shared health owner is invalid: {exc}")

    async def release(self, owner: str) -> bool | RedisFailure:
        held: Final = await self.owner()
        if isinstance(held, RedisFailure):
            return held
        if held is None or held.value != owner:
            return False
        deleted: Final = await self.transport.delete(lock_key())
        return deleted if isinstance(deleted, RedisFailure) else bool(deleted)

    async def put(self, snapshot: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure:
        if ttl <= timedelta(0):
            return RedisFailure(kind="server", message="invalid expire time in 'set' command")
        try:
            encoded: Final = json.dumps(safe_dumps(snapshot))
        except Exception as exc:
            return RedisFailure(kind="decode", message=f"shared health snapshot could not be encoded: {exc}")
        written: Final = await self.transport.set(results_key(), encoded, ttl=ttl)
        return written if isinstance(written, RedisFailure) else None

    async def get(self) -> CachedHealthCheckResults | None | RedisFailure:
        raw: Final = await self.transport.get(results_key())
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            value: Final = _decode_cache_value(raw)
            if value is None:
                return None
            return _RESULTS.validate_python(json.loads(value) if isinstance(value, str) else value)
        except Exception as exc:
            return RedisFailure(kind="decode", message=f"shared health snapshot is invalid: {exc}")
