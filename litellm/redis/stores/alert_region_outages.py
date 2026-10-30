import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport
from litellm.types.integrations.slack_alerting import ProviderRegionOutageModel

_VALUE: Final = TypeAdapter(object)
_OUTAGE: Final = TypeAdapter(ProviderRegionOutageModel)
_MAPPING: Final = TypeAdapter(Mapping[str, object])
_IDS: Final = TypeAdapter(tuple[str, ...])


@dataclass(frozen=True, slots=True)
class RegionOutage:
    provider: str
    region: str


@dataclass(frozen=True, slots=True)
class InvalidRegionOutage:
    failure: RedisFailure
    error: ValidationError


@dataclass(frozen=True, slots=True)
class RegionOutageState:
    outage: ProviderRegionOutageModel
    original: Mapping[str, object]


def region_outage_cache_value(state: RegionOutageState) -> Mapping[str, object]:
    return {**state.original, **prepare_region_outage(state.outage)}


def region_outage_key(region: RegionOutage) -> str:
    return region.provider + region.region


def prepare_region_outage(value: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: list(_IDS.validate_python(item)) if key == "deployment_ids" and isinstance(item, set) else item
        for key, item in value.items()
    }


def restore_region_outage(value: object) -> ProviderRegionOutageModel:
    mapping: Final = _MAPPING.validate_python(value, strict=True)
    restored: Final = {
        key: set(_IDS.validate_python(item)) if key == "deployment_ids" and isinstance(item, list) else item
        for key, item in mapping.items()
    }
    return _OUTAGE.validate_python(restored, strict=True)


class AlertRegionOutagesStore(Protocol):
    async def read_region_outage(
        self, region: RegionOutage
    ) -> RegionOutageState | InvalidRegionOutage | None | RedisFailure: ...

    async def replace_region_outage(
        self, region: RegionOutage, value: RegionOutageState, *, ttl: timedelta | None
    ) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportAlertRegionOutagesStore:
    transport: RedisTransport

    async def read_region_outage(
        self, region: RegionOutage
    ) -> RegionOutageState | InvalidRegionOutage | None | RedisFailure:
        raw: Final = await self.transport.get(region_outage_key(region))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            return _outage_value(_VALUE.validate_json(raw))
        except ValidationError:
            try:
                return _outage_value(_VALUE.validate_python(ast.literal_eval(raw.decode("utf-8"))))
            except (UnicodeError, ValueError, SyntaxError, ValidationError):
                return RedisFailure(kind="decode", message="region outage is not a literal")

    async def replace_region_outage(
        self, region: RegionOutage, value: RegionOutageState, *, ttl: timedelta | None
    ) -> None | RedisFailure:
        if ttl is not None and ttl <= timedelta(0):
            return RedisFailure(kind="server", message="invalid expiry for region outage")
        raw: Final = json.dumps(region_outage_cache_value(value)).encode("utf-8")
        reply: Final = await self.transport.set(region_outage_key(region), raw, ttl=ttl)
        return reply if isinstance(reply, RedisFailure) else None


def region_outage_state(value: object) -> RegionOutageState:
    return RegionOutageState(restore_region_outage(value), _MAPPING.validate_python(value, strict=True))


def _outage_value(value: object) -> RegionOutageState | InvalidRegionOutage | None:
    if value is None:
        return None
    try:
        return region_outage_state(value)
    except ValidationError as error:
        return InvalidRegionOutage(RedisFailure(kind="decode", message="invalid region outage record"), error)
