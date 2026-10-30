import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport
from litellm.types.integrations.slack_alerting import OutageModel

_VALUE: Final = TypeAdapter(object)
_OUTAGE: Final = TypeAdapter(OutageModel)
_MAPPING: Final = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class DeploymentOutageState:
    outage: OutageModel
    original: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class InvalidDeploymentOutage:
    failure: RedisFailure
    error: ValidationError


def deployment_outage_key(deployment_id: str) -> str:
    return deployment_id


def deployment_outage_state(value: object) -> DeploymentOutageState:
    return DeploymentOutageState(
        _OUTAGE.validate_python(value, strict=True), _MAPPING.validate_python(value, strict=True)
    )


def deployment_outage_cache_value(state: DeploymentOutageState) -> Mapping[str, object]:
    return {**state.original, **state.outage}


class AlertDeploymentOutagesStore(Protocol):
    async def read_deployment_outage(
        self, deployment_id: str
    ) -> DeploymentOutageState | InvalidDeploymentOutage | None | RedisFailure: ...

    async def replace_deployment_outage(
        self, deployment_id: str, value: DeploymentOutageState, *, ttl: timedelta | None
    ) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportAlertDeploymentOutagesStore:
    transport: RedisTransport

    async def read_deployment_outage(
        self, deployment_id: str
    ) -> DeploymentOutageState | InvalidDeploymentOutage | None | RedisFailure:
        raw: Final = await self.transport.get(deployment_outage_key(deployment_id))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            return _outage_value(_VALUE.validate_json(raw))
        except ValidationError:
            try:
                return _outage_value(_VALUE.validate_python(ast.literal_eval(raw.decode("utf-8"))))
            except (UnicodeError, ValueError, SyntaxError, ValidationError):
                return RedisFailure(kind="decode", message="deployment outage is not a literal")

    async def replace_deployment_outage(
        self, deployment_id: str, value: DeploymentOutageState, *, ttl: timedelta | None
    ) -> None | RedisFailure:
        if ttl is not None and ttl <= timedelta(0):
            return RedisFailure(kind="server", message="invalid expiry for deployment outage")
        raw: Final = json.dumps(deployment_outage_cache_value(value)).encode("utf-8")
        reply: Final = await self.transport.set(deployment_outage_key(deployment_id), raw, ttl=ttl)
        return reply if isinstance(reply, RedisFailure) else None


def _outage_value(value: object) -> DeploymentOutageState | InvalidDeploymentOutage | None:
    if value is None:
        return None
    try:
        return deployment_outage_state(value)
    except ValidationError as error:
        return InvalidDeploymentOutage(RedisFailure(kind="decode", message="invalid deployment outage record"), error)
