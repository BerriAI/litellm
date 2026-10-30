import ast
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_VALUE: Final = TypeAdapter(object)
FAILED_REQUESTS_SUFFIX: Final = "failed_requests_daily_metrics"
LATENCY_SUFFIX: Final = "latency_daily_metrics"


@dataclass(frozen=True, slots=True)
class DeploymentMetric:
    deployment_id: str
    kind: Literal["failed_requests", "latency"]


@dataclass(frozen=True, slots=True)
class NonNumericMetric:
    pass


def metric_key(metric: DeploymentMetric) -> str:
    suffix: Final = FAILED_REQUESTS_SUFFIX if metric.kind == "failed_requests" else LATENCY_SUFFIX
    return f"{metric.deployment_id}:{suffix}"


def metric_number(value: object) -> int | float | None:
    return value if isinstance(value, (int, float)) else None


class AlertDailyMetricsStore(Protocol):
    async def add_deployment_metrics(
        self, metric: DeploymentMetric, amount: float, *, ttl: timedelta | None
    ) -> float | RedisFailure: ...

    async def read_deployment_metrics(
        self, metrics: Sequence[DeploymentMetric]
    ) -> tuple[int | float | NonNumericMetric | None, ...] | RedisFailure: ...

    async def reset_deployment_metrics(
        self, metrics: Sequence[DeploymentMetric], *, ttl: timedelta | None
    ) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportAlertDailyMetricsStore:
    transport: RedisTransport

    async def add_deployment_metrics(
        self, metric: DeploymentMetric, amount: float, *, ttl: timedelta | None
    ) -> float | RedisFailure:
        key: Final = metric_key(metric)
        if ttl is None:
            return await self.transport.incrbyfloat(key, amount)
        batch: Final = self.transport.batch()
        increment: Final = batch.incrbyfloat(key, amount)
        expiry: Final = batch.ttl(key)
        result: Final = await increment
        remaining: Final = await expiry
        if isinstance(result, RedisFailure):
            return result
        if isinstance(remaining, RedisFailure):
            return remaining
        if remaining == -1:
            expired: Final = (
                await self.transport.delete(key) if ttl <= timedelta(0) else await self.transport.expire(key, ttl)
            )
            if isinstance(expired, RedisFailure):
                return expired
        return result

    async def read_deployment_metrics(
        self, metrics: Sequence[DeploymentMetric]
    ) -> tuple[int | float | NonNumericMetric | None, ...] | RedisFailure:
        if not metrics:
            return ()
        raw: Final = await self.transport.mget(tuple(metric_key(metric) for metric in metrics))
        if isinstance(raw, RedisFailure):
            return raw
        values: Final = tuple(_decode(value) for value in raw)
        failure: Final = next((value for value in values if isinstance(value, RedisFailure)), None)
        if failure is not None:
            return tuple(None if value is None else NonNumericMetric() for value in raw)
        return tuple(value for value in values if not isinstance(value, RedisFailure))

    async def reset_deployment_metrics(
        self, metrics: Sequence[DeploymentMetric], *, ttl: timedelta | None
    ) -> None | RedisFailure:
        if ttl is not None and ttl <= timedelta(0):
            return RedisFailure(kind="server", message="invalid expiry for deployment metric reset")
        batch: Final = self.transport.batch()
        pending: Final = tuple(batch.set(metric_key(metric), "0", ttl=ttl) for metric in metrics)
        replies: Final = tuple([await reply for reply in pending])
        return next((reply for reply in replies if isinstance(reply, RedisFailure)), None)


def _decode(raw: bytes | None) -> int | float | NonNumericMetric | None | RedisFailure:
    if raw is None:
        return None
    try:
        return _metric_value(_VALUE.validate_json(raw))
    except ValidationError:
        try:
            return _metric_value(_VALUE.validate_python(ast.literal_eval(raw.decode("utf-8"))))
        except (ValueError, SyntaxError, UnicodeError):
            return RedisFailure(kind="decode", message="daily deployment metric is not a literal")


def _metric_value(value: object) -> int | float | NonNumericMetric | None:
    if value is None or isinstance(value, (int, float)):
        return value
    return NonNumericMetric()
