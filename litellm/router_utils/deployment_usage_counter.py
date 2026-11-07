"""The one writer of the per-deployment ``{id}:{model}:rpm|tpm:{HH-MM}`` usage counters that
usage-based-routing-v2 selection and ``enforce_model_rate_limits`` read.

The router charges one request before the routing pre-call checks run and hands it back when a check
refuses the attempt. The tokens of a served call follow the router's own usage write, as the twin of its
``global_router:`` tpm increment (``charge_tokens``). Nothing else writes these keys.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, TypeAdapter

from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.redis.stores.router_usage import ROUTER_USAGE_TARGET
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.router import RouterCacheEnum
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from opentelemetry.trace import Span

USAGE_TTL: Final = 60
_STR_MAP: Final = TypeAdapter(Mapping[str, object])
_ROUTER_USAGE_PREFIX: Final = RouterCacheEnum.TPM.value.split("{", 1)[0]


@dataclass(frozen=True, slots=True)
class UsageCharge:
    """One request charged to a deployment's rpm counter: the key it went to, and what the counter
    read right after, None when the cache could not say."""

    key: str
    count: int | None


_ADMITTED: Final[ContextVar[UsageCharge | None]] = ContextVar("deployment_usage_charge", default=None)
_SERVED_FROM_CACHE: Final[ContextVar[bool]] = ContextVar("deployment_usage_served_from_cache", default=False)


def note_served_from_cache(cache_hit: bool) -> None:
    """A response the cache answered used no deployment tokens; the usage write that follows in this
    attempt's context leaves the enforcement counters alone (``usage_twins``)."""
    _SERVED_FROM_CACHE.set(cache_hit)


def charged_rpm(rpm_key: str) -> int | None:
    """What this attempt's own charge counted on ``rpm_key``, so a check compares the atomic
    increment it was admitted with rather than a cache read another attempt may have overtaken."""
    charge: Final = _ADMITTED.get()
    return charge.count if charge is not None and charge.key == rpm_key else None


def end_attempt(charge: UsageCharge) -> None:
    if _ADMITTED.get() is charge:
        _ADMITTED.set(None)


def reads_deployment_usage(callbacks: Iterable[object]) -> bool:
    """Whether one of ``callbacks`` reads the per-deployment usage counters (it says so with a
    ``reads_deployment_usage`` attribute). Without a reader there is nothing to count."""
    return any(getattr(callback, "reads_deployment_usage", False) is True for callback in callbacks)


def _fields(value: object) -> Mapping[str, object]:
    """A deployment section as a mapping, whether it came as a dict or as its pydantic model."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    return _STR_MAP.validate_python(value or {})


def usage_counter_keys(deployment: Mapping[str, object], minute: str) -> tuple[str, str]:
    """The ``(tpm_key, rpm_key)`` a deployment's usage is counted under this minute."""
    model_id: Final = _fields(deployment.get("model_info")).get("id")
    model: Final = _fields(deployment.get("litellm_params")).get("model")
    return f"{model_id}:{model}:tpm:{minute}", f"{model_id}:{model}:rpm:{minute}"


def enforcement_tpm_operation(operation: RedisPipelineIncrementOperation) -> RedisPipelineIncrementOperation | None:
    """The ``{id}:{model}:tpm:{HH-MM}`` increment that mirrors a ``global_router:`` tpm increment of the
    router's usage write, None for any other key."""
    key: Final = operation["key"]
    if not key.startswith(_ROUTER_USAGE_PREFIX) or ":tpm:" not in key:
        return None
    return RedisPipelineIncrementOperation(
        key=key.removeprefix(_ROUTER_USAGE_PREFIX), increment_value=operation["increment_value"], ttl=USAGE_TTL
    )


def usage_twins(
    operations: Iterable[RedisPipelineIncrementOperation], counted: bool
) -> tuple[RedisPipelineIncrementOperation, ...]:
    """The enforcement increments that ride the router's usage write, none when nothing reads them or
    when the cache answered the call."""
    if not counted or _SERVED_FROM_CACHE.get():
        return ()
    return tuple(twin for twin in map(enforcement_tpm_operation, operations) if twin is not None)


class DeploymentUsageCounter:
    """Counts requests against a deployment in the router's dual cache, through one Redis pipeline per
    write so every worker then holds the shared total."""

    def __init__(self, dual_cache: DualCache, clock: Callable[[], datetime] = get_utc_datetime) -> None:
        self.dual_cache: Final = dual_cache
        self.clock: Final = clock

    def _minute(self) -> str:
        return self.clock().strftime("%H-%M")

    def rpm_key(self, deployment: Mapping[str, object]) -> str:
        return usage_counter_keys(deployment, self._minute())[1]

    @with_service_target(ROUTER_USAGE_TARGET)
    async def charge_request(
        self, deployment: Mapping[str, object], parent_otel_span: Span | None = None
    ) -> UsageCharge:
        """One more request this minute, remembered for the checks of this attempt (``charged_rpm``)."""
        key: Final = self.rpm_key(deployment)
        charge: Final = UsageCharge(key, await self._async_increment(key, 1, parent_otel_span))
        _ADMITTED.set(charge)
        return charge

    @with_service_target(ROUTER_USAGE_TARGET)
    async def refund_request(self, charge: UsageCharge, parent_otel_span: Span | None = None) -> None:
        """A refused attempt does not count, on the minute it was charged to."""
        _ = await self._async_increment(charge.key, -1, parent_otel_span)
        end_attempt(charge)

    @with_service_target(ROUTER_USAGE_TARGET)
    def charge_request_sync(self, deployment: Mapping[str, object]) -> UsageCharge:
        key: Final = self.rpm_key(deployment)
        charge: Final = UsageCharge(key, self._increment(key, 1))
        _ADMITTED.set(charge)
        return charge

    @with_service_target(ROUTER_USAGE_TARGET)
    def refund_request_sync(self, charge: UsageCharge) -> None:
        _ = self._increment(charge.key, -1)
        end_attempt(charge)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def charge_tokens(
        self,
        operations: Iterable[RedisPipelineIncrementOperation],
        counted: bool,
        parent_otel_span: Span | None = None,
    ) -> None:
        """The served call's tokens, as the twins of the router's ``global_router:`` tpm increments: counted
        in this worker's memory now, in Redis on the request's post-call pipeline."""
        twins: Final = usage_twins(operations, counted)
        if not twins:
            return
        try:
            await self.dual_cache.async_increment_cache_pipeline_post_call(twins, parent_otel_span)
        except Exception as e:  # noqa: BLE001  # see _async_increment
            verbose_router_logger.debug("DeploymentUsageCounter: tokens were not recorded. Got - %s", e)

    def _held(self, key: str, value: int, count: int) -> int:
        """What this worker's memory keeps: a charge never lowers what a later charge already wrote."""
        held: Final = self.dual_cache.in_memory_cache.get_cache(key)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # InMemoryCache reads are untyped
        return max(count, int(held)) if value > 0 and isinstance(held, int | float) else count

    async def _async_increment(self, key: str, value: int, parent_otel_span: Span | None) -> int | None:
        operation: Final = RedisPipelineIncrementOperation(key=key, increment_value=value, ttl=USAGE_TTL)
        try:
            values: Final = await self.dual_cache.async_increment_cache_pipeline(
                increment_list=[operation], parent_otel_span=parent_otel_span
            )
        except Exception as e:  # noqa: BLE001  # a cache that cannot count never fails the call
            verbose_router_logger.debug("DeploymentUsageCounter: usage was not recorded. Got - %s", e)
            return None
        if values is None or len(values) != 1:
            return None
        count: Final = int(values[0])
        if self.dual_cache.redis_cache is not None:
            await self.dual_cache.async_set_cache(key, self._held(key, value, count), local_only=True, ttl=USAGE_TTL)  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
        return count

    def _increment(self, key: str, value: int) -> int | None:
        try:
            count: Final = int(
                self.dual_cache.increment_cache(key=key, value=value, ttl=USAGE_TTL)  # pyright: ignore[reportUnknownMemberType]  # DualCache.increment_cache is untyped
            )
        except Exception as e:  # noqa: BLE001  # see _async_increment
            verbose_router_logger.debug("DeploymentUsageCounter: usage was not recorded. Got - %s", e)
            return None
        if self.dual_cache.redis_cache is not None:
            self.dual_cache.set_cache(key, self._held(key, value, count), local_only=True, ttl=USAGE_TTL)  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
        return count


__all__ = [
    "ROUTER_USAGE_TARGET",
    "USAGE_TTL",
    "DeploymentUsageCounter",
    "UsageCharge",
    "charged_rpm",
    "end_attempt",
    "enforcement_tpm_operation",
    "note_served_from_cache",
    "reads_deployment_usage",
    "usage_counter_keys",
    "usage_twins",
]
