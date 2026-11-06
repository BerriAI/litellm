"""The one writer of the per-deployment ``{id}:{model}:rpm|tpm:{HH-MM}`` usage counters that
usage-based-routing-v2 selection and ``enforce_model_rate_limits`` read.

The router charges one request before the routing pre-call checks run, hands it back when a check
refuses the attempt, and charges the tokens of each logged success. Nothing else writes these keys.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Final

from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.batch_utils import is_batch_retrieve_call_type
from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.router_utils.pre_call_checks.model_rate_limit_check import ModelRateLimitingCheck
from litellm.router_utils.routing_read_batch import ROUTER_USAGE_TARGET
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.utils import StandardLoggingPayload
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from opentelemetry.trace import Span

USAGE_TTL: Final = 60


def reads_deployment_usage(callbacks: Iterable[object]) -> bool:
    """Whether one of ``callbacks`` reads the per-deployment usage counters: usage-based-routing-v2
    selection or ``enforce_model_rate_limits``. Without a reader there is nothing to count."""
    return any(isinstance(callback, LowestTPMLoggingHandler_v2 | ModelRateLimitingCheck) for callback in callbacks)


def usage_counter_keys(deployment: Mapping[str, object], minute: str) -> tuple[str, str]:
    """The ``(tpm_key, rpm_key)`` a deployment's usage is counted under this minute."""
    model_info: Final = deployment.get("model_info")
    params: Final = deployment.get("litellm_params")
    model_id: Final = model_info.get("id") if isinstance(model_info, Mapping) else None
    model: Final = params.get("model") if isinstance(params, Mapping) else None
    return f"{model_id}:{model}:tpm:{minute}", f"{model_id}:{model}:rpm:{minute}"


class DeploymentUsageCounter(CustomLogger):
    """Counts requests and tokens against a deployment in the router's dual cache, through one Redis
    pipeline per write so every worker then holds the shared total. Requests are charged by the
    pre-call checks that run a reader; ``charges`` says whether a registered callback reads the
    counters, so a logged success counts its tokens only then."""

    def __init__(
        self,
        dual_cache: DualCache,
        charges: Callable[[], bool] = lambda: True,
        clock: Callable[[], datetime] = get_utc_datetime,
    ) -> None:
        super().__init__()
        self.dual_cache: Final = dual_cache
        self.charges: Final = charges
        self.clock: Final = clock

    def _minute(self) -> str:
        return self.clock().strftime("%H-%M")

    def rpm_key(self, deployment: Mapping[str, object]) -> str:
        return usage_counter_keys(deployment, self._minute())[1]

    @with_service_target(ROUTER_USAGE_TARGET)
    async def charge_request(
        self, deployment: Mapping[str, object], parent_otel_span: Span | None = None
    ) -> int | None:
        """One more request this minute; returns the count, or None when the cache could not say."""
        return await self._async_increment(self.rpm_key(deployment), 1, parent_otel_span)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def refund_request(self, deployment: Mapping[str, object], parent_otel_span: Span | None = None) -> None:
        """A refused attempt does not count."""
        _ = await self._async_increment(self.rpm_key(deployment), -1, parent_otel_span)

    @with_service_target(ROUTER_USAGE_TARGET)
    def charge_request_sync(self, deployment: Mapping[str, object]) -> int | None:
        return self._increment(self.rpm_key(deployment), 1)

    @with_service_target(ROUTER_USAGE_TARGET)
    def refund_request_sync(self, deployment: Mapping[str, object]) -> None:
        _ = self._increment(self.rpm_key(deployment), -1)

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
            await self.dual_cache.async_set_cache(key, count, local_only=True, ttl=USAGE_TTL)  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
        return count

    def _increment(self, key: str, value: int) -> int | None:
        try:
            count: Final = int(self.dual_cache.increment_cache(key=key, value=value, ttl=USAGE_TTL))
        except Exception as e:  # noqa: BLE001  # see _async_increment
            verbose_router_logger.debug("DeploymentUsageCounter: usage was not recorded. Got - %s", e)
            return None
        if self.dual_cache.redis_cache is not None:
            self.dual_cache.set_cache(key, count, local_only=True, ttl=USAGE_TTL)  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
        return count

    def _logged_tokens(self, kwargs: Mapping[str, object]) -> tuple[str, int] | None:
        """The ``(tpm_key prefix, total_tokens)`` of a logged success, or None when it counts nothing."""
        if is_batch_retrieve_call_type(kwargs.get("call_type")) or not self.charges():
            return None
        logged: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")  # pyright: ignore[reportAssignmentType]  # the logging payload as CustomLogger receives it
        if logged is None:
            return None
        model_id: Final = logged.get("model_id")
        model: Final = logged.get("hidden_params", {}).get("litellm_model_name")
        tokens: Final = logged.get("total_tokens")
        if model_id is None or not model or not tokens:
            return None
        return f"{model_id}:{model}", int(tokens)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        counted: Final = self._logged_tokens(kwargs)
        if counted is None:
            return
        prefix, tokens = counted
        _ = await self._async_increment(
            f"{prefix}:tpm:{self._minute()}", tokens, get_parent_otel_span_from_kwargs(kwargs)
        )

    @with_service_target(ROUTER_USAGE_TARGET)
    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        counted: Final = self._logged_tokens(kwargs)
        if counted is None:
            return
        prefix, tokens = counted
        _ = self._increment(f"{prefix}:tpm:{self._minute()}", tokens)


__all__ = ["USAGE_TTL", "DeploymentUsageCounter", "reads_deployment_usage", "usage_counter_keys"]
