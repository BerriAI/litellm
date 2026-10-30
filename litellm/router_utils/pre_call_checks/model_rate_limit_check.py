"""
Enforce TPM/RPM or separate ITPM/OTPM rate limits set on model deployments.

When enabled via router_settings.optional_pre_call_checks: ["enforce_model_rate_limits"]

- tpm/rpm: combined TPM + optional RPM (legacy)
- itpm/otpm: separate input/output tokens per minute
- tpd/rpd: tokens and requests per UTC calendar day

When a deployment sets both itpm/otpm and tpm/rpm, both are enforced. A warning
is logged the first time such a deployment is seen.
"""

import contextlib
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import (
    ITPM_RESERVED_KEY,
    async_io_token_pre_call_check,
    async_io_token_reconcile_success,
    async_io_token_refund_failure,
    deployment_has_io_token_limits,
    get_io_token_rate_limit_request_kwargs,
    io_token_pre_call_check,
    io_token_reconcile_success,
    io_token_refund_failure,
)
from litellm.router_utils.routing_read_batch import ROUTER_USAGE_TARGET
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.router import RouterErrors, parse_daily_reset_time
from litellm.types.utils import StandardLoggingPayload
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


class RoutingArgs:
    ttl: int = 60  # 1min (RPM/TPM expire key)
    daily_ttl: int = 86400


DAY_FORMAT: Final = "%Y-%m-%d"


def daily_window_day(now: datetime, reset_offset: timedelta) -> str:
    return (now - reset_offset).strftime(DAY_FORMAT)


def seconds_until_daily_reset(now: datetime, reset_offset: timedelta = timedelta(0)) -> int:
    window_start: Final = (now - reset_offset).replace(hour=0, minute=0, second=0, microsecond=0)
    next_reset: Final = window_start + timedelta(days=1) + reset_offset
    return max(1, int((next_reset - now).total_seconds()))


class _DailyLimitParams(BaseModel):
    """rpd, tpd and daily_reset_time live in litellm_params only."""

    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)
    model: str | None = None
    tpd: int | None = None
    rpd: int | None = None
    daily_reset_time: str | None = None

    def reset_offset(self) -> timedelta:
        return parse_daily_reset_time(self.daily_reset_time)


class _DailyLimitModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)
    id: str | None = None


class _DailyLimitDeployment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)
    model_name: str = ""
    litellm_params: _DailyLimitParams = _DailyLimitParams()
    model_info: _DailyLimitModelInfo = _DailyLimitModelInfo()

    def limits(self) -> tuple[int | None, int | None]:
        return self.litellm_params.tpd, self.litellm_params.rpd

    def reset_offset(self) -> timedelta:
        return self.litellm_params.reset_offset()

    def cache_keys(self, now: datetime) -> tuple[str, str]:
        prefix: Final = f"{self.model_info.id}:{self.litellm_params.model}"
        current_day: Final = daily_window_day(now, self.reset_offset())
        return f"{prefix}:tpd:{current_day}", f"{prefix}:rpd:{current_day}"

    def limit_error(self, kind: Literal["tpd", "rpd"], limit: int, usage: int, now: datetime) -> litellm.RateLimitError:
        return litellm.RateLimitError(
            message=f"Model rate limit exceeded. {kind.upper()} limit={limit}, current usage={usage}",
            llm_provider="",
            model=self.litellm_params.model,
            response=httpx.Response(
                status_code=429,
                content=f"{RouterErrors.user_defined_ratelimit_error.value} {kind} limit={limit}. current usage={usage}. id={self.model_info.id}, model_group={self.model_name}",
                headers={"retry-after": str(seconds_until_daily_reset(now, self.reset_offset()))},
                request=httpx.Request(method="model_rate_limit_check", url="https://github.com/BerriAI/litellm"),
            ),
        )


def _reset_offset_from_kwargs(kwargs: Mapping[str, object]) -> timedelta:
    litellm_params: Final = kwargs.get("litellm_params")
    if litellm_params is None:
        return timedelta(0)
    return _DailyLimitParams.model_validate(litellm_params).reset_offset()


class ModelRateLimitingCheck(CustomLogger):
    """
    Pre-call check that enforces TPM/RPM or ITPM/OTPM limits on model deployments.

    This check runs before each request and raises a RateLimitError
    if the deployment has exceeded its configured TPM or RPM limits.

    Unlike the usage-based-routing strategy which uses limits for routing decisions,
    this check actively enforces those limits across ALL routing strategies.
    """

    def __init__(self, dual_cache: DualCache, clock: Callable[[], datetime] = get_utc_datetime):
        self.dual_cache = dual_cache
        self.clock = clock
        # model_ids already warned about itpm/otpm + tpm/rpm on the same deployment,
        # so the warning is logged once per deployment rather than per request.
        self._io_token_conflict_warned_ids: set[str] = set()

    def _increment(self, key: str, value: int, ttl: int) -> int:
        return self.dual_cache.increment_cache(key=key, value=value, ttl=ttl)

    async def _async_increment(
        self, key: str, value: int, ttl: int, parent_otel_span: Span | None = None
    ) -> int | None:
        count: Final = await self.dual_cache.async_increment_cache(
            key=key, value=value, ttl=ttl, parent_otel_span=parent_otel_span
        )
        return None if count is None else int(count)

    @staticmethod
    def _request_count_operations(
        rpm_key: str | None, rpd_key: str | None, increment_value: int
    ) -> tuple[RedisPipelineIncrementOperation, ...]:
        return tuple(
            RedisPipelineIncrementOperation(key=key, increment_value=increment_value, ttl=ttl)
            for key, ttl in ((rpm_key, RoutingArgs.ttl), (rpd_key, RoutingArgs.daily_ttl))
            if key is not None
        )

    async def _async_refund_request_counts(
        self, rpm_key: str | None, rpd_key: str | None, parent_otel_span: Span | None
    ) -> None:
        operations: Final = self._request_count_operations(rpm_key, rpd_key, -1)
        if operations:
            await self.dual_cache.async_increment_cache_pipeline(
                increment_list=list(operations), parent_otel_span=parent_otel_span
            )

    async def _async_increment_request_counts(
        self, rpm_key: str | None, rpd_key: str | None, parent_otel_span: Span | None
    ) -> tuple[int | None, int | None]:
        operations: Final = self._request_count_operations(rpm_key, rpd_key, 1)
        if not operations:
            return None, None
        counts: Final = await self.dual_cache.async_increment_cache_pipeline(
            increment_list=list(operations), parent_otel_span=parent_otel_span
        )
        if counts is None or len(counts) != len(operations):
            return None, None
        count_by_key: Final = {op["key"]: int(count) for op, count in zip(operations, counts)}
        return (
            None if rpm_key is None else count_by_key.get(rpm_key),
            None if rpd_key is None else count_by_key.get(rpd_key),
        )

    def _warn_io_token_and_tpm_rpm_coexist_once(self, deployment: dict) -> None:
        tpm_limit, rpm_limit = self._get_deployment_limits(deployment)
        if tpm_limit is None and rpm_limit is None:
            return
        model_id: Final = deployment.get("model_info", {}).get("id")
        # Dedup per deployment id; if there is no id (degenerate config) don't
        # collapse every such deployment onto one key - warn each time instead.
        if model_id is not None:
            if model_id in self._io_token_conflict_warned_ids:
                return
            self._io_token_conflict_warned_ids.add(str(model_id))
        verbose_router_logger.warning(
            "Deployment '%s' configures itpm/otpm alongside tpm/rpm; both limit types are enforced on this deployment",
            model_id,
        )

    def _refund_io_token_reservation_if_any(self) -> None:
        request_kwargs: Final = get_io_token_rate_limit_request_kwargs()
        if request_kwargs is not None:
            io_token_refund_failure(self.dual_cache, request_kwargs)

    async def _async_refund_io_token_reservation_if_any(
        self,
        parent_otel_span: Span | None = None,
    ) -> None:
        request_kwargs: Final = get_io_token_rate_limit_request_kwargs()
        if request_kwargs is not None:
            await async_io_token_refund_failure(
                self.dual_cache,
                request_kwargs,
                parent_otel_span=parent_otel_span,
            )

    def _get_deployment_limits(self, deployment: dict) -> tuple[int | None, int | None]:
        """
        Extract TPM and RPM limits from a deployment configuration.

        Checks in order:
        1. Top-level 'tpm'/'rpm' fields
        2. litellm_params.tpm/rpm
        3. model_info.tpm/rpm

        Returns:
            Tuple of (tpm_limit, rpm_limit)
        """
        # Check top-level
        tpm = deployment.get("tpm")
        rpm = deployment.get("rpm")

        # Check litellm_params
        if tpm is None:
            tpm = deployment.get("litellm_params", {}).get("tpm")
        if rpm is None:
            rpm = deployment.get("litellm_params", {}).get("rpm")

        # Check model_info
        if tpm is None:
            tpm = deployment.get("model_info", {}).get("tpm")
        if rpm is None:
            rpm = deployment.get("model_info", {}).get("rpm")

        return tpm, rpm

    def _check_daily_limits(self, deployment: _DailyLimitDeployment, now: datetime) -> None:
        tpd_limit, rpd_limit = deployment.limits()
        if tpd_limit is None and rpd_limit is None:
            return
        tpd_key, rpd_key = deployment.cache_keys(now)
        if tpd_limit is not None:
            current_tpd: Final = self._get_current_tpm(tpd_key, tpd_limit)
            if current_tpd is not None and current_tpd >= tpd_limit:
                raise deployment.limit_error("tpd", tpd_limit, current_tpd, now)
        if rpd_limit is not None:
            current_rpd: Final = self._increment(rpd_key, 1, RoutingArgs.daily_ttl)
            if current_rpd > rpd_limit:
                self._increment(rpd_key, -1, RoutingArgs.daily_ttl)
                raise deployment.limit_error("rpd", rpd_limit, current_rpd, now)

    def _get_cache_keys(self, deployment: dict, current_minute: str) -> tuple[str, str]:
        """Get the cache keys for TPM and RPM tracking."""
        model_id: Final = deployment.get("model_info", {}).get("id")
        deployment_name: Final = deployment.get("litellm_params", {}).get("model")

        tpm_key: Final = f"{model_id}:{deployment_name}:tpm:{current_minute}"
        rpm_key: Final = f"{model_id}:{deployment_name}:rpm:{current_minute}"

        return tpm_key, rpm_key

    @with_service_target(ROUTER_USAGE_TARGET)
    def _get_current_tpm(self, tpm_key: str, tpm_limit: int) -> int | None:
        local_tpm: Final = self.dual_cache.get_cache(key=tpm_key, local_only=True)
        redis_cache: Final = self.dual_cache.redis_cache
        if redis_cache is None or (local_tpm is not None and local_tpm >= tpm_limit):
            return local_tpm
        try:
            return redis_cache.get_cache(key=tpm_key)
        except RedisCircuitBreakerOpenError:
            return local_tpm

    @with_service_target(ROUTER_USAGE_TARGET)
    async def _async_get_current_tpm(self, tpm_key: str, tpm_limit: int, parent_otel_span: Span | None) -> int | None:
        local_tpm: Final = await self.dual_cache.async_get_cache(key=tpm_key, local_only=True)
        redis_cache: Final = self.dual_cache.redis_cache
        if redis_cache is None or (local_tpm is not None and local_tpm >= tpm_limit):
            return local_tpm
        try:
            return await redis_cache.async_get_cache(key=tpm_key, parent_otel_span=parent_otel_span)
        except RedisCircuitBreakerOpenError:
            return local_tpm

    @with_service_target(ROUTER_USAGE_TARGET)
    def pre_call_check(self, deployment: dict) -> dict | None:
        """
        Synchronous pre-call check for model rate limits.

        Raises RateLimitError if deployment exceeds TPM/RPM limits.
        """
        try:
            io_reservation_made = False
            if deployment_has_io_token_limits(deployment):
                self._warn_io_token_and_tpm_rpm_coexist_once(deployment)
                io_token_pre_call_check(
                    self.dual_cache,
                    deployment,
                )
                io_reservation_made = True

            dt: Final = self.clock()
            daily: Final = _DailyLimitDeployment.model_validate(deployment)
            tpm_limit, rpm_limit = self._get_deployment_limits(deployment)

            current_minute: Final = dt.strftime("%H-%M")
            tpm_key, rpm_key = self._get_cache_keys(deployment, current_minute)

            model_id: Final = deployment.get("model_info", {}).get("id")
            model_name: Final = deployment.get("litellm_params", {}).get("model")
            model_group: Final = deployment.get("model_name", "")

            # Check TPM limit
            if tpm_limit is not None:
                current_tpm: Final = self._get_current_tpm(tpm_key, tpm_limit)
                if current_tpm is not None and current_tpm >= tpm_limit:
                    raise litellm.RateLimitError(
                        message=f"Model rate limit exceeded. TPM limit={tpm_limit}, current usage={current_tpm}",
                        llm_provider="",
                        model=model_name,
                        response=httpx.Response(
                            status_code=429,
                            content=f"{RouterErrors.user_defined_ratelimit_error.value} tpm limit={tpm_limit}. current usage={current_tpm}. id={model_id}, model_group={model_group}",
                            headers={"retry-after": str(60)},
                            request=httpx.Request(
                                method="model_rate_limit_check",
                                url="https://github.com/BerriAI/litellm",
                            ),
                        ),
                    )

            # Check RPM limit (atomic increment-first to avoid race conditions)
            if rpm_limit is not None:
                current_rpm: Final = self._increment(rpm_key, 1, RoutingArgs.ttl)
                if current_rpm is not None and current_rpm > rpm_limit:
                    self._increment(rpm_key, -1, RoutingArgs.ttl)
                    raise litellm.RateLimitError(
                        message=f"Model rate limit exceeded. RPM limit={rpm_limit}, current usage={current_rpm}",
                        llm_provider="",
                        model=model_name,
                        response=httpx.Response(
                            status_code=429,
                            content=f"{RouterErrors.user_defined_ratelimit_error.value} rpm limit={rpm_limit}. current usage={current_rpm}. id={model_id}, model_group={model_group}",
                            headers={"retry-after": str(60)},
                            request=httpx.Request(
                                method="model_rate_limit_check",
                                url="https://github.com/BerriAI/litellm",
                            ),
                        ),
                    )

            try:
                self._check_daily_limits(daily, dt)
            except litellm.RateLimitError:
                if rpm_limit is not None:
                    self._increment(rpm_key, -1, RoutingArgs.ttl)
                raise
            return deployment

        except litellm.RateLimitError:
            if io_reservation_made:
                self._refund_io_token_reservation_if_any()
            raise
        except Exception as e:
            verbose_router_logger.debug("Error in ModelRateLimitingCheck.pre_call_check: %s", e)
            # Don't fail the request if rate limit check fails
            return deployment

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_pre_call_check(self, deployment: dict, parent_otel_span: Span | None = None) -> dict | None:
        """
        Async pre-call check for model rate limits.

        Raises RateLimitError if deployment exceeds TPM/RPM or ITPM/OTPM limits.
        """
        try:
            io_reservation_made = False
            if deployment_has_io_token_limits(deployment):
                self._warn_io_token_and_tpm_rpm_coexist_once(deployment)
                await async_io_token_pre_call_check(
                    self.dual_cache,
                    deployment,
                    parent_otel_span=parent_otel_span,
                )
                io_reservation_made = True

            dt: Final = self.clock()
            daily: Final = _DailyLimitDeployment.model_validate(deployment)
            tpd_limit, rpd_limit = daily.limits()
            tpm_limit, rpm_limit = self._get_deployment_limits(deployment)

            current_minute: Final = dt.strftime("%H-%M")
            tpm_key, rpm_key = self._get_cache_keys(deployment, current_minute)
            tpd_key, rpd_key = daily.cache_keys(dt)

            model_id: Final = deployment.get("model_info", {}).get("id")
            model_name: Final = deployment.get("litellm_params", {}).get("model")
            model_group: Final = deployment.get("model_name", "")

            # Check TPM limit
            if tpm_limit is not None:
                current_tpm: Final = await self._async_get_current_tpm(tpm_key, tpm_limit, parent_otel_span)
                if current_tpm is not None and current_tpm >= tpm_limit:
                    raise litellm.RateLimitError(
                        message=f"Model rate limit exceeded. TPM limit={tpm_limit}, current usage={current_tpm}",
                        llm_provider="",
                        model=model_name,
                        response=httpx.Response(
                            status_code=429,
                            content=f"{RouterErrors.user_defined_ratelimit_error.value} tpm limit={tpm_limit}. current usage={current_tpm}. id={model_id}, model_group={model_group}",
                            headers={"retry-after": str(60)},
                            request=httpx.Request(
                                method="model_rate_limit_check",
                                url="https://github.com/BerriAI/litellm",
                            ),
                        ),
                        num_retries=0,  # Don't retry - return 429 immediately
                    )

            if tpd_limit is not None:
                current_tpd: Final = await self._async_get_current_tpm(tpd_key, tpd_limit, parent_otel_span)
                if current_tpd is not None and current_tpd >= tpd_limit:
                    raise daily.limit_error("tpd", tpd_limit, current_tpd, dt)

            charged_rpm_key: Final = rpm_key if rpm_limit is not None else None
            charged_rpd_key: Final = rpd_key if rpd_limit is not None else None
            current_rpm, current_rpd = await self._async_increment_request_counts(
                charged_rpm_key, charged_rpd_key, parent_otel_span
            )
            if rpm_limit is not None and current_rpm is not None and current_rpm > rpm_limit:
                await self._async_refund_request_counts(charged_rpm_key, charged_rpd_key, parent_otel_span)
                raise litellm.RateLimitError(
                    message=f"Model rate limit exceeded. RPM limit={rpm_limit}, current usage={current_rpm}",
                    llm_provider="",
                    model=model_name,
                    response=httpx.Response(
                        status_code=429,
                        content=f"{RouterErrors.user_defined_ratelimit_error.value} rpm limit={rpm_limit}. current usage={current_rpm}. id={model_id}, model_group={model_group}",
                        headers={"retry-after": str(60)},
                        request=httpx.Request(
                            method="model_rate_limit_check",
                            url="https://github.com/BerriAI/litellm",
                        ),
                    ),
                    num_retries=0,  # Don't retry - return 429 immediately
                )

            if rpd_limit is not None and current_rpd is not None and current_rpd > rpd_limit:
                await self._async_refund_request_counts(charged_rpm_key, charged_rpd_key, parent_otel_span)
                raise daily.limit_error("rpd", rpd_limit, current_rpd, dt)

            return deployment

        except litellm.RateLimitError:
            if io_reservation_made:
                await self._async_refund_io_token_reservation_if_any(parent_otel_span=parent_otel_span)
            raise
        except Exception as e:
            verbose_router_logger.debug("Error in ModelRateLimitingCheck.async_pre_call_check: %s", e)
            # Don't fail the request if rate limit check fails
            return deployment

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.litellm_core_utils.core_helpers import (
            get_parent_otel_span_from_kwargs,
        )

        try:
            standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")

            # IO token reconciliation works purely from the cache keys stashed in
            # kwargs/metadata, so it must run before the model_id guard below
            # (which only the TPM-tracking path needs). Otherwise a request whose
            # standard_logging_object lacks model_id would never return its
            # reservation, leaving the counter elevated until the TTL expires.
            slo_metadata: Final = (standard_logging_object.get("metadata") or {}) if standard_logging_object else {}
            kwargs_metadata: Final = kwargs.get("metadata") or {}
            if ITPM_RESERVED_KEY in slo_metadata or ITPM_RESERVED_KEY in kwargs_metadata:
                await async_io_token_reconcile_success(
                    self.dual_cache,
                    kwargs,
                    response_obj,
                    parent_otel_span=get_parent_otel_span_from_kwargs(kwargs),
                )
                # Fall through: a deployment can also configure tpm/rpm alongside
                # itpm/otpm, and that path's pre-call check reads the tpm_key
                # counter tracked below, so it must still be incremented here.

            if standard_logging_object is None:
                return

            model_id: Final = standard_logging_object.get("model_id")
            if model_id is None:
                return

            total_tokens: Final = standard_logging_object.get("total_tokens", 0)
            model: Final = standard_logging_object.get("hidden_params", {}).get("litellm_model_name")

            verbose_router_logger.debug(
                "[TPM TRACKING] model_id=%s, total_tokens=%s, model=%s", model_id, total_tokens, model
            )

            if not model or not total_tokens:
                return

            dt: Final = self.clock()
            current_minute: Final = dt.strftime("%H-%M")
            tpm_key: Final = f"{model_id}:{model}:tpm:{current_minute}"
            reset_offset: Final = _reset_offset_from_kwargs(kwargs)  # pyright: ignore[reportUnknownArgumentType]  # CustomLogger passes an untyped kwargs dict
            tpd_key: Final = f"{model_id}:{model}:tpd:{daily_window_day(dt, reset_offset)}"

            verbose_router_logger.debug("[TPM TRACKING] Incrementing %s by %s", tpm_key, total_tokens)

            tokens: Final = int(total_tokens)
            await self._async_increment(tpm_key, tokens, RoutingArgs.ttl)
            await self._async_increment(tpd_key, tokens, RoutingArgs.daily_ttl)

        except Exception as e:
            verbose_router_logger.debug("Error in ModelRateLimitingCheck.async_log_success_event: %s", e)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.litellm_core_utils.core_helpers import (
            get_parent_otel_span_from_kwargs,
        )

        # Never fail the primary logging pipeline over an io-token refund error.
        with contextlib.suppress(Exception):
            await async_io_token_refund_failure(
                self.dual_cache,
                kwargs,
                parent_otel_span=get_parent_otel_span_from_kwargs(kwargs),
            )

    @with_service_target(ROUTER_USAGE_TARGET)
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Sync version of tracking TPM usage after successful request.
        Always tracks tokens - the pre-call check handles enforcement.
        """
        try:
            standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")
            slo_metadata: Final = (standard_logging_object.get("metadata") or {}) if standard_logging_object else {}
            kwargs_metadata: Final = kwargs.get("metadata") or {}
            if ITPM_RESERVED_KEY in slo_metadata or ITPM_RESERVED_KEY in kwargs_metadata:
                io_token_reconcile_success(
                    self.dual_cache,
                    kwargs,
                    response_obj,
                )
                # Fall through: a deployment can also configure tpm/rpm alongside
                # itpm/otpm, and that path's pre-call check reads the tpm_key
                # counter tracked below, so it must still be incremented here.

            if standard_logging_object is None:
                return

            model_id: Final = standard_logging_object.get("model_id")
            if model_id is None:
                return

            total_tokens: Final = standard_logging_object.get("total_tokens", 0)
            model: Final = standard_logging_object.get("hidden_params", {}).get("litellm_model_name")

            if not model or not total_tokens:
                return

            dt: Final = self.clock()
            current_minute: Final = dt.strftime("%H-%M")
            tpm_key: Final = f"{model_id}:{model}:tpm:{current_minute}"
            reset_offset: Final = _reset_offset_from_kwargs(kwargs)  # pyright: ignore[reportUnknownArgumentType]  # CustomLogger passes an untyped kwargs dict
            tpd_key: Final = f"{model_id}:{model}:tpd:{daily_window_day(dt, reset_offset)}"

            tokens: Final = int(total_tokens)
            self._increment(tpm_key, tokens, RoutingArgs.ttl)
            self._increment(tpd_key, tokens, RoutingArgs.daily_ttl)

        except Exception as e:
            verbose_router_logger.debug("Error in ModelRateLimitingCheck.log_success_event: %s", e)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        with contextlib.suppress(Exception):
            io_token_refund_failure(
                self.dual_cache,
                kwargs,
            )
