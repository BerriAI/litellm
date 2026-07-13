"""
Enforce TPM/RPM or separate ITPM/OTPM rate limits set on model deployments.

When enabled via router_settings.optional_pre_call_checks: ["enforce_model_rate_limits"]

- tpm/rpm: combined TPM + optional RPM (legacy)
- itpm/otpm: separate input/output tokens per minute

When a deployment sets both itpm/otpm and tpm/rpm, both are enforced. A warning
is logged the first time such a deployment is seen.
"""

import contextlib
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import (
    async_io_token_pre_call_check,
    async_io_token_reconcile_success,
    async_io_token_refund_failure,
    deployment_has_io_token_limits,
    get_io_token_rate_limit_request_kwargs,
    io_token_pre_call_check,
    io_token_reconcile_success,
    io_token_refund_failure,
    ITPM_RESERVED_KEY,
)
from litellm.types.router import RouterErrors
from litellm.types.utils import StandardLoggingPayload
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class RoutingArgs:
    ttl: int = 60  # 1min (RPM/TPM expire key)


class ModelRateLimitingCheck(CustomLogger):
    """
    Pre-call check that enforces TPM/RPM or ITPM/OTPM limits on model deployments.

    This check runs before each request and raises a RateLimitError
    if the deployment has exceeded its configured TPM or RPM limits.

    Unlike the usage-based-routing strategy which uses limits for routing decisions,
    this check actively enforces those limits across ALL routing strategies.
    """

    def __init__(self, dual_cache: DualCache):
        self.dual_cache = dual_cache
        # model_ids already warned about itpm/otpm + tpm/rpm on the same deployment,
        # so the warning is logged once per deployment rather than per request.
        self._io_token_conflict_warned_ids: set[str] = set()

    def _warn_io_token_and_tpm_rpm_coexist_once(self, deployment: dict) -> None:
        tpm_limit, rpm_limit = self._get_deployment_limits(deployment)
        if tpm_limit is None and rpm_limit is None:
            return
        model_id = deployment.get("model_info", {}).get("id")
        # Dedup per deployment id; if there is no id (degenerate config) don't
        # collapse every such deployment onto one key - warn each time instead.
        if model_id is not None:
            if model_id in self._io_token_conflict_warned_ids:
                return
            self._io_token_conflict_warned_ids.add(str(model_id))
        verbose_router_logger.warning(
            f"Deployment '{model_id}' configures itpm/otpm alongside tpm/rpm; "
            "both limit types are enforced on this deployment"
        )

    def _refund_io_token_reservation_if_any(self) -> None:
        request_kwargs = get_io_token_rate_limit_request_kwargs()
        if request_kwargs is not None:
            io_token_refund_failure(self.dual_cache, request_kwargs)

    async def _async_refund_io_token_reservation_if_any(
        self,
        parent_otel_span: Optional[Span] = None,
    ) -> None:
        request_kwargs = get_io_token_rate_limit_request_kwargs()
        if request_kwargs is not None:
            await async_io_token_refund_failure(
                self.dual_cache,
                request_kwargs,
                parent_otel_span=parent_otel_span,
            )

    def _get_deployment_limits(self, deployment: Dict) -> tuple[Optional[int], Optional[int]]:
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

    def _get_cache_keys(self, deployment: Dict, current_minute: str) -> tuple[str, str]:
        """Get the cache keys for TPM and RPM tracking."""
        model_id = deployment.get("model_info", {}).get("id")
        deployment_name = deployment.get("litellm_params", {}).get("model")

        tpm_key = f"{model_id}:{deployment_name}:tpm:{current_minute}"
        rpm_key = f"{model_id}:{deployment_name}:rpm:{current_minute}"

        return tpm_key, rpm_key

    def pre_call_check(self, deployment: Dict) -> Optional[Dict]:
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

            tpm_limit, rpm_limit = self._get_deployment_limits(deployment)

            # If no limits are set, allow the request
            if tpm_limit is None and rpm_limit is None:
                return deployment

            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")
            tpm_key, rpm_key = self._get_cache_keys(deployment, current_minute)

            model_id = deployment.get("model_info", {}).get("id")
            model_name = deployment.get("litellm_params", {}).get("model")
            model_group = deployment.get("model_name", "")

            # Check TPM limit
            if tpm_limit is not None:
                # First check local cache
                current_tpm = self.dual_cache.get_cache(key=tpm_key, local_only=True)
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
                current_rpm = self.dual_cache.increment_cache(key=rpm_key, value=1, ttl=RoutingArgs.ttl)
                if current_rpm is not None and current_rpm > rpm_limit:
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

            return deployment

        except litellm.RateLimitError:
            if io_reservation_made:
                self._refund_io_token_reservation_if_any()
            raise
        except Exception as e:
            verbose_router_logger.debug(f"Error in ModelRateLimitingCheck.pre_call_check: {str(e)}")
            # Don't fail the request if rate limit check fails
            return deployment

    async def async_pre_call_check(self, deployment: Dict, parent_otel_span: Optional[Span] = None) -> Optional[Dict]:
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

            tpm_limit, rpm_limit = self._get_deployment_limits(deployment)

            # If no limits are set, allow the request
            if tpm_limit is None and rpm_limit is None:
                return deployment

            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")
            tpm_key, rpm_key = self._get_cache_keys(deployment, current_minute)

            model_id = deployment.get("model_info", {}).get("id")
            model_name = deployment.get("litellm_params", {}).get("model")
            model_group = deployment.get("model_name", "")

            # Check TPM limit
            if tpm_limit is not None:
                # First check local cache
                current_tpm = await self.dual_cache.async_get_cache(key=tpm_key, local_only=True)
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

            # Check RPM limit (atomic increment-first to avoid race conditions)
            if rpm_limit is not None:
                current_rpm = await self.dual_cache.async_increment_cache(
                    key=rpm_key,
                    value=1,
                    ttl=RoutingArgs.ttl,
                    parent_otel_span=parent_otel_span,
                )
                if current_rpm is not None and current_rpm > rpm_limit:
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

            return deployment

        except litellm.RateLimitError:
            if io_reservation_made:
                await self._async_refund_io_token_reservation_if_any(parent_otel_span=parent_otel_span)
            raise
        except Exception as e:
            verbose_router_logger.debug(f"Error in ModelRateLimitingCheck.async_pre_call_check: {str(e)}")
            # Don't fail the request if rate limit check fails
            return deployment

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.litellm_core_utils.core_helpers import (
            _get_parent_otel_span_from_kwargs,
        )

        try:
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object")

            # IO token reconciliation works purely from the cache keys stashed in
            # kwargs/metadata, so it must run before the model_id guard below
            # (which only the TPM-tracking path needs). Otherwise a request whose
            # standard_logging_object lacks model_id would never return its
            # reservation, leaving the counter elevated until the TTL expires.
            slo_metadata = (standard_logging_object.get("metadata") or {}) if standard_logging_object else {}
            kwargs_metadata = kwargs.get("metadata") or {}
            if ITPM_RESERVED_KEY in slo_metadata or ITPM_RESERVED_KEY in kwargs_metadata:
                await async_io_token_reconcile_success(
                    self.dual_cache,
                    kwargs,
                    response_obj,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                )
                # Fall through: a deployment can also configure tpm/rpm alongside
                # itpm/otpm, and that path's pre-call check reads the tpm_key
                # counter tracked below, so it must still be incremented here.

            if standard_logging_object is None:
                return

            model_id = standard_logging_object.get("model_id")
            if model_id is None:
                return

            total_tokens = standard_logging_object.get("total_tokens", 0)
            model = standard_logging_object.get("hidden_params", {}).get("litellm_model_name")

            verbose_router_logger.debug(
                f"[TPM TRACKING] model_id={model_id}, total_tokens={total_tokens}, model={model}"
            )

            if not model or not total_tokens:
                return

            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")
            tpm_key = f"{model_id}:{model}:tpm:{current_minute}"

            verbose_router_logger.debug(f"[TPM TRACKING] Incrementing {tpm_key} by {total_tokens}")

            await self.dual_cache.async_increment_cache(
                key=tpm_key,
                value=total_tokens,
                ttl=RoutingArgs.ttl,
            )

        except Exception as e:
            verbose_router_logger.debug(f"Error in ModelRateLimitingCheck.async_log_success_event: {str(e)}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.litellm_core_utils.core_helpers import (
            _get_parent_otel_span_from_kwargs,
        )

        # Never fail the primary logging pipeline over an io-token refund error.
        with contextlib.suppress(Exception):
            await async_io_token_refund_failure(
                self.dual_cache,
                kwargs,
                parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
            )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Sync version of tracking TPM usage after successful request.
        Always tracks tokens - the pre-call check handles enforcement.
        """
        try:
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object")
            slo_metadata = (standard_logging_object.get("metadata") or {}) if standard_logging_object else {}
            kwargs_metadata = kwargs.get("metadata") or {}
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

            model_id = standard_logging_object.get("model_id")
            if model_id is None:
                return

            total_tokens = standard_logging_object.get("total_tokens", 0)
            model = standard_logging_object.get("hidden_params", {}).get("litellm_model_name")

            if not model or not total_tokens:
                return

            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")
            tpm_key = f"{model_id}:{model}:tpm:{current_minute}"

            self.dual_cache.increment_cache(
                key=tpm_key,
                value=total_tokens,
                ttl=RoutingArgs.ttl,
            )

        except Exception as e:
            verbose_router_logger.debug(f"Error in ModelRateLimitingCheck.log_success_event: {str(e)}")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        with contextlib.suppress(Exception):
            io_token_refund_failure(
                self.dual_cache,
                kwargs,
            )
