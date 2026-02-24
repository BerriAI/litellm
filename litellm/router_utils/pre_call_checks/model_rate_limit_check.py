"""
Enforce TPM/RPM rate limits set on model deployments.

This pre-call check ensures that model-level TPM/RPM limits are enforced
across all requests, regardless of routing strategy.

When enabled via `enforce_model_rate_limits: true` in litellm_settings,
requests that exceed the configured TPM/RPM limits will receive a 429 error.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
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
    Pre-call check that enforces TPM/RPM limits on model deployments.

    This check runs before each request and raises a RateLimitError
    if the deployment has exceeded its configured TPM or RPM limits.

    Unlike the usage-based-routing strategy which uses limits for routing decisions,
    this check actively enforces those limits across ALL routing strategies.
    """

    def __init__(self, dual_cache: DualCache):
        self.dual_cache = dual_cache

    def _get_deployment_limits(
        self, deployment: Dict
    ) -> tuple[Optional[int], Optional[int]]:
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

            # Check RPM limit
            if rpm_limit is not None:
                # First check local cache
                current_rpm = self.dual_cache.get_cache(key=rpm_key, local_only=True)
                if current_rpm >= rpm_limit:
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

                # Check redis cache and increment
                current_rpm = self.dual_cache.increment_cache(
                    key=rpm_key, value=1, ttl=RoutingArgs.ttl
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
                    )

            return deployment

        except litellm.RateLimitError:
            raise
        except Exception as e:
            verbose_router_logger.debug(
                f"Error in ModelRateLimitingCheck.pre_call_check: {str(e)}"
            )
            # Don't fail the request if rate limit check fails
            return deployment

    async def async_pre_call_check(
        self, deployment: Dict, parent_otel_span: Optional[Span] = None
    ) -> Optional[Dict]:
        """
        Async pre-call check for model rate limits.

        Raises RateLimitError if deployment exceeds TPM/RPM limits.
        """
        try:
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
                current_tpm = await self.dual_cache.async_get_cache(
                    key=tpm_key, local_only=True
                )
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

            # Check RPM limit
            if rpm_limit is not None:
                # First check local cache
                current_rpm = await self.dual_cache.async_get_cache(
                    key=rpm_key, local_only=True
                )
                if current_rpm is not None and current_rpm >= rpm_limit:
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

                # Check redis cache and increment
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
            raise
        except Exception as e:
            verbose_router_logger.debug(
                f"Error in ModelRateLimitingCheck.async_pre_call_check: {str(e)}"
            )
            # Don't fail the request if rate limit check fails
            return deployment

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Track TPM usage after successful request.

        This updates the TPM counter with the actual tokens used.
        Always tracks tokens - the pre-call check handles enforcement.
        """
        try:
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )
            if standard_logging_object is None:
                return

            model_id = standard_logging_object.get("model_id")
            if model_id is None:
                return

            total_tokens = standard_logging_object.get("total_tokens", 0)
            model = standard_logging_object.get("hidden_params", {}).get(
                "litellm_model_name"
            )

            verbose_router_logger.debug(
                f"[TPM TRACKING] model_id={model_id}, total_tokens={total_tokens}, model={model}"
            )

            if not model or not total_tokens:
                return

            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")
            tpm_key = f"{model_id}:{model}:tpm:{current_minute}"

            verbose_router_logger.debug(
                f"[TPM TRACKING] Incrementing {tpm_key} by {total_tokens}"
            )

            await self.dual_cache.async_increment_cache(
                key=tpm_key,
                value=total_tokens,
                ttl=RoutingArgs.ttl,
            )

        except Exception as e:
            verbose_router_logger.debug(
                f"Error in ModelRateLimitingCheck.async_log_success_event: {str(e)}"
            )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Sync version of tracking TPM usage after successful request.
        Always tracks tokens - the pre-call check handles enforcement.
        """
        try:
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )
            if standard_logging_object is None:
                return

            model_id = standard_logging_object.get("model_id")
            if model_id is None:
                return

            total_tokens = standard_logging_object.get("total_tokens", 0)
            model = standard_logging_object.get("hidden_params", {}).get(
                "litellm_model_name"
            )

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
            verbose_router_logger.debug(
                f"Error in ModelRateLimitingCheck.log_success_event: {str(e)}"
            )
