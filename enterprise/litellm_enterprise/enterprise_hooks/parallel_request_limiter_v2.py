"""
V2 Implementation of Parallel Requests, TPM, RPM Limiting on the proxy

Designed to work on a multi-instance setup, where multiple instances are writing to redis simultaneously
"""
import asyncio
import sys
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

from fastapi import HTTPException

import litellm
from litellm import DualCache, ModelResponse
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import (
    get_key_model_rpm_limit,
    get_key_model_tpm_limit,
)
from litellm.router_strategy.base_routing_strategy import BaseRoutingStrategy

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    Span = Union[_Span, Any]
    InternalUsageCache = _InternalUsageCache
else:
    Span = Any
    InternalUsageCache = Any


class CacheObject(TypedDict):
    current_global_requests: Optional[dict]
    request_count_api_key: Optional[int]
    request_count_api_key_model: Optional[dict]
    request_count_user_id: Optional[dict]
    request_count_team_id: Optional[dict]
    request_count_end_user_id: Optional[dict]
    rpm_api_key: Optional[int]
    tpm_api_key: Optional[int]


RateLimitGroups = Literal["request_count", "tpm", "rpm"]
RateLimitTypes = Literal["key", "model_per_key", "user", "customer", "team"]


class _PROXY_MaxParallelRequestsHandler(BaseRoutingStrategy, CustomLogger):
    # Class variables or attributes
    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache
        BaseRoutingStrategy.__init__(
            self,
            dual_cache=internal_usage_cache.dual_cache,
            should_batch_redis_writes=True,
            default_sync_interval=0.01,
        )

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except Exception:
            pass

    @property
    def prefix(self) -> str:
        return "parallel_request_limiter_v2"

    def _get_current_usage_key(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        precise_minute: str,
        model: Optional[str],
        rate_limit_type: Literal["key", "model_per_key", "user", "customer", "team"],
        group: RateLimitGroups,
    ) -> Optional[str]:
        if rate_limit_type == "key" and user_api_key_dict.api_key is not None:
            return (
                f"{self.prefix}::{user_api_key_dict.api_key}::{precise_minute}::{group}"
            )
        elif (
            rate_limit_type == "model_per_key"
            and model is not None
            and user_api_key_dict.api_key is not None
        ):
            return f"{self.prefix}::{user_api_key_dict.api_key}::{model}::{precise_minute}::{group}"
        elif rate_limit_type == "user" and user_api_key_dict.user_id is not None:
            return (
                f"{self.prefix}::{user_api_key_dict.user_id}::{precise_minute}::{group}"
            )
        elif (
            rate_limit_type == "customer" and user_api_key_dict.end_user_id is not None
        ):
            return f"{self.prefix}::{user_api_key_dict.end_user_id}::{precise_minute}::{group}"
        elif rate_limit_type == "team" and user_api_key_dict.team_id is not None:
            return (
                f"{self.prefix}::{user_api_key_dict.team_id}::{precise_minute}::{group}"
            )
        elif rate_limit_type == "model_per_key" and model is not None:
            return f"{self.prefix}::{user_api_key_dict.api_key}::{model}::{precise_minute}::{group}"
        else:
            return None

    def get_key_pattern_to_sync(self) -> Optional[str]:
        return self.prefix + "::"

    async def check_key_in_limits_v2(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        max_parallel_requests: Optional[int],
        precise_minute: str,
        tpm_limit: Optional[int],
        rpm_limit: Optional[int],
        rate_limit_type: Literal["key", "model_per_key", "user", "customer", "team"],
    ):
        ## INCREMENT CURRENT USAGE
        increment_list: List[Tuple[str, int]] = []
        increment_value_by_group = {
            "request_count": 1,
            "tpm": 0,
            "rpm": 1,
        }
        for group in ["request_count", "rpm", "tpm"]:
            key = self._get_current_usage_key(
                user_api_key_dict=user_api_key_dict,
                precise_minute=precise_minute,
                model=data.get("model", None),
                rate_limit_type=rate_limit_type,
                group=cast(RateLimitGroups, group),
            )
            if key is None:
                continue
            increment_list.append((key, increment_value_by_group[group]))

        if (
            not max_parallel_requests and not rpm_limit and not tpm_limit
        ):  # no rate limits
            return

        results = await self._increment_value_list_in_current_window(
            increment_list=increment_list,
            ttl=60,
        )
        should_raise_error = False
        if max_parallel_requests is not None:
            should_raise_error = results[0] > max_parallel_requests
        if rpm_limit is not None:
            should_raise_error = should_raise_error or results[1] > rpm_limit
        if tpm_limit is not None:
            should_raise_error = should_raise_error or results[2] > tpm_limit
        if should_raise_error:
            raise self.raise_rate_limit_error(
                additional_details=f"{CommonProxyErrors.max_parallel_request_limit_reached.value}. Hit limit for {rate_limit_type}. Current usage: max_parallel_requests: {results[0]}, current_rpm: {results[1]}, current_tpm: {results[2]}. Current limits: max_parallel_requests: {max_parallel_requests}, rpm_limit: {rpm_limit}, tpm_limit: {tpm_limit}."
            )

    def time_to_next_minute(self) -> float:
        # Get the current time
        now = datetime.now()

        # Calculate the next minute
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)

        # Calculate the difference in seconds
        seconds_to_next_minute = (next_minute - now).total_seconds()

        return seconds_to_next_minute

    def raise_rate_limit_error(
        self, additional_details: Optional[str] = None
    ) -> HTTPException:
        """
        Raise an HTTPException with a 429 status code and a retry-after header
        """
        error_message = "Max parallel request limit reached"
        if additional_details is not None:
            error_message = error_message + " " + additional_details
        raise HTTPException(
            status_code=429,
            detail=f"Max parallel request limit reached {additional_details}",
            headers={"retry-after": str(self.time_to_next_minute())},
        )

    async def async_pre_call_hook(  # noqa: PLR0915
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        self.print_verbose("Inside Max Parallel Request Pre-Call Hook")
        api_key = user_api_key_dict.api_key
        max_parallel_requests = user_api_key_dict.max_parallel_requests
        if max_parallel_requests is None:
            max_parallel_requests = sys.maxsize
        if data is None:
            data = {}
        global_max_parallel_requests = data.get("metadata", {}).get(
            "global_max_parallel_requests", None
        )
        tpm_limit = getattr(user_api_key_dict, "tpm_limit", sys.maxsize)
        if tpm_limit is None:
            tpm_limit = sys.maxsize
        rpm_limit = getattr(user_api_key_dict, "rpm_limit", sys.maxsize)
        if rpm_limit is None:
            rpm_limit = sys.maxsize
        # ------------
        # Setup values
        # ------------
        if global_max_parallel_requests is not None:
            # get value from cache
            _key = "global_max_parallel_requests"
            current_global_requests = await self.internal_usage_cache.async_get_cache(
                key=_key,
                local_only=True,
                litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            )
            # check if below limit
            if current_global_requests is None:
                current_global_requests = 1
            # if above -> raise error
            if current_global_requests >= global_max_parallel_requests:
                return self.raise_rate_limit_error(
                    additional_details=f"Hit Global Limit: Limit={global_max_parallel_requests}, current: {current_global_requests}"
                )
            # if below -> increment
            else:
                await self.internal_usage_cache.async_increment_cache(
                    key=_key,
                    value=1,
                    local_only=True,
                    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                )
        requested_model = data.get("model", None)

        current_date = datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.now().strftime("%H")
        current_minute = datetime.now().strftime("%M")
        precise_minute = f"{current_date}-{current_hour}-{current_minute}"

        tasks = []
        if api_key is not None:
            # CHECK IF REQUEST ALLOWED for key
            tasks.append(
                self.check_key_in_limits_v2(
                    user_api_key_dict=user_api_key_dict,
                    data=data,
                    max_parallel_requests=max_parallel_requests,
                    precise_minute=precise_minute,
                    tpm_limit=tpm_limit,
                    rpm_limit=rpm_limit,
                    rate_limit_type="key",
                )
            )
        if user_api_key_dict.user_id is not None:
            # CHECK IF REQUEST ALLOWED for key
            tasks.append(
                self.check_key_in_limits_v2(
                    user_api_key_dict=user_api_key_dict,
                    data=data,
                    max_parallel_requests=None,
                    precise_minute=precise_minute,
                    tpm_limit=user_api_key_dict.user_tpm_limit,
                    rpm_limit=user_api_key_dict.user_rpm_limit,
                    rate_limit_type="user",
                )
            )
        if user_api_key_dict.team_id is not None:
            tasks.append(
                self.check_key_in_limits_v2(
                    user_api_key_dict=user_api_key_dict,
                    data=data,
                    max_parallel_requests=None,
                    precise_minute=precise_minute,
                    tpm_limit=user_api_key_dict.team_tpm_limit,
                    rpm_limit=user_api_key_dict.team_rpm_limit,
                    rate_limit_type="team",
                )
            )
        if user_api_key_dict.end_user_id is not None:
            tasks.append(
                self.check_key_in_limits_v2(
                    user_api_key_dict=user_api_key_dict,
                    data=data,
                    max_parallel_requests=None,
                    precise_minute=precise_minute,
                    tpm_limit=user_api_key_dict.end_user_tpm_limit,
                    rpm_limit=user_api_key_dict.end_user_rpm_limit,
                    rate_limit_type="customer",
                )
            )
        if requested_model and (
            get_key_model_tpm_limit(user_api_key_dict) is not None
            or get_key_model_rpm_limit(user_api_key_dict) is not None
        ):
            _tpm_limit_for_key_model = get_key_model_tpm_limit(user_api_key_dict) or {}
            _rpm_limit_for_key_model = get_key_model_rpm_limit(user_api_key_dict) or {}

            should_check_rate_limit = False
            if requested_model in _tpm_limit_for_key_model:
                should_check_rate_limit = True
            elif requested_model in _rpm_limit_for_key_model:
                should_check_rate_limit = True

            if should_check_rate_limit:
                model_specific_tpm_limit: Optional[int] = None
                model_specific_rpm_limit: Optional[int] = None
                if requested_model in _tpm_limit_for_key_model:
                    model_specific_tpm_limit = _tpm_limit_for_key_model[requested_model]
                if requested_model in _rpm_limit_for_key_model:
                    model_specific_rpm_limit = _rpm_limit_for_key_model[requested_model]
                tasks.append(
                    self.check_key_in_limits_v2(
                        user_api_key_dict=user_api_key_dict,
                        data=data,
                        max_parallel_requests=None,
                        precise_minute=precise_minute,
                        tpm_limit=model_specific_tpm_limit,
                        rpm_limit=model_specific_rpm_limit,
                        rate_limit_type="model_per_key",
                    )
                )
        await asyncio.gather(*tasks)

        return

    async def _update_usage_in_cache_post_call(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        precise_minute: str,
        model: Optional[str],
        total_tokens: int,
        litellm_parent_otel_span: Union[Span, None] = None,
    ):
        increment_list: List[Tuple[str, int]] = []
        increment_value_by_group = {
            "request_count": -1,
            "tpm": total_tokens,
            "rpm": 0,
        }

        rate_limit_types = ["key", "user", "customer", "team", "model_per_key"]
        for rate_limit_type in rate_limit_types:
            for group in ["request_count", "rpm", "tpm"]:
                key = self._get_current_usage_key(
                    user_api_key_dict=user_api_key_dict,
                    precise_minute=precise_minute,
                    model=model,
                    rate_limit_type=cast(RateLimitTypes, rate_limit_type),
                    group=cast(RateLimitGroups, group),
                )
                if key is None:
                    continue
                increment_list.append((key, increment_value_by_group[group]))

        if increment_list:  # Only call if we have values to increment
            await self._increment_value_list_in_current_window(
                increment_list=increment_list,
                ttl=60,
            )

    async def async_log_success_event(  # noqa: PLR0915
        self, kwargs, response_obj, start_time, end_time
    ):
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,
        )

        litellm_parent_otel_span: Union[Span, None] = _get_parent_otel_span_from_kwargs(
            kwargs=kwargs
        )
        try:
            self.print_verbose("INSIDE parallel request limiter ASYNC SUCCESS LOGGING")

            # ------------
            # Setup values
            # ------------

            global_max_parallel_requests = kwargs["litellm_params"]["metadata"].get(
                "global_max_parallel_requests", None
            )
            user_api_key = kwargs["litellm_params"]["metadata"]["user_api_key"]
            user_api_key_user_id = kwargs["litellm_params"]["metadata"].get(
                "user_api_key_user_id", None
            )
            user_api_key_team_id = kwargs["litellm_params"]["metadata"].get(
                "user_api_key_team_id", None
            )
            user_api_key_end_user_id = kwargs.get("user") or kwargs["litellm_params"][
                "metadata"
            ].get("user_api_key_end_user_id", None)

            # ------------
            # Setup values
            # ------------

            if global_max_parallel_requests is not None:
                # get value from cache
                _key = "global_max_parallel_requests"
                # decrement
                await self.internal_usage_cache.async_increment_cache(
                    key=_key,
                    value=-1,
                    local_only=True,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                )

            current_date = datetime.now().strftime("%Y-%m-%d")
            current_hour = datetime.now().strftime("%H")
            current_minute = datetime.now().strftime("%M")
            precise_minute = f"{current_date}-{current_hour}-{current_minute}"
            model_group = get_model_group_from_litellm_kwargs(kwargs)
            total_tokens = 0

            if isinstance(response_obj, ModelResponse):
                total_tokens = response_obj.usage.total_tokens  # type: ignore

            # ------------
            # Update usage - API Key
            # ------------

            await self._update_usage_in_cache_post_call(
                user_api_key_dict=UserAPIKeyAuth(
                    api_key=user_api_key,
                    user_id=user_api_key_user_id,
                    team_id=user_api_key_team_id,
                    end_user_id=user_api_key_end_user_id,
                ),
                precise_minute=precise_minute,
                model=model_group,
                total_tokens=total_tokens,
            )

        except Exception as e:
            verbose_proxy_logger.exception(
                "Inside Parallel Request Limiter: An exception occurred - {}".format(
                    str(e)
                )
            )

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
    ):
        try:
            self.print_verbose("Inside Max Parallel Request Failure Hook")

            model_group = request_data.get("model", None)
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_hour = datetime.now().strftime("%H")
            current_minute = datetime.now().strftime("%M")
            precise_minute = f"{current_date}-{current_hour}-{current_minute}"

            ## decrement call count if call failed
            await self._update_usage_in_cache_post_call(
                user_api_key_dict=user_api_key_dict,
                precise_minute=precise_minute,
                model=model_group,
                total_tokens=0,
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                "Inside Parallel Request Limiter: An exception occurred - {}".format(
                    str(e)
                )
            )
