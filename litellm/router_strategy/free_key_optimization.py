#### What this does ####
#   Free Key Optimization routing strategy with multi-window rate limiting
#   Tracks TPM/TPH/TPD and RPM/RPH/RPD across minute, hour, and day windows
#   Selects deployment with lowest token usage while respecting ALL rate limits
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm import token_counter
from litellm._logging import verbose_logger, verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.types.router import RouterErrors
from litellm.types.utils import LiteLLMPydanticObjectBase, StandardLoggingPayload
from litellm.utils import get_utc_datetime

from .base_routing_strategy import BaseRoutingStrategy

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class RoutingArgs(LiteLLMPydanticObjectBase):
    ttl: int = 1 * 60  # 1min (RPM/TPM expire key)
    hour_ttl: int = 60 * 60  # 1 hour (for hourly rate limits)
    day_ttl: int = 24 * 60 * 60  # 24 hours (for daily rate limits)


class FreeKeyOptimizationHandler(BaseRoutingStrategy, CustomLogger):
    """
    Free Key Optimization routing strategy with multi-window rate limiting.

    Features:
    - Multi-window tracking: minute, hour, day
    - Supports TPM/TPH/TPD and RPM/RPH/RPD limits
    - AND logic: all configured limits must be respected
    - Selects deployment with lowest current token usage
    - Optimized for free API key usage patterns
    """

    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour

    def __init__(
        self, router_cache: DualCache, model_list: list, routing_args: dict = {}
    ):
        self.router_cache = router_cache
        self.model_list = model_list
        self.routing_args = RoutingArgs(**routing_args)
        BaseRoutingStrategy.__init__(
            self,
            dual_cache=router_cache,
            should_batch_redis_writes=True,
            default_sync_interval=0.1,
        )

    def _get_time_windows(self, dt):
        """
        Get all time window identifiers for the given datetime

        Returns:
            dict: Dictionary with minute, hour, and day time window strings
        """
        return {
            "minute": dt.strftime("%H-%M"),
            "hour": dt.strftime("%Y-%m-%d-%H"),
            "day": dt.strftime("%Y-%m-%d"),
        }

    def _get_deployment_limits(self, deployment: Dict):
        """
        Extract all rate limits from deployment configuration

        Args:
            deployment: Deployment dictionary

        Returns:
            dict: Dictionary with all rate limit values (rpm, rph, rpd, tpm, tph, tpd)
        """
        limits = {}
        limit_types = ["rpm", "rph", "rpd", "tpm", "tph", "tpd"]

        for limit_type in limit_types:
            limit_value = None
            # Check deployment level first
            if limit_value is None:
                limit_value = deployment.get(limit_type)
            # Check litellm_params level
            if limit_value is None:
                limit_value = deployment.get("litellm_params", {}).get(limit_type)
            # Check model_info level
            if limit_value is None:
                limit_value = deployment.get("model_info", {}).get(limit_type)
            # Default to infinity if not set
            limits[limit_type] = (
                limit_value if limit_value is not None else float("inf")
            )

        return limits

    def _generate_cache_keys(
        self, model_id: str, deployment_name: str, time_windows: dict
    ):
        """
        Generate cache keys for all time windows and metrics

        Args:
            model_id: Model ID
            deployment_name: Deployment name
            time_windows: Dictionary of time window identifiers

        Returns:
            dict: Dictionary of cache keys organized by metric and window type
        """
        keys = {}
        for window_type, window_value in time_windows.items():
            keys[f"rpm_{window_type}"] = (
                f"{model_id}:{deployment_name}:rpm:{window_type}:{window_value}"
            )
            keys[f"tpm_{window_type}"] = (
                f"{model_id}:{deployment_name}:tpm:{window_type}:{window_value}"
            )
        return keys

    def _get_ttl_for_window(self, window_type: str) -> int:
        """
        Get the appropriate TTL value for a given time window type

        Args:
            window_type: Type of time window ('minute', 'hour', 'day')

        Returns:
            int: TTL value in seconds
        """
        if window_type == "minute":
            return self.routing_args.ttl
        elif window_type == "hour":
            return self.routing_args.hour_ttl
        elif window_type == "day":
            return self.routing_args.day_ttl
        else:
            return self.routing_args.ttl  # Default to minute TTL

    def pre_call_check(self, deployment: Dict) -> Optional[Dict]:
        """
        Pre-call check + update model rpm/rph/rpd counters

        Returns - deployment (always, no rate limit checking here)

        Note: Rate limit validation happens in get_available_deployments.
        This method only increments request counters after deployment selection.
        """
        try:
            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            model_id = deployment.get("model_info", {}).get("id")
            deployment_name = deployment.get("litellm_params", {}).get("model")

            # Get all time windows and cache keys
            time_windows = self._get_time_windows(dt)
            cache_keys = self._generate_cache_keys(
                model_id, deployment_name, time_windows
            )

            # Increment request counters for all time windows
            request_counter_keys = [
                (cache_keys["rpm_minute"], self._get_ttl_for_window("minute")),
                (cache_keys["rpm_hour"], self._get_ttl_for_window("hour")),
                (cache_keys["rpm_day"], self._get_ttl_for_window("day")),
            ]

            for cache_key, ttl in request_counter_keys:
                self.router_cache.increment_cache(key=cache_key, value=1, ttl=ttl)

            return deployment
        except Exception as e:
            verbose_logger.exception(
                "litellm.router_strategy.free_key_optimization.py::pre_call_check(): Exception occurred - {}".format(
                    str(e)
                )
            )
            return deployment  # don't fail calls if eg. redis fails to connect

    async def async_pre_call_check(
        self, deployment: Dict, parent_otel_span: Optional[Span]
    ) -> Optional[Dict]:
        """
        Pre-call check + update model rpm/rph/rpd counters
        - Used inside semaphore

        Why? solves concurrency issue - https://github.com/BerriAI/litellm/issues/2994

        Returns - deployment (always, no rate limit checking here)

        Note: Rate limit validation happens in async_get_available_deployments.
        This method only increments request counters after deployment selection.
        """
        try:
            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            model_id = deployment.get("model_info", {}).get("id")
            deployment_name = deployment.get("litellm_params", {}).get("model")

            # Get all time windows and cache keys
            time_windows = self._get_time_windows(dt)
            cache_keys = self._generate_cache_keys(
                model_id, deployment_name, time_windows
            )

            # Increment request counters for all time windows
            request_counter_keys = [
                (cache_keys["rpm_minute"], self._get_ttl_for_window("minute")),
                (cache_keys["rpm_hour"], self._get_ttl_for_window("hour")),
                (cache_keys["rpm_day"], self._get_ttl_for_window("day")),
            ]

            for cache_key, ttl in request_counter_keys:
                await self._increment_value_in_current_window(
                    key=cache_key, value=1, ttl=ttl
                )

            return deployment
        except Exception as e:
            verbose_logger.exception(
                "litellm.router_strategy.free_key_optimization.py::async_pre_call_check(): Exception occurred - {}".format(
                    str(e)
                )
            )
            return deployment  # don't fail calls if eg. redis fails to connect

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update TPM/TPH/TPD usage on success
            """
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )
            if standard_logging_object is None:
                raise ValueError("standard_logging_object not passed in.")
            model_group = standard_logging_object.get("model_group")
            model = standard_logging_object["hidden_params"].get("litellm_model_name")
            id = standard_logging_object.get("model_id")
            if model_group is None or id is None or model is None:
                return
            elif isinstance(id, int):
                id = str(id)

            total_tokens = standard_logging_object.get("total_tokens")

            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            time_windows = self._get_time_windows(dt)

            # Generate TPM cache keys for all time windows
            tpm_keys = {}
            for window_type, window_value in time_windows.items():
                tpm_keys[window_type] = f"{id}:{model}:tpm:{window_type}:{window_value}"

            # ------------
            # Update usage for all time windows
            # ------------
            # Update TPM for minute window
            self.router_cache.increment_cache(
                key=tpm_keys["minute"], value=total_tokens, ttl=self.routing_args.ttl
            )

            # Update TPM for hour window
            self.router_cache.increment_cache(
                key=tpm_keys["hour"], value=total_tokens, ttl=self.routing_args.hour_ttl
            )

            # Update TPM for day window
            self.router_cache.increment_cache(
                key=tpm_keys["day"], value=total_tokens, ttl=self.routing_args.day_ttl
            )

            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.router_strategy.free_key_optimization.py::log_success_event(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update TPM/TPH/TPD usage on success
            """
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )
            if standard_logging_object is None:
                raise ValueError("standard_logging_object not passed in.")
            model_group = standard_logging_object.get("model_group")
            model = standard_logging_object["hidden_params"]["litellm_model_name"]
            id = standard_logging_object.get("model_id")
            if model_group is None or id is None:
                return
            elif isinstance(id, int):
                id = str(id)
            total_tokens = standard_logging_object.get("total_tokens")

            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            time_windows = self._get_time_windows(dt)
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)

            # Generate TPM cache keys for all time windows
            tpm_keys = {}
            for window_type, window_value in time_windows.items():
                tpm_keys[window_type] = f"{id}:{model}:tpm:{window_type}:{window_value}"

            # ------------
            # Update usage for all time windows
            # ------------
            # Update TPM for minute window
            await self.router_cache.async_increment_cache(
                key=tpm_keys["minute"],
                value=total_tokens,
                ttl=self.routing_args.ttl,
                parent_otel_span=parent_otel_span,
            )

            # Update TPM for hour window
            await self.router_cache.async_increment_cache(
                key=tpm_keys["hour"],
                value=total_tokens,
                ttl=self.routing_args.hour_ttl,
                parent_otel_span=parent_otel_span,
            )

            # Update TPM for day window
            await self.router_cache.async_increment_cache(
                key=tpm_keys["day"],
                value=total_tokens,
                ttl=self.routing_args.day_ttl,
                parent_otel_span=parent_otel_span,
            )

            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.router_strategy.free_key_optimization.py::async_log_success_event(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    def _filter_deployments_by_limits(
        self,
        healthy_deployments: List[Dict],
        usage_data: Dict,
        input_tokens: int = 0,
    ) -> List[Dict]:
        """
        Filter deployments that exceed any of their configured rate limits

        Args:
            healthy_deployments: List of healthy deployment dicts
            usage_data: Dictionary containing current usage for all deployments
            input_tokens: Number of tokens for the current request

        Returns:
            List of deployments that are within all their rate limits
        """
        eligible_deployments = []

        for deployment in healthy_deployments:
            deployment_id = deployment.get("model_info", {}).get("id")
            if not deployment_id:
                continue

            # Get deployment limits
            limits = self._get_deployment_limits(deployment)

            # Get current usage for this deployment
            current_usage = usage_data.get(deployment_id, {})

            # Check all rate limits (AND logic - all must pass)
            within_limits = True

            # Check TPM limits
            current_tpm = current_usage.get("tpm_minute", 0) or 0
            if current_tpm + input_tokens > limits["tpm"]:
                within_limits = False
                verbose_router_logger.warning(
                    f"Deployment {deployment_id} excluded due to rate limit: tpm"
                )

            current_tph = current_usage.get("tpm_hour", 0) or 0
            if current_tph + input_tokens > limits["tph"]:
                within_limits = False
                verbose_router_logger.warning(
                    f"Deployment {deployment_id} excluded due to rate limit: tph"
                )

            current_tpd = current_usage.get("tpm_day", 0) or 0
            if current_tpd + input_tokens > limits["tpd"]:
                within_limits = False
                verbose_router_logger.warning(
                    f"Deployment {deployment_id} excluded due to rate limit: tpd"
                )

            # Check RPM limits
            current_rpm = current_usage.get("rpm_minute", 0) or 0
            if current_rpm + 1 > limits["rpm"]:
                within_limits = False
                verbose_router_logger.warning(
                    f"Deployment {deployment_id} excluded due to rate limit: rpm"
                )

            current_rph = current_usage.get("rpm_hour", 0) or 0
            if current_rph + 1 > limits["rph"]:
                within_limits = False
                verbose_router_logger.warning(
                    f"Deployment {deployment_id} excluded due to rate limit: rph"
                )

            current_rpd = current_usage.get("rpm_day", 0) or 0
            if current_rpd + 1 > limits["rpd"]:
                within_limits = False
                verbose_router_logger.warning(
                    f"Deployment {deployment_id} excluded due to rate limit: rpd"
                )

            if within_limits:
                eligible_deployments.append(deployment)

        return eligible_deployments

    def _get_cost_information(self, deployment: Dict) -> Dict:
        """
        Extract cost information from deployment configuration.

        Priority order:
        1. litellm_params (user override)
        2. model_info (deployment-specific)
        3. global model cost map (LiteLLM database)

        Returns:
            Dict with input_cost_per_token, output_cost_per_token, and cost_data_source
        """
        input_cost = None
        output_cost = None
        cost_data_source = "unknown"

        # 1. Check litellm_params (highest priority)
        litellm_params = deployment.get("litellm_params", {})
        if (
            "input_cost_per_token" in litellm_params
            or "output_cost_per_token" in litellm_params
        ):
            input_cost = litellm_params.get("input_cost_per_token")
            output_cost = litellm_params.get("output_cost_per_token")
            cost_data_source = "litellm_params_override"

        # 2. Check model_info (second priority)
        if input_cost is None or output_cost is None:
            model_info = deployment.get("model_info", {})
            if (
                "input_cost_per_token" in model_info
                or "output_cost_per_token" in model_info
            ):
                if input_cost is None:
                    input_cost = model_info.get("input_cost_per_token")
                if output_cost is None:
                    output_cost = model_info.get("output_cost_per_token")
                if cost_data_source == "unknown":
                    cost_data_source = "model_info"

        # 3. Check global model cost map (lowest priority)
        if input_cost is None or output_cost is None:
            model_name = litellm_params.get("model")
            if model_name:
                model_cost_info = litellm.model_cost.get(model_name, {})
                if input_cost is None:
                    input_cost = model_cost_info.get("input_cost_per_token")
                if output_cost is None:
                    output_cost = model_cost_info.get("output_cost_per_token")
                if (
                    input_cost is not None or output_cost is not None
                ) and cost_data_source == "unknown":
                    cost_data_source = f"global_model_cost_map[{model_name}]"

        # Convert None to 0, but preserve explicit 0.0 values
        input_cost = 0 if input_cost is None else input_cost
        output_cost = 0 if output_cost is None else output_cost

        if cost_data_source == "unknown":
            cost_data_source = "no_cost_data_found"

        return {
            "input_cost_per_token": input_cost,
            "output_cost_per_token": output_cost,
            "cost_data_source": cost_data_source,
        }

    def _calculate_cost_metric(
        self, cost_info: Dict, usage_data: Dict, deployment_id: str, input_tokens: int
    ) -> Dict:
        """
        Calculate the cost metric for deployment selection.

        Returns:
            Dict with cost_metric, cost_source, and calculation details
        """
        input_cost_per_token = cost_info["input_cost_per_token"]
        output_cost_per_token = cost_info["output_cost_per_token"]

        # Always use cost calculation if we have ANY cost data (including 0.0)
        if cost_info["cost_data_source"] != "no_cost_data_found":
            # Calculate input cost based on actual input tokens
            input_cost = input_cost_per_token * input_tokens

            # Estimate output cost assuming typical 1:1 input:output ratio
            estimated_output_tokens = input_tokens
            output_cost = output_cost_per_token * estimated_output_tokens

            estimated_total_cost = input_cost + output_cost

            return {
                "cost_metric": estimated_total_cost,
                "cost_source": "cost_calculation",
                "estimated_output_tokens": estimated_output_tokens,
                "estimated_total_cost": estimated_total_cost,
            }
        else:
            # Fallback to token usage only when no cost data is available
            current_tpm = usage_data.get(deployment_id, {}).get("tpm_minute", 0) or 0
            return {
                "cost_metric": current_tpm,
                "cost_source": "token_usage_fallback",
                "estimated_output_tokens": None,
                "estimated_total_cost": None,
            }

    def _select_lowest_cost_deployment(
        self,
        eligible_deployments: List[Dict],
        usage_data: Dict,
        input_tokens: int = 100,
    ) -> Optional[Dict]:
        """
        Select deployment with lowest estimated cost from eligible deployments

        Args:
            eligible_deployments: List of deployments within rate limits
            usage_data: Dictionary containing current usage for all deployments
            input_tokens: Estimated input tokens for cost calculation

        Returns:
            Deployment with lowest estimated cost, or None if no deployments available
        """
        if not eligible_deployments:
            verbose_router_logger.warning(
                "free_key_optimization: No eligible deployments available"
            )
            return None

        lowest_cost = float("inf")
        selected_deployment = None
        deployment_costs = []  # Track all deployment costs for logging

        verbose_router_logger.info(
            f"free_key_optimization: Evaluating {len(eligible_deployments)} eligible deployments for cost optimization"
        )

        for deployment in eligible_deployments:
            deployment_id = deployment.get("model_info", {}).get("id")
            if not deployment_id:
                continue

            # Get cost information using the new helper method
            cost_info = self._get_cost_information(deployment)
            input_cost_per_token = cost_info["input_cost_per_token"]
            output_cost_per_token = cost_info["output_cost_per_token"]
            cost_data_source = cost_info["cost_data_source"]

            # Get current usage data for logging
            current_usage_data = usage_data.get(deployment_id, {})
            current_tpm = current_usage_data.get("tpm_minute", 0) or 0
            current_rpm = current_usage_data.get("rpm_minute", 0) or 0
            current_tph = current_usage_data.get("tpm_hour", 0) or 0
            current_rph = current_usage_data.get("rpm_hour", 0) or 0
            current_tpd = current_usage_data.get("tpm_day", 0) or 0
            current_rpd = current_usage_data.get("rpm_day", 0) or 0

            # Calculate cost metric using the new helper method
            cost_calculation = self._calculate_cost_metric(
                cost_info, usage_data, deployment_id, input_tokens
            )
            cost_metric = cost_calculation["cost_metric"]
            cost_source = cost_calculation["cost_source"]
            estimated_total_cost = cost_calculation["estimated_total_cost"]
            estimated_output_tokens = cost_calculation["estimated_output_tokens"]

            # Store deployment info for logging
            deployment_info = {
                "deployment_id": deployment_id,
                "model_name": deployment.get("litellm_params", {}).get(
                    "model", "unknown"
                ),
                "cost_metric": cost_metric,
                "cost_source": cost_source,
                "cost_data_source": cost_data_source,
                "input_cost_per_token": input_cost_per_token,
                "output_cost_per_token": output_cost_per_token,
                "input_tokens": input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "estimated_total_cost": estimated_total_cost,
                "current_tpm": current_tpm,
                "current_rpm": current_rpm,
                "current_tph": current_tph,
                "current_rph": current_rph,
                "current_tpd": current_tpd,
                "current_rpd": current_rpd,
            }
            deployment_costs.append(deployment_info)

            if cost_metric < lowest_cost:
                lowest_cost = cost_metric
                selected_deployment = deployment
            elif cost_metric == lowest_cost and selected_deployment is not None:
                # If cost is equal, randomly select to avoid bias
                if random.choice([True, False]):
                    selected_deployment = deployment

        if selected_deployment is None:  # no deployments available - bail out.
            verbose_router_logger.warning(
                "free_key_optimization: No deployment selected"
            )
            return None

        selected_id = selected_deployment.get("model_info", {}).get("id")
        selected_info = next(
            (d for d in deployment_costs if d["deployment_id"] == selected_id), None
        )

        verbose_router_logger.info(
            f"free_key_optimization: Selected deployment {selected_id} "
            f"(model: {selected_info['model_name'] if selected_info else 'unknown'}) "
            f"with cost_metric: {lowest_cost:.8f} "
            f"(source: {selected_info['cost_source'] if selected_info else 'unknown'})"
        )

        verbose_router_logger.debug("free_key_optimization: All deployment costs:")
        for dep_info in deployment_costs:
            verbose_router_logger.warning(dep_info)

        return selected_deployment

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        parent_otel_span: Optional[Span] = None,
    ):
        """
        Returns a deployment with the lowest TPM usage that respects all rate limits.
        """
        verbose_router_logger.debug(
            f"get_available_deployments - Free Key Optimization. model_group: {model_group}, healthy_deployments: {healthy_deployments}"
        )

        dt = get_utc_datetime()
        time_windows = self._get_time_windows(dt)

        # Generate cache keys for all deployments and time windows
        all_cache_keys = []
        deployment_key_mapping = {}

        for deployment in healthy_deployments:
            if isinstance(deployment, dict):
                deployment_id = deployment.get("model_info", {}).get("id")
                deployment_name = deployment.get("litellm_params", {}).get("model")

                if deployment_id and deployment_name:
                    cache_keys = self._generate_cache_keys(
                        deployment_id, deployment_name, time_windows
                    )
                    deployment_key_mapping[deployment_id] = cache_keys
                    all_cache_keys.extend(cache_keys.values())

        # Batch get all usage data
        usage_values = self.router_cache.batch_get_cache(
            keys=all_cache_keys, parent_otel_span=parent_otel_span
        )

        # Organize usage data by deployment
        usage_data: Dict[str, Dict[str, int]] = {}
        if usage_values:
            key_value_map = dict(zip(all_cache_keys, usage_values))

            for deployment_id, cache_keys in deployment_key_mapping.items():
                usage_data[deployment_id] = {}
                for metric_window, cache_key in cache_keys.items():
                    usage_data[deployment_id][metric_window] = (
                        key_value_map.get(cache_key, 0) or 0
                    )

        # Calculate input tokens
        try:
            input_tokens = token_counter(messages=messages, text=input)
        except Exception:
            input_tokens = 0
        verbose_router_logger.debug(f"input_tokens={input_tokens}")

        # Filter deployments by rate limits
        eligible_deployments = self._filter_deployments_by_limits(
            healthy_deployments=healthy_deployments,
            usage_data=usage_data,
            input_tokens=input_tokens,
        )

        # Select deployment with lowest cost
        selected_deployment = self._select_lowest_cost_deployment(
            eligible_deployments=eligible_deployments,
            usage_data=usage_data,
            input_tokens=input_tokens,
        )

        if selected_deployment is None:
            # Create detailed error message showing why no deployments are available
            deployment_details = {}
            for deployment in healthy_deployments:
                deployment_id = deployment.get("model_info", {}).get("id")
                if deployment_id:
                    limits = self._get_deployment_limits(deployment)
                    current_usage = usage_data.get(deployment_id, {})
                    deployment_details[deployment_id] = {
                        "limits": limits,
                        "current_usage": current_usage,
                    }

            raise litellm.RateLimitError(
                message=f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. All deployments exceed rate limits. Deployment details: {deployment_details}",
                llm_provider="",
                model=model_group,
                response=httpx.Response(
                    status_code=429,
                    content="",
                    headers={"retry-after": str(60)},  # type: ignore
                    request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
            )

        verbose_router_logger.info(
            f"get_available_deployment for model: {model_group}, Selected deployment: {selected_deployment.get('model_info', {}).get('id')} for model: {model_group}"
        )
        return selected_deployment

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ):
        """
        Async implementation of get deployments with free key optimization.

        Reduces time to retrieve usage values from cache and applies multi-window rate limiting.
        """
        verbose_router_logger.debug(
            f"async_get_available_deployments - Free Key Optimization. model_group: {model_group}, healthy_deployments: {healthy_deployments}"
        )

        dt = get_utc_datetime()
        time_windows = self._get_time_windows(dt)

        # Generate cache keys for all deployments and time windows
        all_cache_keys = []
        deployment_key_mapping = {}

        for deployment in healthy_deployments:
            if isinstance(deployment, dict):
                deployment_id = deployment.get("model_info", {}).get("id")
                deployment_name = deployment.get("litellm_params", {}).get("model")

                if deployment_id and deployment_name:
                    cache_keys = self._generate_cache_keys(
                        deployment_id, deployment_name, time_windows
                    )
                    deployment_key_mapping[deployment_id] = cache_keys
                    all_cache_keys.extend(cache_keys.values())

        # Batch get all usage data
        usage_values = await self.router_cache.async_batch_get_cache(
            keys=all_cache_keys
        )

        # Organize usage data by deployment
        usage_data: Dict[str, Dict[str, int]] = {}
        if usage_values:
            key_value_map = dict(zip(all_cache_keys, usage_values))

            for deployment_id, cache_keys in deployment_key_mapping.items():
                usage_data[deployment_id] = {}
                for metric_window, cache_key in cache_keys.items():
                    usage_data[deployment_id][metric_window] = (
                        key_value_map.get(cache_key, 0) or 0
                    )

        # Calculate input tokens
        try:
            input_tokens = token_counter(messages=messages, text=input)
        except Exception:
            input_tokens = 0
        verbose_router_logger.warning(f"input_tokens={input_tokens}")

        # Filter deployments by rate limits
        eligible_deployments = self._filter_deployments_by_limits(
            healthy_deployments=healthy_deployments,
            usage_data=usage_data,
            input_tokens=input_tokens,
        )

        # Select deployment with lowest cost
        selected_deployment = self._select_lowest_cost_deployment(
            eligible_deployments=eligible_deployments,
            usage_data=usage_data,
            input_tokens=input_tokens,
        )

        if selected_deployment is None:
            # Create detailed error message showing why no deployments are available
            deployment_details = {}
            for deployment in healthy_deployments:
                deployment_id = deployment.get("model_info", {}).get("id")
                if deployment_id:
                    limits = self._get_deployment_limits(deployment)
                    current_usage = usage_data.get(deployment_id, {})
                    deployment_details[deployment_id] = {
                        "limits": limits,
                        "current_usage": current_usage,
                    }

            raise litellm.RateLimitError(
                message=f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. All deployments exceed rate limits. Deployment details: {deployment_details}",
                llm_provider="",
                model=model_group,
                response=httpx.Response(
                    status_code=429,
                    content="",
                    headers={"retry-after": str(60)},  # type: ignore
                    request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
            )

        verbose_router_logger.info(
            f"async_get_available_deployment for model: {model_group}, Selected deployment: {selected_deployment.get('model_info', {}).get('id')} for model: {model_group}"
        )
        return selected_deployment
