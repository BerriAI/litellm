#### What this does ####
#   picks based on response time (for streaming, this is time to first token)
import random
import statistics
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import litellm
from litellm import ModelResponse, token_counter, verbose_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import safe_divide_seconds
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.types.utils import LiteLLMPydanticObjectBase

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class RoutingArgs(LiteLLMPydanticObjectBase):
    ttl: float = 1 * 60 * 60  # 1 hour
    lowest_latency_buffer: float = 0
    max_latency_list_size: int = 10
    min_tokens_for_latency: int = 5  # ignore per-token normalization for tiny outputs
    max_latency_seconds_per_token: float = 60.0  # clamp outliers
    max_ttft_seconds: float = 60.0  # clamp outliers


class LowestLatencyLoggingHandler(CustomLogger):
    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __init__(
        self, router_cache: DualCache, routing_args: dict = {}
    ):
        self.router_cache = router_cache
        self.routing_args = RoutingArgs(**routing_args)

    def _append_metric(self, request_count_dict: Dict, id: str, key: str, value: Optional[float]) -> None:
        if value is None:
            return
        request_count_dict[id].setdefault(key, [])
        if (
            len(request_count_dict[id][key])
            < self.routing_args.max_latency_list_size
        ):
            request_count_dict[id][key].append(value)
        else:
            request_count_dict[id][key] = request_count_dict[id][key][
                : self.routing_args.max_latency_list_size - 1
            ] + [value]

    def _clamp(self, value: Optional[float], cap: float) -> Optional[float]:
        if value is None:
            return None
        if value < 0:
            return 0.0
        if value > cap:
            return cap
        return value

    def _to_seconds(self, delta: Union[float, timedelta]) -> float:
        return delta.total_seconds() if isinstance(delta, timedelta) else float(delta)

    def _robust_average(self, values: List[float]) -> float:
        if not values:
            return float("inf")
        try:
            return float(statistics.median(values))
        except Exception:
            return float(sum(values) / len(values))

    def _get_limit_from_deployment(self, deployment: Dict, key: str) -> float:
        for candidate in (
            deployment.get(key, None),
            deployment.get("litellm_params", {}).get(key, None),
            deployment.get("model_info", {}).get(key, None),
        ):
            if candidate is not None:
                return candidate
        return float("inf")

    def _prepare_success_metrics(
        self, kwargs, response_obj, start_time, end_time
    ) -> Optional[Dict[str, Any]]:
        metadata_field = self._select_metadata_field(kwargs)
        if kwargs["litellm_params"].get(metadata_field) is None:
            return None

        model_group = kwargs["litellm_params"][metadata_field].get(
            "model_group", None
        )

        id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
        if model_group is None or id is None:
            return None
        if isinstance(id, int):
            id = str(id)

        response_seconds = self._to_seconds(end_time - start_time)
        ttft_seconds: Optional[float] = None
        if kwargs.get("stream", None) is not None and kwargs["stream"] is True:
            ttft_seconds = self._to_seconds(
                kwargs.get("completion_start_time", end_time) - start_time
            )

        per_token_latency: Optional[float] = response_seconds
        total_tokens = 0
        completion_tokens = 0

        if isinstance(response_obj, ModelResponse):
            _usage = getattr(response_obj, "usage", None)
            if _usage is not None:
                completion_tokens = _usage.completion_tokens or 0
                total_tokens = _usage.total_tokens or 0

        # For very small completions, fall back to raw elapsed to avoid noisy per-token spikes
        if completion_tokens >= self.routing_args.min_tokens_for_latency:
            per_token_latency = safe_divide_seconds(
                response_seconds, completion_tokens, default=response_seconds
            )
        else:
            per_token_latency = response_seconds

        per_token_latency = self._clamp(
            per_token_latency, self.routing_args.max_latency_seconds_per_token
        )

        if ttft_seconds is not None:
            ttft_seconds = self._clamp(ttft_seconds, self.routing_args.max_ttft_seconds)

        return {
            "model_group": model_group,
            "id": id,
            "per_token_latency": per_token_latency,
            "ttft_seconds": ttft_seconds,
            "total_tokens": total_tokens,
        }

    def log_success_event(  # noqa: PLR0915
        self, kwargs, response_obj, start_time, end_time
    ):
        try:
            """
            Update latency usage on success
            """
            metrics = self._prepare_success_metrics(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
            if metrics is None:
                return

            model_group = metrics["model_group"]
            id = metrics["id"]
            latency_key = f"{model_group}_map"

            current_date = datetime.now().strftime("%Y-%m-%d")
            current_hour = datetime.now().strftime("%H")
            current_minute = datetime.now().strftime("%M")
            precise_minute = f"{current_date}-{current_hour}-{current_minute}"

            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            request_count_dict = (
                self.router_cache.get_cache(
                    key=latency_key, parent_otel_span=parent_otel_span
                )
                or {}
            )

            if id not in request_count_dict:
                request_count_dict[id] = {}

            self._append_metric(
                request_count_dict, id, "latency", metrics["per_token_latency"]
            )
            self._append_metric(
                request_count_dict, id, "time_to_first_token", metrics["ttft_seconds"]
            )

            if precise_minute not in request_count_dict[id]:
                request_count_dict[id][precise_minute] = {}

            request_count_dict[id][precise_minute]["tpm"] = (
                request_count_dict[id][precise_minute].get("tpm", 0)
                + metrics["total_tokens"]
            )

            request_count_dict[id][precise_minute]["rpm"] = (
                request_count_dict[id][precise_minute].get("rpm", 0) + 1
            )

            self.router_cache.set_cache(
                key=latency_key, value=request_count_dict, ttl=self.routing_args.ttl
            )  # reset map within window

            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.prompt_injection_detection.py::async_pre_call_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    def _should_penalize_exception(self, exc: Optional[BaseException]) -> bool:
        """
        Decide whether to penalize a deployment for a given exception.

        We penalize on transient/unavailability style failures so the router
        quickly prefers other instances even before cooldown kicks in.

        Penalize when:
        - Timeout
        - 5xx server errors
        - 408 Request Timeout
        - 429 Rate limit
        - API connection errors / service unavailable
        - 401/404 (misconfig on this deployment) — nudge router away
        """
        try:
            if exc is None:
                return False
            # Explicit type checks for common transient errors
            if isinstance(
                exc,
                (
                    litellm.Timeout,
                    litellm.APIConnectionError,
                    litellm.ServiceUnavailableError,
                    litellm.InternalServerError,
                    litellm.APIError,
                    litellm.RateLimitError,
                ),
            ):
                return True

            status = getattr(exc, "status_code", None)
            if isinstance(status, str):
                try:
                    status = int(status)
                except Exception:
                    status = None
            if isinstance(status, int):
                if status >= 500:
                    return True
                if status in (408, 429, 401, 404):
                    return True
            return False
        except Exception:
            return False

    def _penalize_deployment_latency(self, request_count_dict: Dict, id: str) -> None:
        """Append a high latency value to penalize a deployment in selection."""
        penalty_value = 1000.0
        if (
            len(request_count_dict[id].get("latency", []))
            < self.routing_args.max_latency_list_size
        ):
            request_count_dict[id].setdefault("latency", []).append(penalty_value)
        else:
            request_count_dict[id]["latency"] = request_count_dict[id]["latency"][
                : self.routing_args.max_latency_list_size - 1
            ] + [penalty_value]

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Sync failure hook: penalize deployment latency for transient/unavailable errors.
        """
        try:
            metadata_field = self._select_metadata_field(kwargs)
            _exception = kwargs.get("exception", None)
            if not self._should_penalize_exception(_exception):
                return

            if kwargs["litellm_params"].get(metadata_field) is None:
                return

            model_group = kwargs["litellm_params"][metadata_field].get(
                "model_group", None
            )
            id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
            if model_group is None or id is None:
                return
            if isinstance(id, int):
                id = str(id)

            latency_key = f"{model_group}_map"
            request_count_dict = self.router_cache.get_cache(key=latency_key) or {}
            if id not in request_count_dict:
                request_count_dict[id] = {}
            self._penalize_deployment_latency(request_count_dict, id)
            self.router_cache.set_cache(
                key=latency_key, value=request_count_dict, ttl=self.routing_args.ttl
            )
            if self.test_flag:
                self.logged_failure += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.router_strategy.lowest_latency.py::log_failure_event(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Async failure hook: penalize deployment latency for transient/unavailable errors.
        """
        try:
            metadata_field = self._select_metadata_field(kwargs)
            _exception = kwargs.get("exception", None)
            if not self._should_penalize_exception(_exception):
                return

            if kwargs["litellm_params"].get(metadata_field) is None:
                return

            model_group = kwargs["litellm_params"][metadata_field].get(
                "model_group", None
            )
            id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
            if model_group is None or id is None:
                return
            if isinstance(id, int):
                id = str(id)

            latency_key = f"{model_group}_map"
            request_count_dict = (
                await self.router_cache.async_get_cache(key=latency_key) or {}
            )
            if id not in request_count_dict:
                request_count_dict[id] = {}

            self._penalize_deployment_latency(request_count_dict, id)
            await self.router_cache.async_set_cache(
                key=latency_key,
                value=request_count_dict,
                ttl=self.routing_args.ttl,
            )
            if self.test_flag:
                self.logged_failure += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.router_strategy.lowest_latency.py::async_log_failure_event(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    async def async_log_success_event(  # noqa: PLR0915
        self, kwargs, response_obj, start_time, end_time
    ):
        try:
            """
            Update latency usage on success
            """
            metrics = self._prepare_success_metrics(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
            if metrics is None:
                return

            model_group = metrics["model_group"]
            id = metrics["id"]
            latency_key = f"{model_group}_map"

            current_date = datetime.now().strftime("%Y-%m-%d")
            current_hour = datetime.now().strftime("%H")
            current_minute = datetime.now().strftime("%M")
            precise_minute = f"{current_date}-{current_hour}-{current_minute}"

            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            request_count_dict = (
                await self.router_cache.async_get_cache(
                    key=latency_key,
                    parent_otel_span=parent_otel_span,
                    local_only=True,
                )
                or {}
            )

            if id not in request_count_dict:
                request_count_dict[id] = {}

            self._append_metric(
                request_count_dict, id, "latency", metrics["per_token_latency"]
            )
            self._append_metric(
                request_count_dict, id, "time_to_first_token", metrics["ttft_seconds"]
            )

            if precise_minute not in request_count_dict[id]:
                request_count_dict[id][precise_minute] = {}

            request_count_dict[id][precise_minute]["tpm"] = (
                request_count_dict[id][precise_minute].get("tpm", 0)
                + metrics["total_tokens"]
            )

            request_count_dict[id][precise_minute]["rpm"] = (
                request_count_dict[id][precise_minute].get("rpm", 0) + 1
            )

            await self.router_cache.async_set_cache(
                key=latency_key, value=request_count_dict, ttl=self.routing_args.ttl
            )  # reset map within window

            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.router_strategy.lowest_latency.py::async_log_success_event(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    def _get_available_deployments(  # noqa: PLR0915
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
        request_count_dict: Optional[Dict] = None,
    ):
        """Common logic for both sync and async get_available_deployments"""

        # -----------------------
        # Find lowest used model
        # ----------------------
        _latency_per_deployment = {}
        lowest_latency = float("inf")

        current_date = datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.now().strftime("%H")
        current_minute = datetime.now().strftime("%M")
        precise_minute = f"{current_date}-{current_hour}-{current_minute}"

        deployment = None

        if request_count_dict is None:  # base case
            return

        all_deployments = request_count_dict
        for d in healthy_deployments:
            ## if healthy deployment not yet used
            if d["model_info"]["id"] not in all_deployments:
                all_deployments[d["model_info"]["id"]] = {
                    "latency": [0],
                    precise_minute: {"tpm": 0, "rpm": 0},
                }

        try:
            input_tokens = token_counter(messages=messages, text=input)
        except Exception:
            input_tokens = 0

        # randomly sample from all_deployments, incase all deployments have latency=0.0
        _items = all_deployments.items()

        _all_deployments = random.sample(list(_items), len(_items))
        all_deployments = dict(_all_deployments)
        ### GET AVAILABLE DEPLOYMENTS ### filter out any deployments > tpm/rpm limits

        potential_deployments = []
        for item, item_map in all_deployments.items():
            ## get the item from model list
            _deployment = None
            for m in healthy_deployments:
                if item == m["model_info"]["id"]:
                    _deployment = m

            if _deployment is None:
                continue  # skip to next one

            _deployment_tpm = self._get_limit_from_deployment(_deployment, "tpm")
            _deployment_rpm = self._get_limit_from_deployment(_deployment, "rpm")

            latency_samples = item_map.get("latency", [])
            item_ttft_latency = item_map.get("time_to_first_token", [])
            item_rpm = item_map.get(precise_minute, {}).get("rpm", 0)
            item_tpm = item_map.get(precise_minute, {}).get("tpm", 0)

            use_ttft = (
                request_kwargs is not None
                and request_kwargs.get("stream", None) is True
                and len(item_ttft_latency) > 0
            )

            latency_score = self._robust_average(
                [
                    self._to_seconds(_latency_obj)
                    if isinstance(_latency_obj, timedelta)
                    else _latency_obj
                    for _latency_obj in latency_samples
                    if _latency_obj is not None
                ]
            )

            ttft_score = self._robust_average(
                [
                    self._to_seconds(_val)
                    if isinstance(_val, timedelta)
                    else _val
                    for _val in item_ttft_latency
                    if _val is not None
                ]
            )

            # For streaming, prefer TTFT first, then throughput (latency_score is seconds/token surrogate)
            sort_latency = (ttft_score if use_ttft else latency_score)
            secondary_score = latency_score if use_ttft else ttft_score

            # -------------- #
            # Debugging Logic
            # -------------- #
            # We use _latency_per_deployment to log to langfuse, slack - this is not used to make a decision on routing
            # this helps a user to debug why the router picked a specfic deployment      #
            _deployment_api_base = _deployment.get("litellm_params", {}).get(
                "api_base", ""
            )
            if _deployment_api_base is not None:
                _latency_per_deployment[_deployment_api_base] = {
                    "latency": latency_score,
                    "ttft": ttft_score if ttft_score != float("inf") else None,
                }
            # -------------- #
            # End of Debugging Logic
            # -------------- #

            if (
                item_tpm + input_tokens > _deployment_tpm
                or item_rpm + 1 > _deployment_rpm
            ):  # if user passed in tpm / rpm in the model_list
                continue
            else:
                potential_deployments.append(
                    (_deployment, sort_latency, secondary_score)
                )

        if len(potential_deployments) == 0:
            return None

        if not healthy_deployments: # This check was already present and is good
            return None

        # Sort potential deployments by latency
        sorted_deployments = sorted(potential_deployments, key=lambda x: (x[1], x[2]))

        if not sorted_deployments: # Add check for empty sorted_deployments
            return None
            
        # Get the lowest latency
        lowest_latency = sorted_deployments[0][1]

        # Find deployments within buffer of lowest latency
        buffer = self.routing_args.lowest_latency_buffer * lowest_latency

        # If no deployments within buffer, fall back to all sorted deployments
        valid_deployments = [
            x for x in sorted_deployments if x[1] <= lowest_latency + buffer
        ] or sorted_deployments # This fallback is good

        if not valid_deployments: # Add check for empty valid_deployments after fallback
             return None
            
        # Pick a random deployment from valid deployments
        random_valid_deployment = random.choice(valid_deployments)
        deployment = random_valid_deployment[0]
        metadata_field = self._select_metadata_field(request_kwargs)
        if request_kwargs is not None and metadata_field in request_kwargs:
            request_kwargs[metadata_field][
                "_latency_per_deployment"
            ] = _latency_per_deployment
        return deployment

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ):
        # get list of potential deployments
        latency_key = f"{model_group}_map"

        parent_otel_span: Optional[Span] = _get_parent_otel_span_from_kwargs(
            request_kwargs
        )
        request_count_dict = (
            await self.router_cache.async_get_cache(
                key=latency_key, parent_otel_span=parent_otel_span
            )
            or {}
        )

        return self._get_available_deployments(
            model_group,
            healthy_deployments,
            messages,
            input,
            request_kwargs,
            request_count_dict,
        )

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ):
        """
        Returns a deployment with the lowest latency
        """
        # get list of potential deployments
        latency_key = f"{model_group}_map"

        parent_otel_span: Optional[Span] = _get_parent_otel_span_from_kwargs(
            request_kwargs
        )
        request_count_dict = (
            self.router_cache.get_cache(
                key=latency_key, parent_otel_span=parent_otel_span
            )
            or {}
        )

        return self._get_available_deployments(
            model_group,
            healthy_deployments,
            messages,
            input,
            request_kwargs,
            request_count_dict,
        )
