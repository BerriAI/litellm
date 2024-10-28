# used for /metrics endpoint on LiteLLM Proxy
#### What this does ####
#    On success, log events to Prometheus
import os
import subprocess
import sys
import traceback
import uuid
from datetime import date, datetime, timedelta
from typing import Optional, TypedDict, Union

import dotenv
import requests  # type: ignore

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.integrations.prometheus import *
from litellm.types.utils import StandardLoggingPayload


class PrometheusLogger(CustomLogger):
    # Class variables or attributes
    def __init__(
        self,
        **kwargs,
    ):
        try:
            from prometheus_client import Counter, Gauge, Histogram

            from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

            if premium_user is not True:
                verbose_logger.warning(
                    f"🚨🚨🚨 Prometheus Metrics is on LiteLLM Enterprise\n🚨 {CommonProxyErrors.not_premium_user.value}"
                )
                self.litellm_not_a_premium_user_metric = Counter(
                    name="litellm_not_a_premium_user_metric",
                    documentation=f"🚨🚨🚨 Prometheus Metrics is on LiteLLM Enterprise. 🚨 {CommonProxyErrors.not_premium_user.value}",
                )
                return

            self.litellm_proxy_failed_requests_metric = Counter(
                name="litellm_proxy_failed_requests_metric",
                documentation="Total number of failed responses from proxy - the client did not get a success response from litellm proxy",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    REQUESTED_MODEL,
                    "team",
                    "team_alias",
                    "user",
                ]
                + EXCEPTION_LABELS,
            )

            self.litellm_proxy_total_requests_metric = Counter(
                name="litellm_proxy_total_requests_metric",
                documentation="Total number of requests made to the proxy server - track number of client side requests",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    REQUESTED_MODEL,
                    "team",
                    "team_alias",
                    "user",
                ],
            )

            # request latency metrics
            self.litellm_request_total_latency_metric = Histogram(
                "litellm_request_total_latency_metric",
                "Total latency (seconds) for a request to LiteLLM",
                labelnames=[
                    "model",
                    "hashed_api_key",
                    "api_key_alias",
                    "team",
                    "team_alias",
                ],
                buckets=LATENCY_BUCKETS,
            )

            self.litellm_llm_api_latency_metric = Histogram(
                "litellm_llm_api_latency_metric",
                "Total latency (seconds) for a models LLM API call",
                labelnames=[
                    "model",
                    "hashed_api_key",
                    "api_key_alias",
                    "team",
                    "team_alias",
                ],
                buckets=LATENCY_BUCKETS,
            )

            self.litellm_llm_api_time_to_first_token_metric = Histogram(
                "litellm_llm_api_time_to_first_token_metric",
                "Time to first token for a models LLM API call",
                labelnames=[
                    "model",
                    "hashed_api_key",
                    "api_key_alias",
                    "team",
                    "team_alias",
                ],
                buckets=LATENCY_BUCKETS,
            )

            # Counter for spend
            self.litellm_spend_metric = Counter(
                "litellm_spend_metric",
                "Total spend on LLM requests",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    "model",
                    "team",
                    "team_alias",
                    "user",
                ],
            )

            # Counter for total_output_tokens
            self.litellm_tokens_metric = Counter(
                "litellm_total_tokens",
                "Total number of input + output tokens from LLM requests",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    "model",
                    "team",
                    "team_alias",
                    "user",
                ],
            )

            self.litellm_input_tokens_metric = Counter(
                "litellm_input_tokens",
                "Total number of input tokens from LLM requests",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    "model",
                    "team",
                    "team_alias",
                    "user",
                ],
            )
            self.litellm_output_tokens_metric = Counter(
                "litellm_output_tokens",
                "Total number of output tokens from LLM requests",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    "model",
                    "team",
                    "team_alias",
                    "user",
                ],
            )

            # Remaining Budget for Team
            self.litellm_remaining_team_budget_metric = Gauge(
                "litellm_remaining_team_budget_metric",
                "Remaining budget for team",
                labelnames=["team_id", "team_alias"],
            )

            # Remaining Budget for API Key
            self.litellm_remaining_api_key_budget_metric = Gauge(
                "litellm_remaining_api_key_budget_metric",
                "Remaining budget for api key",
                labelnames=["hashed_api_key", "api_key_alias"],
            )

            ########################################
            # LiteLLM Virtual API KEY metrics
            ########################################
            # Remaining MODEL RPM limit for API Key
            self.litellm_remaining_api_key_requests_for_model = Gauge(
                "litellm_remaining_api_key_requests_for_model",
                "Remaining Requests API Key can make for model (model based rpm limit on key)",
                labelnames=["hashed_api_key", "api_key_alias", "model"],
            )

            # Remaining MODEL TPM limit for API Key
            self.litellm_remaining_api_key_tokens_for_model = Gauge(
                "litellm_remaining_api_key_tokens_for_model",
                "Remaining Tokens API Key can make for model (model based tpm limit on key)",
                labelnames=["hashed_api_key", "api_key_alias", "model"],
            )

            ########################################
            # LLM API Deployment Metrics / analytics
            ########################################

            # Remaining Rate Limit for model
            self.litellm_remaining_requests_metric = Gauge(
                "litellm_remaining_requests",
                "LLM Deployment Analytics - remaining requests for model, returned from LLM API Provider",
                labelnames=[
                    "model_group",
                    "api_provider",
                    "api_base",
                    "litellm_model_name",
                    "hashed_api_key",
                    "api_key_alias",
                ],
            )

            self.litellm_remaining_tokens_metric = Gauge(
                "litellm_remaining_tokens",
                "remaining tokens for model, returned from LLM API Provider",
                labelnames=[
                    "model_group",
                    "api_provider",
                    "api_base",
                    "litellm_model_name",
                    "hashed_api_key",
                    "api_key_alias",
                ],
            )
            # Get all keys
            _logged_llm_labels = [
                "litellm_model_name",
                "model_id",
                "api_base",
                "api_provider",
            ]
            team_and_key_labels = [
                "hashed_api_key",
                "api_key_alias",
                "team",
                "team_alias",
            ]

            # Metric for deployment state
            self.litellm_deployment_state = Gauge(
                "litellm_deployment_state",
                "LLM Deployment Analytics - The state of the deployment: 0 = healthy, 1 = partial outage, 2 = complete outage",
                labelnames=_logged_llm_labels,
            )

            self.litellm_deployment_cooled_down = Counter(
                "litellm_deployment_cooled_down",
                "LLM Deployment Analytics - Number of times a deployment has been cooled down by LiteLLM load balancing logic. exception_status is the status of the exception that caused the deployment to be cooled down",
                labelnames=_logged_llm_labels + [EXCEPTION_STATUS],
            )

            self.litellm_deployment_success_responses = Counter(
                name="litellm_deployment_success_responses",
                documentation="LLM Deployment Analytics - Total number of successful LLM API calls via litellm",
                labelnames=[REQUESTED_MODEL] + _logged_llm_labels + team_and_key_labels,
            )
            self.litellm_deployment_failure_responses = Counter(
                name="litellm_deployment_failure_responses",
                documentation="LLM Deployment Analytics - Total number of failed LLM API calls for a specific LLM deploymeny. exception_status is the status of the exception from the llm api",
                labelnames=[REQUESTED_MODEL]
                + _logged_llm_labels
                + EXCEPTION_LABELS
                + team_and_key_labels,
            )
            self.litellm_deployment_total_requests = Counter(
                name="litellm_deployment_total_requests",
                documentation="LLM Deployment Analytics - Total number of LLM API calls via litellm - success + failure",
                labelnames=[REQUESTED_MODEL] + _logged_llm_labels + team_and_key_labels,
            )

            # Deployment Latency tracking
            team_and_key_labels = [
                "hashed_api_key",
                "api_key_alias",
                "team",
                "team_alias",
            ]
            self.litellm_deployment_latency_per_output_token = Histogram(
                name="litellm_deployment_latency_per_output_token",
                documentation="LLM Deployment Analytics - Latency per output token",
                labelnames=_logged_llm_labels + team_and_key_labels,
            )

            self.litellm_deployment_successful_fallbacks = Counter(
                "litellm_deployment_successful_fallbacks",
                "LLM Deployment Analytics - Number of successful fallback requests from primary model -> fallback model",
                [REQUESTED_MODEL, "fallback_model"]
                + team_and_key_labels
                + EXCEPTION_LABELS,
            )
            self.litellm_deployment_failed_fallbacks = Counter(
                "litellm_deployment_failed_fallbacks",
                "LLM Deployment Analytics - Number of failed fallback requests from primary model -> fallback model",
                [REQUESTED_MODEL, "fallback_model"]
                + team_and_key_labels
                + EXCEPTION_LABELS,
            )

            self.litellm_llm_api_failed_requests_metric = Counter(
                name="litellm_llm_api_failed_requests_metric",
                documentation="deprecated - use litellm_proxy_failed_requests_metric",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    "model",
                    "team",
                    "team_alias",
                    "user",
                ],
            )

            self.litellm_requests_metric = Counter(
                name="litellm_requests_metric",
                documentation="deprecated - use litellm_proxy_total_requests_metric. Total number of LLM calls to litellm - track total per API Key, team, user",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    "model",
                    "team",
                    "team_alias",
                    "user",
                ],
            )

        except Exception as e:
            print_verbose(f"Got exception on init prometheus client {str(e)}")
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Define prometheus client
        from litellm.types.utils import StandardLoggingPayload

        verbose_logger.debug(
            f"prometheus Logging - Enters success logging function for kwargs {kwargs}"
        )

        # unpack kwargs
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )

        if standard_logging_payload is None or not isinstance(
            standard_logging_payload, dict
        ):
            raise ValueError(
                f"standard_logging_object is required, got={standard_logging_payload}"
            )

        model = kwargs.get("model", "")
        litellm_params = kwargs.get("litellm_params", {}) or {}
        _metadata = litellm_params.get("metadata", {})
        proxy_server_request = litellm_params.get("proxy_server_request") or {}
        end_user_id = proxy_server_request.get("body", {}).get("user", None)
        user_id = standard_logging_payload["metadata"]["user_api_key_user_id"]
        user_api_key = standard_logging_payload["metadata"]["user_api_key_hash"]
        user_api_key_alias = standard_logging_payload["metadata"]["user_api_key_alias"]
        user_api_team = standard_logging_payload["metadata"]["user_api_key_team_id"]
        user_api_team_alias = standard_logging_payload["metadata"][
            "user_api_key_team_alias"
        ]
        output_tokens = standard_logging_payload["completion_tokens"]
        tokens_used = standard_logging_payload["total_tokens"]
        response_cost = standard_logging_payload["response_cost"]

        print_verbose(
            f"inside track_prometheus_metrics, model {model}, response_cost {response_cost}, tokens_used {tokens_used}, end_user_id {end_user_id}, user_api_key {user_api_key}"
        )

        if (
            user_api_key is not None
            and isinstance(user_api_key, str)
            and user_api_key.startswith("sk-")
        ):
            from litellm.proxy.utils import hash_token

            user_api_key = hash_token(user_api_key)

        # increment total LLM requests and spend metric
        self._increment_top_level_request_and_spend_metrics(
            end_user_id=end_user_id,
            user_api_key=user_api_key,
            user_api_key_alias=user_api_key_alias,
            model=model,
            user_api_team=user_api_team,
            user_api_team_alias=user_api_team_alias,
            user_id=user_id,
            response_cost=response_cost,
        )

        # input, output, total token metrics
        self._increment_token_metrics(
            standard_logging_payload=standard_logging_payload,
            end_user_id=end_user_id,
            user_api_key=user_api_key,
            user_api_key_alias=user_api_key_alias,
            model=model,
            user_api_team=user_api_team,
            user_api_team_alias=user_api_team_alias,
            user_id=user_id,
        )

        # remaining budget metrics
        self._increment_remaining_budget_metrics(
            user_api_team=user_api_team,
            user_api_team_alias=user_api_team_alias,
            user_api_key=user_api_key,
            user_api_key_alias=user_api_key_alias,
            litellm_params=litellm_params,
        )

        # set proxy virtual key rpm/tpm metrics
        self._set_virtual_key_rate_limit_metrics(
            user_api_key=user_api_key,
            user_api_key_alias=user_api_key_alias,
            kwargs=kwargs,
            metadata=_metadata,
        )

        # set latency metrics
        self._set_latency_metrics(
            kwargs=kwargs,
            model=model,
            user_api_key=user_api_key,
            user_api_key_alias=user_api_key_alias,
            user_api_team=user_api_team,
            user_api_team_alias=user_api_team_alias,
            standard_logging_payload=standard_logging_payload,
        )

        # set x-ratelimit headers
        self.set_llm_deployment_success_metrics(
            kwargs, start_time, end_time, output_tokens
        )
        pass

    def _increment_token_metrics(
        self,
        standard_logging_payload: StandardLoggingPayload,
        end_user_id: Optional[str],
        user_api_key: Optional[str],
        user_api_key_alias: Optional[str],
        model: Optional[str],
        user_api_team: Optional[str],
        user_api_team_alias: Optional[str],
        user_id: Optional[str],
    ):
        # token metrics
        self.litellm_tokens_metric.labels(
            end_user_id,
            user_api_key,
            user_api_key_alias,
            model,
            user_api_team,
            user_api_team_alias,
            user_id,
        ).inc(standard_logging_payload["total_tokens"])

        self.litellm_input_tokens_metric.labels(
            end_user_id,
            user_api_key,
            user_api_key_alias,
            model,
            user_api_team,
            user_api_team_alias,
            user_id,
        ).inc(standard_logging_payload["prompt_tokens"])

        self.litellm_output_tokens_metric.labels(
            end_user_id,
            user_api_key,
            user_api_key_alias,
            model,
            user_api_team,
            user_api_team_alias,
            user_id,
        ).inc(standard_logging_payload["completion_tokens"])

    def _increment_remaining_budget_metrics(
        self,
        user_api_team: Optional[str],
        user_api_team_alias: Optional[str],
        user_api_key: Optional[str],
        user_api_key_alias: Optional[str],
        litellm_params: dict,
    ):
        _team_spend = litellm_params.get("metadata", {}).get(
            "user_api_key_team_spend", None
        )
        _team_max_budget = litellm_params.get("metadata", {}).get(
            "user_api_key_team_max_budget", None
        )
        _remaining_team_budget = self._safe_get_remaining_budget(
            max_budget=_team_max_budget, spend=_team_spend
        )

        _api_key_spend = litellm_params.get("metadata", {}).get(
            "user_api_key_spend", None
        )
        _api_key_max_budget = litellm_params.get("metadata", {}).get(
            "user_api_key_max_budget", None
        )
        _remaining_api_key_budget = self._safe_get_remaining_budget(
            max_budget=_api_key_max_budget, spend=_api_key_spend
        )
        # Remaining Budget Metrics
        self.litellm_remaining_team_budget_metric.labels(
            user_api_team, user_api_team_alias
        ).set(_remaining_team_budget)

        self.litellm_remaining_api_key_budget_metric.labels(
            user_api_key, user_api_key_alias
        ).set(_remaining_api_key_budget)

    def _increment_top_level_request_and_spend_metrics(
        self,
        end_user_id: Optional[str],
        user_api_key: Optional[str],
        user_api_key_alias: Optional[str],
        model: Optional[str],
        user_api_team: Optional[str],
        user_api_team_alias: Optional[str],
        user_id: Optional[str],
        response_cost: float,
    ):
        self.litellm_requests_metric.labels(
            end_user_id,
            user_api_key,
            user_api_key_alias,
            model,
            user_api_team,
            user_api_team_alias,
            user_id,
        ).inc()
        self.litellm_spend_metric.labels(
            end_user_id,
            user_api_key,
            user_api_key_alias,
            model,
            user_api_team,
            user_api_team_alias,
            user_id,
        ).inc(response_cost)

    def _set_virtual_key_rate_limit_metrics(
        self,
        user_api_key: Optional[str],
        user_api_key_alias: Optional[str],
        kwargs: dict,
        metadata: dict,
    ):
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,
        )

        # Set remaining rpm/tpm for API Key + model
        # see parallel_request_limiter.py - variables are set there
        model_group = get_model_group_from_litellm_kwargs(kwargs)
        remaining_requests_variable_name = (
            f"litellm-key-remaining-requests-{model_group}"
        )
        remaining_tokens_variable_name = f"litellm-key-remaining-tokens-{model_group}"

        remaining_requests = metadata.get(remaining_requests_variable_name, sys.maxsize)
        remaining_tokens = metadata.get(remaining_tokens_variable_name, sys.maxsize)

        self.litellm_remaining_api_key_requests_for_model.labels(
            user_api_key, user_api_key_alias, model_group
        ).set(remaining_requests)

        self.litellm_remaining_api_key_tokens_for_model.labels(
            user_api_key, user_api_key_alias, model_group
        ).set(remaining_tokens)

    def _set_latency_metrics(
        self,
        kwargs: dict,
        model: Optional[str],
        user_api_key: Optional[str],
        user_api_key_alias: Optional[str],
        user_api_team: Optional[str],
        user_api_team_alias: Optional[str],
        standard_logging_payload: StandardLoggingPayload,
    ):
        # latency metrics
        model_parameters: dict = standard_logging_payload["model_parameters"]
        end_time: datetime = kwargs.get("end_time") or datetime.now()
        start_time: Optional[datetime] = kwargs.get("start_time")
        api_call_start_time = kwargs.get("api_call_start_time", None)

        completion_start_time = kwargs.get("completion_start_time", None)

        if (
            completion_start_time is not None
            and isinstance(completion_start_time, datetime)
            and model_parameters.get("stream")
            is True  # only emit for streaming requests
        ):
            time_to_first_token_seconds = (
                completion_start_time - api_call_start_time
            ).total_seconds()
            self.litellm_llm_api_time_to_first_token_metric.labels(
                model,
                user_api_key,
                user_api_key_alias,
                user_api_team,
                user_api_team_alias,
            ).observe(time_to_first_token_seconds)
        else:
            verbose_logger.debug(
                "Time to first token metric not emitted, stream option in model_parameters is not True"
            )
        if api_call_start_time is not None and isinstance(
            api_call_start_time, datetime
        ):
            api_call_total_time: timedelta = end_time - api_call_start_time
            api_call_total_time_seconds = api_call_total_time.total_seconds()
            self.litellm_llm_api_latency_metric.labels(
                model,
                user_api_key,
                user_api_key_alias,
                user_api_team,
                user_api_team_alias,
            ).observe(api_call_total_time_seconds)

        # total request latency
        if start_time is not None and isinstance(start_time, datetime):
            total_time: timedelta = end_time - start_time
            total_time_seconds = total_time.total_seconds()
            self.litellm_request_total_latency_metric.labels(
                model,
                user_api_key,
                user_api_key_alias,
                user_api_team,
                user_api_team_alias,
            ).observe(total_time_seconds)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.types.utils import StandardLoggingPayload

        verbose_logger.debug(
            f"prometheus Logging - Enters failure logging function for kwargs {kwargs}"
        )

        # unpack kwargs
        model = kwargs.get("model", "")
        litellm_params = kwargs.get("litellm_params", {}) or {}
        standard_logging_payload: StandardLoggingPayload = kwargs.get(
            "standard_logging_object", {}
        )
        proxy_server_request = litellm_params.get("proxy_server_request") or {}

        end_user_id = proxy_server_request.get("body", {}).get("user", None)
        user_id = standard_logging_payload["metadata"]["user_api_key_user_id"]
        user_api_key = standard_logging_payload["metadata"]["user_api_key_hash"]
        user_api_key_alias = standard_logging_payload["metadata"]["user_api_key_alias"]
        user_api_team = standard_logging_payload["metadata"]["user_api_key_team_id"]
        user_api_team_alias = standard_logging_payload["metadata"][
            "user_api_key_team_alias"
        ]
        kwargs.get("exception", None)

        try:
            self.litellm_llm_api_failed_requests_metric.labels(
                end_user_id,
                user_api_key,
                user_api_key_alias,
                model,
                user_api_team,
                user_api_team_alias,
                user_id,
            ).inc()
            self.set_llm_deployment_failure_metrics(kwargs)
        except Exception as e:
            verbose_logger.exception(
                "prometheus Layer Error(): Exception occured - {}".format(str(e))
            )
            pass
        pass

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
    ):
        """
        Track client side failures

        Proxy level tracking - failed client side requests

        labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    REQUESTED_MODEL,
                    "team",
                    "team_alias",
                ] + EXCEPTION_LABELS,
        """
        try:
            self.litellm_proxy_failed_requests_metric.labels(
                end_user=user_api_key_dict.end_user_id,
                hashed_api_key=user_api_key_dict.api_key,
                api_key_alias=user_api_key_dict.key_alias,
                requested_model=request_data.get("model", ""),
                team=user_api_key_dict.team_id,
                team_alias=user_api_key_dict.team_alias,
                user=user_api_key_dict.user_id,
                exception_status=getattr(original_exception, "status_code", None),
                exception_class=str(original_exception.__class__.__name__),
            ).inc()

            self.litellm_proxy_total_requests_metric.labels(
                user_api_key_dict.end_user_id,
                user_api_key_dict.api_key,
                user_api_key_dict.key_alias,
                request_data.get("model", ""),
                user_api_key_dict.team_id,
                user_api_key_dict.team_alias,
                user_api_key_dict.user_id,
            ).inc()
            pass
        except Exception as e:
            verbose_logger.exception(
                "prometheus Layer Error(): Exception occured - {}".format(str(e))
            )
            pass

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ):
        """
        Proxy level tracking - triggered when the proxy responds with a success response to the client
        """
        try:
            self.litellm_proxy_total_requests_metric.labels(
                user_api_key_dict.end_user_id,
                user_api_key_dict.api_key,
                user_api_key_dict.key_alias,
                data.get("model", ""),
                user_api_key_dict.team_id,
                user_api_key_dict.team_alias,
                user_api_key_dict.user_id,
            ).inc()
        except Exception as e:
            verbose_logger.exception(
                "prometheus Layer Error(): Exception occured - {}".format(str(e))
            )
            pass

    def set_llm_deployment_failure_metrics(self, request_kwargs: dict):
        try:
            verbose_logger.debug("setting remaining tokens requests metric")
            standard_logging_payload: StandardLoggingPayload = request_kwargs.get(
                "standard_logging_object", {}
            )
            _response_headers = request_kwargs.get("response_headers")
            _litellm_params = request_kwargs.get("litellm_params", {}) or {}
            _metadata = _litellm_params.get("metadata", {})
            litellm_model_name = request_kwargs.get("model", None)
            api_base = _metadata.get("api_base", None)
            model_group = _metadata.get("model_group", None)
            if api_base is None:
                api_base = _litellm_params.get("api_base", None)
            llm_provider = _litellm_params.get("custom_llm_provider", None)
            _model_info = _metadata.get("model_info") or {}
            model_id = _model_info.get("id", None)
            exception: Exception = request_kwargs.get("exception", None)

            """
            log these labels
            ["litellm_model_name", "model_id", "api_base", "api_provider"]
            """
            self.set_deployment_partial_outage(
                litellm_model_name=litellm_model_name,
                model_id=model_id,
                api_base=api_base,
                api_provider=llm_provider,
            )
            self.litellm_deployment_failure_responses.labels(
                litellm_model_name=litellm_model_name,
                model_id=model_id,
                api_base=api_base,
                api_provider=llm_provider,
                exception_status=str(getattr(exception, "status_code", None)),
                exception_class=exception.__class__.__name__,
                requested_model=model_group,
                hashed_api_key=standard_logging_payload["metadata"][
                    "user_api_key_hash"
                ],
                api_key_alias=standard_logging_payload["metadata"][
                    "user_api_key_alias"
                ],
                team=standard_logging_payload["metadata"]["user_api_key_team_id"],
                team_alias=standard_logging_payload["metadata"][
                    "user_api_key_team_alias"
                ],
            ).inc()

            self.litellm_deployment_total_requests.labels(
                litellm_model_name=litellm_model_name,
                model_id=model_id,
                api_base=api_base,
                api_provider=llm_provider,
                requested_model=model_group,
                hashed_api_key=standard_logging_payload["metadata"][
                    "user_api_key_hash"
                ],
                api_key_alias=standard_logging_payload["metadata"][
                    "user_api_key_alias"
                ],
                team=standard_logging_payload["metadata"]["user_api_key_team_id"],
                team_alias=standard_logging_payload["metadata"][
                    "user_api_key_team_alias"
                ],
            ).inc()

            pass
        except Exception:
            pass

    def set_llm_deployment_success_metrics(
        self,
        request_kwargs: dict,
        start_time,
        end_time,
        output_tokens: float = 1.0,
    ):
        try:
            verbose_logger.debug("setting remaining tokens requests metric")
            standard_logging_payload: StandardLoggingPayload = request_kwargs.get(
                "standard_logging_object", {}
            )
            model_group = standard_logging_payload["model_group"]
            api_base = standard_logging_payload["api_base"]
            _response_headers = request_kwargs.get("response_headers")
            _litellm_params = request_kwargs.get("litellm_params", {}) or {}
            _metadata = _litellm_params.get("metadata", {})
            litellm_model_name = request_kwargs.get("model", None)
            llm_provider = _litellm_params.get("custom_llm_provider", None)
            _model_info = _metadata.get("model_info") or {}
            model_id = _model_info.get("id", None)

            remaining_requests = None
            remaining_tokens = None
            # OpenAI / OpenAI Compatible headers
            if (
                _response_headers
                and "x-ratelimit-remaining-requests" in _response_headers
            ):
                remaining_requests = _response_headers["x-ratelimit-remaining-requests"]
            if (
                _response_headers
                and "x-ratelimit-remaining-tokens" in _response_headers
            ):
                remaining_tokens = _response_headers["x-ratelimit-remaining-tokens"]
            verbose_logger.debug(
                f"remaining requests: {remaining_requests}, remaining tokens: {remaining_tokens}"
            )

            if remaining_requests:
                """
                "model_group",
                "api_provider",
                "api_base",
                "litellm_model_name"
                """
                self.litellm_remaining_requests_metric.labels(
                    model_group,
                    llm_provider,
                    api_base,
                    litellm_model_name,
                    standard_logging_payload["metadata"]["user_api_key_hash"],
                    standard_logging_payload["metadata"]["user_api_key_alias"],
                ).set(remaining_requests)

            if remaining_tokens:
                self.litellm_remaining_tokens_metric.labels(
                    model_group,
                    llm_provider,
                    api_base,
                    litellm_model_name,
                    standard_logging_payload["metadata"]["user_api_key_hash"],
                    standard_logging_payload["metadata"]["user_api_key_alias"],
                ).set(remaining_tokens)

            """
            log these labels
            ["litellm_model_name", "requested_model", model_id", "api_base", "api_provider"]
            """
            self.set_deployment_healthy(
                litellm_model_name=litellm_model_name,
                model_id=model_id,
                api_base=api_base,
                api_provider=llm_provider,
            )

            self.litellm_deployment_success_responses.labels(
                litellm_model_name=litellm_model_name,
                model_id=model_id,
                api_base=api_base,
                api_provider=llm_provider,
                requested_model=model_group,
                hashed_api_key=standard_logging_payload["metadata"][
                    "user_api_key_hash"
                ],
                api_key_alias=standard_logging_payload["metadata"][
                    "user_api_key_alias"
                ],
                team=standard_logging_payload["metadata"]["user_api_key_team_id"],
                team_alias=standard_logging_payload["metadata"][
                    "user_api_key_team_alias"
                ],
            ).inc()

            self.litellm_deployment_total_requests.labels(
                litellm_model_name=litellm_model_name,
                model_id=model_id,
                api_base=api_base,
                api_provider=llm_provider,
                requested_model=model_group,
                hashed_api_key=standard_logging_payload["metadata"][
                    "user_api_key_hash"
                ],
                api_key_alias=standard_logging_payload["metadata"][
                    "user_api_key_alias"
                ],
                team=standard_logging_payload["metadata"]["user_api_key_team_id"],
                team_alias=standard_logging_payload["metadata"][
                    "user_api_key_team_alias"
                ],
            ).inc()

            # Track deployment Latency
            response_ms: timedelta = end_time - start_time
            time_to_first_token_response_time: Optional[timedelta] = None

            if (
                request_kwargs.get("stream", None) is not None
                and request_kwargs["stream"] is True
            ):
                # only log ttft for streaming request
                time_to_first_token_response_time = (
                    request_kwargs.get("completion_start_time", end_time) - start_time
                )

            # use the metric that is not None
            # if streaming - use time_to_first_token_response
            # if not streaming - use response_ms
            _latency: timedelta = time_to_first_token_response_time or response_ms
            _latency_seconds = _latency.total_seconds()

            # latency per output token
            latency_per_token = None
            if output_tokens is not None and output_tokens > 0:
                latency_per_token = _latency_seconds / output_tokens
                self.litellm_deployment_latency_per_output_token.labels(
                    litellm_model_name=litellm_model_name,
                    model_id=model_id,
                    api_base=api_base,
                    api_provider=llm_provider,
                    hashed_api_key=standard_logging_payload["metadata"][
                        "user_api_key_hash"
                    ],
                    api_key_alias=standard_logging_payload["metadata"][
                        "user_api_key_alias"
                    ],
                    team=standard_logging_payload["metadata"]["user_api_key_team_id"],
                    team_alias=standard_logging_payload["metadata"][
                        "user_api_key_team_alias"
                    ],
                ).observe(latency_per_token)

        except Exception as e:
            verbose_logger.error(
                "Prometheus Error: set_llm_deployment_success_metrics. Exception occured - {}".format(
                    str(e)
                )
            )
            return

    async def log_success_fallback_event(
        self, original_model_group: str, kwargs: dict, original_exception: Exception
    ):
        """

        Logs a successful LLM fallback event on prometheus

        """
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingMetadata,
            StandardLoggingPayloadSetup,
        )

        verbose_logger.debug(
            "Prometheus: log_success_fallback_event, original_model_group: %s, kwargs: %s",
            original_model_group,
            kwargs,
        )
        _metadata = kwargs.get("metadata", {})
        standard_metadata: StandardLoggingMetadata = (
            StandardLoggingPayloadSetup.get_standard_logging_metadata(
                metadata=_metadata
            )
        )
        _new_model = kwargs.get("model")
        self.litellm_deployment_successful_fallbacks.labels(
            requested_model=original_model_group,
            fallback_model=_new_model,
            hashed_api_key=standard_metadata["user_api_key_hash"],
            api_key_alias=standard_metadata["user_api_key_alias"],
            team=standard_metadata["user_api_key_team_id"],
            team_alias=standard_metadata["user_api_key_team_alias"],
            exception_status=str(getattr(original_exception, "status_code", None)),
            exception_class=str(original_exception.__class__.__name__),
        ).inc()

    async def log_failure_fallback_event(
        self, original_model_group: str, kwargs: dict, original_exception: Exception
    ):
        """
        Logs a failed LLM fallback event on prometheus
        """
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingMetadata,
            StandardLoggingPayloadSetup,
        )

        verbose_logger.debug(
            "Prometheus: log_failure_fallback_event, original_model_group: %s, kwargs: %s",
            original_model_group,
            kwargs,
        )
        _new_model = kwargs.get("model")
        _metadata = kwargs.get("metadata", {})
        standard_metadata: StandardLoggingMetadata = (
            StandardLoggingPayloadSetup.get_standard_logging_metadata(
                metadata=_metadata
            )
        )
        self.litellm_deployment_failed_fallbacks.labels(
            requested_model=original_model_group,
            fallback_model=_new_model,
            hashed_api_key=standard_metadata["user_api_key_hash"],
            api_key_alias=standard_metadata["user_api_key_alias"],
            team=standard_metadata["user_api_key_team_id"],
            team_alias=standard_metadata["user_api_key_team_alias"],
            exception_status=str(getattr(original_exception, "status_code", None)),
            exception_class=str(original_exception.__class__.__name__),
        ).inc()

    def set_litellm_deployment_state(
        self,
        state: int,
        litellm_model_name: str,
        model_id: str,
        api_base: str,
        api_provider: str,
    ):
        self.litellm_deployment_state.labels(
            litellm_model_name, model_id, api_base, api_provider
        ).set(state)

    def set_deployment_healthy(
        self,
        litellm_model_name: str,
        model_id: str,
        api_base: str,
        api_provider: str,
    ):
        self.set_litellm_deployment_state(
            0, litellm_model_name, model_id, api_base, api_provider
        )

    def set_deployment_partial_outage(
        self,
        litellm_model_name: str,
        model_id: str,
        api_base: str,
        api_provider: str,
    ):
        self.set_litellm_deployment_state(
            1, litellm_model_name, model_id, api_base, api_provider
        )

    def set_deployment_complete_outage(
        self,
        litellm_model_name: str,
        model_id: str,
        api_base: str,
        api_provider: str,
    ):
        self.set_litellm_deployment_state(
            2, litellm_model_name, model_id, api_base, api_provider
        )

    def increment_deployment_cooled_down(
        self,
        litellm_model_name: str,
        model_id: str,
        api_base: str,
        api_provider: str,
        exception_status: str,
    ):
        """
        increment metric when litellm.Router / load balancing logic places a deployment in cool down
        """
        self.litellm_deployment_cooled_down.labels(
            litellm_model_name, model_id, api_base, api_provider, exception_status
        ).inc()

    def _safe_get_remaining_budget(
        self, max_budget: Optional[float], spend: Optional[float]
    ) -> float:
        if max_budget is None:
            return float("inf")

        if spend is None:
            return max_budget

        return max_budget - spend
