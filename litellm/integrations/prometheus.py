# used for /metrics endpoint on LiteLLM Proxy
#### What this does ####
#    On success, log events to Prometheus
import os
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Optional, TypedDict, Union

import dotenv
import requests  # type: ignore

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.integrations.custom_logger import CustomLogger


class PrometheusLogger(CustomLogger):
    # Class variables or attributes
    def __init__(
        self,
        **kwargs,
    ):
        try:
            from prometheus_client import Counter, Gauge, Histogram

            from litellm.proxy.proxy_server import premium_user

            verbose_logger.warning(
                "🚨🚨🚨 Prometheus Metrics will be moving to LiteLLM Enterprise on September 15th, 2024.\n🚨 Contact us here to get a license https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat \n🚨 Enterprise Pricing: https://www.litellm.ai/#pricing"
            )

            self.litellm_llm_api_failed_requests_metric = Counter(
                name="litellm_llm_api_failed_requests_metric",
                documentation="Total number of failed LLM API calls via litellm - track fails per API Key, team, user",
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
                documentation="Total number of LLM calls to litellm - track total per API Key, team, user",
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

            # request latency metrics
            self.litellm_request_total_latency_metric = Histogram(
                "litellm_request_total_latency_metric",
                "Total latency (seconds) for a request to LiteLLM",
                labelnames=[
                    "model",
                ],
            )

            self.litellm_llm_api_latency_metric = Histogram(
                "litellm_llm_api_latency_metric",
                "Total latency (seconds) for a models LLM API call",
                labelnames=[
                    "model",
                ],
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

            # Litellm-Enterprise Metrics
            if premium_user is True:

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
                    ],
                )
                # Get all keys
                _logged_llm_labels = [
                    "litellm_model_name",
                    "model_id",
                    "api_base",
                    "api_provider",
                ]

                # Metric for deployment state
                self.litellm_deployment_state = Gauge(
                    "litellm_deployment_state",
                    "LLM Deployment Analytics - The state of the deployment: 0 = healthy, 1 = partial outage, 2 = complete outage",
                    labelnames=_logged_llm_labels,
                )

                self.litellm_deployment_success_responses = Counter(
                    name="litellm_deployment_success_responses",
                    documentation="LLM Deployment Analytics - Total number of successful LLM API calls via litellm",
                    labelnames=_logged_llm_labels,
                )
                self.litellm_deployment_failure_responses = Counter(
                    name="litellm_deployment_failure_responses",
                    documentation="LLM Deployment Analytics - Total number of failed LLM API calls via litellm",
                    labelnames=_logged_llm_labels,
                )
                self.litellm_deployment_total_requests = Counter(
                    name="litellm_deployment_total_requests",
                    documentation="LLM Deployment Analytics - Total number of LLM API calls via litellm - success + failure",
                    labelnames=_logged_llm_labels,
                )

                # Deployment Latency tracking
                self.litellm_deployment_latency_per_output_token = Histogram(
                    name="litellm_deployment_latency_per_output_token",
                    documentation="LLM Deployment Analytics - Latency per output token",
                    labelnames=_logged_llm_labels,
                )

                self.litellm_deployment_successful_fallbacks = Counter(
                    "litellm_deployment_successful_fallbacks",
                    "LLM Deployment Analytics - Number of successful fallback requests from primary model -> fallback model",
                    ["primary_model", "fallback_model"],
                )
                self.litellm_deployment_failed_fallbacks = Counter(
                    "litellm_deployment_failed_fallbacks",
                    "LLM Deployment Analytics - Number of failed fallback requests from primary model -> fallback model",
                    ["primary_model", "fallback_model"],
                )

        except Exception as e:
            print_verbose(f"Got exception on init prometheus client {str(e)}")
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Define prometheus client
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,
        )
        from litellm.proxy.proxy_server import premium_user

        verbose_logger.debug(
            f"prometheus Logging - Enters success logging function for kwargs {kwargs}"
        )

        # unpack kwargs
        model = kwargs.get("model", "")
        response_cost = kwargs.get("response_cost", 0.0) or 0
        litellm_params = kwargs.get("litellm_params", {}) or {}
        _metadata = litellm_params.get("metadata", {})
        proxy_server_request = litellm_params.get("proxy_server_request") or {}
        end_user_id = proxy_server_request.get("body", {}).get("user", None)
        user_id = litellm_params.get("metadata", {}).get("user_api_key_user_id", None)
        user_api_key = litellm_params.get("metadata", {}).get("user_api_key", None)
        user_api_key_alias = litellm_params.get("metadata", {}).get(
            "user_api_key_alias", None
        )
        user_api_team = litellm_params.get("metadata", {}).get(
            "user_api_key_team_id", None
        )
        user_api_team_alias = litellm_params.get("metadata", {}).get(
            "user_api_key_team_alias", None
        )

        _team_spend = litellm_params.get("metadata", {}).get(
            "user_api_key_team_spend", None
        )
        _team_max_budget = litellm_params.get("metadata", {}).get(
            "user_api_key_team_max_budget", None
        )
        _remaining_team_budget = safe_get_remaining_budget(
            max_budget=_team_max_budget, spend=_team_spend
        )

        _api_key_spend = litellm_params.get("metadata", {}).get(
            "user_api_key_spend", None
        )
        _api_key_max_budget = litellm_params.get("metadata", {}).get(
            "user_api_key_max_budget", None
        )
        _remaining_api_key_budget = safe_get_remaining_budget(
            max_budget=_api_key_max_budget, spend=_api_key_spend
        )
        output_tokens = 1.0
        if response_obj is not None:
            tokens_used = response_obj.get("usage", {}).get("total_tokens", 0)
            output_tokens = response_obj.get("usage", {}).get("completion_tokens", 0)
        else:
            tokens_used = 0

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
        self.litellm_tokens_metric.labels(
            end_user_id,
            user_api_key,
            user_api_key_alias,
            model,
            user_api_team,
            user_api_team_alias,
            user_id,
        ).inc(tokens_used)

        self.litellm_remaining_team_budget_metric.labels(
            user_api_team, user_api_team_alias
        ).set(_remaining_team_budget)

        self.litellm_remaining_api_key_budget_metric.labels(
            user_api_key, user_api_key_alias
        ).set(_remaining_api_key_budget)

        # Set remaining rpm/tpm for API Key + model
        # see parallel_request_limiter.py - variables are set there
        model_group = get_model_group_from_litellm_kwargs(kwargs)
        remaining_requests_variable_name = (
            f"litellm-key-remaining-requests-{model_group}"
        )
        remaining_tokens_variable_name = f"litellm-key-remaining-tokens-{model_group}"

        remaining_requests = _metadata.get(
            remaining_requests_variable_name, sys.maxsize
        )
        remaining_tokens = _metadata.get(remaining_tokens_variable_name, sys.maxsize)

        self.litellm_remaining_api_key_requests_for_model.labels(
            user_api_key, user_api_key_alias, model_group
        ).set(remaining_requests)

        self.litellm_remaining_api_key_tokens_for_model.labels(
            user_api_key, user_api_key_alias, model_group
        ).set(remaining_tokens)

        # latency metrics
        total_time: timedelta = kwargs.get("end_time") - kwargs.get("start_time")
        total_time_seconds = total_time.total_seconds()
        api_call_total_time: timedelta = kwargs.get("end_time") - kwargs.get(
            "api_call_start_time"
        )

        api_call_total_time_seconds = api_call_total_time.total_seconds()

        # log metrics
        self.litellm_request_total_latency_metric.labels(model).observe(
            total_time_seconds
        )
        self.litellm_llm_api_latency_metric.labels(model).observe(
            api_call_total_time_seconds
        )

        # set x-ratelimit headers
        if premium_user is True:
            self.set_llm_deployment_success_metrics(
                kwargs, start_time, end_time, output_tokens
            )
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.proxy.proxy_server import premium_user

        verbose_logger.debug(
            f"prometheus Logging - Enters success logging function for kwargs {kwargs}"
        )

        # unpack kwargs
        model = kwargs.get("model", "")
        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_server_request = litellm_params.get("proxy_server_request") or {}
        end_user_id = proxy_server_request.get("body", {}).get("user", None)
        user_id = litellm_params.get("metadata", {}).get("user_api_key_user_id", None)
        user_api_key = litellm_params.get("metadata", {}).get("user_api_key", None)
        user_api_key_alias = litellm_params.get("metadata", {}).get(
            "user_api_key_alias", None
        )
        user_api_team = litellm_params.get("metadata", {}).get(
            "user_api_key_team_id", None
        )
        user_api_team_alias = litellm_params.get("metadata", {}).get(
            "user_api_key_team_alias", None
        )

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

    def set_llm_deployment_failure_metrics(self, request_kwargs: dict):
        try:
            verbose_logger.debug("setting remaining tokens requests metric")
            _response_headers = request_kwargs.get("response_headers")
            _litellm_params = request_kwargs.get("litellm_params", {}) or {}
            _metadata = _litellm_params.get("metadata", {})
            litellm_model_name = request_kwargs.get("model", None)
            api_base = _metadata.get("api_base", None)
            llm_provider = _litellm_params.get("custom_llm_provider", None)
            model_id = _metadata.get("model_id")

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
            ).inc()

            self.litellm_deployment_total_requests.labels(
                litellm_model_name=litellm_model_name,
                model_id=model_id,
                api_base=api_base,
                api_provider=llm_provider,
            ).inc()

            pass
        except:
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
            _response_headers = request_kwargs.get("response_headers")
            _litellm_params = request_kwargs.get("litellm_params", {}) or {}
            _metadata = _litellm_params.get("metadata", {})
            litellm_model_name = request_kwargs.get("model", None)
            model_group = _metadata.get("model_group", None)
            api_base = _metadata.get("api_base", None)
            llm_provider = _litellm_params.get("custom_llm_provider", None)
            model_id = _metadata.get("model_id")

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
                    model_group, llm_provider, api_base, litellm_model_name
                ).set(remaining_requests)

            if remaining_tokens:
                self.litellm_remaining_tokens_metric.labels(
                    model_group, llm_provider, api_base, litellm_model_name
                ).set(remaining_tokens)

            """
            log these labels
            ["litellm_model_name", "model_id", "api_base", "api_provider"]
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
            ).inc()

            self.litellm_deployment_total_requests.labels(
                litellm_model_name=litellm_model_name,
                model_id=model_id,
                api_base=api_base,
                api_provider=llm_provider,
            ).inc()

            # Track deployment Latency
            response_ms: timedelta = end_time - start_time
            time_to_first_token_response_time: Optional[timedelta] = None

            if (
                request_kwargs.get("stream", None) is not None
                and request_kwargs["stream"] == True
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
                ).observe(latency_per_token)

        except Exception as e:
            verbose_logger.error(
                "Prometheus Error: set_llm_deployment_success_metrics. Exception occured - {}".format(
                    str(e)
                )
            )
            return

    async def log_success_fallback_event(self, original_model_group: str, kwargs: dict):
        verbose_logger.debug(
            "Prometheus: log_success_fallback_event, original_model_group: %s, kwargs: %s",
            original_model_group,
            kwargs,
        )
        _new_model = kwargs.get("model")
        self.litellm_deployment_successful_fallbacks.labels(
            primary_model=original_model_group, fallback_model=_new_model
        ).inc()

    async def log_failure_fallback_event(self, original_model_group: str, kwargs: dict):
        verbose_logger.debug(
            "Prometheus: log_failure_fallback_event, original_model_group: %s, kwargs: %s",
            original_model_group,
            kwargs,
        )
        _new_model = kwargs.get("model")
        self.litellm_deployment_failed_fallbacks.labels(
            primary_model=original_model_group, fallback_model=_new_model
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


def safe_get_remaining_budget(
    max_budget: Optional[float], spend: Optional[float]
) -> float:
    if max_budget is None:
        return float("inf")

    if spend is None:
        return max_budget

    return max_budget - spend
