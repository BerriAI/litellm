# used for /metrics endpoint on LiteLLM Proxy
#### What this does ####
#    On success, log events to Prometheus

import datetime
import os
import subprocess
import sys
import traceback
import uuid
from typing import Optional, Union

import dotenv
import requests  # type: ignore

import litellm
from litellm._logging import print_verbose, verbose_logger


class PrometheusLogger:
    # Class variables or attributes
    def __init__(
        self,
        **kwargs,
    ):
        try:
            from prometheus_client import Counter, Gauge

            from litellm.proxy.proxy_server import premium_user

            self.litellm_llm_api_failed_requests_metric = Counter(
                name="litellm_llm_api_failed_requests_metric",
                documentation="Total number of failed LLM API calls via litellm",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "model",
                    "team",
                    "team_alias",
                    "user",
                ],
            )

            self.litellm_requests_metric = Counter(
                name="litellm_requests_metric",
                documentation="Total number of LLM calls to litellm",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "model",
                    "team",
                    "team_alias",
                    "user",
                ],
            )

            # Counter for spend
            self.litellm_spend_metric = Counter(
                "litellm_spend_metric",
                "Total spend on LLM requests",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
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

            # Litellm-Enterprise Metrics
            if premium_user is True:
                # Remaining Rate Limit for model
                self.litellm_remaining_requests_metric = Gauge(
                    "litellm_remaining_requests",
                    "remaining requests for model, returned from LLM API Provider",
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

        except Exception as e:
            print_verbose(f"Got exception on init prometheus client {str(e)}")
            raise e

    async def _async_log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose, user_id
    ):
        self.log_event(
            kwargs, response_obj, start_time, end_time, user_id, print_verbose
        )

    def log_event(
        self, kwargs, response_obj, start_time, end_time, user_id, print_verbose
    ):
        try:
            # Define prometheus client
            from litellm.proxy.proxy_server import premium_user

            verbose_logger.debug(
                f"prometheus Logging - Enters logging function for model {kwargs}"
            )

            # unpack kwargs
            model = kwargs.get("model", "")
            response_cost = kwargs.get("response_cost", 0.0) or 0
            litellm_params = kwargs.get("litellm_params", {}) or {}
            proxy_server_request = litellm_params.get("proxy_server_request") or {}
            end_user_id = proxy_server_request.get("body", {}).get("user", None)
            user_id = litellm_params.get("metadata", {}).get(
                "user_api_key_user_id", None
            )
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

            if response_obj is not None:
                tokens_used = response_obj.get("usage", {}).get("total_tokens", 0)
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
                model,
                user_api_team,
                user_api_team_alias,
                user_id,
            ).inc()
            self.litellm_spend_metric.labels(
                end_user_id,
                user_api_key,
                model,
                user_api_team,
                user_api_team_alias,
                user_id,
            ).inc(response_cost)
            self.litellm_tokens_metric.labels(
                end_user_id,
                user_api_key,
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

            # set x-ratelimit headers
            if premium_user is True:
                self.set_remaining_tokens_requests_metric(kwargs)

            ### FAILURE INCREMENT ###
            if "exception" in kwargs:
                self.litellm_llm_api_failed_requests_metric.labels(
                    end_user_id,
                    user_api_key,
                    model,
                    user_api_team,
                    user_api_team_alias,
                    user_id,
                ).inc()
        except Exception as e:
            verbose_logger.error(
                "prometheus Layer Error(): Exception occured - {}".format(str(e))
            )
            verbose_logger.debug(traceback.format_exc())
            pass

    def set_remaining_tokens_requests_metric(self, request_kwargs: dict):
        try:
            verbose_logger.debug("setting remaining tokens requests metric")
            _response_headers = request_kwargs.get("response_headers")
            _litellm_params = request_kwargs.get("litellm_params", {}) or {}
            _metadata = _litellm_params.get("metadata", {})
            litellm_model_name = request_kwargs.get("model", None)
            model_group = _metadata.get("model_group", None)
            api_base = _metadata.get("api_base", None)
            llm_provider = _litellm_params.get("custom_llm_provider", None)

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

        except Exception as e:
            verbose_logger.error(
                "Prometheus Error: set_remaining_tokens_requests_metric. Exception occured - {}".format(
                    str(e)
                )
            )
            return


def safe_get_remaining_budget(
    max_budget: Optional[float], spend: Optional[float]
) -> float:
    if max_budget is None:
        return float("inf")

    if spend is None:
        return max_budget

    return max_budget - spend
