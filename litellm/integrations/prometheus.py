# used for /metrics endpoint on LiteLLM Proxy
#### What this does ####
#    On success, log events to Prometheus

import dotenv, os
import requests  # type: ignore
import traceback
import datetime, subprocess, sys
import litellm, uuid
from litellm._logging import print_verbose, verbose_logger


class PrometheusLogger:
    # Class variables or attributes
    def __init__(
        self,
        **kwargs,
    ):
        try:
            from prometheus_client import Counter

            self.litellm_llm_api_failed_requests_metric = Counter(
                name="litellm_llm_api_failed_requests_metric",
                documentation="Total number of failed LLM API calls via litellm",
                labelnames=["end_user", "hashed_api_key", "model", "team", "user"],
            )

            self.litellm_requests_metric = Counter(
                name="litellm_requests_metric",
                documentation="Total number of LLM calls to litellm",
                labelnames=["end_user", "hashed_api_key", "model", "team", "user"],
            )

            # Counter for spend
            self.litellm_spend_metric = Counter(
                "litellm_spend_metric",
                "Total spend on LLM requests",
                labelnames=["end_user", "hashed_api_key", "model", "team", "user"],
            )

            # Counter for total_output_tokens
            self.litellm_tokens_metric = Counter(
                "litellm_total_tokens",
                "Total number of input + output tokens from LLM requests",
                labelnames=["end_user", "hashed_api_key", "model", "team", "user"],
            )
        except Exception as e:
            print_verbose(f"Got exception on init prometheus client {str(e)}")
            raise e

    async def _async_log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose, user_id
    ):
        self.log_event(kwargs, response_obj, start_time, end_time, print_verbose)

    def log_event(
        self, kwargs, response_obj, start_time, end_time, user_id, print_verbose
    ):
        try:
            # Define prometheus client
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
            user_api_team = litellm_params.get("metadata", {}).get(
                "user_api_key_team_id", None
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
                end_user_id, user_api_key, model, user_api_team, user_id
            ).inc()
            self.litellm_spend_metric.labels(
                end_user_id, user_api_key, model, user_api_team, user_id
            ).inc(response_cost)
            self.litellm_tokens_metric.labels(
                end_user_id, user_api_key, model, user_api_team, user_id
            ).inc(tokens_used)

            ### FAILURE INCREMENT ###
            if "exception" in kwargs:
                self.litellm_llm_api_failed_requests_metric.labels(
                    end_user_id, user_api_key, model, user_api_team, user_id
                ).inc()
        except Exception as e:
            traceback.print_exc()
            verbose_logger.debug(
                f"prometheus Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass
