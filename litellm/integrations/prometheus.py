# used for /metrics endpoint on LiteLLM Proxy
#### What this does ####
#    On success + failure, log events to Supabase

import dotenv, os
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
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
            verbose_logger.debug(f"in init prometheus metrics")
            from prometheus_client import Counter

            self.litellm_requests_metric = Counter(
                name="litellm_requests_metric",
                documentation="Total number of LLM calls to litellm",
                labelnames=["end_user", "key", "model", "team"],
            )

            # Counter for spend
            self.litellm_spend_metric = Counter(
                "litellm_spend_metric",
                "Total spend on LLM requests",
                labelnames=["end_user", "key", "model", "team"],
            )

            # Counter for total_output_tokens
            self.litellm_tokens_metric = Counter(
                "litellm_total_tokens",
                "Total number of input + output tokens from LLM requests",
                labelnames=["end_user", "key", "model", "team"],
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
            response_cost = kwargs.get("response_cost", 0.0)
            litellm_params = kwargs.get("litellm_params", {}) or {}
            proxy_server_request = litellm_params.get("proxy_server_request") or {}
            end_user_id = proxy_server_request.get("body", {}).get("user", None)
            user_api_key = litellm_params.get("metadata", {}).get("user_api_key", None)
            user_api_team = litellm_params.get("metadata", {}).get(
                "user_api_key_team_id", None
            )
            tokens_used = response_obj.get("usage", {}).get("total_tokens", 0)

            print_verbose(
                f"inside track_prometheus_metrics, model {model}, response_cost {response_cost}, tokens_used {tokens_used}, end_user_id {end_user_id}, user_api_key {user_api_key}"
            )

            self.litellm_requests_metric.labels(
                end_user_id, user_api_key, model, user_api_team
            ).inc()
            self.litellm_spend_metric.labels(
                end_user_id, user_api_key, model, user_api_team
            ).inc(response_cost)
            self.litellm_tokens_metric.labels(
                end_user_id, user_api_key, model, user_api_team
            ).inc(tokens_used)
        except Exception as e:
            traceback.print_exc()
            verbose_logger.debug(
                f"prometheus Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass
