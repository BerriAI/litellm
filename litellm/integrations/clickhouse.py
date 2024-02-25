# callback to make a request to an API endpoint

#### What this does ####
#    On success, logs events to Promptlayer
import dotenv, os
import requests

from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache

from typing import Literal, Union

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback


#### What this does ####
#    On success + failure, log events to Supabase

import dotenv, os
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
import datetime, subprocess, sys
import litellm, uuid
from litellm._logging import print_verbose, verbose_logger


class ClickhouseLogger:
    # Class variables or attributes
    def __init__(self, endpoint=None, headers=None):
        import clickhouse_connect

        print_verbose(
            f"ClickhouseLogger init, host {os.getenv('CLICKHOUSE_HOST')}, port {os.getenv('CLICKHOUSE_PORT')}, username {os.getenv('CLICKHOUSE_USERNAME')}"
        )

        client = clickhouse_connect.get_client(
            host=os.getenv("CLICKHOUSE_HOST"),
            port=os.getenv("CLICKHOUSE_PORT"),
            username=os.getenv("CLICKHOUSE_USERNAME"),
            password=os.getenv("CLICKHOUSE_PASSWORD"),
        )
        self.client = client

    # This is sync, because we run this in a separate thread. Running in a sepearate thread ensures it will never block an LLM API call
    # Experience with s3, Langfuse shows that async logging events are complicated and can block LLM calls
    def log_event(
        self, kwargs, response_obj, start_time, end_time, user_id, print_verbose
    ):
        try:
            verbose_logger.debug(
                f"ClickhouseLogger Logging - Enters logging function for model {kwargs}"
            )

            # construct payload to send custom logger
            # follows the same params as langfuse.py
            litellm_params = kwargs.get("litellm_params", {})
            metadata = (
                litellm_params.get("metadata", {}) or {}
            )  # if litellm_params['metadata'] == None
            messages = kwargs.get("messages")
            cost = kwargs.get("response_cost", 0.0)
            optional_params = kwargs.get("optional_params", {})
            call_type = kwargs.get("call_type", "litellm.completion")
            cache_hit = kwargs.get("cache_hit", False)
            usage = response_obj["usage"]
            id = response_obj.get("id", str(uuid.uuid4()))

            from litellm.proxy.utils import get_logging_payload

            payload = get_logging_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
            # Build the initial payload

            # Ensure everything in the payload is converted to str
            for key, value in payload.items():
                try:
                    payload[key] = str(value)
                except:
                    # non blocking if it can't cast to a str
                    pass

            print_verbose(f"\nGeneric Logger - Logging payload = {payload}")

            # make request to endpoint with payload
            # print_verbose(
            #     f"Generic Logger - final response status = {response_status}, response text = {response_text}"
            # )
            # return response
        except Exception as e:
            traceback.print_exc()
            verbose_logger.debug(f"Generic - {str(e)}\n{traceback.format_exc()}")
            pass
