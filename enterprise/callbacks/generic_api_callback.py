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


class GenericAPILogger:
    # Class variables or attributes
    def __init__(self, endpoint=None, headers=None):
        try:
            if endpoint == None:
                # check env for "GENERIC_LOGGER_ENDPOINT"
                if os.getenv("GENERIC_LOGGER_ENDPOINT"):
                    # Do something with the endpoint
                    endpoint = os.getenv("GENERIC_LOGGER_ENDPOINT")
                else:
                    # Handle the case when the endpoint is not found in the environment variables
                    raise ValueError(
                        f"endpoint not set for GenericAPILogger, GENERIC_LOGGER_ENDPOINT not found in environment variables"
                    )
            headers = headers or litellm.generic_logger_headers
            self.endpoint = endpoint
            self.headers = headers

            verbose_logger.debug(
                f"in init GenericAPILogger, endpoint {self.endpoint}, headers {self.headers}"
            )

            pass

        except Exception as e:
            print_verbose(f"Got exception on init GenericAPILogger client {str(e)}")
            raise e

    # This is sync, because we run this in a separate thread. Running in a sepearate thread ensures it will never block an LLM API call
    # Experience with s3, Langfuse shows that async logging events are complicated and can block LLM calls
    def log_event(
        self, kwargs, response_obj, start_time, end_time, user_id, print_verbose
    ):
        try:
            verbose_logger.debug(
                f"GenericAPILogger Logging - Enters logging function for model {kwargs}"
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

            # Build the initial payload
            payload = {
                "id": id,
                "call_type": call_type,
                "cache_hit": cache_hit,
                "startTime": start_time,
                "endTime": end_time,
                "model": kwargs.get("model", ""),
                "user": kwargs.get("user", ""),
                "modelParameters": optional_params,
                "messages": messages,
                "response": response_obj,
                "usage": usage,
                "metadata": metadata,
                "cost": cost,
            }

            # Ensure everything in the payload is converted to str
            for key, value in payload.items():
                try:
                    payload[key] = str(value)
                except:
                    # non blocking if it can't cast to a str
                    pass

            import json

            data = {
                "data": payload,
            }
            data = json.dumps(data)
            print_verbose(f"\nGeneric Logger - Logging payload = {data}")

            # make request to endpoint with payload
            response = requests.post(self.endpoint, json=data, headers=self.headers)

            response_status = response.status_code
            response_text = response.text

            print_verbose(
                f"Generic Logger - final response status = {response_status}, response text = {response_text}"
            )
            return response
        except Exception as e:
            traceback.print_exc()
            verbose_logger.debug(f"Generic - {str(e)}\n{traceback.format_exc()}")
            pass
