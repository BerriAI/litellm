#### What this does ####
#    On success + failure, log events to Datadog

import dotenv, os
import requests  # type: ignore
import traceback
import datetime, subprocess, sys
import litellm, uuid
from litellm._logging import print_verbose, verbose_logger


def make_json_serializable(payload):
    for key, value in payload.items():
        try:
            if isinstance(value, dict):
                # recursively sanitize dicts
                payload[key] = make_json_serializable(value.copy())
            elif not isinstance(value, (str, int, float, bool, type(None))):
                # everything else becomes a string
                payload[key] = str(value)
        except:
            # non blocking if it can't cast to a str
            pass
    return payload


class DataDogLogger:
    # Class variables or attributes
    def __init__(
        self,
        **kwargs,
    ):
        from datadog_api_client import ApiClient, Configuration

        # check if the correct env variables are set
        if os.getenv("DD_API_KEY", None) is None:
            raise Exception("DD_API_KEY is not set, set 'DD_API_KEY=<>")
        if os.getenv("DD_SITE", None) is None:
            raise Exception("DD_SITE is not set in .env, set 'DD_SITE=<>")
        self.configuration = Configuration()

        try:
            verbose_logger.debug(f"in init datadog logger")
            pass

        except Exception as e:
            print_verbose(f"Got exception on init s3 client {str(e)}")
            raise e

    async def _async_log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose, user_id
    ):
        self.log_event(kwargs, response_obj, start_time, end_time, print_verbose)

    def log_event(
        self, kwargs, response_obj, start_time, end_time, user_id, print_verbose
    ):
        try:
            # Define DataDog client
            from datadog_api_client.v2.api.logs_api import LogsApi
            from datadog_api_client.v2 import ApiClient
            from datadog_api_client.v2.models import HTTPLogItem, HTTPLog

            verbose_logger.debug(
                f"datadog Logging - Enters logging function for model {kwargs}"
            )
            litellm_params = kwargs.get("litellm_params", {})
            metadata = (
                litellm_params.get("metadata", {}) or {}
            )  # if litellm_params['metadata'] == None
            messages = kwargs.get("messages")
            optional_params = kwargs.get("optional_params", {})
            call_type = kwargs.get("call_type", "litellm.completion")
            cache_hit = kwargs.get("cache_hit", False)
            usage = response_obj["usage"]
            id = response_obj.get("id", str(uuid.uuid4()))
            usage = dict(usage)
            try:
                response_time = (end_time - start_time).total_seconds() * 1000
            except:
                response_time = None

            try:
                response_obj = dict(response_obj)
            except:
                response_obj = response_obj

            # Clean Metadata before logging - never log raw metadata
            # the raw metadata can contain circular references which leads to infinite recursion
            # we clean out all extra litellm metadata params before logging
            clean_metadata = {}
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    # clean litellm metadata before logging
                    if key in [
                        "endpoint",
                        "caching_groups",
                        "previous_models",
                    ]:
                        continue
                    else:
                        clean_metadata[key] = value

            # Build the initial payload
            payload = {
                "id": id,
                "call_type": call_type,
                "cache_hit": cache_hit,
                "start_time": start_time,
                "end_time": end_time,
                "response_time": response_time,
                "model": kwargs.get("model", ""),
                "user": kwargs.get("user", ""),
                "model_parameters": optional_params,
                "spend": kwargs.get("response_cost", 0),
                "messages": messages,
                "response": response_obj,
                "usage": usage,
                "metadata": clean_metadata,
            }

            make_json_serializable(payload)
            import json

            payload = json.dumps(payload)

            print_verbose(f"\ndd Logger - Logging payload = {payload}")

            with ApiClient(self.configuration) as api_client:
                api_instance = LogsApi(api_client)
                body = HTTPLog(
                    [
                        HTTPLogItem(
                            ddsource="litellm",
                            message=payload,
                            service="litellm-server",
                        ),
                    ]
                )
                response = api_instance.submit_log(body)

            print_verbose(
                f"Datadog Layer Logging - final response object: {response_obj}"
            )
        except Exception as e:
            verbose_logger.debug(
                f"Datadog Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass
