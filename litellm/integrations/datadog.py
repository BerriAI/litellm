#### What this does ####
#    On success + failure, log events to Datadog

import datetime
import os
import subprocess
import sys
import traceback
import uuid
from typing import TypedDict

import dotenv
import requests  # type: ignore

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)


class DatadogPayload(TypedDict, total=False):
    ddsource: str
    ddtags: str
    hostname: str
    message: str
    service: str


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


class DataDogLogger(CustomBatchLogger):
    # Class variables or attributes
    def __init__(
        self,
        **kwargs,
    ):

        # check if the correct env variables are set
        if os.getenv("DD_API_KEY", None) is None:
            raise Exception("DD_API_KEY is not set, set 'DD_API_KEY=<>")
        if os.getenv("DD_SITE", None) is None:
            raise Exception("DD_SITE is not set in .env, set 'DD_SITE=<>")
        self.async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        self.DD_API_KEY = os.getenv("DD_API_KEY")

        self.intake_url = f"https://http-intake.logs.{os.getenv('DD_SITE')}/api/v2/logs"

        self.sync_client = _get_httpx_client()
        try:
            verbose_logger.debug(f"in init datadog logger")
            pass

        except Exception as e:
            print_verbose(f"Got exception on init s3 client {str(e)}")
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
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

            dd_payload = DatadogPayload(
                ddsource=os.getenv("DD_SOURCE", "litellm"),
                ddtags="",
                hostname="",
                message=payload,
                service="litellm-server",
            )

            response = await self.async_client.post(
                url=self.intake_url,
                json=dd_payload,
                headers={
                    "DD-API-KEY": self.DD_API_KEY,
                },
            )

            print_verbose("response = ", response)
            print_verbose("status_code = ", response.status_code)
            print_verbose("text = ", response.text)
            print_verbose(
                f"Datadog Layer Logging - final response object: {response_obj}"
            )
        except Exception as e:
            verbose_logger.exception(
                f"Datadog Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass
