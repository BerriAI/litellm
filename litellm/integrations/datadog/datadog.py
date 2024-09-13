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
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .types import DatadogPayload
from .utils import make_json_serializable


class DataDogLogger(CustomBatchLogger):
    # Class variables or attributes
    def __init__(
        self,
        **kwargs,
    ):
        try:
            verbose_logger.debug(f"Datadog: in init datadog logger")
            # check if the correct env variables are set
            if os.getenv("DD_API_KEY", None) is None:
                raise Exception("DD_API_KEY is not set, set 'DD_API_KEY=<>")
            if os.getenv("DD_SITE", None) is None:
                raise Exception("DD_SITE is not set in .env, set 'DD_SITE=<>")
            self.async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )
            self.DD_API_KEY = os.getenv("DD_API_KEY")
            self.intake_url = (
                f"https://http-intake.logs.{os.getenv('DD_SITE')}/api/v2/logs"
            )
            self.sync_client = _get_httpx_client()
        except Exception as e:
            verbose_logger.exception(
                f"Datadog: Got exception on init Datadog client {str(e)}"
            )
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "Datadog: Logging - Enters logging function for model %s", kwargs
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

            verbose_logger.debug("Datadog: Logger - Logging payload = %s", payload)

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

            response.raise_for_status()
            if response.status_code != 202:
                raise Exception(
                    f"Response from datadog API status_code: {response.status_code}, text: {response.text}"
                )

            verbose_logger.debug(
                "Datadog: Response from datadog API status_code: %s, text: %s",
                response.status_code,
                response.text,
            )

        except Exception as e:
            verbose_logger.exception(
                f"Datadog Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass
