# What is this?
## On Success events log usage to Amberflo

import json
import os
from typing import Optional

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)

class AmberfloLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.validate_environment()
        self.async_http_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sync_http_handler = HTTPHandler()

    def validate_environment(self):
        """
        Expects
        AMBERFLO_API_KEY

        in the environment
        """
        missing_keys = []
        if os.getenv("AMBERFLO_API_KEY", None) is None:
            missing_keys.append("AMBERFLO_API_KEY")

        if len(missing_keys) > 0:
            raise Exception("Missing keys={} in environment.".format(missing_keys))

    def _common_logic(self, kwargs: dict, response_obj) -> dict:
        id = response_obj.get("id", kwargs.get("litellm_call_id"))

        dimensions = {
            "model": response_obj.get("model", kwargs.get("model")),
            "object": response_obj.get("object"),
        }

        created = response_obj.get("created", None)

        user = response_obj.get("user", kwargs.get("user"))
        if not user:
            raise Exception("Amberflo: user is required")

        if (
            isinstance(response_obj, litellm.ModelResponse)
            or isinstance(response_obj, litellm.EmbeddingResponse)
        ) and hasattr(response_obj, "usage"):
            usage = response_obj["usage"]
            return {
                "customerId": user,
                "values": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0)
                },
                "meterTimeInMillis": created * 1000,
                "dimensions": dimensions,
                "uniqueId": id,
            }
        else:
            return None

    def _prepare_request_data(self, kwargs, response_obj):
        """
        Common logic for preparing the data, headers, etc.
        """
        _url = "https://app.amberflo.io/ingest"
        api_key = os.getenv("AMBERFLO_API_KEY")
        _headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
        }
        _data = self._common_logic(kwargs, response_obj)
        return _url, _headers, _data

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        _url, _headers, _data = self._prepare_request_data(kwargs, response_obj)
        if _data is None:
            return

        try:
            response = self.sync_http_handler.post(
                url=_url,
                data=json.dumps(_data),
                headers=_headers,
            )

            response.raise_for_status()
        except Exception as e:
            error_response = getattr(e, "response", None)
            if error_response is not None and hasattr(error_response, "text"):
                verbose_logger.debug(f"\nError Message: {error_response.text}")
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        _url, _headers, _data = self._prepare_request_data(kwargs, response_obj)
        if _data is None:
            return

        response: Optional[httpx.Response] = None
        try:
            response = await self.async_http_handler.post(
                url=_url,
                data=json.dumps(_data),
                headers=_headers,
            )

            response.raise_for_status()
        except Exception as e:
            if response is not None and hasattr(response, "text"):
                verbose_logger.debug(f"\nError Message: {response.text}")
            raise e
