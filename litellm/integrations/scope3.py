
# On success events, log the event to the Scope3 API to measure the carbon footprint of the API call.

import json
import os
import traceback
from typing import Literal, Optional
from datetime import datetime, timezone

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)

class Scope3Logger(CustomLogger):
    """
        Expects
        SCOPE3_ACCESS_TOKEN,

        Optional:
        SCOPE3_API_BASE,

        in the environment
    """
    def __init__(self) -> None:
        super().__init__()
        self.async_http_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.async_http_handler = httpx.AsyncClient()
        if self.async_http_handler is None:
            raise AssertionError("Failed to get async httpx client")
        self.sync_http_handler = HTTPHandler()
        self.url = os.getenv("SCOPE3_API_BASE", "https://aiapi.scope3.com")
        if self.url.endswith("/"):
            self.url += "v1/log_json"
        else:
            self.url += "/v1/log_json"
        self.access_token = os.getenv("SCOPE3_ACCESS_TOKEN", "")
        if self.access_token == "":
            raise AssertionError("SCOPE3_ACCESS_TOKEN missing or not set correctly. SCOPE3_ACCESS_TOKEN={}".format(self.access_token))

    def _get_log_row(self, kwargs: dict, response_obj, start_time, end_time) -> dict:
        if isinstance(kwargs["api_call_start_time"], datetime):
            start_time_utc = kwargs["api_call_start_time"].isoformat() + "Z"
        else:
            start_time_utc = datetime.now(timezone.utc).isoformat() + "Z"

        total_cost = litellm.completion_cost(completion_response=response_obj)

        log_row = {
            "start_time_utc": start_time_utc,
            "integration_source": "litellm",
            "request_id": kwargs["litellm_call_id"],
            "request_cost": total_cost,
            "request_currency": "USD",
            "request_duration_ms": (end_time - start_time).total_seconds() * 1000,
            "model_id": kwargs["model"],
            "model_id_used": response_obj["model"],
            "managed_service_id": kwargs["custom_llm_provider"]
        }

        if (isinstance(response_obj, litellm.ModelResponse)) and hasattr(response_obj, "usage"):
            log_row["input_tokens"] = response_obj["usage"].get("prompt_tokens", 0)
            log_row["output_tokens"] = response_obj["usage"].get("completion_tokens", 0)

        if kwargs.get("response_headers") and kwargs["response_headers"].get("openai-processing-ms"):
            log_row["processing_duration_ms"] = float(kwargs["response_headers"]["openai-processing-ms"])

        if kwargs.get("litellm_params") != None and kwargs["litellm_params"].get("metadata") != None:
            metadata = kwargs.get("litellm_params", {}).get("metadata")
            for key, value in metadata.items():
                if key.startswith("scope3_"):  
                    log_row[key.replace("scope3_", "")] = value   

        return {"rows": [log_row]}

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        _data = self._get_log_row(kwargs=kwargs, response_obj=response_obj, start_time=start_time, end_time=end_time)
        _headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.access_token),
        }

        try:
            response = self.sync_http_handler.post(
                url=self.url,
                data=json.dumps(_data),
                headers=_headers,
            )
            response.raise_for_status()
        except Exception as e:
            error_response = getattr(e, "response", None)
            if error_response is not None and hasattr(error_response, "text"):
                verbose_logger.error(f"\nError Message: {error_response.text}")
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        _data = self._get_log_row(kwargs=kwargs, response_obj=response_obj, start_time=start_time, end_time=end_time)
        _headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.access_token),
        }

        response: Optional[httpx.Response] = None
        try:
            response = await self.async_http_handler.post(
                url=self.url,
                data=json.dumps(_data),
                headers=_headers,
            )
            if response is not None:
                response.raise_for_status()
                verbose_logger.debug(f"Logged Scope3 Object: {response.text}")
        except Exception as e:
            if response is not None and hasattr(response, "text"):
                verbose_logger.error(f"\nError Message: {response.text}")
            raise e
