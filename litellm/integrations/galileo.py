import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)


# from here: https://docs.rungalileo.io/galileo/gen-ai-studio-products/galileo-observe/how-to/logging-data-via-restful-apis#structuring-your-records
class LLMResponse(BaseModel):
    latency_ms: int
    status_code: int
    input_text: str
    output_text: str
    node_type: str
    model: str
    num_input_tokens: int
    num_output_tokens: int
    output_logprobs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional. When available, logprobs are used to compute Uncertainty.",
    )
    created_at: str = Field(
        ..., description='timestamp constructed in "%Y-%m-%dT%H:%M:%S" format'
    )
    tags: Optional[List[str]] = None
    user_metadata: Optional[Dict[str, Any]] = None


class GalileoObserve(CustomLogger):
    def __init__(self) -> None:
        self.in_memory_records: List[dict] = []
        self.batch_size = 1
        self.base_url = os.getenv("GALILEO_BASE_URL", None)
        self.project_id = os.getenv("GALILEO_PROJECT_ID", None)
        self.headers: Optional[Dict[str, str]] = None
        self.async_httpx_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        pass

    def set_galileo_headers(self):
        # following https://docs.rungalileo.io/galileo/gen-ai-studio-products/galileo-observe/how-to/logging-data-via-restful-apis#logging-your-records

        headers = {
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        galileo_login_response = litellm.module_level_client.post(
            url=f"{self.base_url}/login",
            headers=headers,
            data={
                "username": os.getenv("GALILEO_USERNAME"),
                "password": os.getenv("GALILEO_PASSWORD"),
            },
        )

        access_token = galileo_login_response.json()["access_token"]

        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

    def get_output_str_from_response(self, response_obj, kwargs):
        output = None
        if response_obj is not None and (
            kwargs.get("call_type", None) == "embedding"
            or isinstance(response_obj, litellm.EmbeddingResponse)
        ):
            output = None
        elif response_obj is not None and isinstance(
            response_obj, litellm.ModelResponse
        ):
            output = response_obj["choices"][0]["message"].json()
        elif response_obj is not None and isinstance(
            response_obj, litellm.TextCompletionResponse
        ):
            output = response_obj.choices[0].text
        elif response_obj is not None and isinstance(
            response_obj, litellm.ImageResponse
        ):
            output = response_obj["data"]

        return output

    async def async_log_success_event(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ):
        verbose_logger.debug("On Async Success")

        _latency_ms = int((end_time - start_time).total_seconds() * 1000)
        _call_type = kwargs.get("call_type", "litellm")
        input_text = litellm.utils.get_formatted_prompt(
            data=kwargs, call_type=_call_type
        )

        _usage = response_obj.get("usage", {}) or {}
        num_input_tokens = _usage.get("prompt_tokens", 0)
        num_output_tokens = _usage.get("completion_tokens", 0)

        output_text = self.get_output_str_from_response(
            response_obj=response_obj, kwargs=kwargs
        )

        if output_text is not None:
            request_record = LLMResponse(
                latency_ms=_latency_ms,
                status_code=200,
                input_text=input_text,
                output_text=output_text,
                node_type=_call_type,
                model=kwargs.get("model", "-"),
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                created_at=start_time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),  # timestamp str constructed in "%Y-%m-%dT%H:%M:%S" format
            )

            # dump to dict
            request_dict = request_record.model_dump()
            self.in_memory_records.append(request_dict)

            if len(self.in_memory_records) >= self.batch_size:
                await self.flush_in_memory_records()

    async def flush_in_memory_records(self):
        verbose_logger.debug("flushing in memory records")
        response = await self.async_httpx_handler.post(
            url=f"{self.base_url}/projects/{self.project_id}/observe/ingest",
            headers=self.headers,
            json={"records": self.in_memory_records},
        )

        if response.status_code == 200:
            verbose_logger.debug(
                "Galileo Logger:successfully flushed in memory records"
            )
            self.in_memory_records = []
        else:
            verbose_logger.debug("Galileo Logger: failed to flush in memory records")
            verbose_logger.debug(
                "Galileo Logger error=%s, status code=%s",
                response.text,
                response.status_code,
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        verbose_logger.debug("On Async Failure")
