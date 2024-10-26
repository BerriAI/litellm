import asyncio
import os
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from httpx import Response

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.datadog_llm_obs import *
from litellm.types.utils import StandardLoggingPayload


class DataDogLLMObsLogger(CustomBatchLogger):
    def __init__(self, **kwargs):
        try:
            verbose_logger.debug("DataDogLLMObs: Initializing logger")
            if os.getenv("DD_API_KEY", None) is None:
                raise Exception("DD_API_KEY is not set, set 'DD_API_KEY=<>'")

            self.async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )
            self.DD_API_KEY = os.getenv("DD_API_KEY")
            self.intake_url = (
                "https://api.us5.datadoghq.com/api/intake/llm-obs/v1/trace/spans"
            )

            # Optional - only used for testing
            dd_base_url = (
                os.getenv("_DATADOG_BASE_URL")
                or os.getenv("DATADOG_BASE_URL")
                or os.getenv("DD_BASE_URL")
            )
            if dd_base_url:
                self.intake_url = f"{dd_base_url}/api/intake/llm-obs/v1/trace/spans"

            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()
            super().__init__(**kwargs, flush_lock=self.flush_lock)
        except Exception as e:
            verbose_logger.exception(f"DataDogLLMObs: Error initializing - {str(e)}")
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                f"DataDogLLMObs: Logging success event for model {kwargs.get('model', 'unknown')}"
            )
            payload = self.create_llm_obs_payload(
                kwargs, response_obj, start_time, end_time
            )
            verbose_logger.debug(f"DataDogLLMObs: Payload: {payload}")
            self.log_queue.append(payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()
        except Exception as e:
            verbose_logger.exception(
                f"DataDogLLMObs: Error logging success event - {str(e)}"
            )

    async def async_send_batch(self):
        try:
            if not self.log_queue:
                return

            verbose_logger.debug(
                f"DataDogLLMObs: Flushing {len(self.log_queue)} events"
            )

            # Prepare the payload
            payload = {
                "data": {
                    "type": "span",
                    "attributes": {
                        "ml_app": "litellm",
                        "tags": [
                            "service:litellm",
                            f"env:{os.getenv('DD_ENV', 'production')}",
                        ],
                        "spans": self.log_queue,
                    },
                }
            }

            response = await self.async_client.post(
                url=self.intake_url,
                json=payload,
                headers={
                    "DD-API-KEY": self.DD_API_KEY,
                    "Content-Type": "application/json",
                },
            )

            response.raise_for_status()
            if response.status_code != 202:
                raise Exception(
                    f"DataDogLLMObs: Unexpected response - status_code: {response.status_code}, text: {response.text}"
                )

            verbose_logger.debug(
                f"DataDogLLMObs: Successfully sent batch - status_code: {response.status_code}"
            )
            self.log_queue.clear()
        except Exception as e:
            verbose_logger.exception(f"DataDogLLMObs: Error sending batch - {str(e)}")
            import traceback

            traceback.print_exc()

    from litellm.types.integrations.datadog_llm_obs import (
        InputMeta,
        LLMMetrics,
        LLMObsPayload,
        Meta,
        OutputMeta,
    )

    def create_llm_obs_payload(
        self, kwargs: Dict, response_obj: Any, start_time: datetime, end_time: datetime
    ) -> LLMObsPayload:
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )
        if standard_logging_payload is None:
            raise Exception("DataDogLLMObs: standard_logging_object is not set")

        messages = standard_logging_payload["messages"]
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})

        input_meta = InputMeta(messages=messages)
        output_meta = OutputMeta(
            messages=[
                {
                    "content": response_obj.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", ""),
                    "role": "assistant",
                }
            ]
        )

        meta = Meta(kind="llm", input=input_meta, output=output_meta)

        # Calculate metrics (you may need to adjust these based on available data)
        metrics = LLMMetrics(
            input_tokens=float(standard_logging_payload.get("prompt_tokens", 0)),
            output_tokens=float(standard_logging_payload.get("completion_tokens", 0)),
            total_tokens=float(standard_logging_payload.get("total_tokens", 0)),
        )

        return LLMObsPayload(
            parent_id=metadata.get("parent_id", "undefined"),
            trace_id=metadata.get("trace_id", str(uuid.uuid4())),
            span_id=metadata.get("span_id", str(uuid.uuid4())),
            name=metadata.get("name", "litellm_llm_call"),
            meta=meta,
            start_ns=int(start_time.timestamp() * 1e9),
            duration=int((end_time - start_time).total_seconds() * 1e9),
            metrics=metrics,
        )
