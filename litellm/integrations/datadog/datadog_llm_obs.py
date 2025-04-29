"""
Implements logging integration with Datadog's LLM Observability Service


API Reference: https://docs.datadoghq.com/llm_observability/setup/api/?tab=example#api-standards

"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_any_messages_to_chat_completion_str_messages_conversion,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.datadog_llm_obs import *
from litellm.types.utils import StandardLoggingPayload


class DataDogLLMObsLogger(DataDogLogger, CustomBatchLogger):
    def __init__(self, **kwargs):
        try:
            verbose_logger.debug("DataDogLLMObs: Initializing logger")
            if os.getenv("DD_API_KEY", None) is None:
                raise Exception("DD_API_KEY is not set, set 'DD_API_KEY=<>'")
            if os.getenv("DD_SITE", None) is None:
                raise Exception(
                    "DD_SITE is not set, set 'DD_SITE=<>', example sit = `us5.datadoghq.com`"
                )

            self.async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )
            self.DD_API_KEY = os.getenv("DD_API_KEY")
            self.DD_SITE = os.getenv("DD_SITE")
            self.intake_url = (
                f"https://api.{self.DD_SITE}/api/intake/llm-obs/v1/trace/spans"
            )

            # testing base url
            dd_base_url = os.getenv("DD_BASE_URL")
            if dd_base_url:
                self.intake_url = f"{dd_base_url}/api/intake/llm-obs/v1/trace/spans"

            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()
            self.log_queue: List[LLMObsPayload] = []
            CustomBatchLogger.__init__(self, **kwargs, flush_lock=self.flush_lock)
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
                "data": DDIntakePayload(
                    type="span",
                    attributes=DDSpanAttributes(
                        ml_app=self._get_datadog_service(),
                        tags=[self._get_datadog_tags()],
                        spans=self.log_queue,
                    ),
                ),
            }
            verbose_logger.debug("payload %s", json.dumps(payload, indent=4))
            response = await self.async_client.post(
                url=self.intake_url,
                json=payload,
                headers={
                    "DD-API-KEY": self.DD_API_KEY,
                    "Content-Type": "application/json",
                },
            )

            if response.status_code != 202:
                raise Exception(
                    f"DataDogLLMObs: Unexpected response - status_code: {response.status_code}, text: {response.text}"
                )

            verbose_logger.debug(
                f"DataDogLLMObs: Successfully sent batch - status_code: {response.status_code}"
            )
            self.log_queue.clear()
        except httpx.HTTPStatusError as e:
            verbose_logger.exception(
                f"DataDogLLMObs: Error sending batch - {e.response.text}"
            )
        except Exception as e:
            verbose_logger.exception(f"DataDogLLMObs: Error sending batch - {str(e)}")

    def create_llm_obs_payload(
        self, kwargs: Dict, response_obj: Any, start_time: datetime, end_time: datetime
    ) -> LLMObsPayload:
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )
        if standard_logging_payload is None:
            raise Exception("DataDogLLMObs: standard_logging_object is not set")

        messages = standard_logging_payload["messages"]
        messages = self._ensure_string_content(messages=messages)

        metadata = kwargs.get("litellm_params", {}).get("metadata", {})

        input_meta = InputMeta(
            messages=handle_any_messages_to_chat_completion_str_messages_conversion(
                messages
            )
        )
        output_meta = OutputMeta(messages=self._get_response_messages(response_obj))

        meta = Meta(
            kind="llm",
            input=input_meta,
            output=output_meta,
            metadata=self._get_dd_llm_obs_payload_metadata(standard_logging_payload),
        )

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
            tags=[
                self._get_datadog_tags(standard_logging_object=standard_logging_payload)
            ],
        )

    def _get_response_messages(self, response_obj: Any) -> List[Any]:
        """
        Get the messages from the response object

        for now this handles logging /chat/completions responses
        """
        if isinstance(response_obj, litellm.ModelResponse):
            return [response_obj["choices"][0]["message"].json()]
        return []

    def _ensure_string_content(
        self, messages: Optional[Union[str, List[Any], Dict[Any, Any]]]
    ) -> List[Any]:
        if messages is None:
            return []
        if isinstance(messages, str):
            return [messages]
        elif isinstance(messages, list):
            return [message for message in messages]
        elif isinstance(messages, dict):
            return [str(messages.get("content", ""))]
        return []

    def _get_dd_llm_obs_payload_metadata(
        self, standard_logging_payload: StandardLoggingPayload
    ) -> Dict:
        _metadata = {
            "model_name": standard_logging_payload.get("model", "unknown"),
            "model_provider": standard_logging_payload.get(
                "custom_llm_provider", "unknown"
            ),
        }
        _standard_logging_metadata: dict = (
            dict(standard_logging_payload.get("metadata", {})) or {}
        )
        _metadata.update(_standard_logging_metadata)
        return _metadata
