"""
Opik Logger that logs LLM events to an Opik server
"""

from typing import Any, Dict, Callable, List, Union
import datetime

import litellm
from litellm._logging import verbose_logger
from litellm.types.utils import ModelResponse
from litellm.integrations.custom_logger import CustomLogger
import traceback

from .utils import (
    get_opik_config_variable,
    redact_secrets,
    model_response_to_dict,
    create_uuid7
)

from .types import OpikSpan, OpikTrace
import os
import asyncio
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

from litellm.integrations.custom_batch_logger import CustomBatchLogger

OPIK_MAX_BATCH_SIZE = 1000

class OpikLogger(CustomBatchLogger):
    """
    Opik Logger for logging events to an Opik Server
    """

    def __init__(self, **kwargs):
        self.async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        self.opik_base_url = get_opik_config_variable(
            "url_override",
            user_value=kwargs.get("url", None),
            default_value=""
        )
        self.opik_api_key = get_opik_config_variable(
            "api_key",
            user_value=kwargs.get("api_key", None),
            default_value=""
        )
        self.opik_workspace = get_opik_config_variable(
            "workspace",
            user_value=kwargs.get("workspace", None),
            default_value=""
        )

        self.opik_project_name = get_opik_config_variable(
            "project_name",
            user_value=kwargs.get("project_name", None),
            default_value=""
        )
        
        asyncio.create_task(self.periodic_flush())
        self.flush_lock = asyncio.Lock()

        super().__init__(**kwargs, flush_lock=self.flush_lock, batch_size=OPIK_MAX_BATCH_SIZE)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            opik_payload = self._create_opik_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time
            )

            self.log_queue.extend(opik_payload)
            verbose_logger.debug(f"OpikLogger added event to log_queue - Will flush in {self.flush_interval} seconds...")

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger failed to log success event - {str(e)}\n{traceback.format_exc()}"
            )
    
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            opik_payload = self._create_opik_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time
            )

            self.log_queue.extend(opik_payload)
            verbose_logger.debug(f"OpikLogger added event to log_queue - Will flush in {self.flush_interval} seconds...")

            if len(self.log_queue) >= self.batch_size:
                self.async_send_batch()
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger failed to log success event - {str(e)}\n{traceback.format_exc()}"
            )

    async def _submit_batch(self, url: str, headers: Dict[str, str], batch: List[Union[OpikTrace, OpikSpan]]):
        try:
            response = await self.async_httpx_client.post(
                url=url,
                headers=headers,
                json={
                    "traces": batch
                }
            )
            response.raise_for_status()

            if response.status_code >= 300:
                verbose_logger.error(
                    f"OpikLogger - Error: {response.status_code} - {response.text}"
                )
            else:
                verbose_logger.debug(
                    f"OpikLogger - Batch of {len(self.log_queue)} runs successfully created"
                )
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger failed to send trace batch - {str(e)}\n{traceback.format_exc()}"
            )

    async def async_send_batch(self):
        if not self.log_queue:
            return
        
        trace_url = f"{self.opik_base_url}/api/v1/private/traces/batch"
        span_url = f"{self.opik_base_url}/api/v1/private/spans/batch"

        # Create headers
        headers = {}
        if self.opik_workspace:
            headers["Comet-Workspace"] = self.opik_workspace
        
        if self.opik_api_key:
            headers["authorization"] = f"{self.opik_api_key}"

        # Split the log_queue into traces and spans
        traces = [x for x in self.log_queue if isinstance(x, OpikTrace)]
        spans = [x for x in self.log_queue if isinstance(x, OpikSpan)]

        # Send trace batc
        if len(traces) > 0:
            await self._submit_batch(trace_url, headers, traces)
        if len(spans) > 0:
            await self._submit_batch(span_url, headers, spans)

    def _create_opik_payload(self, kwargs, response_obj, start_time, end_time) -> List[Union[OpikTrace, OpikSpan]]:
        if kwargs.get("stream", False):
            if kwargs.get("complete_streaming_response"):
                response_obj = kwargs["complete_streaming_response"]
            elif kwargs.get("async_complete_streaming_response"):
                response_obj = kwargs["async_complete_streaming_response"]
            else:
                verbose_logger.debug("OpikLogger skipping chunk; waiting for end...")
                return
        
        # litellm metadata:
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        # -----
        litellm_opik_metadata = metadata.get("opik", {})
        project_name = litellm_opik_metadata.get("project_name", self.opik_project_name)
        trace_id = litellm_opik_metadata.get("trace_id", None)
        parent_span_id = litellm_opik_metadata.get("parent_span_id", None)
        opik_metadata = litellm_opik_metadata.get("metadata", None)
        opik_tags = litellm_opik_metadata.get("tags", [])

        span_name = "%s_%s_%s" % (
            response_obj.get("model", "unknown-model"),
            response_obj.get("object", "unknown-object"),
            response_obj.get("created", 0),
        )
        trace_name = response_obj.get("object", "unknown type")

        input_data = redact_secrets(kwargs)
        output_data = model_response_to_dict(response_obj)
        metadata = opik_metadata or {}
        metadata["created_from"] = "litellm"
        if kwargs.get("custom_llm_provider"):
            opik_tags.append(kwargs["custom_llm_provider"])
        if "object" in response_obj:
            metadata["type"] = response_obj["object"]
        if "model" in response_obj:
            metadata["model"] = response_obj["model"]
        if "response_cost" in kwargs:
            metadata["cost"] = {
                "total_tokens": kwargs["response_cost"],
                "currency": "USD"
            }
        
        payload = []
        if trace_id is None:
            trace_id = create_uuid7()

            payload.append(
                OpikTrace(
                    project_name=project_name,
                    trace_id=trace_id,
                    name=trace_name,
                    start_time=start_time,
                    end_time=end_time,
                    input=input_data,
                    output=output_data,
                    metadata=metadata,
                    tags=opik_tags,
                )
            )
        
        payload.append(
            OpikSpan(
                project_name=project_name,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                name=span_name,
                type="llm",
                start_time=start_time,
                end_time=end_time,
                input=input_data,
                output=output_data,
                metadata=metadata,
                tags=opik_tags,
            )
        )
        
        return payload
