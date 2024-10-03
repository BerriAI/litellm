"""
Opik Logger that logs LLM events to an Opik server
"""

from typing import Dict, List
import json

from litellm._logging import verbose_logger
import traceback

from .utils import (
    get_opik_config_variable,
    create_uuid7,
    create_usage_object
)

import asyncio
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

from litellm.integrations.custom_batch_logger import CustomBatchLogger

class OpikLogger(CustomBatchLogger):
    """
    Opik Logger for logging events to an Opik Server
    """

    def __init__(self, **kwargs):
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        self.opik_base_url = get_opik_config_variable(
            "url_override",
            user_value=kwargs.get("url", None),
            default_value="https://www.comet.com/opik/api"
        )
        self.opik_api_key = get_opik_config_variable(
            "api_key",
            user_value=kwargs.get("api_key", None),
            default_value=None
        )
        self.opik_workspace = get_opik_config_variable(
            "workspace",
            user_value=kwargs.get("workspace", None),
            default_value=None
        )

        self.opik_project_name = get_opik_config_variable(
            "project_name",
            user_value=kwargs.get("project_name", None),
            default_value="Default Project"
        )
        
        asyncio.create_task(self.periodic_flush())
        self.flush_lock = asyncio.Lock()

        super().__init__(**kwargs, flush_lock=self.flush_lock)

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
                verbose_logger.debug("OpikLogger - Flushing batch")
                await self.flush_queue()
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
                verbose_logger.debug("OpikLogger - Flushing batch")
                self.flush_queue()
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger failed to log success event - {str(e)}\n{traceback.format_exc()}"
            )

    async def _submit_batch(self, url: str, headers: Dict[str, str], batch: List[Dict]):
        try:
            response = await self.async_httpx_client.post(
                url=url,
                headers=headers,
                json=batch
            )
            response.raise_for_status()

            if response.status_code >= 300:
                verbose_logger.error(
                    f"OpikLogger - Error: {response.status_code} - {response.text}"
                )
            else:
                verbose_logger.debug(
                    f"OpikLogger - {len(self.log_queue)} Opik events submitted"
                )
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger failed to send trace batch - {str(e)}\n{traceback.format_exc()}"
            )

    async def async_send_batch(self):
        verbose_logger.exception("Calling async_send_batch")
        if not self.log_queue:
            return
        
        trace_url = f"{self.opik_base_url}/v1/private/traces/batch"
        span_url = f"{self.opik_base_url}/v1/private/spans/batch"

        # Create headers
        headers = {}
        if self.opik_workspace:
            headers["Comet-Workspace"] = self.opik_workspace
        
        if self.opik_api_key:
            headers["authorization"] = self.opik_api_key

        # Split the log_queue into traces and spans
        def remove_nulls(x):
            x_ = {k:v for k,v in x.items() if v is not None}
            return x_

        traces = [remove_nulls(x) for x in self.log_queue if "type" not in x]
        spans = [remove_nulls(x) for x in self.log_queue if "type" in x]

        # Send trace batch
        if len(traces) > 0:
            await self._submit_batch(trace_url, headers, {"traces": traces})
        if len(spans) > 0:
            await self._submit_batch(span_url, headers, {"spans": spans})

    def _create_opik_payload(self, kwargs, response_obj, start_time, end_time) -> List[Dict]:
        
        
        # Get metadata
        _litellm_params = kwargs.get("litellm_params", {}) or {}
        litellm_params_metadata = _litellm_params.get("metadata", {}) or {}
        
        # Extract opik metadata
        litellm_opik_metadata = litellm_params_metadata.get("opik", {})
        verbose_logger.debug(f"litellm_opik_metadata - {json.dumps(litellm_opik_metadata, default=str)}")
        project_name = litellm_opik_metadata.get("project_name", self.opik_project_name)

        # Extract trace_id and parent_span_id
        current_span_data = litellm_opik_metadata.get("current_span_data", None)
        if current_span_data:
            trace_id = current_span_data.get("trace_id", None)
            parent_span_id = current_span_data.get("id", None)
        else:
            trace_id = None
            parent_span_id = None
        # Create Opik tags
        opik_tags = litellm_opik_metadata.get("tags", [])
        if kwargs.get("custom_llm_provider"):
            opik_tags.append(kwargs["custom_llm_provider"])

        # Use standard_logging_object to create metadata and input/output data
        standard_logging_object = kwargs.get("standard_logging_object", None)
        if standard_logging_object is None:
            verbose_logger.debug("OpikLogger skipping event; no standard_logging_object found")
            return []
        
        # Create input and output data
        input_data = standard_logging_object.get("messages", {})
        output_data = standard_logging_object.get("response", {})
        
        # Create usage object
        usage = create_usage_object(response_obj["usage"])
        
        # Define span and trace names
        span_name = "%s_%s_%s" % (
            response_obj.get("model", "unknown-model"),
            response_obj.get("object", "unknown-object"),
            response_obj.get("created", 0),
        )
        trace_name = response_obj.get("object", "unknown type")
        
        # Create metadata object, we add the opik metadata first and then
        # update it with the standard_logging_object metadata
        metadata = litellm_opik_metadata
        metadata["created_from"] = "litellm"
        
        metadata.update(standard_logging_object.get("metadata", {}))
        if "call_type" in standard_logging_object:
            metadata["type"] = standard_logging_object["call_type"]
        if "status" in standard_logging_object:
            metadata["status"] = standard_logging_object["status"]
        if "response_cost" in kwargs:
            metadata["cost"] = {
                "total_tokens": kwargs["response_cost"],
                "currency": "USD"
            }
        if "response_cost_failure_debug_info" in kwargs:
            metadata["response_cost_failure_debug_info"] = kwargs["response_cost_failure_debug_info"]
        if "model_map_information" in standard_logging_object:
            metadata["model_map_information"] = standard_logging_object["model_map_information"]
        if "model" in standard_logging_object:
            metadata["model"] = standard_logging_object["model"]
        if "model_id" in standard_logging_object:
            metadata["model_id"] = standard_logging_object["model_id"]
        if "model_group" in standard_logging_object:
            metadata["model_group"] = standard_logging_object["model_group"]
        if "api_base" in standard_logging_object:
            metadata["api_base"] = standard_logging_object["api_base"]
        if "cache_hit" in standard_logging_object:
            metadata["cache_hit"] = standard_logging_object["cache_hit"]
        if "saved_cache_cost" in standard_logging_object:
            metadata["saved_cache_cost"] = standard_logging_object["saved_cache_cost"]
        if "error_str" in standard_logging_object:
            metadata["error_str"] = standard_logging_object["error_str"]
        if "model_parameters" in standard_logging_object:
            metadata["model_parameters"] = standard_logging_object["model_parameters"]
        if "hidden_params" in standard_logging_object:
            metadata["hidden_params"] = standard_logging_object["hidden_params"]
        
        payload = []
        if trace_id is None:
            trace_id = create_uuid7()
            verbose_logger.debug(f"OpikLogger creating payload for trace with id {trace_id}")

            payload.append({
                "project_name": project_name,
                "id": trace_id,
                "name": trace_name,
                "start_time": start_time.isoformat() + "Z",
                "end_time": end_time.isoformat() + "Z",
                "input": input_data,
                "output": output_data,
                "metadata": metadata,
                "tags": opik_tags,
            })

        span_id = create_uuid7()
        verbose_logger.debug(f"OpikLogger creating payload for trace with id {trace_id} and span with id {span_id}")
        payload.append({
            "id": span_id,
            "project_name": project_name,
            "trace_id": trace_id,
            "parent_span_id": parent_span_id,
            "name": span_name,
            "type": "llm",
            "start_time": start_time.isoformat() + "Z",
            "end_time": end_time.isoformat() + "Z",
            "input": input_data,
            "output": output_data,
            "metadata": metadata,
            "tags": opik_tags,
            "usage": usage
        })
        verbose_logger.debug(f"Payload: {payload}")
        return payload
