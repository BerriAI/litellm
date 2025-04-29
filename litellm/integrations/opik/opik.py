"""
Opik Logger that logs LLM events to an Opik server
"""

import asyncio
import json
import traceback
from typing import Dict, List

from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .utils import (
    create_usage_object,
    create_uuid7,
    get_opik_config_variable,
    get_traces_and_spans_from_payload,
)


class OpikLogger(CustomBatchLogger):
    """
    Opik Logger for logging events to an Opik Server
    """

    def __init__(self, **kwargs):
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sync_httpx_client = _get_httpx_client()

        self.opik_project_name = get_opik_config_variable(
            "project_name",
            user_value=kwargs.get("project_name", None),
            default_value="Default Project",
        )

        opik_base_url = get_opik_config_variable(
            "url_override",
            user_value=kwargs.get("url", None),
            default_value="https://www.comet.com/opik/api",
        )
        opik_api_key = get_opik_config_variable(
            "api_key", user_value=kwargs.get("api_key", None), default_value=None
        )
        opik_workspace = get_opik_config_variable(
            "workspace", user_value=kwargs.get("workspace", None), default_value=None
        )

        self.trace_url = f"{opik_base_url}/v1/private/traces/batch"
        self.span_url = f"{opik_base_url}/v1/private/spans/batch"

        self.headers = {}
        if opik_workspace:
            self.headers["Comet-Workspace"] = opik_workspace

        if opik_api_key:
            self.headers["authorization"] = opik_api_key

        self.opik_workspace = opik_workspace
        self.opik_api_key = opik_api_key
        try:
            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger - Asynchronous processing not initialized as we are not running in an async context {str(e)}"
            )
            self.flush_lock = None

        super().__init__(**kwargs, flush_lock=self.flush_lock)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            opik_payload = self._create_opik_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )

            self.log_queue.extend(opik_payload)
            verbose_logger.debug(
                f"OpikLogger added event to log_queue - Will flush in {self.flush_interval} seconds..."
            )

            if len(self.log_queue) >= self.batch_size:
                verbose_logger.debug("OpikLogger - Flushing batch")
                await self.flush_queue()
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger failed to log success event - {str(e)}\n{traceback.format_exc()}"
            )

    def _sync_send(self, url: str, headers: Dict[str, str], batch: Dict):
        try:
            response = self.sync_httpx_client.post(
                url=url, headers=headers, json=batch  # type: ignore
            )
            response.raise_for_status()
            if response.status_code != 204:
                raise Exception(
                    f"Response from opik API status_code: {response.status_code}, text: {response.text}"
                )
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger failed to send batch - {str(e)}\n{traceback.format_exc()}"
            )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            opik_payload = self._create_opik_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )

            traces, spans = get_traces_and_spans_from_payload(opik_payload)
            if len(traces) > 0:
                self._sync_send(
                    url=self.trace_url, headers=self.headers, batch={"traces": traces}
                )
            if len(spans) > 0:
                self._sync_send(
                    url=self.span_url, headers=self.headers, batch={"spans": spans}
                )
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger failed to log success event - {str(e)}\n{traceback.format_exc()}"
            )

    async def _submit_batch(self, url: str, headers: Dict[str, str], batch: Dict):
        try:
            response = await self.async_httpx_client.post(
                url=url, headers=headers, json=batch  # type: ignore
            )
            response.raise_for_status()

            if response.status_code >= 300:
                verbose_logger.error(
                    f"OpikLogger - Error: {response.status_code} - {response.text}"
                )
            else:
                verbose_logger.info(
                    f"OpikLogger - {len(self.log_queue)} Opik events submitted"
                )
        except Exception as e:
            verbose_logger.exception(f"OpikLogger failed to send batch - {str(e)}")

    def _create_opik_headers(self):
        headers = {}
        if self.opik_workspace:
            headers["Comet-Workspace"] = self.opik_workspace

        if self.opik_api_key:
            headers["authorization"] = self.opik_api_key
        return headers

    async def async_send_batch(self):
        verbose_logger.info("Calling async_send_batch")
        if not self.log_queue:
            return

        # Split the log_queue into traces and spans
        traces, spans = get_traces_and_spans_from_payload(self.log_queue)

        # Send trace batch
        if len(traces) > 0:
            await self._submit_batch(
                url=self.trace_url, headers=self.headers, batch={"traces": traces}
            )
            verbose_logger.info(f"Sent {len(traces)} traces")
        if len(spans) > 0:
            await self._submit_batch(
                url=self.span_url, headers=self.headers, batch={"spans": spans}
            )
            verbose_logger.info(f"Sent {len(spans)} spans")

    def _create_opik_payload(  # noqa: PLR0915
        self, kwargs, response_obj, start_time, end_time
    ) -> List[Dict]:
        # Get metadata
        _litellm_params = kwargs.get("litellm_params", {}) or {}
        litellm_params_metadata = _litellm_params.get("metadata", {}) or {}

        # Extract opik metadata
        litellm_opik_metadata = litellm_params_metadata.get("opik", {})
        verbose_logger.debug(
            f"litellm_opik_metadata - {json.dumps(litellm_opik_metadata, default=str)}"
        )
        project_name = litellm_opik_metadata.get("project_name", self.opik_project_name)

        # Extract trace_id and parent_span_id
        current_span_data = litellm_opik_metadata.get("current_span_data", None)
        if isinstance(current_span_data, dict):
            trace_id = current_span_data.get("trace_id", None)
            parent_span_id = current_span_data.get("id", None)
        elif current_span_data:
            trace_id = current_span_data.trace_id
            parent_span_id = current_span_data.id
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
            verbose_logger.debug(
                "OpikLogger skipping event; no standard_logging_object found"
            )
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
        if "current_span_data" in metadata:
            del metadata["current_span_data"]
        metadata["created_from"] = "litellm"

        metadata.update(standard_logging_object.get("metadata", {}))
        if "call_type" in standard_logging_object:
            metadata["type"] = standard_logging_object["call_type"]
        if "status" in standard_logging_object:
            metadata["status"] = standard_logging_object["status"]
        if "response_cost" in kwargs:
            metadata["cost"] = {
                "total_tokens": kwargs["response_cost"],
                "currency": "USD",
            }
        if "response_cost_failure_debug_info" in kwargs:
            metadata["response_cost_failure_debug_info"] = kwargs[
                "response_cost_failure_debug_info"
            ]
        if "model_map_information" in standard_logging_object:
            metadata["model_map_information"] = standard_logging_object[
                "model_map_information"
            ]
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
            verbose_logger.debug(
                f"OpikLogger creating payload for trace with id {trace_id}"
            )

            payload.append(
                {
                    "project_name": project_name,
                    "id": trace_id,
                    "name": trace_name,
                    "start_time": start_time.isoformat() + "Z",
                    "end_time": end_time.isoformat() + "Z",
                    "input": input_data,
                    "output": output_data,
                    "metadata": metadata,
                    "tags": opik_tags,
                }
            )

        span_id = create_uuid7()
        verbose_logger.debug(
            f"OpikLogger creating payload for trace with id {trace_id} and span with id {span_id}"
        )
        payload.append(
            {
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
                "usage": usage,
            }
        )
        verbose_logger.debug(f"Payload: {payload}")
        return payload
