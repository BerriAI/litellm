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

from . import opik_payload_builder, utils


class OpikLogger(CustomBatchLogger):
    """
    Opik Logger for logging events to an Opik Server
    """

    def __init__(self, **kwargs):
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sync_httpx_client = _get_httpx_client()

        self.opik_project_name = utils.get_opik_config_variable(
            "project_name",
            user_value=kwargs.get("project_name", None),
            default_value="Default Project",
        )

        opik_base_url = utils.get_opik_config_variable(
            "url_override",
            user_value=kwargs.get("url", None),
            default_value="https://www.comet.com/opik/api",
        )
        opik_api_key = utils.get_opik_config_variable(
            "api_key", user_value=kwargs.get("api_key", None), default_value=None
        )
        opik_workspace = utils.get_opik_config_variable(
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

            traces, spans = utils.get_traces_and_spans_from_payload(opik_payload)
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
        traces, spans = utils.get_traces_and_spans_from_payload(self.log_queue)

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

    def _create_opik_payload(self, kwargs, response_obj, start_time, end_time) -> List[Dict]:
        """Create Opik payload using the payload builder module"""
        return opik_payload_builder.build_opik_payload(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            project_name=self.opik_project_name,
        )
