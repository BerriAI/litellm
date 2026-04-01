"""
Opik Logger that logs LLM events to an Opik server
"""

import asyncio
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)

from . import opik_payload_builder, utils

try:
    from opik.api_objects import opik_client
except Exception:
    opik_client = None


def _should_skip_event(kwargs: Dict[str, Any]) -> bool:
    """Check if event should be skipped due to missing standard_logging_object."""
    if kwargs.get("standard_logging_object") is None:
        verbose_logger.debug(
            "OpikLogger skipping event; no standard_logging_object found"
        )
        return True
    return False


class OpikLogger(CustomBatchLogger):
    """
    Opik Logger for logging events to an Opik Server
    """

    def __init__(self, **kwargs: Any) -> None:
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sync_httpx_client = _get_httpx_client()

        self.opik_project_name: str = (
            utils.get_opik_config_variable(
                "project_name",
                user_value=kwargs.get("project_name", None),
                default_value="Default Project",
            )
            or "Default Project"
        )

        opik_base_url: str = (
            utils.get_opik_config_variable(
                "url_override",
                user_value=kwargs.get("url", None),
                default_value="https://www.comet.com/opik/api",
            )
            or "https://www.comet.com/opik/api"
        )
        opik_api_key: Optional[str] = utils.get_opik_config_variable(
            "api_key", user_value=kwargs.get("api_key", None), default_value=None
        )
        opik_workspace: Optional[str] = utils.get_opik_config_variable(
            "workspace", user_value=kwargs.get("workspace", None), default_value=None
        )

        self.trace_url: str = f"{opik_base_url}/v1/private/traces/batch"
        self.span_url: str = f"{opik_base_url}/v1/private/spans/batch"

        self.headers: Dict[str, str] = {}
        if opik_workspace:
            self.headers["Comet-Workspace"] = opik_workspace

        if opik_api_key:
            self.headers["authorization"] = opik_api_key

        self.opik_workspace: Optional[str] = opik_workspace
        self.opik_api_key: Optional[str] = opik_api_key
        try:
            asyncio.create_task(self.periodic_flush())
            self.flush_lock: Optional[asyncio.Lock] = asyncio.Lock()
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger - Asynchronous processing not initialized as we are not running in an async context {str(e)}"
            )
            self.flush_lock = None

        # Initialize _opik_client attribute
        if opik_client is not None:
            self._opik_client = opik_client.get_client_cached()
        else:
            self._opik_client = None

        super().__init__(**kwargs, flush_lock=self.flush_lock)

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        try:
            if _should_skip_event(kwargs):
                return

            # Build payload using the payload builder
            trace_payload, span_payload = opik_payload_builder.build_opik_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                project_name=self.opik_project_name,
            )

            if self._opik_client is not None:
                # Opik native client is available, use it to send data
                if trace_payload is not None:
                    self._opik_client.trace(
                        id=trace_payload.id,
                        name=trace_payload.name,
                        start_time=datetime.fromisoformat(trace_payload.start_time),
                        end_time=datetime.fromisoformat(trace_payload.end_time),
                        input=trace_payload.input,
                        output=trace_payload.output,
                        metadata=trace_payload.metadata,
                        tags=trace_payload.tags,
                        thread_id=trace_payload.thread_id,
                        project_name=trace_payload.project_name,
                    )

                self._opik_client.span(
                    id=span_payload.id,
                    trace_id=span_payload.trace_id,
                    parent_span_id=span_payload.parent_span_id,
                    name=span_payload.name,
                    type=span_payload.type,
                    model=span_payload.model,
                    start_time=datetime.fromisoformat(span_payload.start_time),
                    end_time=datetime.fromisoformat(span_payload.end_time),
                    input=span_payload.input,
                    output=span_payload.output,
                    metadata=span_payload.metadata,
                    tags=span_payload.tags,
                    usage=span_payload.usage,
                    project_name=span_payload.project_name,
                    provider=span_payload.provider,
                    total_cost=span_payload.total_cost,
                )
            else:
                # Add payloads to LiteLLM queue
                if trace_payload is not None:
                    self.log_queue.append(trace_payload.__dict__)
                self.log_queue.append(span_payload.__dict__)

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

    def _sync_send(
        self, url: str, headers: Dict[str, str], batch: Dict[str, Any]
    ) -> None:
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

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        try:
            if _should_skip_event(kwargs):
                return

            # Build payload using the payload builder
            trace_payload, span_payload = opik_payload_builder.build_opik_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                project_name=self.opik_project_name,
            )
            if self._opik_client is not None:
                # Opik native client is available, use it to send data
                if trace_payload is not None:
                    self._opik_client.trace(
                        id=trace_payload.id,
                        name=trace_payload.name,
                        start_time=datetime.fromisoformat(trace_payload.start_time),
                        end_time=datetime.fromisoformat(trace_payload.end_time),
                        input=trace_payload.input,
                        output=trace_payload.output,
                        metadata=trace_payload.metadata,
                        tags=trace_payload.tags,
                        thread_id=trace_payload.thread_id,
                        project_name=trace_payload.project_name,
                    )

                self._opik_client.span(
                    id=span_payload.id,
                    trace_id=span_payload.trace_id,
                    parent_span_id=span_payload.parent_span_id,
                    name=span_payload.name,
                    type=span_payload.type,
                    model=span_payload.model,
                    start_time=datetime.fromisoformat(span_payload.start_time),
                    end_time=datetime.fromisoformat(span_payload.end_time),
                    input=span_payload.input,
                    output=span_payload.output,
                    metadata=span_payload.metadata,
                    tags=span_payload.tags,
                    usage=span_payload.usage,
                    project_name=span_payload.project_name,
                    provider=span_payload.provider,
                    total_cost=span_payload.total_cost,
                )
            else:
                # Opik native client is not available, use LiteLLM queue to send data
                if trace_payload is not None:
                    self._sync_send(
                        url=self.trace_url,
                        headers=self.headers,
                        batch={"traces": [trace_payload.__dict__]},
                    )

                # Always send span
                self._sync_send(
                    url=self.span_url,
                    headers=self.headers,
                    batch={"spans": [span_payload.__dict__]},
                )
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger failed to log success event - {str(e)}\n{traceback.format_exc()}"
            )

    async def _submit_batch(
        self, url: str, headers: Dict[str, str], batch: Dict[str, Any]
    ) -> None:
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

    def _create_opik_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.opik_workspace:
            headers["Comet-Workspace"] = self.opik_workspace

        if self.opik_api_key:
            headers["authorization"] = self.opik_api_key
        return headers

    async def async_send_batch(self) -> None:
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
