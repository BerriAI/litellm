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
from .opik_payload_builder import types

try:
    from opik.api_objects import opik_client
except Exception:
    opik_client = None


def _should_skip_event(kwargs: Dict[str, Any]) -> bool:
    """Check if event should be skipped due to missing standard_logging_object."""
    if kwargs.get("standard_logging_object") is None:
        verbose_logger.debug(
            "OpikLogger: skipping event; no standard_logging_object found"
        )
        return True
    return False


class OpikLogger(CustomBatchLogger):
    """
    Opik Logger for logging events to an Opik Server.

    Optional environment variables:
        OPIK_API_KEY: Opik API key if you are using Opik cloud.
        OPIK_PROJECT_NAME: Opik project name.
        OPIK_WORKSPACE: Opik workspace name.
        OPIK_URL_OVERRIDE: The URL of the Opik server if a custom installation is used.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        """
        Create an OpikLogger instance.
        
        The Opik configuration can be provided as keyword arguments, as environment variables,
        or as Opik configuration file in user home directory.
        
        Args:
            project_name: Opik project name [Optional].
            workspace: Opik workspace name [Optional].
            api_key: Opik API key if you are using Opik cloud [Optional].
            url_override: The URL of the Opik server if a custom installation is used [Optional].
            
        Example:
            ```python
            import asyncio
            import logging
            import time
            from typing import List, Dict
            
            import litellm
            from litellm.integrations.opik import opik
            
            # Configure logging
            logging.basicConfig(level=logging.DEBUG)
            logger = logging.getLogger(__name__)
            
            opik_handler = opik.OpikLogger()
            litellm.callbacks = [opik_handler]
            
            async def completion(messages: List[Dict[str, str]]):
                for message in messages:
                    response = await litellm.acompletion(
                        model="gpt-3.5-turbo",
                        messages=[message],
                        stream=True
                    )
                    async for chunk in response:
                        print(chunk)
                        continue
            
            async def waiting_for_log_queue_flush(timeout:float=10):
                end_time = time.time() + timeout
                while time.time() < end_time:
                    await asyncio.sleep(1)
                    if len(opik_handler.log_queue) == 0:
                        break
            
            async def main():
                messages = [
                    {"role": "user", "content": "Hi 👋 - i'm openai"},
                    {"role": "user", "content": "What is your name?"},
                ]
                try:
                    await completion(messages)
            
                    await waiting_for_log_queue_flush()
                except Exception as e:
                    logger.error(f"Error during completion: {str(e)}")
                    raise
            
            if __name__ == "__main__":
                asyncio.run(main())
            ```
        """
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

        self.flush_lock = asyncio.Lock()
        self.periodic_started = False
        self.periodic_failed = False

        # Initialize _opik_client attribute
        if opik_client is not None:
            self._opik_client = opik_client.get_client_cached()
        else:
            self._opik_client = None

        super().__init__(**kwargs, flush_lock=self.flush_lock)

    async def _init_periodic_flush_task(self):
        if self.periodic_started or self.periodic_failed:
            return

        try:
            asyncio.create_task(self.periodic_flush())
            self.periodic_started = True
        except Exception:
            verbose_logger.exception(
                "OpikLogger: Asynchronous processing not initialized as we are not running in an asyncio context. "
                "If you are using LiteLLM's async mode, please initialize the OpikLogger within the asyncio context."
            )
            self.periodic_failed = True

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        try:
            await self._log_async_event_unsafe(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger: failed to asynchronously log success event - {str(e)}\n{traceback.format_exc()}"
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
                    f"Response from OPIK API status_code: {response.status_code}, text: {response.text}"
                )
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger: failed to send batch - {str(e)}\n{traceback.format_exc()}"
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
                self._send_via_opik_native_client(
                    trace_payload=trace_payload,
                    span_payload=span_payload
                )
                return

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
                f"OpikLogger: failed to log success event - {str(e)}\n{traceback.format_exc()}"
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
                    f"OpikLogger: Error: {response.status_code} - {response.text}"
                )
            else:
                if "traces" in batch:
                    verbose_logger.info(
                        f"OpikLogger: Successfully sent a batch of {len(batch['traces'])} Opik traces"
                    )
                elif "spans" in batch:
                    verbose_logger.info(
                        f"OpikLogger: Successfully sent a batch of {len(batch['spans'])} Opik spans"
                    )
                else:
                    verbose_logger.info(
                        f"OpikLogger: successfully sent {len(batch)} Opik event(s)"
                    )
        except Exception as e:
            verbose_logger.exception(f"OpikLogger: failed to send batch - {str(e)}")

    def _create_opik_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.opik_workspace:
            headers["Comet-Workspace"] = self.opik_workspace

        if self.opik_api_key:
            headers["authorization"] = self.opik_api_key
        return headers

    async def async_log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        try:
            await self._log_async_event_unsafe(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            verbose_logger.exception(
                f"OpikLogger: failed to asynchronously log failure event - {str(e)}\n{traceback.format_exc()}"
            )

    async def async_send_batch(self) -> None:
        verbose_logger.info("OpikLogger: Calling async_send_batch")
        if not self.log_queue:
            return

        # Split the log_queue into traces and spans
        traces, spans = utils.get_traces_and_spans_from_payload(self.log_queue)

        if len(traces) > 0:
            verbose_logger.info(f"OpikLogger: Attempting to send batch of {len(traces)} traces")
            await self._submit_batch(
                url=self.trace_url, headers=self.headers, batch={"traces": traces}
            )

        if len(spans) > 0:
            verbose_logger.info(f"OpikLogger: Attempting to send batch of {len(spans)} spans")
            await self._submit_batch(
                url=self.span_url, headers=self.headers, batch={"spans": spans}
            )

    async def _log_async_event_unsafe(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        if _should_skip_event(kwargs):
            return

        await self._init_periodic_flush_task()

        # Build payload using the payload builder
        # response_obj might be None or have error information in failure cases
        trace_payload, span_payload = opik_payload_builder.build_opik_payload(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            project_name=self.opik_project_name,
        )

        if self._opik_client is not None:
            # Opik native client is available, use it to send data
            self._send_via_opik_native_client(
                trace_payload=trace_payload,
                span_payload=span_payload
            )
            return

        # Add payloads to LiteLLM queue
        if trace_payload is not None:
            self.log_queue.append(trace_payload.__dict__)
        self.log_queue.append(span_payload.__dict__)

        verbose_logger.debug(
            f"OpikLogger: added event to log_queue - Will flush in {self.flush_interval} seconds..."
        )

        if len(self.log_queue) >= self.batch_size:
            verbose_logger.debug("OpikLogger: Flushing batch")
            await self.flush_queue()

    def _send_via_opik_native_client(
        self,
        trace_payload: Optional[types.TracePayload],
        span_payload: types.SpanPayload
    ) -> None:
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
