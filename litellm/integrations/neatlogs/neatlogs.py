""" Neatlogs integration for litellm  """

import asyncio
import datetime
import json
import os
import traceback
from typing import Any, Optional, Union
import uuid

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.additional_logging_utils import AdditionalLoggingUtils
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus


class NeatlogsLogger(CustomBatchLogger, AdditionalLoggingUtils):
    def __init__(self, neatlogs_api_key: Optional[str] = None, **kwargs):
        try:
            # Get API key from parameter or environment variable
            self.api_key = neatlogs_api_key or os.getenv("NEATLOGS_API_KEY")

            if self.api_key is None:
                raise Exception(
                    "NEATLOGS_API_KEY is not set. Please set the NEATLOGS_API_KEY environment variable or pass neatlogs_api_key parameter."
                )

            self.endpoint = "https://app.neatlogs.com/api/data/v2"

            self.async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )
            self.sync_client = _get_httpx_client()
            self.flush_lock = asyncio.Lock()
            self._periodic_task_started = False
            super().__init__(
                **kwargs,
                flush_lock=self.flush_lock,
                batch_size=litellm.DEFAULT_BATCH_SIZE,
            )

        except Exception as e:
            verbose_logger.exception(
                f"Neatlogs: Got exception on init Neatlogs client {str(e)}"
            )
            raise e

    async def _ensure_periodic_task_started(self):
        """Ensure the periodic flush task is started"""
        if not getattr(self, "_periodic_task_started", False):
            try:
                self._periodic_task = asyncio.create_task(self.periodic_flush())
                self._periodic_task_started = True
                verbose_logger.debug("Neatlogs: Started periodic flush task")
            except RuntimeError:
                # Still no running event loop, skip for now
                verbose_logger.debug(
                    "Neatlogs: No running event loop, skipping periodic task"
                )
                pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "Neatlogs: Logging - Enters logging function for model %s", kwargs
            )
            await self._ensure_periodic_task_started()
            await self._log_async_event(
                kwargs, response_obj, start_time, end_time, status="SUCCESS"
            )

        except Exception as e:
            verbose_logger.exception(
                f"Neatlogs Layer Error - {str(e)}\\{traceback.format_exc()}"
            )
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "Neatlogs: Logging - Enters logging function for model %s", kwargs
            )
            await self._ensure_periodic_task_started()
            await self._log_async_event(
                kwargs, response_obj, start_time, end_time, status="FAILURE"
            )

        except Exception as e:
            verbose_logger.exception(
                f"Neatlogs Layer Error - {str(e)}\\{traceback.format_exc()}"
            )
            pass

    async def async_send_batch(self):
        try:
            if not self.log_queue:
                verbose_logger.exception("Neatlogs: log_queue does not exist")
                return

            verbose_logger.debug(
                "Neatlogs - about to flush %s events on %s",
                len(self.log_queue),
                self.endpoint,
            )

            # Format each payload in the batch according to Neatlogs API specification
            formatted_batch = []
            for payload in self.log_queue:
                api_payload = {
                    "dataDump": json.dumps(payload),
                    "projectAPIKey": self.api_key,
                    "externalTraceId": payload.get("trace_id"),
                    "timestamp": payload.get("start_time"),
                }
                formatted_batch.append(api_payload)

            headers = {
                "Content-Type": "application/json",
            }

            response = await self.async_client.post(
                url=self.endpoint,
                json=formatted_batch,  # type: ignore[arg-type]  # httpx accepts both dict and list for json=; stubs may show warning, but this is correct for batch API.
                headers=headers,
            )

            response.raise_for_status()
            if response.status_code not in [200, 201, 206]:
                raise Exception(
                    f"Response from Neatlogs API status_code: {response.status_code}, text: {response.text}"
                )

            verbose_logger.debug(
                "Neatlogs: Response from Neatlogs API status_code: %s, text: %s",
                response.status_code,
                response.text,
            )
            self.log_queue.clear()
        except Exception as e:
            verbose_logger.exception(
                f"Neatlogs Error sending batch API - {str(e)}\\{traceback.format_exc()}"
            )

    async def _log_async_event(
        self, kwargs, response_obj, start_time, end_time, status
    ):
        payload = self.create_neatlogs_payload(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            status=status,
        )

        self.log_queue.append(payload)
        verbose_logger.debug(
            f"Neatlogs, event added to queue. Will flush in {self.flush_interval} seconds..."
        )

        if len(self.log_queue) >= self.batch_size:
            await self.async_send_batch()

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Sync Log success events to Neatlogs

        - Creates a Neatlogs payload
        - instantly logs it on Neatlogs API
        """
        try:
            payload = self.create_neatlogs_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                status="SUCCESS",
            )

            # Format payload according to Neatlogs API specification
            api_payload = {
                "dataDump": json.dumps(payload),
                "projectAPIKey": self.api_key,
                "externalTraceId": payload.get("trace_id"),
                "timestamp": payload.get("start_time"),
            }

            response = self.sync_client.post(
                url=self.endpoint,
                json=api_payload,
                headers={
                    "Content-Type": "application/json",
                },
            )

            response.raise_for_status()
            if response.status_code not in [200, 201, 206]:
                raise Exception(
                    f"Response from Neatlogs API status_code: {response.status_code}, text: {response.text}"
                )

            verbose_logger.debug(
                "Neatlogs: Response from Neatlogs API status_code: %s, text: %s",
                response.status_code,
                response.text,
            )

        except Exception as e:
            verbose_logger.exception(
                f"Neatlogs Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Sync Log failure events to Neatlogs

        - Creates a Neatlogs payload
        - instantly logs it on Neatlogs API
        """
        try:
            payload = self.create_neatlogs_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                status="FAILURE",
            )

            # Format payload according to Neatlogs API specification
            api_payload = {
                "dataDump": json.dumps(payload),
                "projectAPIKey": self.api_key,
                "externalTraceId": payload.get("trace_id"),
                "timestamp": payload.get("start_time"),
            }

            response = self.sync_client.post(
                url=self.endpoint,
                json=api_payload,
                headers={
                    "Content-Type": "application/json",
                },
            )

            response.raise_for_status()
            if response.status_code not in [200, 201, 206]:
                raise Exception(
                    f"Response from Neatlogs API status_code: {response.status_code}, text: {response.text}"
                )

            verbose_logger.debug(
                "Neatlogs: Response from Neatlogs API status_code: %s, text: %s",
                response.status_code,
                response.text,
            )

        except Exception as e:
            verbose_logger.exception(
                f"Neatlogs Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    def create_neatlogs_payload(
        self,
        kwargs: Union[dict, Any],
        response_obj: Any,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        status: str,
    ) -> dict:
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {}) or {}

        # Try to get completion content
        completion_content = ""
        if (
            status == "SUCCESS"
            and response_obj
            and response_obj.choices
            and response_obj.choices[0].message
        ):
            completion_content = response_obj.choices[0].message.content

        # Try to get usage info
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if status == "SUCCESS" and response_obj and response_obj.usage:
            prompt_tokens = response_obj.usage.prompt_tokens
            completion_tokens = response_obj.usage.completion_tokens
            total_tokens = response_obj.usage.total_tokens

        payload = {
            "session_id": metadata.get("session_id"),
            "agent_id": metadata.get("agent_id"),
            "thread_id": metadata.get("thread_id"),
            "span_id": str(uuid.uuid4()),
            "trace_id": kwargs.get("litellm_call_id"),
            "parent_span_id": metadata.get("parent_span_id"),
            "node_type": "llm_call",
            "node_name": kwargs.get("model"),
            "model": kwargs.get("model"),
            "provider": "litellm",
            "framework": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": kwargs.get("response_cost", 0.0),
            "messages": kwargs.get("messages"),
            "completion": completion_content,
            "timestamp": end_time.isoformat(),
            "start_time": start_time.timestamp(),
            "end_time": end_time.timestamp(),
            "duration": (end_time - start_time).total_seconds(),
            "tags": metadata.get("tags", []),
            "error_report": str(response_obj) if status == "FAILURE" else None,
            "status": status,
            "api_key": self.api_key,
        }
        return payload

    def force_flush(self):
        """Force flush any remaining events in the queue (for testing/debugging)"""
        if self.log_queue:
            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create task
                    asyncio.create_task(self.async_send_batch())
                else:
                    # If no loop running, run synchronously
                    loop.run_until_complete(self.async_send_batch())
            except RuntimeError:
                # No event loop, create new one
                asyncio.run(self.async_send_batch())
        else:
            pass

    async def async_health_check(self) -> IntegrationHealthCheckStatus:
        """
        Check if the service is healthy
        """
        try:
            # Create a simple test payload
            test_payload = {
                "session_id": "health_check",
                "agent_id": "health_check",
                "thread_id": "health_check",
                "span_id": str(uuid.uuid4()),
                "trace_id": str(uuid.uuid4()),
                "parent_span_id": None,
                "node_type": "health_check",
                "node_name": "health_check",
                "model": "health_check",
                "provider": "health_check",
                "framework": "litellm",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "cost": 0.0,
                "messages": [{"role": "user", "content": "health check"}],
                "completion": "OK",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "start_time": datetime.datetime.utcnow().timestamp(),
                "end_time": datetime.datetime.utcnow().timestamp(),
                "duration": 0.1,
                "tags": ["health_check"],
                "error_report": None,
                "status": "SUCCESS",
                "api_key": self.api_key,
            }

            # Format test payload according to Neatlogs API specification
            api_payload = {
                "dataDump": json.dumps(test_payload),
                "projectAPIKey": self.api_key,
                "externalTraceId": test_payload.get("trace_id"),
                "timestamp": test_payload.get("start_time"),
            }

            response = await self.async_client.post(
                url=self.endpoint,
                json=api_payload,
                headers={
                    "Content-Type": "application/json",
                },
            )

            response.raise_for_status()
            return IntegrationHealthCheckStatus(
                status="healthy",
                error_message=None,
            )
        except httpx.HTTPStatusError as e:
            return IntegrationHealthCheckStatus(
                status="unhealthy",
                error_message=e.response.text,
            )
        except Exception as e:
            return IntegrationHealthCheckStatus(
                status="unhealthy",
                error_message=str(e),
            )

    async def get_request_response_payload(
        self,
        request_id: str,
        start_time_utc: Optional[datetime.datetime],
        end_time_utc: Optional[datetime.datetime],
    ) -> Optional[dict]:
        """
        Get the request and response payload for a given `request_id`
        """

        return None
