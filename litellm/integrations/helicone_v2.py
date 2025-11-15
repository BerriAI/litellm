"""
Helicone integration that leverages StandardLoggingPayload and supports batching via CustomBatchLogger.
"""

import asyncio
import json
import os
from typing import Any, Dict, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload

__all__ = ["HeliconeLogger"]


class HeliconeLogger(CustomBatchLogger):
    """Batching Helicone logger that consumes the StandardLoggingPayload."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        base = api_base or os.getenv("HELICONE_API_BASE") or "https://api.hconeai.com"
        self.api_base = base[:-1] if base.endswith("/") else base
        self.api_key = api_key or os.getenv("HELICONE_API_KEY")

        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.flush_lock: Optional[asyncio.Lock] = None
        try:
            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()
        except (
            Exception
        ) as exc:  # pragma: no cover - dependent on runtime loop availability
            verbose_logger.debug(
                "HeliconeLogger async batching disabled; running synchronously. %s",
                exc,
            )
            self.flush_lock = None

        super().__init__(flush_lock=self.flush_lock, **kwargs)

        batch_size_override = os.getenv("HELICONE_BATCH_SIZE")
        if batch_size_override:
            try:
                self.batch_size = int(batch_size_override)
            except ValueError:
                verbose_logger.debug(
                    "HeliconeLogger: ignoring invalid HELICONE_BATCH_SIZE=%s",
                    batch_size_override,
                )

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        try:
            data = self._build_data(kwargs, response_obj, start_time, end_time)
            if data is None:
                return
            self._send_sync(data)
        except Exception:
            verbose_logger.exception("HeliconeLogger: sync logging failed")

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        try:
            data = self._build_data(kwargs, response_obj, start_time, end_time)
            if data is None:
                return

            if self.flush_lock is None:
                await self._send_async(data)
                return

            self.log_queue.append(data)
            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()
        except Exception:
            verbose_logger.exception("HeliconeLogger: async logging failed")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "HeliconeLogger: Async logging - Enters logging function for model %s",
                kwargs,
            )
            data = self._build_data(kwargs, response_obj, start_time, end_time)

            if data is None:
                return

            if self.flush_lock is None:
                await self._send_async(data)
                return

            self.log_queue.append(data)
            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()
        except Exception as e:
            verbose_logger.exception(f"HeliconeLogger Layer Error - {str(e)}")
            pass

    async def async_send_batch(self, *args: Any, **kwargs: Any) -> None:
        if not self.log_queue:
            return

        events = list(self.log_queue)
        for event in events:
            try:
                await self._send_async(event)
            except Exception:
                verbose_logger.exception(
                    "HeliconeLogger: failed to send batched Helicone event"
                )

    def _build_data(
        self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> dict:
        logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )
        if logging_payload is None:
            raise ValueError("standard_logging_object not found in kwargs")

        provider_url = logging_payload.get("api_base", "")
        provider_request = self._pick_request_json(kwargs)
        meta: dict = {}
        providerRequest = {
            "url": provider_url,
            "json": provider_request,
            "meta": meta,
        }

        # provider_response = logging_payload.get("response", {})
        provider_response = self._pick_response(logging_payload)
        # provider_response_header = self._pick_response_headers(logging_payload)
        provider_response_status = self._pick_status_code(logging_payload)
        provider_response = {
            "json": provider_response,
            "headers": {},
            "status": provider_response_status,
        }

        start_time_seconds = int(start_time.timestamp())
        start_time_milliseconds = int(
            (start_time.timestamp() - start_time_seconds) * 1000
        )
        end_time_seconds = int(end_time.timestamp())
        end_time_milliseconds = int((end_time.timestamp() - end_time_seconds) * 1000)
        timing = {
            "startTime": {
                "seconds": start_time_seconds,
                "milliseconds": start_time_milliseconds,
            },
            "endTime": {
                "seconds": end_time_seconds,
                "milliseconds": end_time_milliseconds,
            },
        }

        payload_json = {
            "providerRequest": providerRequest,
            "providerResponse": provider_response,
            "timing": timing,
        }
        return self._sanitize(payload_json)

    def _pick_request_json(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if kwargs:
            additional_args = kwargs.get("additional_args") or {}
            if isinstance(additional_args, dict):
                complete_input_dict = additional_args.get("complete_input_dict")
                if isinstance(complete_input_dict, dict):
                    return complete_input_dict
        return {}

    def _pick_response(self, logging_payload: StandardLoggingPayload) -> Any:
        if logging_payload.get("status") == "success":
            return logging_payload.get("response", {})
        return logging_payload.get("error_str", {})

    def _pick_response_headers(
        self, logging_payload: StandardLoggingPayload
    ) -> Dict[str, Any]:
        headers: Dict[str, Any] = {}
        hidden_params = logging_payload.get("hidden_params")
        if isinstance(hidden_params, dict):
            provider_headers = hidden_params.get("response_headers")
            if isinstance(provider_headers, dict):
                headers.update(provider_headers)
        return headers

    def _pick_status_code(self, logging_payload: StandardLoggingPayload) -> int:
        error_information = logging_payload.get("error_information") or {}
        if isinstance(error_information, dict):
            error_code = error_information.get("error_code")
            if isinstance(error_code, str) and error_code:
                return int(error_code)
        return 200

    @staticmethod
    def _sanitize(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the payload."""
        return json.loads(safe_dumps(payload))

    def _send_sync(self, data: Dict[str, Any]) -> None:
        url = f"{self.api_base}/custom/v1/log"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = litellm.module_level_client.post(
            url=url,
            headers=headers,
            json=data,
        )
        verbose_logger.debug(
            "HeliconeLogger: logged Helicone event (status %s)",
            getattr(response, "status_code", "unknown"),
        )

    async def _send_async(self, data: Dict[str, Any]) -> None:
        url = f"{self.api_base}/custom/v1/log"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = await self.async_httpx_client.post(
            url=url,
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        verbose_logger.debug(
            "HeliconeLogger: logged Helicone event (status %s)",
            response.status_code,
        )
