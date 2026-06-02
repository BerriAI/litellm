import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, cast

from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
    get_content_from_model_response,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.llms.openai import AllMessageValues

GALILEO_CLOUD_API_BASE_URL = "https://api.galileo.ai"
# Cap the in-memory buffer so persistent flush failures (e.g. Galileo
# unavailable, invalid credentials) cannot leak memory unboundedly.
GALILEO_MAX_IN_MEMORY_RECORDS = 1000


class LLMResponse(BaseModel):
    latency_ms: int
    status_code: int
    input_text: str
    output_text: str
    node_type: str
    model: str
    num_input_tokens: int
    num_output_tokens: int
    output_logprobs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional. When available, logprobs are used to compute Uncertainty.",
    )
    created_at: str = Field(
        ..., description='timestamp constructed in "%Y-%m-%dT%H:%M:%S" format'
    )
    tags: Optional[List[str]] = None
    user_metadata: Optional[Dict[str, Any]] = None


class GalileoObserve(CustomLogger):
    def __init__(self) -> None:
        self.in_memory_records: List[dict] = []
        self.batch_size = 1
        self.api_key = os.getenv("GALILEO_API_KEY")
        self.project_id = os.getenv("GALILEO_PROJECT_ID")
        self.log_stream_id = os.getenv("GALILEO_LOG_STREAM_ID")
        self.username = os.getenv("GALILEO_USERNAME")
        self.password = os.getenv("GALILEO_PASSWORD")
        self.base_url = self._normalize_base_url(os.getenv("GALILEO_BASE_URL"))
        if self.api_key and not self.base_url:
            self.base_url = GALILEO_CLOUD_API_BASE_URL
        self.use_v2_api = bool(self.api_key)
        self.headers: Optional[Dict[str, str]] = None
        self.async_httpx_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

    @staticmethod
    def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
        if base_url:
            return base_url.rstrip("/")
        return None

    def _is_configured(self) -> bool:
        if not self.project_id or not self.base_url:
            return False
        if self.use_v2_api:
            return bool(self.api_key)
        return bool(self.username and self.password)

    async def async_set_galileo_headers(self) -> None:
        galileo_login_response = await self.async_httpx_handler.post(
            url=f"{self.base_url}/login",
            headers={
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "username": self.username,
                "password": self.password,
            },
        )
        galileo_login_response.raise_for_status()
        access_token = galileo_login_response.json()["access_token"]
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

    async def _ensure_headers(self) -> bool:
        if self.headers is not None:
            return True

        if self.use_v2_api:
            if not self.api_key:
                return False
            self.headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "Galileo-API-Key": self.api_key,
            }
            return True

        if not (self.username and self.password and self.base_url):
            return False

        try:
            await self.async_set_galileo_headers()
            return True
        except Exception as e:
            verbose_logger.debug("Galileo Logger: failed to authenticate: %s", e)
            return False

    @staticmethod
    def _galileo_input_messages(
        messages: Optional[List[Any]], input_text: str
    ) -> List[Dict[str, str]]:
        if not messages:
            return [{"role": "user", "content": input_text}]

        galileo_messages: List[Dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if not role:
                continue
            galileo_messages.append(
                {
                    "role": str(role),
                    "content": convert_content_list_to_str(
                        message=cast(AllMessageValues, message)
                    ),
                }
            )

        if galileo_messages:
            return galileo_messages
        return [{"role": "user", "content": input_text}]

    @staticmethod
    def _record_to_v2_span(record: Dict[str, Any]) -> Dict[str, Any]:
        created_at = record.get("created_at", "")
        if created_at and not re.search(r"(Z|[+-]\d{2}:?\d{2})$", created_at):
            created_at = f"{created_at}Z"

        span: Dict[str, Any] = {
            "type": "llm",
            "name": record.get("node_type", "litellm"),
            "created_at": created_at,
            "input": GalileoObserve._galileo_input_messages(
                record.get("messages"), record.get("input_text", "")
            ),
            "output": {
                "role": "assistant",
                "content": record.get("output_text", ""),
            },
            "status_code": record.get("status_code", 200),
            "model": record.get("model"),
            "metrics": {
                "duration_ns": int(record.get("latency_ms", 0)) * 1_000_000,
                "num_input_tokens": record.get("num_input_tokens"),
                "num_output_tokens": record.get("num_output_tokens"),
            },
        }
        if record.get("tags"):
            span["tags"] = record["tags"]
        return span

    def _get_ingest_request(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self.base_url or not self.project_id:
            return None

        # Snapshot the records to be sent into a new list so concurrent appends
        # during the network round-trip (across the await points in
        # flush_in_memory_records) aren't silently dropped when we later clear
        # the in-memory buffer.
        records = list(self.in_memory_records)

        if self.use_v2_api:
            payload: Dict[str, Any] = {
                "spans": [self._record_to_v2_span(record) for record in records],
                "reliable": False,
            }
            if self.log_stream_id:
                payload["log_stream_id"] = self.log_stream_id
            return (
                f"{self.base_url}/v2/projects/{self.project_id}/spans",
                payload,
            )

        return (
            f"{self.base_url}/projects/{self.project_id}/observe/ingest",
            {"records": records},
        )

    def get_output_str_from_response(
        self, response_obj: Any, kwargs: Dict[str, Any]
    ) -> Optional[str]:
        if response_obj is None:
            return None
        if kwargs.get("call_type", None) == "embedding" or isinstance(
            response_obj, litellm.EmbeddingResponse
        ):
            return None
        if isinstance(response_obj, litellm.TextCompletionResponse):
            return response_obj.choices[0].text
        if isinstance(response_obj, litellm.ImageResponse):
            return json.dumps(response_obj["data"], default=str)
        if isinstance(response_obj, (litellm.ModelResponse, dict)):
            return get_content_from_model_response(response_obj)
        return None

    async def async_log_success_event(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ):
        verbose_logger.debug("On Async Success")

        if not self._is_configured():
            verbose_logger.debug(
                "Galileo Logger: skipping flush — set GALILEO_PROJECT_ID and "
                "either GALILEO_API_KEY (hosted) or GALILEO_USERNAME/GALILEO_PASSWORD "
                "(enterprise Observe)."
            )
            return

        _latency_ms = int((end_time - start_time).total_seconds() * 1000)
        _call_type = kwargs.get("call_type", "litellm")
        input_text = litellm.utils.get_formatted_prompt(
            data=kwargs, call_type=_call_type
        )

        _usage = response_obj.get("usage", {}) or {}
        num_input_tokens = _usage.get("prompt_tokens", 0)
        num_output_tokens = _usage.get("completion_tokens", 0)

        output_text = self.get_output_str_from_response(
            response_obj=response_obj, kwargs=kwargs
        )

        if output_text is not None:
            request_record = LLMResponse(
                latency_ms=_latency_ms,
                status_code=200,
                input_text=input_text,
                output_text=output_text,
                node_type=_call_type,
                model=kwargs.get("model", "-"),
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                created_at=start_time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),  # timestamp str constructed in "%Y-%m-%dT%H:%M:%S" format
            )

            request_dict = request_record.model_dump()
            messages = kwargs.get("messages")
            if messages:
                request_dict["messages"] = messages
            self.in_memory_records.append(request_dict)

            # Bound the buffer so persistent flush failures cannot grow it
            # without limit. Drop the oldest records once we exceed the cap.
            if len(self.in_memory_records) > GALILEO_MAX_IN_MEMORY_RECORDS:
                dropped = len(self.in_memory_records) - GALILEO_MAX_IN_MEMORY_RECORDS
                self.in_memory_records = self.in_memory_records[
                    -GALILEO_MAX_IN_MEMORY_RECORDS:
                ]
                verbose_logger.warning(
                    "Galileo Logger: in-memory buffer exceeded %s records; "
                    "dropped %s oldest record(s). Check Galileo connectivity/credentials.",
                    GALILEO_MAX_IN_MEMORY_RECORDS,
                    dropped,
                )

            if len(self.in_memory_records) >= self.batch_size:
                await self.flush_in_memory_records()

    async def flush_in_memory_records(self):
        if not self.in_memory_records:
            return

        # Capture the number of records that will be sent BEFORE any await so
        # that concurrent appends made by other asyncio tasks during the
        # network round-trip aren't silently dropped on the success-clear.
        records_in_payload = len(self.in_memory_records)

        ingest_request = self._get_ingest_request()
        if ingest_request is None:
            verbose_logger.debug(
                "Galileo Logger: missing GALILEO_BASE_URL or GALILEO_PROJECT_ID"
            )
            return

        if not await self._ensure_headers():
            verbose_logger.debug("Galileo Logger: could not set request headers")
            return

        url, payload = ingest_request
        verbose_logger.debug("flushing in memory records to %s", url)

        try:
            response = await self.async_httpx_handler.post(
                url=url,
                headers=self.headers,
                json=payload,
            )
        except Exception as e:
            verbose_logger.debug(
                "Galileo Logger: failed to flush in memory records: %s", e
            )
            return

        if response.is_success:
            verbose_logger.debug(
                "Galileo Logger: successfully flushed in memory records"
            )
            del self.in_memory_records[:records_in_payload]
        else:
            verbose_logger.debug("Galileo Logger: failed to flush in memory records")
            verbose_logger.debug(
                "Galileo Logger error=%s, status code=%s",
                response.text,
                response.status_code,
            )
            # Legacy enterprise auth caches a bearer token obtained from
            # /login. If the request was rejected for auth reasons, drop the
            # cached headers so the next flush re-authenticates instead of
            # silently failing forever on a stale token. The v2 API key path
            # uses a long-lived static key, so leave its headers in place.
            if not self.use_v2_api and response.status_code in (401, 403):
                self.headers = None

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        verbose_logger.debug("On Async Failure")
