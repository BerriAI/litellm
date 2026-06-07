from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import httpx
from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
    get_content_from_model_response,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    HttpxBinaryResponseContent,
    ResponsesAPIResponse,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

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
    num_total_tokens: int
    cost: Optional[float] = Field(
        default=None,
        description="Total cost of the LLM call in USD as computed by LiteLLM.",
    )
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
        messages: Optional[Any], input_text: str
    ) -> List[Dict[str, str]]:
        if isinstance(messages, dict):
            messages = messages.get("messages")
        if not messages:
            return [{"role": "user", "content": input_text}]
        if not isinstance(messages, list):
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
    def _local_timezone():
        return datetime.now().astimezone().tzinfo or timezone.utc

    @staticmethod
    def _format_created_at(dt: Union[datetime, Any]) -> str:
        """Serialize timestamps as UTC ISO-8601 for Galileo."""
        if not isinstance(dt, datetime):
            return str(dt)

        if dt.tzinfo is None:
            # LiteLLM often passes naive datetimes in local time; convert to UTC
            # instead of appending Z to local time (which shifts Traces tab sorting).
            dt = dt.replace(tzinfo=GalileoObserve._local_timezone())

        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _normalize_created_at(created_at: str) -> str:
        if created_at and not re.search(r"(Z|[+-]\d{2}:?\d{2})$", created_at):
            return f"{created_at}Z"
        return created_at

    @staticmethod
    def _token_metrics_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
        num_input_tokens = int(record.get("num_input_tokens") or 0)
        num_output_tokens = int(record.get("num_output_tokens") or 0)
        num_total_tokens = int(record.get("num_total_tokens") or 0)
        if num_total_tokens == 0 and (num_input_tokens or num_output_tokens):
            num_total_tokens = num_input_tokens + num_output_tokens
        metrics: Dict[str, Any] = {
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "num_total_tokens": num_total_tokens,
        }
        cost = record.get("cost")
        if cost is not None:
            metrics["cost"] = float(cost)
        return metrics

    @staticmethod
    def _record_to_v2_span(
        record: Dict[str, Any],
        *,
        trace_id: str,
        span_id: str,
    ) -> Dict[str, Any]:
        created_at = GalileoObserve._normalize_created_at(record.get("created_at", ""))

        span: Dict[str, Any] = {
            "type": "llm",
            "id": span_id,
            "trace_id": trace_id,
            "parent_id": trace_id,
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
                **GalileoObserve._token_metrics_from_record(record),
            },
        }
        if record.get("tags"):
            span["tags"] = record["tags"]
        return span

    @staticmethod
    def _record_to_v2_trace(record: Dict[str, Any]) -> Dict[str, Any]:
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        created_at = GalileoObserve._normalize_created_at(record.get("created_at", ""))

        return {
            "type": "trace",
            "id": trace_id,
            "name": record.get("node_type", "litellm"),
            "created_at": created_at,
            "input": record.get("input_text", ""),
            "output": record.get("output_text", ""),
            "status_code": record.get("status_code", 200),
            "metrics": {
                "duration_ns": int(record.get("latency_ms", 0)) * 1_000_000,
                **GalileoObserve._token_metrics_from_record(record),
            },
            "spans": [
                GalileoObserve._record_to_v2_span(
                    record, trace_id=trace_id, span_id=span_id
                )
            ],
        }

    def _build_traces_payload(self, records: List[dict]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "traces": [self._record_to_v2_trace(record) for record in records],
            "logging_method": "api_direct",
            "reliable": False,
            "is_complete": True,
        }
        if self.log_stream_id:
            payload["log_stream_id"] = self.log_stream_id
        return payload

    def _get_ingest_request(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self.base_url or not self.project_id:
            return None

        # Snapshot the records to be sent into a new list so concurrent appends
        # during the network round-trip (across the await points in
        # flush_in_memory_records) aren't silently dropped when we later clear
        # the in-memory buffer.
        records = list(self.in_memory_records)
        payload = self._build_traces_payload(records)

        if self.use_v2_api:
            return (
                f"{self.base_url}/ingest/traces/{self.project_id}",
                payload,
            )

        # Username/password auth logs in for a JWT and uses the standard v2 traces API.
        return (
            f"{self.base_url}/v2/projects/{self.project_id}/traces",
            payload,
        )

    @staticmethod
    def _redact_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        if not headers:
            return {}
        redacted: Dict[str, str] = {}
        for key, value in headers.items():
            if key.lower() in {"authorization", "galileo-api-key"} and value:
                redacted[key] = (
                    f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                )
            else:
                redacted[key] = value
        return redacted

    def _log_flush_config(self) -> None:
        verbose_logger.debug(
            "Galileo Logger flush config: use_v2_api=%s base_url=%s project_id=%s "
            "log_stream_id=%s api_key_set=%s username_set=%s record_count=%s",
            self.use_v2_api,
            self.base_url,
            self.project_id,
            self.log_stream_id,
            bool(self.api_key),
            bool(self.username),
            len(self.in_memory_records),
        )

    @staticmethod
    def _log_v2_payload_validation(payload: Dict[str, Any]) -> None:
        missing_fields: List[str] = []
        traces = payload.get("traces", [])
        if not traces:
            missing_fields.append("traces")

        for trace_index, trace in enumerate(traces):
            if not isinstance(trace, dict):
                continue
            for field in ("id", "type", "spans"):
                if field not in trace:
                    missing_fields.append(f"traces[{trace_index}].{field}")

            trace_id = trace.get("id")
            for span_index, span in enumerate(trace.get("spans", [])):
                if not isinstance(span, dict):
                    continue
                for field in ("id", "trace_id", "parent_id"):
                    if field not in span:
                        missing_fields.append(
                            f"traces[{trace_index}].spans[{span_index}].{field}"
                        )
                if trace_id and span.get("trace_id") != trace_id:
                    missing_fields.append(
                        f"traces[{trace_index}].spans[{span_index}].trace_id mismatch"
                    )

        if missing_fields:
            verbose_logger.debug(
                "Galileo Logger: ingest /traces payload validation issues: %s",
                missing_fields,
            )

    def _log_flush_payload(self, url: str, payload: Dict[str, Any]) -> None:
        traces = payload.get("traces", [])
        verbose_logger.debug(
            "Galileo Logger flush URL: %s trace_count=%s",
            url,
            len(traces) if isinstance(traces, list) else 0,
        )
        if self.use_v2_api and "/ingest/traces/" in url:
            self._log_v2_payload_validation(payload)

    @staticmethod
    def _log_http_status_error(error: httpx.HTTPStatusError, url: str) -> None:
        response = error.response
        verbose_logger.debug(
            "Galileo Logger HTTP error: status=%s url=%s",
            response.status_code,
            url,
        )
        verbose_logger.debug(
            "Galileo Logger HTTP error response body: %s",
            response.text,
        )
        try:
            verbose_logger.debug(
                "Galileo Logger HTTP error response json: %s",
                response.json(),
            )
        except Exception:
            pass

    @staticmethod
    def _build_prompt(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        optional_params = kwargs.get("optional_params", {}) or {}
        prompt: Dict[str, Any] = {"messages": kwargs.get("messages")}
        if optional_params.get("functions") is not None:
            prompt["functions"] = optional_params["functions"]
        if optional_params.get("tools") is not None:
            prompt["tools"] = optional_params["tools"]
        return prompt

    @staticmethod
    def _serialize_galileo_output(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value

        def _json_default(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return str(obj)

        return json.dumps(value, default=_json_default)

    @staticmethod
    def _prompt_to_input_text(prompt: Dict[str, Any]) -> str:
        messages = prompt.get("messages")
        if messages is not None:
            text = GalileoObserve._input_text_from_messages(messages)
            if text:
                return text
        return json.dumps(prompt, default=str)

    @staticmethod
    def _get_chat_content_for_galileo(response_obj: litellm.ModelResponse) -> Any:
        if response_obj.choices and len(response_obj.choices) > 0:
            message = response_obj["choices"][0]["message"]
            if hasattr(message, "json"):
                message_json = message.json()
                if isinstance(message_json, str):
                    return json.loads(message_json)
                return message_json
            return message
        return None

    @staticmethod
    def _get_text_completion_content_for_galileo(
        response_obj: litellm.TextCompletionResponse,
    ) -> Optional[str]:
        if response_obj.choices and len(response_obj.choices) > 0:
            return response_obj.choices[0].text
        return None

    @staticmethod
    def _get_responses_api_content_for_galileo(
        response_obj: ResponsesAPIResponse,
    ) -> Any:
        if hasattr(response_obj, "output") and response_obj.output:
            return response_obj.output
        return None

    @staticmethod
    def _langfuse_style_rerank_prompt(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Match Langfuse rerank input: prompt = {"messages": kwargs.get("messages")}."""
        return {"messages": kwargs.get("messages")}

    def _get_galileo_input_output_content(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
    ) -> Tuple[str, Optional[str], Any]:
        """
        Mirror Langfuse _get_langfuse_input_output_content for Galileo ingest.

        Returns (input_text, output_text, messages_for_span). output_text None skips ingest.
        """
        call_type = kwargs.get("call_type")
        prompt = self._build_prompt(kwargs)

        if (
            level == "ERROR"
            and status_message is not None
            and isinstance(status_message, str)
        ):
            return self._prompt_to_input_text(prompt), status_message, prompt

        if response_obj is not None and (
            call_type == "embedding"
            or isinstance(response_obj, litellm.EmbeddingResponse)
        ):
            return self._prompt_to_input_text(prompt), None, prompt

        if response_obj is not None and isinstance(response_obj, litellm.ModelResponse):
            output = self._get_chat_content_for_galileo(response_obj)
            return (
                self._prompt_to_input_text(prompt),
                self._serialize_galileo_output(output),
                kwargs.get("messages") or [],
            )

        if response_obj is not None and isinstance(
            response_obj, HttpxBinaryResponseContent
        ):
            return self._prompt_to_input_text(prompt), "speech-output", prompt

        if response_obj is not None and isinstance(
            response_obj, litellm.TextCompletionResponse
        ):
            output = self._get_text_completion_content_for_galileo(response_obj)
            return (
                self._prompt_to_input_text(prompt),
                self._serialize_galileo_output(output),
                kwargs.get("messages") or [],
            )

        if response_obj is not None and isinstance(response_obj, litellm.ImageResponse):
            output = response_obj.get("data", None)
            return (
                self._prompt_to_input_text(prompt),
                self._serialize_galileo_output(output),
                prompt,
            )

        if response_obj is not None and isinstance(
            response_obj, litellm.TranscriptionResponse
        ):
            output = response_obj.get("text", None)
            return (
                self._prompt_to_input_text(prompt),
                self._serialize_galileo_output(output),
                prompt,
            )

        if response_obj is not None and isinstance(
            response_obj, litellm.RerankResponse
        ):
            output = response_obj.results
            rerank_prompt = self._langfuse_style_rerank_prompt(kwargs)
            return (
                json.dumps(rerank_prompt, default=str),
                self._serialize_galileo_output(output),
                rerank_prompt,
            )

        if response_obj is not None and isinstance(response_obj, ResponsesAPIResponse):
            output = self._get_responses_api_content_for_galileo(response_obj)
            return (
                self._prompt_to_input_text(prompt),
                self._serialize_galileo_output(output),
                kwargs.get("messages") or [],
            )

        if (
            call_type == "_arealtime"
            and response_obj is not None
            and isinstance(response_obj, list)
        ):
            input_val = kwargs.get("input")
            return (
                self._serialize_galileo_output(input_val) or "",
                self._serialize_galileo_output(response_obj),
                input_val,
            )

        if (
            call_type == "pass_through_endpoint"
            and response_obj is not None
            and isinstance(response_obj, dict)
        ):
            output = response_obj.get("response", "")
            return (
                self._prompt_to_input_text(prompt),
                self._serialize_galileo_output(output),
                prompt,
            )

        if response_obj is not None and isinstance(response_obj, dict):
            output = get_content_from_model_response(response_obj)
            return (
                self._prompt_to_input_text(prompt),
                self._serialize_galileo_output(output),
                kwargs.get("messages") or [],
            )

        return self._prompt_to_input_text(prompt), None, kwargs.get("messages") or []

    def get_output_str_from_response(
        self, response_obj: Any, kwargs: Dict[str, Any]
    ) -> Optional[str]:
        _, output_text, _ = self._get_galileo_input_output_content(
            kwargs=kwargs, response_obj=response_obj
        )
        return output_text

    @staticmethod
    def _input_text_from_messages(messages: Any) -> str:
        """Return a plain-string summary of the input suitable for the trace-level input field."""
        if isinstance(messages, str):
            return messages
        if not isinstance(messages, list):
            return ""
        # Use the last user/human message so the trace table shows the actual prompt
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() in ("user", "human"):
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    )
                if content:
                    return str(content)
        # Fallback: first non-empty content of any role
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    )
                if content:
                    return str(content)
        return ""

    async def async_log_success_event(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ):
        verbose_logger.debug("On Async Success")
        try:
            await self._async_log_success_event_impl(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception:
            verbose_logger.exception(
                "Galileo Logger: unexpected error in async_log_success_event"
            )

    async def _async_log_success_event_impl(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ):
        if not self._is_configured():
            verbose_logger.debug(
                "Galileo Logger: skipping — GALILEO_PROJECT_ID=%s GALILEO_API_KEY=%s GALILEO_BASE_URL=%s",
                bool(self.project_id),
                bool(self.api_key),
                bool(self.base_url),
            )
            return

        slo: Optional[Dict[str, Any]] = kwargs.get("standard_logging_object")
        if slo is None:
            verbose_logger.debug(
                "Galileo Logger: no standard_logging_object in kwargs, skipping"
            )
            return

        _call_type: str = str(
            slo.get("call_type") or kwargs.get("call_type") or "litellm"
        )

        input_text, output_text, messages = self._get_galileo_input_output_content(
            kwargs=kwargs, response_obj=response_obj
        )
        if output_text is None:
            verbose_logger.debug(
                "Galileo Logger: skipping %s — no text output to log", _call_type
            )
            return

        raw_start = slo.get("startTime")
        raw_end = slo.get("endTime")
        if raw_start is None or raw_end is None:
            verbose_logger.debug(
                "Galileo Logger: standard_logging_object missing startTime/endTime, "
                "falling back to start_time/end_time params"
            )
            if not isinstance(start_time, datetime) or not isinstance(
                end_time, datetime
            ):
                return
            start_ts = start_time
            end_ts = end_time
            if start_ts.tzinfo is None:
                start_ts = start_ts.replace(tzinfo=GalileoObserve._local_timezone())
            if end_ts.tzinfo is None:
                end_ts = end_ts.replace(tzinfo=GalileoObserve._local_timezone())
            start_ts = start_ts.astimezone(timezone.utc)
            end_ts = end_ts.astimezone(timezone.utc)
        else:
            start_ts = datetime.fromtimestamp(float(raw_start), tz=timezone.utc)
            end_ts = datetime.fromtimestamp(float(raw_end), tz=timezone.utc)
        _latency_ms = max(0, int((end_ts - start_ts).total_seconds() * 1000))
        num_input_tokens = int(slo.get("prompt_tokens") or 0)
        num_output_tokens = int(slo.get("completion_tokens") or 0)
        num_total_tokens = int(slo.get("total_tokens") or 0)
        if num_total_tokens == 0 and (num_input_tokens or num_output_tokens):
            num_total_tokens = num_input_tokens + num_output_tokens

        request_record = LLMResponse(
            latency_ms=_latency_ms,
            status_code=200,
            input_text=input_text,
            output_text=output_text,
            node_type=_call_type,
            model=str(slo.get("model") or kwargs.get("model") or "-"),
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            num_total_tokens=num_total_tokens,
            cost=slo.get("response_cost"),
            created_at=GalileoObserve._format_created_at(start_ts),
        )

        request_dict = request_record.model_dump()
        if isinstance(messages, dict):
            messages = messages.get("messages")
        if isinstance(messages, list) and messages:
            request_dict["messages"] = messages
        self.in_memory_records.append(request_dict)
        verbose_logger.debug(
            "Galileo Logger: queued record, in_memory=%d", len(self.in_memory_records)
        )

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
                "Galileo Logger: missing GALILEO_BASE_URL or GALILEO_PROJECT_ID — skipping flush"
            )
            return

        if not await self._ensure_headers():
            verbose_logger.debug(
                "Galileo Logger: could not set request headers — skipping flush"
            )
            return

        url, payload = ingest_request
        self._log_flush_config()
        self._log_flush_payload(url=url, payload=payload)
        verbose_logger.debug(
            "Galileo Logger flush headers: %s",
            self._redact_headers(self.headers),
        )
        verbose_logger.debug("flushing in memory records to %s", url)

        try:
            response = await self.async_httpx_handler.post(
                url=url,
                headers=self.headers,
                json=payload,
            )
        except httpx.HTTPStatusError as e:
            self._log_http_status_error(error=e, url=url)
            verbose_logger.debug(
                "Galileo Logger: failed to flush in memory records: %s", e
            )
            return
        except Exception as e:
            verbose_logger.debug(
                "Galileo Logger: failed to flush in memory records: %s", e
            )
            return

        if response.is_success:
            verbose_logger.debug(
                "Galileo Logger: successfully flushed in memory records"
            )
            verbose_logger.debug(
                "Galileo Logger flush response: status=%s body=%s",
                response.status_code,
                response.text,
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
