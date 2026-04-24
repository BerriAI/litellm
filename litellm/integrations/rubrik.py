"""Rubrik LiteLLM Plugin for tool blocking and batch logging."""

import asyncio
import os
import random
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import httpx
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.litellm_core_utils.core_helpers import safe_deep_copy
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Function,
    GenericGuardrailAPIInputs,
    StandardLoggingPayload,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )

_ENDPOINT_ANTHROPIC_MESSAGES = "/messages"
_WEBHOOK_PATH_TOOL_BLOCKING = "/v1/after_completion/openai/v1"
_WEBHOOK_PATH_LOGGING_BATCH = "/v1/litellm/batch"


@dataclass
class BlockedToolsResult:
    """Returned by _extract_blocked_tools when at least one tool was blocked."""

    allowed_tools: list
    explanation: str


class RubrikLogger(CustomGuardrail, CustomBatchLogger):
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ):
        self.flush_lock = asyncio.Lock()
        kwargs.setdefault("guardrail_name", "rubrik")
        kwargs.setdefault("event_hook", GuardrailEventHooks.post_call)
        kwargs.setdefault("default_on", True)
        super().__init__(
            flush_lock=self.flush_lock,
            **kwargs,
        )

        verbose_logger.debug("initializing rubrik logger")

        self.sampling_rate = 1.0
        rbrk_sampling_rate = os.getenv("RUBRIK_SAMPLING_RATE")
        if rbrk_sampling_rate is not None:
            try:
                parsed_rate = float(rbrk_sampling_rate.strip())
                self.sampling_rate = max(0.0, min(1.0, parsed_rate))
                if parsed_rate != self.sampling_rate:
                    verbose_logger.warning(
                        f"RUBRIK_SAMPLING_RATE={parsed_rate} clamped to "
                        f"{self.sampling_rate}"
                    )
            except ValueError:
                verbose_logger.warning(
                    f"Invalid RUBRIK_SAMPLING_RATE: {rbrk_sampling_rate!r}, using 1.0"
                )

        self.key = api_key or os.getenv("RUBRIK_API_KEY")
        if not self.key:
            verbose_logger.warning(
                "Rubrik: No API key configured. Requests will be unauthenticated."
            )
        _batch_size = os.getenv("RUBRIK_BATCH_SIZE")

        if _batch_size:
            try:
                self.batch_size = int(_batch_size)
            except ValueError:
                verbose_logger.warning(
                    f"Invalid RUBRIK_BATCH_SIZE: {_batch_size!r}, using default"
                )

        _webhook_url = api_base or os.getenv("RUBRIK_WEBHOOK_URL")

        if _webhook_url is None:
            raise ValueError(
                "Rubrik webhook URL not configured. "
                "Set RUBRIK_WEBHOOK_URL or pass api_base."
            )

        _webhook_url = _webhook_url.rstrip("/").removesuffix("/v1")
        self.tool_blocking_endpoint = f"{_webhook_url}{_WEBHOOK_PATH_TOOL_BLOCKING}"
        self.logging_endpoint = f"{_webhook_url}{_WEBHOOK_PATH_LOGGING_BATCH}"

        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        self.tool_blocking_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback,
            params={"timeout": httpx.Timeout(5.0, connect=2.0)},
        )

        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.key:
            self._headers["Authorization"] = f"Bearer {self.key}"

        asyncio.create_task(self.periodic_flush())

    async def aclose(self):
        """Close the dedicated tool blocking HTTP client."""
        await self.tool_blocking_client.close()

    # -- Guardrail hook --------------------------------------------------------

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """Validate tool calls against the blocking service (fail-open)."""
        if input_type != "response":
            return inputs

        tool_calls = inputs.get("tool_calls")
        if not tool_calls:
            return inputs

        try:
            return await self._check_tool_calls(
                inputs, tool_calls, request_data, logging_obj
            )
        except ModifyResponseException:
            raise
        except Exception as e:
            verbose_logger.error(
                f"Tool blocking hook failed: {e}. "
                "Returning original response unchanged.",
                exc_info=True,
            )
            return inputs

    async def _check_tool_calls(
        self,
        inputs: GenericGuardrailAPIInputs,
        tool_calls: Any,
        request_data: dict,
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> GenericGuardrailAPIInputs:
        """Send tool calls to blocking service, raise if any are blocked."""
        message_tool_calls = self._normalize_tool_calls(tool_calls)

        call_details = (
            getattr(logging_obj, "model_call_details", {}) if logging_obj else {}
        )
        response = request_data.get("response")
        request_id = getattr(response, "id", None) if response else None
        if logging_obj and not call_details:
            verbose_logger.warning(
                "Rubrik: logging_obj present but model_call_details is empty "
                "-- request context will be missing"
            )

        response_data = self._build_tool_call_payload(message_tool_calls, request_id)
        req_data = self._extract_request_data(call_details)

        service_response = await self._post_to_tool_blocking_service(
            response_data, req_data
        )
        blocked = self._extract_blocked_tools(service_response, message_tool_calls)

        if blocked:
            model = self._resolve_model(request_data, call_details)
            raise ModifyResponseException(
                message=blocked.explanation,
                model=model,
                request_data=request_data,
                guardrail_name=self.guardrail_name,
            )

        return inputs

    @staticmethod
    def _normalize_tool_calls(tool_calls: Any) -> list[ChatCompletionMessageToolCall]:
        """Convert tool_calls from inputs to ChatCompletionMessageToolCall objects."""
        result = []
        for tc in tool_calls:
            if isinstance(tc, ChatCompletionMessageToolCall):
                result.append(tc)
            elif isinstance(tc, dict):
                func = tc.get("function", {})
                result.append(
                    ChatCompletionMessageToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=Function(
                            name=func.get("name", ""),
                            arguments=func.get("arguments", ""),
                        ),
                    )
                )
            elif hasattr(tc, "id") and hasattr(tc, "function"):
                result.append(
                    ChatCompletionMessageToolCall(
                        id=tc.id or "",
                        type=getattr(tc, "type", None) or "function",
                        function=tc.function,
                    )
                )
            else:
                raise TypeError(
                    f"Cannot normalize tool_call of type {type(tc).__name__}: {tc!r}"
                )
        return result

    @staticmethod
    def _build_tool_call_payload(
        tool_calls: list[ChatCompletionMessageToolCall],
        request_id: str | None,
    ) -> dict[str, Any]:
        """Build a full OpenAI ChatCompletion-format dict for the blocking service."""
        return {
            "id": request_id or f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            tc.model_dump(exclude_none=True) for tc in tool_calls
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

    @staticmethod
    def _extract_request_data(call_details: dict[str, Any]) -> dict[str, Any]:
        """Extract original request data from model_call_details."""
        if not call_details:
            return {}
        litellm_params = call_details.get("litellm_params", {}) or {}
        return {
            "messages": call_details.get("messages"),
            "model": call_details.get("model"),
            "proxy_server_request": litellm_params.get("proxy_server_request"),
        }

    @staticmethod
    def _resolve_model(
        request_data: dict[str, Any], call_details: dict[str, Any]
    ) -> str:
        """Get the model name for the ModifyResponseException."""
        response = request_data.get("response")
        if response and hasattr(response, "model"):
            return response.model or "unknown"
        return call_details.get("model", "unknown")

    # -- Logging hooks ---------------------------------------------------------

    async def _prepare_log_payload(
        self, kwargs: dict, event_type: str
    ) -> StandardLoggingPayload | None:
        """Shared logic for success and failure logging."""
        if random.random() > self.sampling_rate:
            verbose_logger.debug(
                f"Skipping Rubrik {event_type} logging "
                f"(sampling_rate={self.sampling_rate})"
            )
            return None

        # Deep-copy so mutations don't affect other callbacks sharing this object
        standard_logging_payload: StandardLoggingPayload = safe_deep_copy(
            kwargs["standard_logging_object"]
        )

        # For Anthropic /v1/messages requests, LiteLLM creates a separate
        # ModelResponse (with a generated chatcmpl-* id) for logging, which
        # differs from the original Anthropic msg-* id on the response dict.
        # Normalize to litellm_call_id so that the logging and tool-blocking
        # endpoints see the same request identifier.
        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_request = litellm_params.get("proxy_server_request", {}) or {}
        url_path = urllib.parse.urlparse(proxy_request.get("url", "")).path
        if url_path.endswith(_ENDPOINT_ANTHROPIC_MESSAGES):
            _litellm_call_id = kwargs.get("litellm_call_id")
            if _litellm_call_id:
                standard_logging_payload["id"] = _litellm_call_id  # type: ignore[literal-required]

        if "system" in kwargs:
            system_prompt_msg_list = kwargs["system"]
            try:
                if system_prompt_msg_list:
                    system_scaffold = {
                        "role": "system",
                        "content": system_prompt_msg_list,
                    }
                    if isinstance(standard_logging_payload["messages"], list):
                        standard_logging_payload["messages"].insert(0, system_scaffold)
                    elif isinstance(standard_logging_payload["messages"], (dict, str)):
                        standard_logging_payload["messages"] = [
                            system_scaffold,
                            standard_logging_payload["messages"],
                        ]
            except Exception as e:
                verbose_logger.warning(
                    f"Rubrik: failed to prepend system prompt: {e}",
                    exc_info=True,
                )

        return standard_logging_payload

    async def _enqueue_log_event(self, kwargs: dict, event_type: str):
        try:
            payload = await self._prepare_log_payload(kwargs, event_type)
            if payload is None:
                return

            self.log_queue.append(payload)

            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()
        except Exception as e:
            verbose_logger.error(
                f"Rubrik {event_type} logging hook failed: {e}. "
                "Skipping logging for this event.",
                exc_info=True,
            )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._enqueue_log_event(kwargs, "success")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await self._enqueue_log_event(kwargs, "failure")

    # -- Batch logging ---------------------------------------------------------

    async def _log_batch_to_rubrik(self, data):
        try:
            response = await self.async_httpx_client.post(
                url=self.logging_endpoint,
                json=data,
                headers=self._headers,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            verbose_logger.exception(
                f"Rubrik HTTP Error: {e.response.status_code} - {e.response.text}"
            )
        except Exception:
            verbose_logger.exception("Rubrik Layer Error")

    async def async_send_batch(self):
        """Handles sending batches of responses to Rubrik."""
        if not self.log_queue:
            return

        await self._log_batch_to_rubrik(
            data=self.log_queue,
        )

    # -- Tool blocking service -------------------------------------------------

    async def _post_to_tool_blocking_service(
        self,
        response_data: dict[str, Any],
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Post a payload to the tool blocking service and return the response.

        Args:
            response_data: The OpenAI-formatted response payload to send.
            request_data: Original LLM request data to include alongside
                the response for additional context. Empty dict if unavailable.

        Raises:
            Exception: If the service is unavailable or returns an error.
        """
        envelope = {
            "request": request_data,
            "response": response_data,
        }
        verbose_logger.debug(
            f"Sending request to tool blocking service: "
            f"{self.tool_blocking_endpoint}"
        )
        http_response = await self.tool_blocking_client.post(
            self.tool_blocking_endpoint,
            json=envelope,
            headers=self._headers,
        )
        http_response.raise_for_status()
        result: dict[str, Any] = http_response.json()
        return result

    @staticmethod
    def _extract_blocked_tools(
        service_response: dict[str, Any],
        all_tool_calls: list[ChatCompletionMessageToolCall],
    ) -> BlockedToolsResult | None:
        """Determine whether any tool calls were blocked by the service.

        Compares the service response (which contains only allowed tools) against
        the full set of tool calls. Returns None if all tools are allowed, or a
        BlockedToolsResult.

        Expects service_response in OpenAI chat completion format:
            {"choices": [{"message": {"tool_calls": [...], "content": "..."}}]}
        """
        choices = service_response.get("choices", [])
        if not choices:
            raise Exception("Tool blocking service returned empty response")

        message = choices[0].get("message", {})
        returned_tool_calls = message.get("tool_calls", [])
        blocking_explanation = message.get("content", "")

        allowed_ids = {tc["id"] for tc in returned_tool_calls if tc.get("id")}
        allowed_tools = [tc for tc in all_tool_calls if tc.id in allowed_ids]

        if len(allowed_tools) == len(all_tool_calls):
            return None

        explanation = blocking_explanation or "Tool call blocked by policy."
        return BlockedToolsResult(
            allowed_tools=allowed_tools,
            explanation=f"\n\n{explanation}",
        )
