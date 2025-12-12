"""Gray Swan Cygnal guardrail integration."""

import os
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import Choices, LLMResponseTypes, ModelResponse


class GraySwanGuardrailMissingSecrets(Exception):
    """Raised when the Gray Swan API key is missing."""


class GraySwanGuardrailAPIError(Exception):
    """Raised when the Gray Swan API returns an error."""


class GraySwanGuardrail(CustomGuardrail):
    """
    Guardrail that calls Gray Swan's Cygnal monitoring endpoint.

    see: https://docs.grayswan.ai/cygnal/monitor-requests
    """

    SUPPORTED_ON_FLAGGED_ACTIONS = {"block", "monitor", "passthrough"}
    DEFAULT_ON_FLAGGED_ACTION = "monitor"
    BASE_API_URL = "https://api.grayswan.ai"
    MONITOR_PATH = "/cygnal/monitor"
    SUPPORTED_REASONING_MODES = {"off", "hybrid", "thinking"}

    def __init__(
        self,
        guardrail_name: Optional[str] = "grayswan",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        on_flagged_action: Optional[str] = None,
        violation_threshold: Optional[float] = None,
        reasoning_mode: Optional[str] = None,
        categories: Optional[Dict[str, str]] = None,
        policy_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        api_key_value = api_key or os.getenv("GRAYSWAN_API_KEY")
        if not api_key_value:
            raise GraySwanGuardrailMissingSecrets(
                "Gray Swan API key missing. Set `GRAYSWAN_API_KEY` or pass `api_key`."
            )
        self.api_key: str = api_key_value

        base = api_base or os.getenv("GRAYSWAN_API_BASE") or self.BASE_API_URL
        self.api_base = base.rstrip("/")
        self.monitor_url = f"{self.api_base}{self.MONITOR_PATH}"

        action = on_flagged_action
        if action and action.lower() in self.SUPPORTED_ON_FLAGGED_ACTIONS:
            self.on_flagged_action = action.lower()
        else:
            if action:
                verbose_proxy_logger.warning(
                    "Gray Swan Guardrail: Unsupported on_flagged_action '%s', defaulting to '%s'.",
                    action,
                    self.DEFAULT_ON_FLAGGED_ACTION,
                )
            self.on_flagged_action = self.DEFAULT_ON_FLAGGED_ACTION

        self.violation_threshold = self._resolve_threshold(violation_threshold)
        self.reasoning_mode = self._resolve_reasoning_mode(reasoning_mode)
        self.categories = categories
        self.policy_id = policy_id

        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Guardrail hook entry points
    # ------------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
            "responses",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.pre_call
            )
            is not True
        ):
            return data

        verbose_proxy_logger.debug("Gray Swan Guardrail: pre-call hook triggered")

        # Get messages - handle both "messages" (Chat Completions/Anthropic) and "input" (Responses API)
        messages = data.get("messages")
        if not messages:
            # Try Responses API format which uses "input" instead of "messages"
            input_data = data.get("input")
            if input_data:
                # input can be a string or list of messages
                if isinstance(input_data, str):
                    messages = [{"role": "user", "content": input_data}]
                elif isinstance(input_data, list):
                    messages = input_data
        if not messages:
            verbose_proxy_logger.debug("Gray Swan Guardrail: No messages in data")
            return data

        dynamic_body = self.get_guardrail_dynamic_request_body_params(data) or {}

        payload = self._prepare_payload(messages, dynamic_body)
        if payload is None:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: no content to scan; skipping request"
            )
            return data

        await self.run_grayswan_guardrail(payload, data, GuardrailEventHooks.pre_call)
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.during_call
            )
            is not True
        ):
            return data

        verbose_proxy_logger.debug("GraySwan Guardrail: during-call hook triggered")

        # Get messages - handle both "messages" (Chat Completions/Anthropic) and "input" (Responses API)
        messages = data.get("messages")
        if not messages:
            # Try Responses API format which uses "input" instead of "messages"
            input_data = data.get("input")
            if input_data:
                # input can be a string or list of messages
                if isinstance(input_data, str):
                    messages = [{"role": "user", "content": input_data}]
                elif isinstance(input_data, list):
                    messages = input_data
        if not messages:
            verbose_proxy_logger.debug("Gray Swan Guardrail: No messages in data")
            return data

        dynamic_body = self.get_guardrail_dynamic_request_body_params(data) or {}

        payload = self._prepare_payload(messages, dynamic_body)
        if payload is None:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: no content to scan; skipping request"
            )
            return data

        await self.run_grayswan_guardrail(
            payload, data, GuardrailEventHooks.during_call
        )
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return response

        verbose_proxy_logger.debug("GraySwan Guardrail: post-call hook triggered")

        # Handle both Pydantic objects and plain dicts (Anthropic Messages returns dict)
        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        elif isinstance(response, dict):
            response_dict = response
        else:
            response_dict = {}

        # Try to extract messages from different response formats
        response_messages: List[Dict[str, Any]] = []

        # 1. OpenAI Chat Completions format (choices[].message)
        for choice in response_dict.get("choices", []):
            if isinstance(choice, dict):
                msg = choice.get("message")
                if msg is not None:
                    response_messages.append(
                        msg if isinstance(msg, dict) else msg.model_dump()
                    )

        # 2. OpenAI Responses API format (output[].content[].text)
        if not response_messages:
            for output_item in response_dict.get("output", []):
                if isinstance(output_item, dict) and output_item.get("type") == "message":
                    content_list = output_item.get("content", [])
                    text_parts: List[str] = []
                    for content_item in content_list:
                        if isinstance(content_item, dict):
                            # Handle output_text type
                            if content_item.get("type") == "output_text":
                                text = content_item.get("text", "")
                                if text:
                                    text_parts.append(text)
                            # Handle text type (alternative format)
                            elif content_item.get("type") == "text":
                                text = content_item.get("text", "")
                                if text:
                                    text_parts.append(text)
                    if text_parts:
                        response_messages.append({
                            "role": output_item.get("role", "assistant"),
                            "content": "\n".join(text_parts)
                        })

        # 3. Anthropic Messages format (content[].text at root level)
        if not response_messages:
            # Try to get content from response_dict or directly from response object
            content_list = response_dict.get("content") or []
            if not content_list and hasattr(response, "content"):
                content_list = response.content or []  # type: ignore

            if content_list and isinstance(content_list, list):
                text_parts_anthropic: List[str] = []
                for content_item in content_list:
                    text: Optional[str] = None

                    # Try multiple ways to extract text from content item
                    # 1. If it's a dict with type "text"
                    if isinstance(content_item, dict):
                        if content_item.get("type") == "text":
                            text = content_item.get("text", "")
                    # 2. If it has model_dump(), convert and check
                    elif hasattr(content_item, "model_dump"):
                        item_dict = content_item.model_dump()
                        if isinstance(item_dict, dict) and item_dict.get("type") == "text":
                            text = item_dict.get("text", "")
                    # 3. If it has text attribute directly (TextBlock object)
                    elif hasattr(content_item, "text"):
                        # Check type if available
                        item_type = getattr(content_item, "type", "text")
                        if item_type == "text":
                            text = getattr(content_item, "text", "")

                    if text:
                        text_parts_anthropic.append(text)

                if text_parts_anthropic:
                    role = response_dict.get("role") or getattr(response, "role", "assistant")
                    response_messages.append({
                        "role": role,
                        "content": "\n".join(text_parts_anthropic)
                    })

        if not response_messages:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: no response messages detected; skipping post-call scan"
            )
            return response

        dynamic_body = self.get_guardrail_dynamic_request_body_params(data) or {}

        payload = self._prepare_payload(response_messages, dynamic_body)
        if payload is None:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: no content to scan; skipping request"
            )
            return response

        await self.run_grayswan_guardrail(payload, data, GuardrailEventHooks.post_call)

        # If passthrough mode and detection info exists, replace response content with violation message
        if self.on_flagged_action == "passthrough" and "metadata" in data:
            guardrail_detections = data.get("metadata", {}).get(
                "guardrail_detections", []
            )
            if guardrail_detections:
                # Replace the model response content with guardrail violation message
                violation_message = self._format_violation_message(
                    guardrail_detections, is_output=True
                )

                # Handle ModelResponse (OpenAI-style chat/text completions)
                # Use isinstance to narrow the type for mypy
                if isinstance(response, ModelResponse) and response.choices:
                    verbose_proxy_logger.debug(
                        "Gray Swan Guardrail: Replacing response content in ModelResponse format"
                    )
                    for choice in response.choices:
                        # Handle chat completion format (message.content)
                        # Choices has message attribute, StreamingChoices has delta
                        if isinstance(choice, Choices) and hasattr(choice, "message") and hasattr(
                            choice.message, "content"
                        ):
                            choice.message.content = violation_message
                        # Handle text completion format (text)
                        # Text attribute might be set dynamically, use setattr
                        elif hasattr(choice, "text"):
                            setattr(choice, "text", violation_message)

                        # Update finish_reason to indicate content filtering
                        if hasattr(choice, "finish_reason"):
                            choice.finish_reason = "content_filter"

                # Handle OpenAI Responses API format (output[].content[])
                elif hasattr(response, "output") and response.output:  # type: ignore
                    verbose_proxy_logger.debug(
                        "Gray Swan Guardrail: Replacing response content in Responses API format"
                    )
                    for output_item in response.output:  # type: ignore
                        # Check if output_item has content attribute (ResponseOutputMessage)
                        if hasattr(output_item, "content") and output_item.content:
                            # Replace all content items with a single text item containing the violation message
                            from litellm.types.responses.main import OutputText
                            output_item.content = [
                                OutputText(
                                    type="output_text",
                                    text=violation_message,
                                    annotations=[],
                                )
                            ]
                        # Also handle dict format
                        elif isinstance(output_item, dict) and "content" in output_item:
                            output_item["content"] = [
                                {"type": "output_text", "text": violation_message, "annotations": []}
                            ]

                # Handle AnthropicMessagesResponse format
                elif hasattr(response, "content") and isinstance(response.content, list):  # type: ignore
                    verbose_proxy_logger.debug(
                        "Gray Swan Guardrail: Replacing response content in Anthropic Messages format"
                    )
                    # Replace content blocks with text block containing violation message
                    response.content = [  # type: ignore
                        {"type": "text", "text": violation_message}
                    ]
                    # Update stop_reason if present
                    if hasattr(response, "stop_reason"):
                        response.stop_reason = "end_turn"  # type: ignore

                else:
                    verbose_proxy_logger.warning(
                        "Gray Swan Guardrail: Passthrough mode enabled but response format not recognized. "
                        "Cannot replace content. Response type: %s",
                        type(response).__name__,
                    )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return response

    @log_guardrail_information
    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterator,
        request_data: dict,
    ) -> AsyncIterator:
        """
        Hook for post-call processing of streaming responses.

        Collects all chunks, extracts content, runs guardrail check,
        and replaces response if violation detected in passthrough mode.
        """
        if (
            self.should_run_guardrail(
                data=request_data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            # If guardrail should not run, yield chunks as-is
            async for chunk in response:
                yield chunk
            return

        verbose_proxy_logger.debug(
            "GraySwan Guardrail: async_post_call_streaming_iterator_hook triggered"
        )

        # Collect all chunks
        collected_chunks: List[Any] = []
        async for chunk in response:
            collected_chunks.append(chunk)

        # Extract content from collected chunks
        content = self._extract_content_from_streaming_chunks(collected_chunks)

        if not content:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: no content extracted from streaming response; yielding original chunks"
            )
            for chunk in collected_chunks:
                yield chunk
            return

        # Prepare payload and run guardrail
        messages = [{"role": "assistant", "content": content}]
        dynamic_body = self.get_guardrail_dynamic_request_body_params(request_data) or {}
        payload = self._prepare_payload(messages, dynamic_body)

        if payload is None:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: no content to scan; yielding original chunks"
            )
            for chunk in collected_chunks:
                yield chunk
            return

        # Run guardrail check
        await self.run_grayswan_guardrail(
            payload, request_data, GuardrailEventHooks.post_call
        )

        # Check if violation was detected in passthrough mode
        if self.on_flagged_action == "passthrough" and "metadata" in request_data:
            guardrail_detections = request_data.get("metadata", {}).get(
                "guardrail_detections", []
            )
            if guardrail_detections:
                verbose_proxy_logger.debug(
                    "Gray Swan Guardrail: Violation detected in streaming response, replacing with violation message"
                )
                violation_message = self._format_violation_message(
                    guardrail_detections, is_output=True
                )

                # Create SSE violation response based on chunk type
                if collected_chunks and isinstance(collected_chunks[0], bytes):
                    # Anthropic SSE bytes format
                    for sse_chunk in self._create_anthropic_sse_violation_response(
                        violation_message, request_data
                    ):
                        yield sse_chunk
                else:
                    # OpenAI/standard format - yield a single violation chunk
                    # This is a fallback; most streaming cases should be handled by the bytes path
                    verbose_proxy_logger.warning(
                        "Gray Swan Guardrail: Non-bytes streaming format, yielding original chunks"
                    )
                    for chunk in collected_chunks:
                        yield chunk
                return

        # No violation or not passthrough mode - yield original chunks
        for chunk in collected_chunks:
            yield chunk

    def _extract_content_from_streaming_chunks(
        self, chunks: List[Any]
    ) -> Optional[str]:
        """
        Extract text content from streaming chunks.

        Handles both bytes (Anthropic SSE) and object (ModelResponseStream) formats.
        """
        if not chunks:
            return None

        # Check if chunks are bytes (Anthropic SSE format)
        if isinstance(chunks[0], bytes):
            return self._extract_content_from_sse_bytes(chunks)

        # Handle ModelResponseStream objects
        text_parts: List[str] = []
        for chunk in chunks:
            if hasattr(chunk, "choices"):
                for choice in chunk.choices:
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        if choice.delta.content:
                            text_parts.append(choice.delta.content)

        return "".join(text_parts) if text_parts else None

    def _extract_content_from_sse_bytes(self, chunks: List[bytes]) -> Optional[str]:
        """
        Extract text content from Anthropic SSE bytes chunks.

        Parses the SSE format and extracts text from content_block_delta events.
        """
        import json

        text_parts: List[str] = []
        for chunk in chunks:
            try:
                chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                for line in chunk_str.split("\n"):
                    line = line.strip()
                    if not line or line.startswith("event:"):
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                            # Handle Anthropic content_block_delta events
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        text_parts.append(text)
                        except json.JSONDecodeError:
                            pass
            except Exception:
                continue

        return "".join(text_parts) if text_parts else None

    def _create_anthropic_sse_violation_response(
        self, violation_message: str, request_data: dict
    ) -> List[bytes]:
        """
        Create Anthropic SSE formatted violation response chunks.

        Returns a list of bytes representing SSE events for the violation message.
        """
        import json
        import uuid

        model = request_data.get("model", "unknown")
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        events: List[bytes] = []

        # message_start event
        message_start = {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        events.append(f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode("utf-8"))

        # content_block_start event
        content_block_start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        events.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode("utf-8"))

        # content_block_delta event with the violation message
        content_block_delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": violation_message},
        }
        events.append(f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode("utf-8"))

        # content_block_stop event
        content_block_stop = {"type": "content_block_stop", "index": 0}
        events.append(f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode("utf-8"))

        # message_delta event
        message_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": len(violation_message.split())},
        }
        events.append(f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode("utf-8"))

        # message_stop event
        message_stop = {"type": "message_stop"}
        events.append(f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n".encode("utf-8"))

        return events

    # ------------------------------------------------------------------
    # Core GraySwan interaction
    # ------------------------------------------------------------------

    async def run_grayswan_guardrail(
        self,
        payload: dict,
        data: Optional[dict] = None,
        hook_type: Optional[GuardrailEventHooks] = None,
    ):
        headers = self._prepare_headers()

        try:
            response = await self.async_handler.post(
                url=self.monitor_url,
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: monitor response %s", safe_dumps(result)
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - depends on HTTP client behaviour
            verbose_proxy_logger.exception(
                "Gray Swan Guardrail: API request failed: %s", exc
            )
            raise GraySwanGuardrailAPIError(str(exc)) from exc

        self._process_grayswan_response(result, data, hook_type)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "grayswan-api-key": self.api_key,
        }

    def _prepare_payload(
        self, messages: list[dict], dynamic_body: dict
    ) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {}
        payload["messages"] = messages

        categories = dynamic_body.get("categories") or self.categories
        if categories:
            payload["categories"] = categories

        policy_id = dynamic_body.get("policy_id") or self.policy_id
        if policy_id:
            payload["policy_id"] = policy_id

        reasoning_mode = dynamic_body.get("reasoning_mode") or self.reasoning_mode
        if reasoning_mode:
            payload["reasoning_mode"] = reasoning_mode

        return payload

    def _process_grayswan_response(
        self,
        response_json: Dict[str, Any],
        data: Optional[dict] = None,
        hook_type: Optional[GuardrailEventHooks] = None,
    ) -> None:
        violation_score = float(response_json.get("violation", 0.0) or 0.0)
        violated_rules = response_json.get("violated_rule_descriptions", [])
        mutation_detected = response_json.get("mutation")
        ipi_detected = response_json.get("ipi")

        flagged = violation_score >= self.violation_threshold
        if not flagged:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: request passed (score=%s, rules=%s)",
                violation_score,
                violated_rules,
            )
            return

        verbose_proxy_logger.warning(
            "Gray Swan Guardrail: violation score %.3f exceeds threshold %.3f",
            violation_score,
            self.violation_threshold,
        )

        if self.on_flagged_action == "block":
            # Determine if violation was in input or output
            violation_location = (
                "output"
                if hook_type == GuardrailEventHooks.post_call
                else "input"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Blocked by Gray Swan Guardrail",
                    "violation_location": violation_location,
                    "violation": violation_score,
                    "violated_rules": violated_rules,
                    "mutation": mutation_detected,
                    "ipi": ipi_detected,
                },
            )
        elif self.on_flagged_action == "monitor":
            verbose_proxy_logger.info(
                "Gray Swan Guardrail: Monitoring mode - allowing flagged content to proceed"
            )
        elif self.on_flagged_action == "passthrough":
            # Store detection info
            detection_info = {
                "guardrail": "grayswan",
                "flagged": True,
                "violation_score": violation_score,
                "violated_rules": violated_rules,
                "mutation": mutation_detected,
                "ipi": ipi_detected,
            }

            # For pre_call and during_call, raise exception to short-circuit LLM call
            if hook_type in (
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
            ):
                verbose_proxy_logger.info(
                    "Gray Swan Guardrail: Passthrough mode - raising exception to short-circuit LLM call"
                )
                violation_message = self._format_violation_message(
                    [detection_info], is_output=False
                )
                self.raise_passthrough_exception(
                    violation_message=violation_message,
                    request_data=data or {},
                    detection_info=detection_info,
                )

            # For post_call, store in metadata to replace response later
            verbose_proxy_logger.info(
                "Gray Swan Guardrail: Passthrough mode - storing detection info in metadata"
            )
            if data is not None:
                if "metadata" not in data:
                    data["metadata"] = {}
                if "guardrail_detections" not in data["metadata"]:
                    data["metadata"]["guardrail_detections"] = []
                data["metadata"]["guardrail_detections"].append(detection_info)

    def _format_violation_message(
        self, guardrail_detections: list, is_output: bool = False
    ) -> str:
        """
        Format guardrail detections into a user-friendly violation message.

        Args:
            guardrail_detections: List of detection info dictionaries
            is_output: True if violation is in model output (post_call), False if in input (pre_call/during_call)

        Returns:
            Formatted violation message string
        """
        if not guardrail_detections:
            return "Content was flagged by guardrail"

        # Get the most recent detection (should be from this guardrail)
        detection = guardrail_detections[-1]

        violation_score = detection.get("violation_score", 0.0)
        violated_rules = detection.get("violated_rules", [])
        mutation = detection.get("mutation", False)
        ipi = detection.get("ipi", False)

        # Indicate whether violation was in input or output
        violation_location = "the model response" if is_output else "input query"

        message_parts = [
            f"Sorry I can't help with that. According to the Gray Swan Cygnal Guardrail, the {violation_location} has a violation score of {violation_score:.2f}.",
        ]

        if violated_rules:
            # Format violated rules - handle both new format (dict with rule/name/description)
            # and legacy format (simple values)
            formatted_rules: List[str] = []
            for rule in violated_rules:
                if isinstance(rule, dict):
                    # New format: {'rule': 6, 'name': 'Illegal Activities...', 'description': '...'}
                    rule_num = rule.get("rule", "")
                    rule_name = rule.get("name", "")
                    rule_desc = rule.get("description", "")
                    if rule_num and rule_name:
                        if rule_desc:
                            formatted_rules.append(f"#{rule_num} {rule_name}: {rule_desc}")
                        else:
                            formatted_rules.append(f"#{rule_num} {rule_name}")
                    elif rule_name:
                        formatted_rules.append(rule_name)
                    else:
                        formatted_rules.append(str(rule))
                else:
                    # Legacy format: simple value (string or number)
                    formatted_rules.append(str(rule))

            if formatted_rules:
                message_parts.append(
                    f"It was violating the rule(s): {', '.join(formatted_rules)}."
                )

        if mutation:
            message_parts.append(
                "Mutation effort to make the harmful intention disguised was DETECTED."
            )

        if ipi:
            message_parts.append("Indirect Prompt Injection was DETECTED.")

        return "\n".join(message_parts)

    def _resolve_threshold(self, threshold: Optional[float]) -> float:
        if threshold is not None:
            return min(max(threshold, 0.0), 1.0)
        return 0.5

    def _resolve_reasoning_mode(self, candidate: Optional[str]) -> Optional[str]:
        if candidate is None:
            return None
        normalised = candidate.strip().lower()
        if normalised in self.SUPPORTED_REASONING_MODES:
            return normalised
        verbose_proxy_logger.warning(
            "Gray Swan Guardrail: ignoring unsupported reasoning_mode '%s'",
            candidate,
        )
        return None

    @staticmethod
    def get_config_model():
        from litellm.types.proxy.guardrails.guardrail_hooks.grayswan import (
            GraySwanGuardrailConfigModel,
        )

        return GraySwanGuardrailConfigModel
