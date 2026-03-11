"""
Rubrik LiteLLM Plugin for logging and tool blocking.

Provides two main features:
1. Tool call blocking: Intercepts LLM responses (streaming and non-streaming),
   validates tool calls against a Rubrik blocking service, and filters disallowed tools.
2. Request/response logging: Batches and logs LLM interactions to a Rubrik backend.

Supports both OpenAI and Anthropic response formats with fail-open error handling.

Configuration:
    Environment variables:
        RUBRIK_WEBHOOK_URL (required): Base URL for the Rubrik service
        RUBRIK_API_KEY (optional): Bearer token for authentication
        RUBRIK_BATCH_SIZE (optional): Batch size for logging (default 512)
        RUBRIK_SAMPLING_RATE (optional): Fraction of requests to sample (0.0-1.0)

    Proxy config (guardrails):
        See https://docs.litellm.ai/docs/adding_provider/simple_guardrail_tutorial

To enable verbose logging, set the environment variable:
    export LITELLM_LOG=DEBUG
"""

import asyncio
import copy
import json
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Union, cast
from urllib.parse import urlparse

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.factory import get_attribute_or_key
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    Function,
    LLMResponseTypes,
    Message,
    ModelResponse,
    ModelResponseStream,
    StandardLoggingPayload,
    StreamingChoices,
)

# Anthropic SSE event types
_EVENT_CONTENT_BLOCK_START = "content_block_start"
_EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
_EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
_EVENT_MESSAGE_DELTA = "message_delta"
_EVENT_MESSAGE_STOP = "message_stop"

# Anthropic content block types
_BLOCK_TYPE_TOOL_USE = "tool_use"
_DELTA_TYPE_INPUT_JSON = "input_json_delta"

# Endpoint URL suffixes for format detection
_ENDPOINT_OPENAI_CHAT_COMPLETIONS = "/chat/completions"
_ENDPOINT_ANTHROPIC_MESSAGES = "/messages"

# Webhook endpoint paths
_WEBHOOK_PATH_TOOL_BLOCKING = "/v1/after_completion/openai/v1"
_WEBHOOK_PATH_LOGGING_BATCH = "/v1/litellm/batch"
_CONTENT_BLOCK_EVENTS = frozenset(
    {_EVENT_CONTENT_BLOCK_START, _EVENT_CONTENT_BLOCK_DELTA, _EVENT_CONTENT_BLOCK_STOP},
)
_TERMINAL_MESSAGE_EVENTS = frozenset({_EVENT_MESSAGE_DELTA, _EVENT_MESSAGE_STOP})


@dataclass
class AnthropicToolCallData:
    """Structured data for accumulated Anthropic tool call information."""

    index: int
    id: str
    name: str
    input: Dict[str, Any] = field(default_factory=dict)
    partial_json: str = ""


class LLMResponseFormat(Enum):
    """Enum representing the format of an LLM response."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    UNKNOWN = "unknown"


class RubrikLogger(CustomGuardrail, CustomBatchLogger):
    def __init__(self, **kwargs: Any) -> None:
        self.flush_lock = asyncio.Lock()
        super().__init__(
            guardrail_name="rubrik",
            flush_lock=self.flush_lock,
            **kwargs,
        )

        verbose_logger.debug("initializing rubrik logger")

        self.sampling_rate = 1.0
        rbrk_sampling_rate = os.getenv("RUBRIK_SAMPLING_RATE")
        if rbrk_sampling_rate is not None:
            try:
                parsed = float(rbrk_sampling_rate.strip())
                if not (0.0 <= parsed <= 1.0):
                    verbose_logger.warning(
                        f"RUBRIK_SAMPLING_RATE={parsed!r} out of range [0.0, 1.0], clamping."
                    )
                    parsed = max(0.0, min(1.0, parsed))
                self.sampling_rate = parsed
            except ValueError:
                verbose_logger.warning(f"Invalid RUBRIK_SAMPLING_RATE: {rbrk_sampling_rate!r}, using 1.0")

        # Initialize helpers for format conversion
        self.anthropic_adapter = LiteLLMAnthropicMessagesAdapter()
        self.anthropic_config = AnthropicConfig()

        self.key = os.getenv("RUBRIK_API_KEY")
        _batch_size = os.getenv("RUBRIK_BATCH_SIZE", None)

        if _batch_size:
            try:
                parsed_batch = int(_batch_size)
                if parsed_batch <= 0:
                    verbose_logger.warning(f"RUBRIK_BATCH_SIZE={parsed_batch} is not positive, using default")
                else:
                    self.batch_size = parsed_batch
            except ValueError:
                verbose_logger.warning(f"Invalid RUBRIK_BATCH_SIZE: {_batch_size!r}, using default")

        _webhook_url = os.getenv("RUBRIK_WEBHOOK_URL")

        if _webhook_url is None:
            raise ValueError("environment variable RUBRIK_WEBHOOK_URL not set")

        _webhook_url = _webhook_url.rstrip("/").removesuffix("/v1")
        self.tool_blocking_endpoint = f"{_webhook_url}{_WEBHOOK_PATH_TOOL_BLOCKING}"
        self.logging_endpoint = f"{_webhook_url}{_WEBHOOK_PATH_LOGGING_BATCH}"

        # Cache the httpx client for logging (uses LiteLLM's shared client)
        self.async_httpx_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)

        # Create a dedicated httpx client for tool blocking to avoid connection pooling issues
        # with LiteLLM's shared client
        self.tool_blocking_client = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        try:
            asyncio.create_task(self.periodic_flush())
        except RuntimeError:
            verbose_logger.debug("No running event loop for periodic flush - will start when proxy runs")

    async def async_send_batch(self, *args: Any, **kwargs: Any) -> None:
        """Send the current log queue to Rubrik. Called by CustomBatchLogger.flush_queue."""
        await self._log_batch_to_rubrik(data=list(self.log_queue))

    async def aclose(self) -> None:
        """Close the dedicated tool-blocking HTTP client."""
        await self.tool_blocking_client.aclose()

    async def async_log_success_event(
        self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        try:
            standard_logging_payload: StandardLoggingPayload = kwargs["standard_logging_object"]
            random_sample = random.random()
            if random_sample > self.sampling_rate:
                verbose_logger.debug(f"Skipping Rubrik logging (sampling_rate={self.sampling_rate})")
                return
            # If the request is an anthropic request, the system prompt _might_ be in kwargs["system"]
            if "system" in kwargs:
                system_prompt_msg_list = kwargs["system"]
                try:
                    if system_prompt_msg_list:
                        system_scaffold = {"role": "system", "content": system_prompt_msg_list}
                        if isinstance(standard_logging_payload["messages"], list):
                            standard_logging_payload["messages"].insert(0, system_scaffold)  # type: ignore[union-attr]
                        elif isinstance(standard_logging_payload["messages"], (dict, str)):
                            standard_logging_payload["messages"] = [
                                system_scaffold,
                                standard_logging_payload["messages"],
                            ]
                except Exception as e:
                    verbose_logger.debug(f"Error adding system prompt to messages: {e}")

            self.log_queue.append(standard_logging_payload)

            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()
        except Exception as e:
            verbose_logger.error(
                f"Rubrik logging hook failed: {e}. Skipping logging for this event.",
                exc_info=True,
            )

    async def _log_batch_to_rubrik(self, data: Any) -> None:
        try:
            headers: Dict[str, str] = {"Content-Type": "application/json"}

            if self.key:
                headers["Authorization"] = f"Bearer {self.key}"

            response = await self.async_httpx_client.post(
                url=self.logging_endpoint,
                json=data,
                headers=headers,
            )

            if response.status_code >= 300:
                verbose_logger.error(f"Rubrik Error: {response.status_code} - {response.text}")
                response.raise_for_status()

        except httpx.HTTPStatusError as e:
            verbose_logger.exception(f"Rubrik HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception:
            verbose_logger.exception("Rubrik Layer Error")

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        """
        Hook called after successful LLM API call (non-streaming).

        Detects response format (OpenAI/Anthropic), sends to tool blocking service,
        and returns modified response with blocked tools removed.
        Falls back to returning original response on any error (fail-open).
        """
        try:
            response_format = RubrikLogger._detect_llm_response_format(data)

            if response_format == LLMResponseFormat.UNKNOWN:
                verbose_logger.warning(
                    f"Received response in unknown format (neither OpenAI nor Anthropic). "
                    f"Returning response untouched. Response type: {type(response)}",
                )
                return response

            if response_format == LLMResponseFormat.ANTHROPIC:
                openai_dict = self._anthropic_response_to_openai_dict(cast(Dict[str, Any], response))
            else:
                openai_dict = cast(ModelResponse, response).model_dump()

            # Skip blocking service call if response has no tool calls
            choices = openai_dict.get("choices", [])
            has_tool_calls = any(choice.get("message", {}).get("tool_calls") for choice in choices)
            if not has_tool_calls:
                return response

            modified_openai_dict = await self._check_and_modify_response(openai_dict)

            if response_format == LLMResponseFormat.ANTHROPIC:
                self._openai_dict_to_anthropic_response(modified_openai_dict, cast(Dict[str, Any], response))
            else:
                for key, value in modified_openai_dict.items():
                    if hasattr(response, key):
                        setattr(response, key, value)

            return response
        except Exception as e:
            verbose_logger.error(
                f"Tool blocking hook failed: {e}. Returning original response unchanged.",
                exc_info=True,
            )
            return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: AsyncGenerator,
        request_data: Dict[str, Any],
    ) -> AsyncGenerator[Union[ModelResponseStream, bytes], None]:
        """
        Intercept streaming responses to block tool calls in real-time.

        Accumulates tool call deltas as they stream, validates them with the blocking
        service, then yields filtered results with blocked tools removed.
        Falls back to passing through buffered + remaining chunks on error (fail-open).
        """
        response_format = self._detect_llm_response_format(request_data)

        handlers: Dict[LLMResponseFormat, Any] = {
            LLMResponseFormat.OPENAI: self._handle_openai_streaming,
            LLMResponseFormat.ANTHROPIC: self._handle_anthropic_streaming,
        }

        handler = handlers.get(response_format)
        if handler is None:
            verbose_logger.info("Streaming response for non-OpenAI/Anthropic endpoint - passing through")
            async for chunk in response:
                yield chunk
            return

        buffered_chunks: List[Union[ModelResponseStream, bytes]] = []

        async def buffering_generator() -> AsyncGenerator[Union[ModelResponseStream, bytes], None]:
            """Wrapper that buffers chunks as they're consumed."""
            async for chunk in response:
                buffered_chunks.append(chunk)
                yield chunk

        try:
            async for chunk in handler(buffering_generator()):
                buffered_chunks.clear()
                yield chunk
        except Exception as e:
            verbose_logger.error(
                f"Streaming tool blocking failed: {e}. Passing through buffered and remaining chunks.",
                exc_info=True,
            )
            for buffered_chunk in buffered_chunks:
                yield buffered_chunk
            async for chunk in response:
                yield chunk

    @staticmethod
    def _detect_llm_response_format(request_data: Dict[str, Any]) -> LLMResponseFormat:
        """Detect the LLM response format (OpenAI/Anthropic) from the proxied endpoint URL."""
        proxy_request = request_data.get("proxy_server_request", {})
        url = proxy_request.get("url", "")
        path = urlparse(url).path

        if path.endswith(_ENDPOINT_OPENAI_CHAT_COMPLETIONS):
            return LLMResponseFormat.OPENAI
        if path.endswith(_ENDPOINT_ANTHROPIC_MESSAGES):
            return LLMResponseFormat.ANTHROPIC
        return LLMResponseFormat.UNKNOWN

    async def _handle_openai_streaming(self, response: Any) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Process OpenAI streaming responses, filtering blocked tool calls.

        Accumulates tool call deltas until finish_reason is received, then validates
        all tools with the blocking service and yields a synthetic response.
        """
        accumulated_tool_calls: Dict[int, ChatCompletionDeltaToolCall] = {}
        chunk_template: Optional[ModelResponseStream] = None

        async for chunk in response:
            if not chunk.choices:
                yield chunk
                continue

            if chunk_template is None:
                chunk_template = copy.deepcopy(chunk)

            choice: StreamingChoices = cast(StreamingChoices, chunk.choices[0])
            has_tool_calls = bool(choice.delta and choice.delta.tool_calls)
            is_finished = bool(choice.finish_reason)

            # 1) Accumulate tool call fragments.
            #    Some models (e.g. GPT-5) repeat the complete arguments in the
            #    finish chunk — skip accumulation when we already have data to
            #    avoid doubling, but still accumulate if this is the first chunk.
            if has_tool_calls and not (is_finished and accumulated_tool_calls):
                for delta in choice.delta.tool_calls or []:
                    self._accumulate_openai_tool_call_delta(
                        cast(ChatCompletionDeltaToolCall, cast(object, delta)),
                        accumulated_tool_calls,
                    )

            # 2) Pass through chunks that aren't part of a tool call sequence.
            if not accumulated_tool_calls:
                yield chunk
                continue

            # 3) Once all tool calls are complete, validate and emit.
            if is_finished and chunk_template is not None:
                yield await self._create_openai_allowed_tools_chunk(chunk_template, accumulated_tool_calls)
                return

        # Stream ended without a finish chunk — yield accumulated calls unblocked (fail-open)
        if accumulated_tool_calls and chunk_template is not None:
            yield await self._create_openai_allowed_tools_chunk(chunk_template, accumulated_tool_calls)

    async def _handle_anthropic_streaming(self, response: Any) -> AsyncGenerator[bytes, None]:
        """
        Process Anthropic streaming responses, filtering blocked tool calls.

        Strategy:
            1. Pass through chunks until the first tool call appears
            2. Buffer all subsequent chunks once a tool call is detected
            3. On message_delta (stream end), validate tools with blocking service
            4. Emit buffered chunks excluding blocked tools (with sequential reindexing)
            5. Append explanation text block if any tools were blocked
            6. Emit final message_delta and message_stop
        """
        parsed_stream = self._parse_anthropic_sse_stream(response)
        accumulated_tools: Dict[str, AnthropicToolCallData] = {}
        buffered_chunks: List[Dict[str, Any]] = []
        is_buffering = False
        current_content_index = -1

        async for chunk in parsed_stream:
            event_type = chunk.get("type")
            is_tool_chunk = self._is_tool_related_anthropic_chunk(chunk)

            # Track content block index for non-tool chunks before buffering starts
            if not is_buffering and not is_tool_chunk and event_type in _CONTENT_BLOCK_EVENTS:
                current_content_index = chunk.get("index", current_content_index)

            if is_tool_chunk:
                is_buffering = True
                self._accumulate_anthropic_tool_call(chunk, accumulated_tools)

            # Pass through non-tool chunks before any tools appear
            if not is_buffering:
                yield self._encode_anthropic_chunk_to_sse(chunk)
                continue

            buffered_chunks.append(chunk)

            # Continue buffering until stream ends
            if event_type != _EVENT_MESSAGE_DELTA:
                continue

            # Stream complete - validate tools and emit filtered results
            blocked_indices, explanation = await self._get_blocked_anthropic_tool_calls(accumulated_tools)

            # Emit allowed chunks with sequential content block reindexing
            for buffered_chunk in buffered_chunks:
                if not self._should_yield_anthropic_chunk(buffered_chunk, blocked_indices):
                    continue

                if buffered_chunk.get("type") in _CONTENT_BLOCK_EVENTS:
                    if buffered_chunk.get("type") == _EVENT_CONTENT_BLOCK_START:
                        current_content_index += 1
                    buffered_chunk["index"] = current_content_index
                yield self._encode_anthropic_chunk_to_sse(buffered_chunk)

            # Add explanation text block if any tools were blocked
            if explanation:
                for explanation_chunk in self._generate_anthropic_text_block(explanation, current_content_index + 1):
                    yield self._encode_anthropic_chunk_to_sse(explanation_chunk)

            # Adjust stop_reason if all tools were blocked
            if accumulated_tools and len(blocked_indices) == len(accumulated_tools):
                chunk["delta"]["stop_reason"] = "end_turn"
            yield self._encode_anthropic_chunk_to_sse(chunk)

            # Drain remaining stream events (e.g., message_stop)
            async for remaining_chunk in parsed_stream:
                yield self._encode_anthropic_chunk_to_sse(remaining_chunk)

            # Normal path completed — clear so post-loop guard doesn't re-emit
            buffered_chunks.clear()

        # Stream ended without message_delta — fail-open: emit buffered chunks unmodified
        if is_buffering and buffered_chunks:
            for buffered_chunk in buffered_chunks:
                if buffered_chunk.get("type") not in _TERMINAL_MESSAGE_EVENTS:
                    yield self._encode_anthropic_chunk_to_sse(buffered_chunk)

    @staticmethod
    def _accumulate_openai_tool_call_delta(
        delta: ChatCompletionDeltaToolCall,
        accumulated_tool_calls: Dict[int, ChatCompletionDeltaToolCall],
    ) -> None:
        """
        Accumulate an OpenAI tool call delta into the state dictionary.

        The first delta for each index contains the header (id, type, function name).
        Subsequent deltas append to the function arguments.
        """
        delta_index = delta.index

        if delta_index not in accumulated_tool_calls:
            accumulated_tool_calls[delta_index] = delta.model_copy(deep=True)
            return

        if not delta.function:
            verbose_logger.warning(f"Tool call delta missing function field: {delta}")
            return

        existing = accumulated_tool_calls[delta_index]
        if existing.function and existing.function.arguments is None:
            existing.function.arguments = ""
        existing.function.arguments += delta.function.arguments or ""

    @staticmethod
    def _is_tool_related_anthropic_chunk(chunk: Dict[str, Any]) -> bool:
        """Check if an Anthropic SSE chunk belongs to a tool_use content block."""
        event_type = chunk.get("type")

        if event_type == _EVENT_CONTENT_BLOCK_START:
            return bool(chunk.get("content_block", {}).get("type") == _BLOCK_TYPE_TOOL_USE)

        if event_type == _EVENT_CONTENT_BLOCK_DELTA:
            return bool(chunk.get("delta", {}).get("type") == _DELTA_TYPE_INPUT_JSON)

        return False

    @staticmethod
    def _should_yield_anthropic_chunk(chunk: Dict[str, Any], blocked_content_indices: Set[int]) -> bool:
        """Determine whether a buffered Anthropic chunk should be emitted to the client."""
        event_type = chunk.get("type")

        if event_type in _TERMINAL_MESSAGE_EVENTS:
            return False

        if event_type in _CONTENT_BLOCK_EVENTS:
            return chunk.get("index") not in blocked_content_indices

        return True

    async def _get_blocked_anthropic_tool_calls(
        self,
        accumulated_tools: Dict[str, AnthropicToolCallData],
    ) -> Tuple[Set[int], Optional[str]]:
        """
        Validate Anthropic tool calls with the blocking service.

        Returns:
            (blocked_content_indices, explanation)
        """
        if not accumulated_tools:
            return set(), None

        openai_format_tools = self._convert_anthropic_tools_to_openai_format(accumulated_tools)
        allowed_tools, explanation = await self._get_allowed_tool_calls(
            {tc.index: tc for tc in openai_format_tools},
        )

        allowed_tool_ids = {tc.id for tc in allowed_tools}
        blocked_content_indices = {
            tool_data.index for tool_id, tool_data in accumulated_tools.items() if tool_id not in allowed_tool_ids
        }

        return blocked_content_indices, explanation

    @staticmethod
    def _accumulate_anthropic_tool_call(
        chunk: Dict[str, Any],
        tool_calls: Dict[str, AnthropicToolCallData],
    ) -> None:
        """Accumulate tool_use data from an Anthropic streaming chunk."""
        event_type = chunk.get("type")

        if event_type == _EVENT_CONTENT_BLOCK_START:
            RubrikLogger._handle_anthropic_tool_start(chunk, tool_calls)
        elif event_type == _EVENT_CONTENT_BLOCK_DELTA:
            RubrikLogger._handle_anthropic_tool_delta(chunk, tool_calls)

    @staticmethod
    def _handle_anthropic_tool_start(
        chunk: Dict[str, Any],
        tool_calls: Dict[str, AnthropicToolCallData],
    ) -> None:
        """Create a new tool entry from a content_block_start event."""
        content_block = chunk.get("content_block", {})
        if content_block.get("type") != _BLOCK_TYPE_TOOL_USE:
            return

        tool_id = content_block.get("id")
        if tool_id is None:
            verbose_logger.warning("Anthropic tool block missing 'id' field — skipping accumulation")
            return
        tool_calls[tool_id] = AnthropicToolCallData(
            index=chunk.get("index", 0),
            id=tool_id,
            name=content_block.get("name"),
            input=content_block.get("input", {}),
            partial_json="",
        )

    @staticmethod
    def _handle_anthropic_tool_delta(
        chunk: Dict[str, Any],
        tool_calls: Dict[str, AnthropicToolCallData],
    ) -> None:
        """Append a JSON fragment to the matching tool entry by index."""
        delta = chunk.get("delta", {})
        if delta.get("type") != _DELTA_TYPE_INPUT_JSON:
            return

        chunk_index = chunk.get("index", 0)
        json_fragment = delta.get("partial_json", "")

        for tool_data in tool_calls.values():
            if tool_data.index == chunk_index:
                tool_data.partial_json += json_fragment
                break

    async def _get_allowed_tool_calls(
        self,
        tool_calls_by_index: Dict[int, ChatCompletionDeltaToolCall],
    ) -> Tuple[List[ChatCompletionDeltaToolCall], Optional[str]]:
        """
        Validate tool calls with the blocking service and return allowed ones.

        Returns:
            (allowed_tools, explanation) - explanation is set only if some tools were blocked.
        """
        all_tool_calls = list(tool_calls_by_index.values())
        service_response = await self._make_tool_blocking_request(all_tool_calls)

        choices = service_response.get("choices", [])
        if not choices:
            verbose_logger.warning(
                "Tool blocking service returned empty choices — allowing all tools (fail-open)"
            )
            return all_tool_calls, None

        message = choices[0].get("message", {})
        returned_tool_calls = message.get("tool_calls", [])
        blocking_explanation = message.get("content", "")

        allowed_ids = {tc["id"] for tc in returned_tool_calls if tc.get("id")}
        allowed_tools = [tc for tc in all_tool_calls if tc.id in allowed_ids]

        if len(allowed_tools) == len(all_tool_calls):
            return allowed_tools, None

        return allowed_tools, f"\n\n{blocking_explanation}" if blocking_explanation else None

    async def _create_openai_allowed_tools_chunk(
        self,
        chunk_template: ModelResponseStream,
        tool_calls_by_index: Dict[int, ChatCompletionDeltaToolCall],
    ) -> ModelResponseStream:
        """Create a synthetic OpenAI chunk containing only allowed tool calls."""
        allowed_tools, explanation = await self._get_allowed_tool_calls(tool_calls_by_index)
        choice: StreamingChoices = cast(StreamingChoices, chunk_template.choices[0])

        # Re-index tools sequentially so clients see contiguous indices
        for new_index, tool in enumerate(allowed_tools):
            tool.index = new_index

        choice.delta = Delta(
            content=explanation,
            tool_calls=allowed_tools if allowed_tools else None,
        )
        choice.finish_reason = "tool_calls" if allowed_tools else "stop"

        return chunk_template

    @staticmethod
    def _convert_anthropic_tools_to_openai_format(
        tool_calls: Dict[str, AnthropicToolCallData],
    ) -> List[ChatCompletionDeltaToolCall]:
        """Convert accumulated Anthropic tool calls to OpenAI format for blocking service."""
        return [
            ChatCompletionDeltaToolCall(
                index=tool_data.index,
                id=tool_id,
                type="function",
                function=Function(name=tool_data.name, arguments=tool_data.partial_json),
            )
            for tool_id, tool_data in tool_calls.items()
        ]

    @staticmethod
    def _generate_anthropic_text_block(text: str, index: int) -> List[Dict[str, Any]]:
        """Generate Anthropic SSE events for a synthetic text content block."""
        return [
            {"type": _EVENT_CONTENT_BLOCK_START, "index": index, "content_block": {"type": "text", "text": ""}},
            {"type": _EVENT_CONTENT_BLOCK_DELTA, "index": index, "delta": {"type": "text_delta", "text": text}},
            {"type": _EVENT_CONTENT_BLOCK_STOP, "index": index},
        ]

    @staticmethod
    async def _parse_anthropic_sse_stream(response: Any) -> AsyncGenerator[Dict[str, Any], None]:
        """Parse raw Anthropic SSE bytes into decoded dict chunks."""
        async for raw_chunk in response:
            for decoded_chunk in RubrikLogger._decode_all_anthropic_sse_events(raw_chunk):
                yield decoded_chunk

    @staticmethod
    def _encode_anthropic_chunk_to_sse(chunk_dict: Dict[str, Any]) -> bytes:
        """Encode an Anthropic dict chunk back to SSE byte format."""
        json_str = json.dumps(chunk_dict, separators=(",", ":"))
        return f"event: {chunk_dict.get('type', '')}\ndata: {json_str}\n\n".encode()

    @staticmethod
    def _decode_all_anthropic_sse_events(raw_chunk: bytes) -> List[Dict[str, Any]]:
        """Decode all Anthropic SSE events from a raw chunk."""
        data_prefix = "data:"
        events: List[Dict[str, Any]] = []

        try:
            text = raw_chunk.decode("utf-8")
        except UnicodeDecodeError:
            verbose_logger.error("Rubrik: Failed to decode SSE chunk as UTF-8, skipping")
            return events

        for line in text.split("\n"):
            stripped_line = line.strip()
            if stripped_line.startswith(data_prefix):
                json_payload = stripped_line[len(data_prefix):].strip()
                try:
                    events.append(json.loads(json_payload))
                except json.JSONDecodeError:
                    verbose_logger.error("Rubrik: Malformed JSON in SSE event: %s", json_payload[:200])

        return events

    async def _check_and_modify_response(
        self,
        response_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send a response to the tool blocking service and return the modified version."""
        headers = self._build_tool_blocking_headers()

        verbose_logger.debug(f"Sending request to tool blocking service: {self.tool_blocking_endpoint}")
        http_response = await self.tool_blocking_client.post(
            self.tool_blocking_endpoint,
            json=response_dict,
            headers=headers,
            timeout=5.0,
        )
        http_response.raise_for_status()

        modified_response: Dict[str, Any] = http_response.json()
        verbose_logger.debug("Received modified response from tool blocking service")
        return modified_response

    async def _make_tool_blocking_request(
        self,
        delta_tool_calls: List[ChatCompletionDeltaToolCall],
    ) -> Dict[str, Any]:
        """Send accumulated tool calls to the blocking service for validation."""
        message_tool_calls = self._convert_deltas_to_message_tool_calls(delta_tool_calls)
        request_payload = ModelResponse(
            choices=[
                Choices(message=Message(role="assistant", content=None, tool_calls=message_tool_calls)),
            ],
        )

        headers = self._build_tool_blocking_headers()
        response = await self.tool_blocking_client.post(
            self.tool_blocking_endpoint,
            json=request_payload.model_dump(exclude_none=True),
            headers=headers,
            timeout=2.0,
        )
        response.raise_for_status()
        response_dict: Dict[str, Any] = response.json()
        return response_dict

    @staticmethod
    def _convert_deltas_to_message_tool_calls(
        delta_tool_calls: List[ChatCompletionDeltaToolCall],
    ) -> List[ChatCompletionMessageToolCall]:
        """Convert streaming delta tool calls to message format for the blocking service."""
        return [
            ChatCompletionMessageToolCall(id=tc.id, type=tc.type or "function", function=tc.function)
            for tc in delta_tool_calls
        ]

    def _build_tool_blocking_headers(self) -> Dict[str, str]:
        """Build HTTP headers for tool blocking service requests."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        return headers

    def _anthropic_response_to_openai_dict(self, response: Any) -> Dict[str, Any]:
        """Convert raw Anthropic /v1/messages response to OpenAI format for the blocking service."""
        anthropic_completion = {
            "content": response["content"],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

        (
            text_content,
            _citations,
            _thinking_blocks,
            _reasoning_content,
            tool_calls,
            _web_search_results,
            _tool_results,
            _compaction_blocks,
        ) = self.anthropic_config.extract_response_content(completion_response=anthropic_completion)

        message: Dict[str, Any] = {"role": "assistant", "content": text_content}
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": response.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.get("model", ""),
            "choices": [{"index": 0, "message": message, "finish_reason": response.get("stop_reason", "stop")}],
            "usage": response.get("usage", {}),
        }

    def _openai_dict_to_anthropic_response(
        self,
        openai_dict: Dict[str, Any],
        original_response: Any,
    ) -> None:
        """Convert OpenAI format dict back to Anthropic format, updating the original in-place.

        Preserves non-tool content blocks (thinking, citations, etc.) from the original
        response and only replaces tool_use blocks based on the blocking service result.
        """
        openai_dict.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        anthropic_response = self.anthropic_adapter.translate_openai_response_to_anthropic(
            response=ModelResponse(**openai_dict),
        )

        # Extract the allowed tool IDs from the translated response
        translated_content = anthropic_response.get("content") or []
        allowed_tool_ids = {
            get_attribute_or_key(block, "id")
            for block in translated_content
            if get_attribute_or_key(block, "type") == "tool_use" and get_attribute_or_key(block, "id") is not None
        }

        # Preserve non-tool blocks from original, filter tool blocks to allowed only
        original_content = original_response.get("content") or []
        filtered_content = []
        for block in original_content:
            if get_attribute_or_key(block, "type") == "tool_use":
                if get_attribute_or_key(block, "id") in allowed_tool_ids:
                    filtered_content.append(block)
            else:
                filtered_content.append(block)

        # Add explanation text block if the blocking service added one
        existing_texts = {
            get_attribute_or_key(b, "text", "")
            for b in original_content
            if get_attribute_or_key(b, "type") == "text"
        }
        for block in translated_content:
            if get_attribute_or_key(block, "type") == "text":
                text = get_attribute_or_key(block, "text", "")
                if text and text not in existing_texts:
                    filtered_content.append(block)

        original_response["content"] = filtered_content
        original_response["stop_reason"] = anthropic_response.get("stop_reason", "end_turn")


# Module-level handler instance for use with litellm_settings.callbacks
try:
    rubrik_handler = RubrikLogger()
except (ValueError, RuntimeError) as e:
    verbose_logger.warning(
        f"Rubrik handler not initialised ({e}). "
        "Set RUBRIK_WEBHOOK_URL to enable the plugin."
    )
    rubrik_handler = None  # type: ignore[assignment]
