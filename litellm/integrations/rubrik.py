"""
Rubrik LiteLLM Plugin for logging and tool blocking.

To enable verbose logging, set the environment variable:
    export LITELLM_LOG=DEBUG

Or in Python before importing litellm:
    import os
    os.environ["LITELLM_LOG"] = "DEBUG"
"""

import asyncio
import json
import os
import random
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, cast

import httpx
from litellm._logging import verbose_logger
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
_BLOCK_TYPE_TEXT = "text"
_BLOCK_TYPE_TOOL_USE = "tool_use"
_DELTA_TYPE_INPUT_JSON = "input_json_delta"

# Endpoint URL suffixes for format detection
_ENDPOINT_OPENAI_CHAT_COMPLETIONS = "/chat/completions"
_ENDPOINT_ANTHROPIC_MESSAGES = "/v1/messages"

# Webhook endpoint paths
_WEBHOOK_PATH_TOOL_BLOCKING = "/v1/after_completion/openai/v1"
_WEBHOOK_PATH_LOGGING_BATCH = "/v1/litellm/batch"

_CONTENT_BLOCK_EVENTS = frozenset(
    {_EVENT_CONTENT_BLOCK_START, _EVENT_CONTENT_BLOCK_DELTA, _EVENT_CONTENT_BLOCK_STOP},
)
_TERMINAL_MESSAGE_EVENTS = frozenset({_EVENT_MESSAGE_DELTA, _EVENT_MESSAGE_STOP})

# Content block types replaced during Anthropic non-streaming round-trip
_REPLACED_BLOCK_TYPES = frozenset({_BLOCK_TYPE_TOOL_USE, _BLOCK_TYPE_TEXT})


@dataclass
class AnthropicToolCallData:
    """Structured data for accumulated Anthropic tool call information."""

    index: int
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    partial_json: str = ""


@dataclass
class BlockedToolsResult:
    """Returned by _get_allowed_tool_calls when at least one tool was blocked."""

    allowed_tools: list
    explanation: str


class LLMResponseFormat(Enum):
    """Enum representing the format of an LLM response."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    UNKNOWN = "unknown"


class RubrikLogger(CustomGuardrail, CustomBatchLogger):
    def __init__(self, api_key: str | None = None, api_base: str | None = None, **kwargs):
        _flush_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=_flush_lock)

        verbose_logger.debug("initializing rubrik logger")

        self.sampling_rate = 1.0
        rbrk_sampling_rate = os.getenv("RUBRIK_SAMPLING_RATE")
        if rbrk_sampling_rate is not None:
            try:
                parsed_rate = float(rbrk_sampling_rate.strip())
                if parsed_rate < 0.0 or parsed_rate > 1.0:
                    verbose_logger.warning(
                        f"RUBRIK_SAMPLING_RATE={parsed_rate} out of range [0.0, 1.0], clamping"
                    )
                    parsed_rate = max(0.0, min(1.0, parsed_rate))
                self.sampling_rate = parsed_rate
            except ValueError:
                verbose_logger.warning(f"Invalid RUBRIK_SAMPLING_RATE: {rbrk_sampling_rate!r}, using 1.0")

        # Initialize helpers for format conversion
        self.anthropic_adapter = LiteLLMAnthropicMessagesAdapter()
        self.anthropic_config = AnthropicConfig()

        self.key = api_key or os.getenv("RUBRIK_API_KEY")
        _batch_size = os.getenv("RUBRIK_BATCH_SIZE")

        if _batch_size:
            # Batch size has a default of 512
            # Queue will be flushed when the queue reaches this size or when
            # the periodic interval is triggered (every 5 seconds by default)
            try:
                parsed_batch_size = int(_batch_size)
                if parsed_batch_size > 0:
                    self.batch_size = parsed_batch_size
                else:
                    verbose_logger.warning(f"RUBRIK_BATCH_SIZE={parsed_batch_size} must be positive, using default")
            except ValueError:
                verbose_logger.warning(f"Invalid RUBRIK_BATCH_SIZE: {_batch_size!r}, using default")

        _webhook_url = api_base or os.getenv("RUBRIK_WEBHOOK_URL")

        if _webhook_url is None:
            raise ValueError("Rubrik webhook URL not configured")

        _webhook_url = _webhook_url.rstrip("/").removesuffix("/v1")
        self.tool_blocking_endpoint = f"{_webhook_url}{_WEBHOOK_PATH_TOOL_BLOCKING}"
        self.logging_endpoint = f"{_webhook_url}{_WEBHOOK_PATH_LOGGING_BATCH}"

        # Cache the httpx client for logging (uses LiteLLM's shared client)
        self.async_httpx_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)

        # Create a dedicated httpx client for tool blocking to avoid connection pooling issues
        # with LiteLLM's shared client
        self.tool_blocking_client = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),  # 2s connect timeout, 5s total timeout
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        self.log_queue = []
        asyncio.create_task(self.periodic_flush())

    async def aclose(self):
        """Close the dedicated httpx client for tool blocking."""
        await self.tool_blocking_client.aclose()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            standard_logging_payload: StandardLoggingPayload = kwargs["standard_logging_object"]
            random_sample = random.random()
            if random_sample > self.sampling_rate:
                verbose_logger.debug(f"Skipping Rubrik logging (sampling_rate={self.sampling_rate})")
                return
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

    async def _log_batch_to_rubrik(self, data):
        try:
            headers = self._build_headers()

            response = await self.async_httpx_client.post(
                url=self.logging_endpoint,
                json=data,
                headers=headers,
            )

            # In practice, this is almost never going to get called as the client.post will
            # usually raise an error
            if response.status_code >= 300:
                verbose_logger.error(f"Rubrik Error: {response.status_code} - {response.text}")
                response.raise_for_status()

        except httpx.HTTPStatusError as e:
            verbose_logger.exception(f"Rubrik HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception:
            verbose_logger.exception("Rubrik Layer Error")

    async def async_send_batch(self):
        """Handles sending batches of responses to Rubrik."""
        if not self.log_queue:
            return

        await self._log_batch_to_rubrik(
            data=list(self.log_queue),
        )


    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        """
        Hook called after successful LLM API call (non-streaming).

        This hook:
        1. Detects whether the response is in Anthropic or OpenAI format
        2. Converts Anthropic format to OpenAI format if needed
        3. Sends the OpenAI-formatted response to the tool blocking service
        4. Receives the modified response with blocked tools removed
        5. Converts back to Anthropic format if the original was Anthropic
        6. Returns the modified response

        If the tool blocking service is unavailable or returns an error, the original
        response is returned unchanged (fail-open behavior).

        Args:
            data: Request data dictionary
            user_api_key_dict: User API key authentication data
            response: LLM response object

        Returns:
            Modified response object with blocked tools removed, or original response on error
        """
        try:
            response_format = RubrikLogger._detect_llm_response_format(data)

            # If unknown format, return untouched
            if response_format == LLMResponseFormat.UNKNOWN:
                verbose_logger.warning(
                    f"Received response in unknown format (neither OpenAI nor Anthropic). "
                    f"Returning response untouched. Response type: {type(response)}",
                )
                return response

            if response_format == LLMResponseFormat.ANTHROPIC:
                # Skip blocking service if the response has no tool calls (text-only)
                content = response.get("content", []) if isinstance(response, dict) else []
                has_tools = any(
                    isinstance(b, dict) and b.get("type") == _BLOCK_TYPE_TOOL_USE
                    for b in content
                )
                if not has_tools:
                    return response
                return await self._handle_anthropic_non_streaming(response)

            openai_dict = response.model_dump()

            # Skip blocking service if the response has no tool calls (text-only)
            choices = openai_dict.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                if not message.get("tool_calls"):
                    return response

            modified_openai_dict = await self._post_to_tool_blocking_service(openai_dict)
            # Update the OpenAI Pydantic model fields with the modified dict entries
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
        request_data: dict[str, Any],
    ) -> AsyncGenerator[ModelResponseStream | bytes, None]:
        """
        Intercept streaming responses to block tool calls in real-time.

        Accumulates tool call deltas as they stream, validates them with the blocking
        service, then yields filtered results with blocked tools removed.

        If the tool blocking service is unavailable or returns an error, the original
        stream is passed through unchanged (fail-open behavior). Any chunks consumed
        before the error are yielded before passing through the remaining stream.
        """
        response_format = self._detect_llm_response_format(request_data)

        if response_format == LLMResponseFormat.OPENAI:
            handler = self._handle_openai_streaming
        elif response_format == LLMResponseFormat.ANTHROPIC:
            handler = self._handle_anthropic_streaming
        else:
            verbose_logger.info("Streaming response for non-OpenAI/Anthropic endpoint - passing through")
            async for chunk in response:
                yield chunk
            return

        # Buffer chunks as they're consumed from the original stream so we can yield them if an error occurs
        buffered_chunks: list[ModelResponseStream | bytes] = []

        async def buffering_generator():
            """Wrapper that buffers chunks as they're consumed."""
            async for item in response:
                buffered_chunks.append(item)
                yield item

        try:
            async for chunk in handler(buffering_generator()):
                yield chunk
            # Handler completed successfully — clear the buffer
            buffered_chunks.clear()
        except Exception as e:
            verbose_logger.error(
                f"Streaming tool blocking failed: {e}. Passing through buffered and remaining chunks.",
                exc_info=True,
            )
            # Yield any buffered chunks that were consumed but not yet yielded
            for buffered_chunk in buffered_chunks:
                yield buffered_chunk

            # Pass through the remaining chunks from the original stream
            async for chunk in response:
                yield chunk

    @staticmethod
    def _detect_llm_response_format(request_data: dict[str, Any]) -> LLMResponseFormat:
        """Detect the LLM response format (OpenAI/Anthropic) from the proxied endpoint URL.

        Query parameters (e.g. ?api-version=2024-10-21 sent by the Azure OpenAI SDK) are
        stripped before the endpoint suffix check to avoid a false UNKNOWN classification.
        """
        proxy_request = request_data.get("proxy_server_request", {})
        url = proxy_request.get("url", "")
        url_path = urllib.parse.urlparse(url).path

        if url_path.endswith(_ENDPOINT_OPENAI_CHAT_COMPLETIONS):
            return LLMResponseFormat.OPENAI
        if url_path.endswith(_ENDPOINT_ANTHROPIC_MESSAGES):
            return LLMResponseFormat.ANTHROPIC
        return LLMResponseFormat.UNKNOWN

    async def _handle_openai_streaming(self, response: Any) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Process OpenAI streaming responses, filtering blocked tool calls.

        Strategy:
            1. Pass through non-tool chunks immediately.
            2. Once tool calls appear, buffer ALL subsequent chunks (tool deltas,
               any interleaved non-tool content, and the finish chunk).
            3. On finish_reason, validate with the blocking service.
            4. Replay buffered chunks:
               - All tools allowed: yield buffered chunks unchanged.
               - Some tools blocked: yield only allowed tool call chunks (re-indexed
                 sequentially) and any non-tool content chunks; yield an explanation
                 content chunk; yield finish chunk with finish_reason unchanged.
               - All tools blocked: yield non-tool content chunks; yield an
                 explanation chunk with finish_reason="stop".
        """
        accumulated_tool_calls: dict[int, ChatCompletionDeltaToolCall] = {}
        buffered_chunks: list[ModelResponseStream] = []

        async for chunk in response:
            if not chunk.choices:
                yield chunk
                continue

            choice: StreamingChoices = cast(StreamingChoices, chunk.choices[0])

            # Only accumulate tool calls from non-finish chunks.
            # Some models (e.g. GPT-5) repeat the full tool call in the finish chunk,
            # which would double the arguments if accumulated.
            has_tool_calls = False
            if not choice.finish_reason:
                has_tool_calls = RubrikLogger._accumulate_openai_tool_calls(choice, accumulated_tool_calls)

            if has_tool_calls:
                buffered_chunks.append(chunk)
                continue

            # No tool calls seen yet — pass through immediately
            if not buffered_chunks:
                yield chunk
                continue

            buffered_chunks.append(chunk)

            if not choice.finish_reason:
                continue

            # Stream finished — validate buffered tool calls and replay
            blocked = await self._get_allowed_tool_calls(accumulated_tool_calls)

            if not blocked:
                for buffered_chunk in buffered_chunks:
                    yield buffered_chunk
                return

            async for filtered_chunk in self._replay_filtered_tool_chunks(
                buffered_chunks,
                blocked.allowed_tools,
                blocked.explanation,
            ):
                yield filtered_chunk

    @staticmethod
    async def _replay_filtered_tool_chunks(
        buffered_chunks: list[ModelResponseStream],
        allowed_tools: list[ChatCompletionDeltaToolCall],
        explanation: str,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Replay buffered chunks with blocked tool calls removed and indices renumbered.

        Non-tool-call, non-finish chunks (e.g. interleaved content) are yielded unchanged.
        """
        allowed_indices = {tc.index for tc in allowed_tools}
        # Map original tool_call indices to dense 0-based indices so the
        # consumer sees a contiguous sequence after blocked calls are removed.
        index_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(allowed_indices))}

        for buffered_chunk in buffered_chunks:
            buffered_choice: StreamingChoices = buffered_chunk.choices[0]

            if buffered_choice.delta and buffered_choice.delta.tool_calls:
                filtered_calls = [
                    tc.model_copy(update={"index": index_remap[tc.index]})
                    for tc in buffered_choice.delta.tool_calls
                    if tc.index in allowed_indices
                ]
                if not filtered_calls:
                    continue
                buffered_choice.delta.tool_calls = filtered_calls
                yield buffered_chunk
                continue

            if not buffered_choice.finish_reason:
                # Non-tool-call, non-finish chunk - just yield
                yield buffered_chunk
                continue

            if not allowed_tools:
                # All tools blocked — explanation chunk is the terminal chunk
                yield RubrikLogger._create_openai_explanation_chunk(
                    buffered_chunk,
                    explanation,
                    finish_reason="stop",
                )
                return

            # Some tools allowed — yield explanation then original finish chunk
            if explanation:
                yield RubrikLogger._create_openai_explanation_chunk(buffered_chunk, explanation)
            yield buffered_chunk

    @staticmethod
    def _create_openai_explanation_chunk(
        template_chunk: ModelResponseStream,
        explanation: str,
        finish_reason: str | None = None,
    ) -> ModelResponseStream:
        """Create a synthetic streaming chunk containing the blocking explanation text."""
        return ModelResponseStream(
            id=template_chunk.id,
            created=template_chunk.created,
            model=template_chunk.model,
            object=template_chunk.object,
            system_fingerprint=template_chunk.system_fingerprint,
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=explanation),
                    finish_reason=finish_reason,
                ),
            ],
        )

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

        Returns:
            SSE-encoded bytes ready to be sent to the client.
        """
        parsed_stream = self._parse_anthropic_sse_stream(response)
        accumulated_tools: dict[str, AnthropicToolCallData] = {}
        index_to_tool: dict[int, AnthropicToolCallData] = {}
        buffered_chunks: list[dict[str, Any]] = []
        is_buffering = False
        replay_index_base = -1

        async for chunk in parsed_stream:
            event_type = chunk.get("type")
            is_tool_chunk = self._is_tool_related_anthropic_chunk(chunk)

            if not is_buffering and not is_tool_chunk and event_type in _CONTENT_BLOCK_EVENTS:
                replay_index_base = chunk.get("index", 0)

            if is_tool_chunk:
                is_buffering = True
                self._accumulate_anthropic_tool_call(chunk, accumulated_tools, index_to_tool)

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
                        replay_index_base += 1
                    buffered_chunk["index"] = replay_index_base
                yield self._encode_anthropic_chunk_to_sse(buffered_chunk)

            # Add explanation text block if any tools were blocked
            if explanation:
                for explanation_chunk in self._generate_anthropic_text_block(explanation, replay_index_base + 1):
                    yield self._encode_anthropic_chunk_to_sse(explanation_chunk)

            # Adjust stop_reason if all tools were blocked
            if accumulated_tools and len(blocked_indices) == len(accumulated_tools):
                chunk["delta"]["stop_reason"] = "end_turn"
            yield self._encode_anthropic_chunk_to_sse(chunk)

            # Successfully processed — clear buffer so fail-open guard doesn't re-emit
            buffered_chunks.clear()
            is_buffering = False

            # Drain remaining stream events (e.g., message_stop)
            async for remaining_chunk in parsed_stream:
                yield self._encode_anthropic_chunk_to_sse(remaining_chunk)

        # Post-loop fail-open: if we were still buffering when the stream ended
        # (e.g., no message_delta was received), emit buffered non-terminal chunks as-is
        if is_buffering and buffered_chunks:
            verbose_logger.warning("Anthropic stream ended while still buffering — emitting buffered chunks (fail-open)")
            for buffered_chunk in buffered_chunks:
                event_type = buffered_chunk.get("type")
                if event_type not in _TERMINAL_MESSAGE_EVENTS:
                    yield self._encode_anthropic_chunk_to_sse(buffered_chunk)

    @staticmethod
    def _accumulate_openai_tool_calls(
        choice: StreamingChoices,
        accumulated_tool_calls: dict[int, ChatCompletionDeltaToolCall],
    ) -> bool:
        """Accumulate tool call deltas from a streaming choice. Returns True if tool calls were present."""
        if not choice.delta or not choice.delta.tool_calls:
            return False

        for delta in choice.delta.tool_calls:
            delta_index = delta.index
            if delta_index is None:
                verbose_logger.warning(f"Tool call delta has None index, skipping: {delta}")
                continue

            # First delta for this index: store full copy as base
            if delta_index not in accumulated_tool_calls:
                accumulated_tool_calls[delta_index] = delta.model_copy()
                continue

            # Subsequent deltas: append arguments
            existing = accumulated_tool_calls[delta_index]
            if not delta.function:
                verbose_logger.warning(f"Tool call delta missing function field: {delta}")
                continue

            if existing.function is None:
                verbose_logger.warning(f"Accumulated tool call missing function field at index {delta_index}")
                continue

            existing.function.arguments = (existing.function.arguments or "") + (delta.function.arguments or "")

        return True

    @staticmethod
    def _is_tool_related_anthropic_chunk(chunk: dict[str, Any]) -> bool:
        """Check if an Anthropic SSE chunk belongs to a tool_use content block."""
        event_type = chunk.get("type")

        if event_type == _EVENT_CONTENT_BLOCK_START:
            return bool(chunk.get("content_block", {}).get("type") == _BLOCK_TYPE_TOOL_USE)

        if event_type == _EVENT_CONTENT_BLOCK_DELTA:
            return bool(chunk.get("delta", {}).get("type") == _DELTA_TYPE_INPUT_JSON)

        return False

    @staticmethod
    def _should_yield_anthropic_chunk(chunk: dict[str, Any], blocked_content_indices: set[int]) -> bool:
        """
        Determine whether a buffered Anthropic chunk should be emitted to the client.

        We yield everything but blocked tool calls and stop messages.
        """
        event_type = chunk.get("type")

        # Terminal events are handled separately after filtering
        if event_type in _TERMINAL_MESSAGE_EVENTS:
            return False

        # Suppress content blocks at blocked indices
        if event_type in _CONTENT_BLOCK_EVENTS:
            return chunk.get("index") not in blocked_content_indices

        return True

    async def _get_blocked_anthropic_tool_calls(
        self,
        accumulated_tools: dict[str, AnthropicToolCallData],
    ) -> tuple[set[int], str | None]:
        """
        Validate Anthropic tool calls with the blocking service.

        Returns:
            (blocked_content_indices, explanation) - indices to filter and explanation if tools were blocked.
        """
        if not accumulated_tools:
            return set(), None

        openai_format_tools = self._convert_anthropic_tools_to_openai_format(accumulated_tools)
        blocked = await self._get_allowed_tool_calls(
            {tc.index: tc for tc in openai_format_tools},
        )

        if not blocked:
            return set(), None

        # Compute blocked content block indices
        allowed_tool_ids = {tc.id for tc in blocked.allowed_tools}
        blocked_content_indices = {
            tool_data.index for tool_id, tool_data in accumulated_tools.items() if tool_id not in allowed_tool_ids
        }

        return blocked_content_indices, blocked.explanation

    @staticmethod
    def _accumulate_anthropic_tool_call(
        chunk: dict[str, Any],
        tool_calls: dict[str, AnthropicToolCallData],
        index_to_tool: dict[int, AnthropicToolCallData],
    ) -> None:
        """
        Accumulate tool_use data from an Anthropic streaming chunk.

        Handles content_block_start (new tool entry) and content_block_delta (JSON fragments).
        """
        event_type = chunk.get("type")

        if event_type == _EVENT_CONTENT_BLOCK_START:
            RubrikLogger._handle_anthropic_tool_start(chunk, tool_calls, index_to_tool)
        elif event_type == _EVENT_CONTENT_BLOCK_DELTA:
            RubrikLogger._handle_anthropic_tool_delta(chunk, index_to_tool)

    @staticmethod
    def _handle_anthropic_tool_start(
        chunk: dict[str, Any],
        tool_calls: dict[str, AnthropicToolCallData],
        index_to_tool: dict[int, AnthropicToolCallData],
    ) -> None:
        """Create a new tool entry from a content_block_start event."""
        content_block = chunk.get("content_block", {})
        if content_block.get("type") != _BLOCK_TYPE_TOOL_USE:
            return

        tool_id = content_block.get("id")
        if tool_id is None:
            verbose_logger.warning("Anthropic tool_use content_block_start missing 'id', skipping")
            return

        tool_data = AnthropicToolCallData(
            index=chunk.get("index", 0),
            id=tool_id,
            name=content_block.get("name"),
            input=content_block.get("input", {}),
            partial_json="",
        )
        tool_calls[tool_id] = tool_data
        index_to_tool[tool_data.index] = tool_data

    @staticmethod
    def _handle_anthropic_tool_delta(
        chunk: dict[str, Any],
        index_to_tool: dict[int, AnthropicToolCallData],
    ) -> None:
        """Append a JSON fragment to the matching tool entry by index."""
        delta = chunk.get("delta", {})
        if delta.get("type") != _DELTA_TYPE_INPUT_JSON:
            return

        chunk_index = chunk.get("index", 0)
        json_fragment = delta.get("partial_json", "")

        tool_data = index_to_tool.get(chunk_index)
        if tool_data is not None:
            tool_data.partial_json += json_fragment

    async def _get_allowed_tool_calls(
        self,
        tool_calls_by_index: dict[int, ChatCompletionDeltaToolCall],
    ) -> BlockedToolsResult | None:
        """
        Validate tool calls with the blocking service.

        Returns:
            None when all tools are allowed, otherwise a BlockedToolsResult
            containing the allowed subset and the explanation text.

        Raises:
            Exception: If the blocking service returns an empty response.
        """
        all_tool_calls = list(tool_calls_by_index.values())
        message_tool_calls = [
            ChatCompletionMessageToolCall(id=tc.id, type=tc.type or "function", function=tc.function)
            for tc in all_tool_calls
        ]
        payload = ModelResponse(
            choices=[Choices(message=Message(role="assistant", content=None, tool_calls=message_tool_calls))],
        ).model_dump(exclude_none=True)
        service_response = await self._post_to_tool_blocking_service(payload)
        return self._extract_allowed_tools(service_response, all_tool_calls)

    @staticmethod
    def _extract_allowed_tools(
        service_response: dict[str, Any],
        all_tool_calls: list[ChatCompletionDeltaToolCall],
    ) -> BlockedToolsResult | None:
        """Extract allowed tools and explanation from the blocking service response.

        Expects service_response in OpenAI chat completion format:
            {"choices": [{"message": {"tool_calls": [...], "content": "..."}}]}
        where tool_calls contains only the allowed tools and content holds the blocking explanation.
        """
        choices = service_response.get("choices", [])
        if not choices:
            verbose_logger.warning("Tool blocking service returned empty response — allowing all tools (fail-open)")
            return None

        message = choices[0].get("message", {})
        returned_tool_calls = message.get("tool_calls", [])
        blocking_explanation = message.get("content", "")

        allowed_ids = {tc["id"] for tc in returned_tool_calls if tc.get("id")}
        allowed_tools = [tc for tc in all_tool_calls if tc.id in allowed_ids]

        if len(allowed_tools) == len(all_tool_calls):
            return None

        explanation = f"\n\n{blocking_explanation}" if blocking_explanation else ""
        return BlockedToolsResult(allowed_tools=allowed_tools, explanation=explanation)

    @staticmethod
    def _convert_anthropic_tools_to_openai_format(
        tool_calls: dict[str, AnthropicToolCallData],
    ) -> list[ChatCompletionDeltaToolCall]:
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
    def _generate_anthropic_text_block(text: str, index: int) -> list[dict[str, Any]]:
        """Generate Anthropic SSE events for a synthetic text content block."""
        return [
            {"type": _EVENT_CONTENT_BLOCK_START, "index": index, "content_block": {"type": _BLOCK_TYPE_TEXT, "text": ""}},
            {"type": _EVENT_CONTENT_BLOCK_DELTA, "index": index, "delta": {"type": "text_delta", "text": text}},
            {"type": _EVENT_CONTENT_BLOCK_STOP, "index": index},
        ]

    @staticmethod
    async def _parse_anthropic_sse_stream(response: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Parse raw Anthropic SSE bytes into decoded dict chunks."""
        async for raw_chunk in response:
            for decoded_chunk in RubrikLogger._decode_all_anthropic_sse_events(raw_chunk):
                yield decoded_chunk

    @staticmethod
    def _encode_anthropic_chunk_to_sse(chunk_dict: dict[str, Any]) -> bytes:
        """Encode an Anthropic dict chunk back to SSE byte format."""
        json_str = json.dumps(chunk_dict, separators=(",", ":"))
        event_type = str(chunk_dict.get("type", "")).replace("\r", "").replace("\n", "")
        return f"event: {event_type}\ndata: {json_str}\n\n".encode()

    @staticmethod
    def _decode_all_anthropic_sse_events(raw_chunk: bytes) -> list[dict[str, Any]]:
        """
        Decode all Anthropic SSE events from a raw chunk.

        A single raw chunk may contain multiple SSE events separated by blank lines.
        """
        data_prefix = "data:"
        events: list[dict[str, Any]] = []

        for line in raw_chunk.decode("utf-8").split("\n"):
            stripped_line = line.strip()
            if stripped_line.startswith(data_prefix):
                json_payload = stripped_line[len(data_prefix) :].strip()
                events.append(json.loads(json_payload))

        return events

    async def _post_to_tool_blocking_service(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post a payload to the tool blocking service and return the response.

        Raises:
            Exception: If the service is unavailable or returns an error.
        """
        headers = self._build_headers()
        verbose_logger.debug(f"Sending request to tool blocking service: {self.tool_blocking_endpoint}")
        http_response = await self.tool_blocking_client.post(
            self.tool_blocking_endpoint,
            json=payload,
            headers=headers,
        )
        http_response.raise_for_status()
        result: dict[str, Any] = http_response.json()
        return result

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for tool blocking service requests."""
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        return headers

    async def _handle_anthropic_non_streaming(self, response: Any) -> Any:
        """Handle Anthropic non-streaming: convert to OpenAI, call blocking service, convert back.

        The response is a raw dict from the Anthropic passthrough endpoint, but typed as Any
        since the caller receives it as LLMResponseTypes which doesn't include dict.
        """
        openai_dict = self._anthropic_response_to_openai_dict(response)
        modified_openai_dict = await self._post_to_tool_blocking_service(openai_dict)
        self._openai_dict_to_anthropic_response(modified_openai_dict, response)
        return response

    def _anthropic_response_to_openai_dict(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert raw Anthropic /v1/messages response to OpenAI format for the blocking service."""
        anthropic_completion = {
            "content": response.get("content", []),
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

        message: dict[str, Any] = {"role": "assistant", "content": text_content}
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
        openai_dict: dict[str, Any],
        original_response: dict[str, Any],
    ) -> None:
        """Convert OpenAI format dict back to Anthropic format, updating the original in-place.

        Only replaces tool_use blocks from the original response with the filtered set from the
        blocking service. Non-tool blocks (text, thinking, citations, etc.) are preserved.
        """
        openai_dict.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        anthropic_response = self.anthropic_adapter.translate_openai_response_to_anthropic(
            response=ModelResponse(**openai_dict),
        )

        new_content = anthropic_response.get("content", []) or []

        # Preserve non-tool, non-text blocks (thinking, citations, etc.) from the original response.
        # Text and tool_use blocks are taken from the converted response since the blocking service
        # may have modified text (added explanation) and removed blocked tool calls.
        original_content = original_response.get("content", [])
        preserved_blocks = [
            block for block in original_content
            if not (isinstance(block, dict) and block.get("type") in _REPLACED_BLOCK_TYPES)
        ]

        original_response["content"] = preserved_blocks + new_content
        original_response["stop_reason"] = anthropic_response.get("stop_reason", "end_turn")


