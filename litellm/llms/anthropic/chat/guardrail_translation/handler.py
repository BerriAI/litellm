"""
Anthropic Message Handler for Unified Guardrails

This module provides a class-based handler for Anthropic-format messages.
The class methods can be overridden for custom behavior.

Pattern Overview:
-----------------
1. Extract text content from messages/responses (both string and list formats)
2. Create async tasks to apply guardrails to each text segment
3. Track mappings to know where each response belongs
4. Apply guardrail responses back to the original structure
"""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.base_llm.guardrail_translation.utils import (
    anthropic_tool_name,
    effective_scan_only_tool_results_for_guardrail,
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    merge_guardrailed_scoped_messages,
    merge_returned_tools_into_request_tools,
    scoped_structured_message_indices,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthropicMessagesRequest,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionRequest,
    ChatCompletionToolCallChunk,
    ChatCompletionToolParam,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    GenericGuardrailAPIInputs,
    ModelResponse,
)

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import (
        CustomGuardrail,
        ModifyResponseException,
    )
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.anthropic_messages.anthropic_response import (
        AnthropicMessagesResponse,
    )


@dataclass(frozen=True, slots=True)
class MessageContentTarget:
    msg_idx: int


@dataclass(frozen=True, slots=True)
class ContentBlockTextTarget:
    msg_idx: int
    content_idx: int


@dataclass(frozen=True, slots=True)
class ToolResultStringTarget:
    msg_idx: int
    content_idx: int


@dataclass(frozen=True, slots=True)
class ToolResultBlockTextTarget:
    msg_idx: int
    content_idx: int
    block_idx: int


InputWriteBackTarget = (
    MessageContentTarget | ContentBlockTextTarget | ToolResultStringTarget | ToolResultBlockTextTarget
)


@dataclass(frozen=True, slots=True)
class ScannedText:
    text: str
    target: InputWriteBackTarget


@dataclass(frozen=True, slots=True)
class ExtractedInput:
    scanned: tuple[ScannedText, ...]
    images: tuple[str, ...]


EMPTY_EXTRACTED_INPUT: Final = ExtractedInput(scanned=(), images=())


class AnthropicMessagesHandler(BaseTranslation):
    """Process Anthropic messages with guardrails.

    In-sequence system entries are untrusted client input. This handler scans and preserves
    them through guardrail rewrites; downstream provider handling is out of scope.
    """

    def __init__(self):
        super().__init__()
        self.adapter = LiteLLMAnthropicMessagesAdapter()

    @staticmethod
    def _build_streaming_usage_response(
        responses_so_far: list[object],
        request_data: dict | None,
    ) -> ModelResponse | None:
        chunks: Final = tuple(response for response in responses_so_far if isinstance(response, (str, bytes)))
        if not chunks:
            return None
        try:
            return AnthropicPassthroughLoggingHandler._build_usage_only_response_from_chunks(
                all_chunks=chunks,
                model=str((request_data or {}).get("model") or ""),
            )
        except (AttributeError, TypeError, ValueError):
            return None

    def build_block_sse_chunks(
        self,
        exc: "ModifyResponseException",
        stream_started: bool = False,
        responses_so_far: list[object] | None = None,
    ) -> list[bytes]:
        """
        Build an Anthropic SSE sequence delivering the guardrail block message
        and terminating the stream cleanly.

        - ``stream_started`` False (buffered / pre-stream): nothing has been
          sent, so emit a complete standalone message (message_start ->
          content_block_* -> message_delta -> message_stop) via
          FakeAnthropicMessagesStreamIterator, the same converter the
          /v1/messages pre-stream block handler uses.
        - ``stream_started`` True (sampling / detect-only end-of-stream): real
          chunks were already sent, so *continue* the in-progress message --
          close the open content block, append the block message as a new text
          block, then end the message. Emitting a second ``message_start`` here
          would make Anthropic clients reject the stream.
        """
        if stream_started:
            return self._block_continuation_chunks(exc, responses_so_far or [])
        return self._standalone_block_chunks(exc)

    def _standalone_block_chunks(self, exc: "ModifyResponseException") -> list[bytes]:
        import uuid

        from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
            FakeAnthropicMessagesStreamIterator,
        )
        from litellm.llms.base_llm.guardrail_translation.utils import (
            blocked_response_usage,
        )
        from litellm.types.utils import AnthropicMessagesResponse

        block_response: Final = AnthropicMessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": exc.message}],
            model=exc.model,
            stop_reason="end_turn",
            usage=blocked_response_usage(getattr(exc, "original_response", None)),
        )
        return list(FakeAnthropicMessagesStreamIterator(response=block_response))

    def _block_continuation_chunks(self, exc: "ModifyResponseException", responses_so_far: list[object]) -> list[bytes]:
        """Continue an already-started message: close the open content block,
        append the block message as a new text block, then end the message --
        without a second message_start."""

        from litellm.llms.base_llm.guardrail_translation.utils import (
            blocked_response_usage,
        )

        def _sse(event_type: str, payload: dict) -> bytes:
            return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()

        output_tokens: Final = blocked_response_usage(getattr(exc, "original_response", None))["output_tokens"]
        open_index, max_index = self._content_block_state(responses_so_far)
        new_index: Final = (max_index + 1) if max_index is not None else 0
        chunks: list[bytes] = []
        if open_index is not None:
            chunks.append(_sse("content_block_stop", {"type": "content_block_stop", "index": open_index}))
        chunks += [
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": new_index,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": new_index,
                    "delta": {"type": "text_delta", "text": exc.message},
                },
            ),
            _sse("content_block_stop", {"type": "content_block_stop", "index": new_index}),
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            ),
            _sse("message_stop", {"type": "message_stop"}),
        ]
        return chunks

    @staticmethod
    def _content_block_state(
        responses_so_far: list[object],
    ) -> tuple[int | None, int | None]:
        """From the SSE chunks already sent to the client, return (open
        content-block index or None, highest content-block index seen or None).

        A single streamed item may bundle multiple SSE events (raw bytes) or be
        an already-parsed event dict, so every event across every item is
        considered -- matching how ``get_streaming_string_so_far`` reads the
        same stream."""
        open_indices: Final[set[int]] = set()
        max_index: int | None = None
        for item in responses_so_far:
            for data in AnthropicMessagesHandler._iter_sse_events(item):
                event_type = data.get("type")
                index = data.get("index")
                if not isinstance(index, int):
                    continue
                if event_type == "content_block_start":
                    open_indices.add(index)
                    max_index = index if max_index is None else max(max_index, index)
                elif event_type == "content_block_stop":
                    open_indices.discard(index)
        open_index: Final = max(open_indices) if open_indices else None
        return open_index, max_index

    @staticmethod
    def _iter_sse_events(item: object) -> list[dict[str, object]]:
        """Yield the event-data dicts in one stream chunk.

        Handles both formats this stream can carry (see
        ``get_streaming_string_so_far``): raw SSE ``bytes`` -- which may bundle
        several events separated by a blank line -- and an already-parsed event
        ``dict``."""
        if isinstance(item, dict):
            return [item]
        if not isinstance(item, (bytes, bytearray)):
            return []
        events: Final[list[dict[str, object]]] = []
        for block in item.decode("utf-8", errors="replace").split("\n\n"):
            for line in block.split("\n"):
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                try:
                    parsed: str | int | float | bool | None | Sequence[object] | Mapping[str, object] = json.loads(
                        line[len("data:") :].strip()
                    )
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    events.append(parsed)
        return events

    def _translate_to_openai(self, data: dict) -> ChatCompletionRequest:
        """Translate Anthropic request to OpenAI chat completion format."""
        (
            chat_completion_compatible_request,
            _tool_name_mapping,
        ) = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            anthropic_message_request=cast(AnthropicMessagesRequest, data.copy())
        )
        return chat_completion_compatible_request

    def get_structured_messages(self, data: dict) -> list[AllMessageValues] | None:
        """
        Convert Anthropic messages request data to OpenAI-spec structured messages.

        Uses the Anthropic-to-OpenAI adapter to translate message format.
        """
        messages: Final = data.get("messages")
        if messages is None:
            return None
        chat_completion_compatible_request: Final = self._translate_to_openai(data)
        result: Final = cast(
            list[AllMessageValues],
            chat_completion_compatible_request.get("messages", []),
        )
        return result if result else None

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> Any:
        """
        Process input messages by applying guardrails to text content.
        """
        messages: Final = data.get("messages")
        if messages is None:
            return data

        skip_system: Final = effective_skip_system_message_for_guardrail(guardrail_to_apply)
        skip_tool: Final = effective_skip_tool_message_for_guardrail(guardrail_to_apply)
        scan_only_tool_results: Final = effective_scan_only_tool_results_for_guardrail(guardrail_to_apply)

        # Exclude only the trusted top-level prompt. In-sequence system entries are untrusted
        # and must stay aligned with texts_to_check for positional masking. When the top-level
        # prompt is included, the pre-existing count mismatch disables positional masking.
        translation_source: Final = {  # mutable-ok: API message payload
            key: value for key, value in data.items() if key != "system"
        }
        chat_completion_compatible_request: Final = self._translate_to_openai(translation_source)

        full_structured_messages: Final = cast(
            list[AllMessageValues],
            chat_completion_compatible_request.get("messages", []),
        )
        has_midturn_system_message: Final = any(
            str(message.get("role") or "").lower() == "system" for message in full_structured_messages
        )
        hoisted_system_message: Final = None if skip_system else self._hoisted_top_level_system_message(data)
        if hoisted_system_message is not None:
            full_structured_messages.insert(0, hoisted_system_message)
        # skip_system already excluded the trusted top-level prompt (it is simply not hoisted);
        # in-sequence system entries are untrusted and always stay in scope.
        scoped_message_indices: Final = scoped_structured_message_indices(
            full_structured_messages,
            scan_only_tool_results=scan_only_tool_results,
            skip_system=False,
            skip_tool=skip_tool,
        )
        structured_messages: Final = [full_structured_messages[index] for index in scoped_message_indices]

        tools_to_check: Final[list[ChatCompletionToolParam]] = (
            [] if scan_only_tool_results else chat_completion_compatible_request.get("tools", [])
        )

        # Step 1: Extract all text content and images
        extracted: Final = tuple(
            self._extract_input_text_and_images(
                message=message,
                msg_idx=msg_idx,
                skip_system_message=skip_system,
                skip_tool_message=skip_tool,
                scan_only_tool_results=scan_only_tool_results,
            )
            for msg_idx, message in enumerate(messages)
        )
        scanned: Final = tuple(item for one_message in extracted for item in one_message.scanned)
        texts_to_check: Final = [item.text for item in scanned]  # mutable-ok: GenericGuardrailAPIInputs takes list[str]
        images_to_check: Final = [
            image for one_message in extracted for image in one_message.images
        ]  # mutable-ok: GenericGuardrailAPIInputs takes list[str]

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check:
            inputs: Final = GenericGuardrailAPIInputs(texts=texts_to_check)
            if images_to_check:
                inputs["images"] = images_to_check
            if tools_to_check:
                inputs["tools"] = tools_to_check
            original_structured_messages: Final = structured_messages
            if structured_messages:
                inputs["structured_messages"] = structured_messages
            # Include model information if available
            model: Final = data.get("model")
            if model:
                inputs["model"] = model
            guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=data,
                input_type="request",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts: Final = guardrailed_inputs.get("texts", [])
            guardrailed_tools: Final = guardrailed_inputs.get("tools")
            if guardrailed_tools is not None:
                # Convert tools back from OpenAI format to Anthropic format
                anthropic_config: Final = AnthropicConfig()
                anthropic_tools: Final[list[AllAnthropicToolsValues]] = []
                for tool in guardrailed_tools:
                    converted_tool, mcp_server = anthropic_config._map_tool_helper(tool)
                    if converted_tool is not None:
                        anthropic_tools.append(converted_tool)
                    # Note: MCP servers are handled separately in the main transformation
                data["tools"] = (
                    merge_returned_tools_into_request_tools(
                        request_tools=data.get("tools"),
                        returned_tools=anthropic_tools,
                        tool_name=anthropic_tool_name,
                    )
                    if scan_only_tool_results
                    else anthropic_tools
                )

            guardrailed_structured_messages: Final = guardrailed_inputs.get("structured_messages")
            if (
                guardrailed_structured_messages is not None
                and guardrailed_structured_messages is not original_structured_messages
            ):
                self._write_back_structured_messages(
                    data,
                    guardrailed_structured_messages
                    if guardrail_to_apply.structured_messages_cover_full_request()
                    else merge_guardrailed_scoped_messages(
                        full_messages=full_structured_messages,
                        scoped_indices=scoped_message_indices,
                        guardrailed_scoped=guardrailed_structured_messages,
                    ),
                    hoisted_system_message=hoisted_system_message,
                    preserve_system_messages=has_midturn_system_message,
                )
            else:
                # Step 3: Map guardrail responses back to original message structure
                await self._apply_guardrail_responses_to_input(
                    messages=messages,
                    responses=guardrailed_texts,
                    scanned=scanned,
                )

        verbose_proxy_logger.debug("Anthropic Messages: Processed input messages: %s", messages)

        return data

    def _hoisted_top_level_system_message(
        self, data: dict
    ) -> AllMessageValues | None:  # mutable-ok: API message payload
        """Return the system message produced by translating the top-level prompt."""
        system: Final = data.get("system")
        if not system:
            return None
        probe: Final = self._translate_to_openai(
            {  # mutable-ok: API message payload
                "model": data.get("model") or "",
                "messages": [],  # mutable-ok: API message payload
                "system": system,
            }
        )
        hoisted: Final = probe.get("messages") or []  # mutable-ok: API message payload
        return hoisted[0] if hoisted else None

    @staticmethod
    def _openai_system_message_to_anthropic(
        message: dict[str, object],
    ) -> dict[str, object] | None:  # mutable-ok: API message payload
        """Convert an OpenAI system message to the client's Anthropic-shaped entry."""
        content: Final = message.get("content")
        if isinstance(content, str):
            return (
                {"role": "system", "content": content} if content else None  # mutable-ok: API message payload
            )  # mutable-ok: API message payload
        if not isinstance(content, list):
            return None
        blocks: Final[list[dict[str, object]]] = []  # mutable-ok: API message payload
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text")
            if not isinstance(text, str) or not text:
                continue
            anthropic_block: dict[str, object] = {  # mutable-ok: API message payload
                "type": "text",
                "text": text,
            }  # mutable-ok: API message payload
            cache_control = block.get("cache_control")
            if cache_control:
                anthropic_block["cache_control"] = deepcopy(cache_control)
            blocks.append(anthropic_block)
        return (
            {"role": "system", "content": blocks} if blocks else None  # mutable-ok: API message payload
        )  # mutable-ok: API message payload

    @staticmethod
    def _fold_leading_systems_into_top_level(
        data: dict[str, object],  # mutable-ok: API message payload
        leading_systems: Sequence[object],
        include_existing_system: bool,
    ) -> None:
        """Deliver leading system rows through Anthropic's top-level system param, which rejects them in messages."""
        existing: Final = data.get("system") if include_existing_system else None
        existing_blocks: Final[list[object]] = (  # mutable-ok: API message payload
            [{"type": "text", "text": existing}]
            if isinstance(existing, str) and existing
            else list(existing)
            if isinstance(existing, list)
            else []
        )
        converted_rows: Final = tuple(
            AnthropicMessagesHandler._openai_system_message_to_anthropic(message)
            for message in leading_systems
            if isinstance(message, dict)
        )
        folded: Final[list[object]] = existing_blocks + [  # mutable-ok: API message payload
            block
            for row in converted_rows
            if row is not None
            for block in (
                [{"type": "text", "text": row["content"]}] if isinstance(row["content"], str) else row["content"]
            )
        ]
        if folded:
            data["system"] = folded  # rebind-ok: write-back mutates the request payload in place
        else:
            data.pop("system", None)

    @staticmethod
    def _is_hoisted_top_level_system(message: object, hoisted_system_message: object) -> bool:
        """Match the hoisted prompt by identity, or by value after serialization."""
        if hoisted_system_message is None:
            return False
        if message is hoisted_system_message:
            return True
        return (
            isinstance(message, dict) and isinstance(hoisted_system_message, dict) and message == hoisted_system_message
        )

    @staticmethod
    def _is_system(message: object) -> bool:
        """Whether the row is an in-sequence system message."""
        return isinstance(message, dict) and str(message.get("role") or "").lower() == "system"

    @staticmethod
    def _defer_systems_inside_tool_exchanges(
        structured_messages: list,  # mutable-ok: API message payload
    ) -> list:
        """Hold a system row until the tool exchange around it completes so the call/result pair converts together."""
        from litellm.litellm_core_utils.prompt_templates.factory import group_tool_exchanges

        non_system_positions: Final[list[int]] = [
            index
            for index, message in enumerate(structured_messages)
            if not AnthropicMessagesHandler._is_system(message)
        ]
        exchange_end_for_start: Final[dict[int, int]] = {
            non_system_positions[group[0]]: non_system_positions[group[-1]]
            for group in group_tool_exchanges([structured_messages[index] for index in non_system_positions])
            if len(group) > 1
        }
        ordered: Final[list] = []  # mutable-ok: API message payload
        deferred_systems: Final[list] = []  # mutable-ok: API message payload
        open_exchange_end = -1  # rebind-ok: advances to the enclosing exchange's last index
        for index, message in enumerate(structured_messages):
            if AnthropicMessagesHandler._is_system(message) and index < open_exchange_end:
                deferred_systems.append(message)
                continue
            open_exchange_end = exchange_end_for_start.get(index, open_exchange_end)
            ordered.append(message)
            if index >= open_exchange_end and deferred_systems:
                ordered.extend(deferred_systems)
                deferred_systems.clear()
        ordered.extend(deferred_systems)
        return ordered

    @staticmethod
    def _write_back_structured_messages(
        data: dict,  # mutable-ok: API message payload
        structured_messages: list,  # mutable-ok: API message payload
        hoisted_system_message: object = None,
        preserve_system_messages: bool = False,
    ) -> None:
        """Write a guardrail's structured-message rewrite back without losing corrections."""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            anthropic_messages_pt,
            group_tool_exchanges,
        )

        _is_system: Final = AnthropicMessagesHandler._is_system
        model: Final = str(data.get("model") or "")
        converted: Final[list] = []  # mutable-ok: API message payload

        def _convert_run(run: list) -> None:  # mutable-ok: API message payload
            for group in group_tool_exchanges(run):
                converted.extend(
                    anthropic_messages_pt(
                        messages=[run[index] for index in group],  # mutable-ok: API message payload
                        model=model,
                        llm_provider="anthropic",
                    )
                )

        ordered: Final = AnthropicMessagesHandler._defer_systems_inside_tool_exchanges(structured_messages)
        leading_count: Final = next(
            (index for index, message in enumerate(ordered) if not _is_system(message)),
            len(ordered),
        )
        leading_systems: Final = ordered[:leading_count]
        hoisted_in_leading: Final = any(
            AnthropicMessagesHandler._is_hoisted_top_level_system(message, hoisted_system_message)
            for message in leading_systems
        )
        if leading_systems and not (leading_count == 1 and hoisted_in_leading):
            AnthropicMessagesHandler._fold_leading_systems_into_top_level(
                data,
                leading_systems,
                include_existing_system=hoisted_system_message is None,
            )
        run: Final[list] = []  # mutable-ok: API message payload
        hoisted_dropped = hoisted_in_leading  # rebind-ok: flips once the hoisted prompt is dropped
        for message in ordered[leading_count:]:
            if not _is_system(message):
                run.append(message)
                continue
            _convert_run(run)
            run.clear()
            if not hoisted_dropped and AnthropicMessagesHandler._is_hoisted_top_level_system(
                message, hoisted_system_message
            ):
                hoisted_dropped = True
                continue
            if preserve_system_messages:
                anthropic_system = AnthropicMessagesHandler._openai_system_message_to_anthropic(message)
                if anthropic_system is not None:
                    converted.append(anthropic_system)
        _convert_run(run)
        if not any(not _is_system(message) for message in converted):
            converted.extend(anthropic_messages_pt(messages=[], model=model, llm_provider="anthropic"))
        for msg in converted:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        block.pop("cache_control", None)
        data["messages"] = converted

    @staticmethod
    def _extract_midturn_system_text(
        message: Mapping[str, object],
        msg_idx: int,
    ) -> ExtractedInput:
        """Match the adapter's filtering so positional guardrail write-back stays aligned."""
        content: Final = message.get("content")
        if isinstance(content, str):
            if not content:
                return EMPTY_EXTRACTED_INPUT
            return ExtractedInput(scanned=(ScannedText(content, MessageContentTarget(msg_idx)),), images=())
        if not isinstance(content, list):
            return EMPTY_EXTRACTED_INPUT
        return ExtractedInput(
            scanned=tuple(
                ScannedText(text_str, ContentBlockTextTarget(msg_idx, content_idx))
                for content_idx, content_item in enumerate(content)
                if isinstance(content_item, dict)
                and content_item.get("type") == "text"
                and isinstance(text_str := content_item.get("text"), str)
                and text_str
            ),
            images=(),
        )

    def extract_request_tool_names(self, data: dict) -> list[str]:
        """Extract tool names from Anthropic messages request (tools[].name)."""
        names: Final[list[str]] = []
        for tool in data.get("tools") or []:
            if isinstance(tool, dict) and tool.get("name"):
                names.append(str(tool["name"]))
        return names

    @classmethod
    def _extract_input_text_and_images(
        cls,
        message: Mapping[str, object],
        msg_idx: int,
        skip_system_message: bool = False,
        skip_tool_message: bool = False,
        scan_only_tool_results: bool = False,
    ) -> ExtractedInput:
        """Extract text content and images from a message.

        In-sequence system entries are scanned even when ``skip_system_message`` is set:
        that flag covers only the trusted top-level prompt, which never appears here.
        """
        role: Final = str(message.get("role") or "")
        if role == "system":
            if scan_only_tool_results:
                return EMPTY_EXTRACTED_INPUT
            return cls._extract_midturn_system_text(message=message, msg_idx=msg_idx)
        if skip_tool_message and role.lower() == "tool":
            return EMPTY_EXTRACTED_INPUT

        content: Final = message.get("content", None)
        if isinstance(content, str):
            if scan_only_tool_results:
                return EMPTY_EXTRACTED_INPUT
            return ExtractedInput(scanned=(ScannedText(content, MessageContentTarget(msg_idx)),), images=())
        if not isinstance(content, list):
            return EMPTY_EXTRACTED_INPUT

        blocks: Final = tuple(
            cls._extract_content_block(
                content_item=content_item,
                msg_idx=msg_idx,
                content_idx=content_idx,
                skip_tool_message=skip_tool_message,
                scan_only_tool_results=scan_only_tool_results,
            )
            for content_idx, content_item in enumerate(content)
            if isinstance(content_item, dict)
        )
        return ExtractedInput(
            scanned=tuple(item for block in blocks for item in block.scanned),
            images=tuple(image for block in blocks for image in block.images),
        )

    @classmethod
    def _extract_content_block(
        cls,
        content_item: Mapping[str, Any],
        msg_idx: int,
        content_idx: int,
        skip_tool_message: bool,
        scan_only_tool_results: bool = False,
    ) -> ExtractedInput:
        if content_item.get("type") == "tool_result":
            if skip_tool_message:
                return EMPTY_EXTRACTED_INPUT
            return cls._extract_tool_result(content_item=content_item, msg_idx=msg_idx, content_idx=content_idx)

        if scan_only_tool_results:
            return EMPTY_EXTRACTED_INPUT

        text_str: Final = content_item.get("text", None)
        return ExtractedInput(
            scanned=(
                () if text_str is None else (ScannedText(text_str, ContentBlockTextTarget(msg_idx, content_idx)),)
            ),
            images=cls._image_sources(content_item) if content_item.get("type") == "image" else (),
        )

    @classmethod
    def _extract_tool_result(
        cls,
        content_item: Mapping[str, object],
        msg_idx: int,
        content_idx: int,
    ) -> ExtractedInput:
        tool_result_content: Final = content_item.get("content")

        if isinstance(tool_result_content, str):
            return ExtractedInput(
                scanned=(ScannedText(tool_result_content, ToolResultStringTarget(msg_idx, content_idx)),),
                images=(),
            )
        if not isinstance(tool_result_content, list):
            return EMPTY_EXTRACTED_INPUT

        blocks: Final = tuple(
            (block_idx, block) for block_idx, block in enumerate(tool_result_content) if isinstance(block, dict)
        )
        return ExtractedInput(
            scanned=tuple(
                ScannedText(block["text"], ToolResultBlockTextTarget(msg_idx, content_idx, block_idx))
                for block_idx, block in blocks
                if isinstance(block.get("text"), str)
            ),
            images=tuple(
                image for _, block in blocks if block.get("type") == "image" for image in cls._image_sources(block)
            ),
        )

    @staticmethod
    def _image_sources(block: Mapping[str, object]) -> tuple[str, ...]:
        source: Final = block.get("source")
        if not isinstance(source, Mapping):
            return ()
        # Could be base64 or url
        data: Final = source.get("data")
        return (data,) if data else ()

    async def _apply_guardrail_responses_to_input(
        self,
        messages: list[dict[str, object]],
        responses: list[str],
        scanned: tuple[ScannedText, ...],
    ) -> None:
        """
        Apply guardrail responses back to input messages.
        """
        for item, guardrail_response in zip(scanned, responses):
            target = item.target
            message = messages[target.msg_idx]
            content = message.get("content", None)
            if content is None:
                continue

            match target:
                case MessageContentTarget():
                    if isinstance(content, str):
                        message["content"] = (
                            guardrail_response  # mutable-ok: guardrails rewrite the caller's request payload in place
                        )
                case ContentBlockTextTarget(content_idx=content_idx):
                    if isinstance(content, list):
                        content[content_idx]["text"] = (
                            guardrail_response  # mutable-ok: guardrails rewrite the caller's request payload in place
                        )
                case ToolResultStringTarget(content_idx=content_idx):
                    if isinstance(content, list):
                        content[content_idx]["content"] = (
                            guardrail_response  # mutable-ok: guardrails rewrite the caller's request payload in place
                        )
                case ToolResultBlockTextTarget(content_idx=content_idx, block_idx=block_idx):
                    if isinstance(content, list):
                        content[content_idx]["content"][block_idx]["text"] = (
                            guardrail_response  # mutable-ok: guardrails rewrite the caller's request payload in place
                        )
                case _:
                    assert_never(target)

    async def process_output_response(
        self,
        response: "AnthropicMessagesResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict | None = None,
    ) -> "AnthropicMessagesResponse":
        """
        Process output response by applying guardrails to text content and tool calls.

        Args:
            response: Anthropic MessagesResponse object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

        Returns:
            Modified response with guardrail applied to content

        Response Format Support:
            - List content: response.content = [
                {"type": "text", "text": "text here"},
                {"type": "tool_use", "id": "...", "name": "...", "input": {...}},
                ...
            ]
        """
        texts_to_check: Final[list[str]] = []
        images_to_check: Final[list[str]] = []
        tool_calls_to_check: Final[list[ChatCompletionToolCallChunk]] = []
        task_mappings: Final[list[tuple[int, int | None]]] = []

        response_content: Final = self._get_response_content(response)
        if not response_content:
            return response

        # Step 1: Extract all text content and tool calls from response
        self._extract_from_content_blocks(
            response_content,
            texts_to_check,
            images_to_check,
            task_mappings,
            tool_calls_to_check,
        )

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check or tool_calls_to_check:
            request_data = self._prepare_request_data(
                request_data,
                response,
                user_api_key_dict,
                key="response",
            )

            inputs: Final = self._build_guardrail_inputs(
                texts_to_check,
                images_to_check,
                tool_calls_to_check,
                response,
            )

            guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts: Final = guardrailed_inputs.get("texts", [])

            # Step 3: Map guardrail responses back to original response structure
            await self._apply_guardrail_responses_to_output(
                response=response,
                responses=guardrailed_texts,
                task_mappings=task_mappings,
            )

        verbose_proxy_logger.debug("Anthropic Messages: Processed output response: %s", response)

        return response

    async def process_output_streaming_response(
        self,
        responses_so_far: list[Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict | None = None,
    ) -> list[Any]:
        """
        Process output streaming response by applying guardrails to text content.

        Get the string so far, check the apply guardrail to the string so far, and return the list of responses so far.
        """
        from litellm.integrations.custom_guardrail import ModifyResponseException

        has_ended: Final = self._check_streaming_has_ended(responses_so_far)
        if has_ended:
            # build the model response from the responses_so_far
            built_response: Final = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
                all_chunks=responses_so_far,
                litellm_logging_obj=cast("LiteLLMLoggingObj", litellm_logging_obj),
                model="",
            )

            # Check if model_response is valid and has choices before accessing
            if built_response is not None and hasattr(built_response, "choices") and built_response.choices:
                model_response: Final = cast(ModelResponse, built_response)
                first_choice: Final = cast(Choices, model_response.choices[0])
                tool_calls_list: Final = cast(
                    list[ChatCompletionMessageToolCall] | None,
                    first_choice.message.tool_calls,
                )
                string_so_far = first_choice.message.content
                guardrail_inputs: Final = GenericGuardrailAPIInputs()
                if string_so_far:
                    guardrail_inputs["texts"] = [string_so_far]
                if tool_calls_list:
                    guardrail_inputs["tool_calls"] = tool_calls_list

                try:
                    prepared_request_data = self._prepare_request_data(
                        request_data,
                        model_response,
                        user_api_key_dict,
                        key="response",
                    )
                    _guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                        inputs=guardrail_inputs,
                        request_data=prepared_request_data,
                        input_type="response",
                        logging_obj=litellm_logging_obj,
                    )
                except ModifyResponseException as e:
                    if e.original_response is None:
                        e.original_response = built_response or self._build_streaming_usage_response(
                            responses_so_far, request_data
                        )
                    raise
            else:
                verbose_proxy_logger.debug("Skipping output guardrail - model response has no choices")
            return responses_so_far

        string_so_far = self.get_streaming_string_so_far(responses_so_far)
        try:
            prepared_request_data = self._prepare_request_data(
                request_data,
                responses_so_far,
                user_api_key_dict,
                key="responses",
            )
            _guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs={"texts": [string_so_far]},
                request_data=prepared_request_data,
                input_type="response",
                logging_obj=litellm_logging_obj,
            )
        except ModifyResponseException as e:
            if e.original_response is None:
                e.original_response = self._build_streaming_usage_response(responses_so_far, request_data)
            raise
        return responses_so_far

    def _prepare_request_data(
        self,
        request_data: dict | None,
        response: object,
        user_api_key_dict: "UserAPIKeyAuth | None",
        key: str,
    ) -> dict:
        """Ensure request_data has the response/responses_so_far key and metadata."""
        if request_data is None:
            request_data = {key: response}
        else:
            if key not in request_data:
                request_data[key] = response

        if "litellm_metadata" not in request_data:
            user_metadata: Final = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
            if user_metadata:
                request_data["litellm_metadata"] = user_metadata
        return request_data

    @staticmethod
    def _get_response_content(response: object) -> list[Any]:
        """Extract content list from a dict or object response."""
        if isinstance(response, dict):
            return response.get("content", []) or []
        elif hasattr(response, "content"):
            return getattr(response, "content", None) or []
        return []

    def _extract_from_content_blocks(
        self,
        response_content: list[Any],
        texts_to_check: list[str],
        images_to_check: list[str],
        task_mappings: list[tuple[int, int | None]],
        tool_calls_to_check: list["ChatCompletionToolCallChunk"],
    ) -> None:
        """Extract text, images, and tool calls from content blocks."""
        for content_idx, content_block in enumerate(response_content):
            block_dict: dict[str, object] = {}
            if isinstance(content_block, dict):
                block_type = content_block.get("type")
                block_dict = cast(dict[str, object], content_block)
            elif hasattr(content_block, "type"):
                block_type = getattr(content_block, "type", None)
                if hasattr(content_block, "model_dump"):
                    block_dict = content_block.model_dump()
                else:
                    block_dict = {
                        "type": block_type,
                        "text": getattr(content_block, "text", None),
                    }
            else:
                continue

            if block_type in ["text", "tool_use"]:
                self._extract_output_text_and_images(
                    content_block=block_dict,
                    content_idx=content_idx,
                    texts_to_check=texts_to_check,
                    images_to_check=images_to_check,
                    task_mappings=task_mappings,
                    tool_calls_to_check=tool_calls_to_check,
                )

    @staticmethod
    def _build_guardrail_inputs(
        texts_to_check: list[str],
        images_to_check: list[str],
        tool_calls_to_check: list["ChatCompletionToolCallChunk"],
        response: object,
    ) -> "GenericGuardrailAPIInputs":
        """Build GenericGuardrailAPIInputs with optional images, tool calls, model."""
        inputs: Final = GenericGuardrailAPIInputs(texts=texts_to_check)
        if images_to_check:
            inputs["images"] = images_to_check
        if tool_calls_to_check:
            inputs["tool_calls"] = tool_calls_to_check
        response_model = None
        if isinstance(response, dict):
            response_model = response.get("model")
        elif hasattr(response, "model"):
            response_model = getattr(response, "model", None)
        if response_model:
            inputs["model"] = response_model
        return inputs

    def get_streaming_string_so_far(self, responses_so_far: list[Any]) -> str:
        """
        Parse streaming responses and extract accumulated text content.

        Handles two formats:
        1. Raw bytes in SSE (Server-Sent Events) format from Anthropic API
        2. Parsed dict objects (for backwards compatibility)

        SSE format example:
            b'event: content_block_delta\\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" curious"}}\\n\\n'

        Dict format example:
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": " curious"
                }
            }
        """
        text_so_far = ""
        for response in responses_so_far:
            # Handle raw bytes in SSE format
            if isinstance(response, bytes):
                text_so_far += self._extract_text_from_sse(response)
            # Handle already-parsed dict format
            elif isinstance(response, dict):
                delta = response.get("delta") if response.get("delta") else None
                if delta and delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        text_so_far += text
        return text_so_far

    def _extract_text_from_sse(self, sse_bytes: bytes) -> str:
        """
        Extract text content from Server-Sent Events (SSE) format.

        Args:
            sse_bytes: Raw bytes in SSE format

        Returns:
            Accumulated text from all content_block_delta events
        """
        text = ""
        try:
            # Decode bytes to string
            sse_string: Final = sse_bytes.decode("utf-8")

            # Split by double newline to get individual events
            events: Final = sse_string.split("\n\n")

            for event in events:
                if not event.strip():
                    continue

                # Parse event lines
                lines = event.strip().split("\n")
                event_type = None
                data_line = None

                for line in lines:
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_line = line[5:].strip()

                # Only process content_block_delta events
                if event_type == "content_block_delta" and data_line:
                    try:
                        data = json.loads(data_line)
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text += delta.get("text", "")
                    except json.JSONDecodeError:
                        verbose_proxy_logger.warning("Failed to parse JSON from SSE data: %s", data_line)

        except Exception as e:
            verbose_proxy_logger.error("Error extracting text from SSE: %s", e)

        return text

    def _check_streaming_has_ended(self, responses_so_far: list[Any]) -> bool:
        """
        Check if streaming response has ended by looking for non-null stop_reason.

        Handles two formats:
        1. Raw bytes in SSE (Server-Sent Events) format from Anthropic API
        2. Parsed dict objects (for backwards compatibility)

        SSE format example:
            b'event: message_delta\\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},...}\\n\\n'

        Dict format example:
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use",
                    "stop_sequence": null
                }
            }

        Returns:
            True if stop_reason is set to a non-null value, indicating stream has ended
        """
        for response in responses_so_far:
            # Handle raw bytes in SSE format
            if isinstance(response, bytes):
                try:
                    # Decode bytes to string
                    sse_string = response.decode("utf-8")

                    # Split by double newline to get individual events
                    events = sse_string.split("\n\n")

                    for event in events:
                        if not event.strip():
                            continue

                        # Parse event lines
                        lines = event.strip().split("\n")
                        event_type = None
                        data_line = None

                        for line in lines:
                            if line.startswith("event:"):
                                event_type = line[6:].strip()
                            elif line.startswith("data:"):
                                data_line = line[5:].strip()

                        # Check for message_delta event with stop_reason
                        if event_type == "message_delta" and data_line:
                            try:
                                data = json.loads(data_line)
                                delta = data.get("delta", {})
                                stop_reason = delta.get("stop_reason")
                                if stop_reason is not None:
                                    return True
                            except json.JSONDecodeError:
                                verbose_proxy_logger.warning("Failed to parse JSON from SSE data: %s", data_line)

                except Exception as e:
                    verbose_proxy_logger.error("Error checking streaming end in SSE: %s", e)

            # Handle already-parsed dict format
            elif isinstance(response, dict):
                if response.get("type") == "message_delta":
                    delta = response.get("delta", {})
                    stop_reason = delta.get("stop_reason")
                    if stop_reason is not None:
                        return True

        return False

    def _has_text_content(self, response: "AnthropicMessagesResponse") -> bool:
        """
        Check if response has any text content to process.

        Override this method to customize text content detection.
        """
        if isinstance(response, dict):
            response_content = response.get("content", [])
        else:
            response_content = getattr(response, "content", None) or []

        if not response_content:
            return False
        for content_block in response_content:
            # Check if this is a text block by checking the 'type' field
            if isinstance(content_block, dict) and content_block.get("type") == "text":
                content_text = content_block.get("text")
                if content_text and isinstance(content_text, str):
                    return True
        return False

    def _extract_output_text_and_images(
        self,
        content_block: dict[str, object],
        content_idx: int,
        texts_to_check: list[str],
        images_to_check: list[str],
        task_mappings: list[tuple[int, int | None]],
        tool_calls_to_check: list[ChatCompletionToolCallChunk] | None = None,
    ) -> None:
        """
        Extract text content, images, and tool calls from a response content block.

        Override this method to customize text/image/tool extraction logic.
        """
        content_type: Final = content_block.get("type")

        # Extract text content
        if content_type == "text":
            content_text: Final = content_block.get("text")
            if content_text and isinstance(content_text, str):
                # Simple string content
                texts_to_check.append(content_text)
                task_mappings.append((content_idx, None))

        # Extract tool calls
        elif content_type == "tool_use":
            tool_call: Final = AnthropicConfig.convert_tool_use_to_openai_format(
                anthropic_tool_content=content_block,
                index=content_idx,
            )
            if tool_calls_to_check is None:
                tool_calls_to_check = []
            tool_calls_to_check.append(tool_call)

    async def _apply_guardrail_responses_to_output(
        self,
        response: "AnthropicMessagesResponse",
        responses: list[str],
        task_mappings: list[tuple[int, int | None]],
    ) -> None:
        """
        Apply guardrail responses back to output response.

        Override this method to customize how responses are applied.
        """
        for task_idx, guardrail_response in enumerate(responses):
            mapping = task_mappings[task_idx]
            content_idx = cast(int, mapping[0])

            # Handle both dict and object responses
            response_content: list[Any] = []
            if isinstance(response, dict):
                response_content = response.get("content", []) or []
            elif hasattr(response, "content"):
                content = getattr(response, "content", None)
                response_content = content or []
            else:
                continue

            if not response_content:
                continue

            # Get the content block at the index
            if content_idx >= len(response_content):
                continue

            content_block = response_content[content_idx]

            # Verify it's a text block and update the text field
            # Handle both dict and Pydantic object content blocks
            if isinstance(content_block, dict):
                if content_block.get("type") == "text":
                    cast(dict[str, object], content_block)["text"] = guardrail_response
            elif hasattr(content_block, "type") and getattr(content_block, "type", None) == "text":
                # Update Pydantic object's text attribute
                if hasattr(content_block, "text"):
                    content_block.text = guardrail_response
