"""
Transformation layer: Anthropic /v1/messages <-> OpenAI Responses API.

This module owns all format conversions for the direct v1/messages -> Responses API
path used for OpenAI and Azure models.
"""

import json
from collections.abc import Iterable, Mapping
from itertools import groupby
from typing import Any, Final, cast

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    TOOL_RESULT_IMAGE_BOUNDARY,
    TOOL_RESULT_IMAGE_PLACEHOLDER,
    responses_reasoning_item_from_thinking_blocks,
    with_prompt_cache_breakpoint,
)
from litellm.litellm_core_utils.reasoning_effort_utils import (
    reasoning_effort_from_thinking_budget,
)
from litellm.llms.anthropic.experimental_pass_through.utils import (
    is_reasoning_auto_summary_enabled,
    prompt_cache_key_from_user_id,
)
from litellm.types.llms.anthropic import (
    AllAnthropicPassThroughMessageValues,
    AllAnthropicToolsValues,
    AnthropicFinishReason,
    AnthropicMessagesRequest,
    AnthropicMessagesToolChoice,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockThinking,
    AnthropicResponseContentBlockToolUse,
    AnthropicSystemMessageContent,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from litellm.types.llms.openai import (
    ChatCompletionThinkingBlock,
    ResponseAPIUsage,
    ResponsesAPIResponse,
)


class LiteLLMAnthropicToResponsesAPIAdapter:
    """
    Converts Anthropic /v1/messages requests to OpenAI Responses API format and
    converts Responses API responses back to Anthropic format.
    """

    @staticmethod
    def translate_responses_api_usage_to_anthropic_usage(
        raw_usage: ResponseAPIUsage | None,
    ) -> AnthropicUsage:
        """Map Responses API usage onto Anthropic usage, where ``input_tokens``
        excludes the cache-read and cache-write tokens reported alongside it.
        """
        if raw_usage is None:
            return AnthropicUsage(input_tokens=0, output_tokens=0)

        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
        from litellm.responses.utils import ResponseAPILoggingUtils

        chat_usage = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(raw_usage)
        return LiteLLMAnthropicMessagesAdapter._translate_openai_usage_to_anthropic_usage(chat_usage)

    # ------------------------------------------------------------------ #
    # Request translation: Anthropic -> Responses API                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _translate_anthropic_image_source_to_url(source: object) -> str | None:
        """Convert Anthropic image source to a URL string."""
        if not isinstance(source, dict):
            return None
        source_type: Final = source.get("type")
        if source_type == "base64":
            media_type: Final = source.get("media_type", "image/jpeg")
            data: Final = source.get("data", "")
            return f"data:{media_type};base64,{data}" if data else None
        elif source_type == "url":
            return source.get("url")
        return None

    @staticmethod
    def _translate_anthropic_document_block_to_file_part(
        block: Mapping[str, object],
    ) -> dict[str, str] | None:  # mutable-ok: API message payload
        """Convert an Anthropic document block to a Responses input_file part."""
        raw_source: Final = block.get("source")
        if not isinstance(raw_source, Mapping):
            return None
        source: Final = cast(Mapping[str, object], raw_source)  # cast-ok: untrusted client payload
        source_type: Final = source.get("type")
        if source_type == "base64":
            data: Final = source.get("data")
            if not isinstance(data, str) or not data:
                return None
            raw_media_type: Final = source.get("media_type")
            media_type: Final = (
                raw_media_type if isinstance(raw_media_type, str) and raw_media_type else "application/pdf"
            )
            raw_title: Final = block.get("title")
            filename: Final = raw_title if isinstance(raw_title, str) and raw_title else "document.pdf"
            return {  # mutable-ok: API message payload
                "type": "input_file",
                "filename": filename,
                "file_data": f"data:{media_type};base64,{data}",
            }
        if source_type == "url":
            url: Final = source.get("url")
            if not isinstance(url, str) or not url:
                return None
            return {"type": "input_file", "file_url": url}  # mutable-ok: API message payload
        return None

    @staticmethod
    def _tool_result_output_value(
        output_text: str,
        file_parts: tuple[dict[str, str], ...],  # mutable-ok: json content parts
    ) -> str | list[dict[str, str]]:  # mutable-ok: API message payload
        """Plain string output, or a part list when document file parts are present."""
        if not file_parts:
            return output_text
        text_parts: Final = (
            [{"type": "input_text", "text": output_text}] if output_text else []  # mutable-ok: API message payload
        )
        return [*text_parts, *file_parts]  # mutable-ok: API message payload

    @staticmethod
    def _translate_midturn_system_content_to_responses(
        content: str | Iterable[AnthropicSystemMessageContent],
    ) -> list[dict[str, object]]:  # mutable-ok: API message payload
        """Convert in-sequence system content to Responses input-text parts."""
        if isinstance(content, str):
            return (
                [{"type": "input_text", "text": content}] if content else []  # mutable-ok: API message payload
            )  # mutable-ok: API message payload
        if not isinstance(content, list):
            return []  # mutable-ok: API message payload
        return [  # mutable-ok: API message payload
            with_prompt_cache_breakpoint(
                {"type": "input_text", "text": text}, block.get("prompt_cache_breakpoint")
            )  # mutable-ok: API message payload
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and (text := block.get("text"))  # pyright: ignore[reportUnnecessaryIsInstance]  # untrusted client payload
        ]

    @staticmethod
    def _summary_part_text(part: object) -> str:
        if isinstance(part, Mapping):
            mapping: Final = cast(Mapping[str, Any], part)  # cast-ok: summary parts are untyped provider json
            return str(mapping.get("text") or "")
        return str(getattr(part, "text", None) or "")

    @classmethod
    def _thinking_blocks_from_reasoning_item(
        cls,
        summary: Iterable[object],
    ) -> tuple[dict[str, Any], ...]:  # mutable-ok: API message payload
        """Anthropic thinking blocks for one Responses reasoning item.

        The signature stays empty: only Anthropic can sign a thinking block, and a stand-in
        value would be replayed as a real one and rejected by every backend that verifies it.
        """
        return tuple(
            AnthropicResponseContentBlockThinking(
                type="thinking",
                thinking=text,
                signature=None,
            ).model_dump()
            for part in summary
            if (text := cls._summary_part_text(part))
        )

    @staticmethod
    def _assistant_block_group_key(indexed_block: tuple[int, Mapping[str, Any]]) -> str:
        """Group a run of consecutive thinking blocks together; keep every other block alone."""
        index, block = indexed_block
        return "thinking" if block.get("type") == "thinking" else f"block:{index}"

    @classmethod
    def _assistant_group_to_input_item(
        cls, group: tuple[Mapping[str, Any], ...]
    ) -> dict[str, Any] | None:  # mutable-ok: API message payload
        first: Final = group[0]
        btype: Final = first.get("type")
        if btype == "thinking":
            blocks: Final = cast(tuple[ChatCompletionThinkingBlock, ...], group)  # cast-ok: untrusted client payload
            reasoning_item: Final = responses_reasoning_item_from_thinking_blocks(blocks)
            return None if reasoning_item is None else dict(reasoning_item)  # mutable-ok: API message payload
        if btype == "tool_use":
            return {  # mutable-ok: API message payload
                "type": "function_call",
                "call_id": first.get("id", ""),
                "name": first.get("name", ""),
                "arguments": json.dumps(first.get("input", {})),  # mutable-ok: API message payload
            }
        return None

    def translate_messages_to_responses_input(
        self,
        messages: list[AllAnthropicPassThroughMessageValues],
    ) -> list[dict[str, Any]]:
        """
        Convert Anthropic messages list to Responses API `input` items.

        Mapping:
          system text        -> message(role=system, input_text)
          user text          -> message(role=user, input_text)
          user image         -> message(role=user, input_image)
          user document      -> message(role=user, input_file)
          user tool_result   -> function_call_output
          assistant text     -> message(role=assistant, output_text)
          assistant thinking -> reasoning
          assistant tool_use -> function_call
        """
        input_items: Final[list[dict[str, Any]]] = []

        for m in messages:
            if m["role"] == "system":
                system_parts = self._translate_midturn_system_content_to_responses(m.get("content"))
                if system_parts:
                    input_items.append(
                        {  # mutable-ok: API message payload
                            "type": "message",
                            "role": "system",
                            "content": system_parts,
                        }
                    )
                continue

            role = m["role"]
            content = m.get("content")

            if role == "user":
                if isinstance(content, str):
                    input_items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": content}],
                        }
                    )
                elif isinstance(content, list):
                    user_parts: list[dict[str, Any]] = []
                    tool_image_parts: list[dict[str, Any]] = []  # mutable-ok: json content parts
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            user_parts.append(
                                with_prompt_cache_breakpoint(
                                    {"type": "input_text", "text": block.get("text", "")},
                                    block.get("prompt_cache_breakpoint"),
                                )
                            )
                        elif btype == "image":
                            url = self._translate_anthropic_image_source_to_url(cast(dict, block.get("source", {})))
                            if url:
                                user_parts.append(
                                    with_prompt_cache_breakpoint(
                                        {"type": "input_image", "image_url": url}, block.get("prompt_cache_breakpoint")
                                    )
                                )
                        elif btype == "document":
                            file_part = self._translate_anthropic_document_block_to_file_part(block)
                            if file_part:
                                user_parts.append(
                                    with_prompt_cache_breakpoint(file_part, block.get("prompt_cache_breakpoint"))
                                )
                        elif btype == "tool_result":
                            tool_use_id = block.get("tool_use_id", "")
                            inner = block.get("content")
                            document_candidates = (
                                tuple(
                                    self._translate_anthropic_document_block_to_file_part(c)
                                    for c in inner
                                    if isinstance(c, dict) and c.get("type") == "document"
                                )
                                if isinstance(inner, list)
                                else ()
                            )
                            tool_file_parts = tuple(part for part in document_candidates if part is not None)
                            if inner is None:
                                output_text = ""
                            elif isinstance(inner, str):
                                output_text = inner
                            elif isinstance(inner, list):
                                parts = [
                                    c.get("text", "") for c in inner if isinstance(c, dict) and c.get("type") == "text"
                                ]
                                output_text = "\n".join(parts)
                                image_candidates = tuple(
                                    self._translate_anthropic_image_source_to_url(c.get("source"))
                                    for c in inner
                                    if isinstance(c, dict) and c.get("type") == "image"
                                )
                                image_urls = tuple(url for url in image_candidates if url)
                                if image_urls:
                                    output_text = (
                                        f"{output_text}\n{TOOL_RESULT_IMAGE_PLACEHOLDER}"
                                        if output_text
                                        else TOOL_RESULT_IMAGE_PLACEHOLDER
                                    )
                                    tool_image_parts.extend(
                                        {"type": "input_image", "image_url": url}  # mutable-ok: json content part
                                        for url in image_urls
                                    )
                            else:
                                output_text = str(inner)
                            # tool_result is a top-level item, not inside the message
                            input_items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": tool_use_id,
                                    "output": self._tool_result_output_value(output_text, tool_file_parts),
                                }
                            )
                    if tool_image_parts:
                        boundary_part = {  # mutable-ok: json content part
                            "type": "input_text",
                            "text": TOOL_RESULT_IMAGE_BOUNDARY,
                        }
                        input_items.append(
                            {  # mutable-ok: json input item
                                "type": "message",
                                "role": "user",
                                "content": [boundary_part, *tool_image_parts],  # mutable-ok: json content list
                            }
                        )
                    if user_parts:
                        input_items.append(
                            {
                                "type": "message",
                                "role": "user",
                                "content": user_parts,
                            }
                        )

            elif role == "assistant":
                if isinstance(content, str):
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    )
                elif isinstance(content, list):
                    blocks = tuple(block for block in content if isinstance(block, dict))
                    input_items.extend(
                        item
                        for _, group in groupby(enumerate(blocks), key=self._assistant_block_group_key)
                        if (item := self._assistant_group_to_input_item(tuple(block for _, block in group))) is not None
                    )
                    asst_parts: list[dict[str, Any]] = [  # mutable-ok: API message payload
                        {"type": "output_text", "text": block.get("text", "")}  # mutable-ok: API message payload
                        for block in blocks
                        if block.get("type") == "text"
                    ]
                    if asst_parts:
                        input_items.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": asst_parts,
                            }
                        )

        return input_items

    def translate_tools_to_responses_api(
        self,
        tools: list[AllAnthropicToolsValues],
    ) -> list[dict[str, Any]]:
        """Convert Anthropic tool definitions to Responses API function tools."""
        result: Final[list[dict[str, Any]]] = []
        for tool in tools:
            tool_dict = cast(dict[str, Any], tool)
            tool_type = tool_dict.get("type", "")
            tool_name = tool_dict.get("name", "")
            # web_search tool
            if (isinstance(tool_type, str) and tool_type.startswith("web_search")) or tool_name == "web_search":
                result.append({"type": "web_search_preview"})
                continue
            # Responses turns strict mode on when `strict` is omitted, silently rewriting
            # `required` to every property. Anthropic tools are non-strict unless asked.
            func_tool: dict[str, Any] = {
                "type": "function",
                "name": tool_name,
                "strict": bool(tool_dict.get("strict")),
            }
            if "description" in tool_dict:
                func_tool["description"] = tool_dict["description"]
            if "input_schema" in tool_dict:
                func_tool["parameters"] = tool_dict["input_schema"]
            result.append(func_tool)
        return result

    @staticmethod
    def translate_tool_choice_to_responses_api(
        tool_choice: AnthropicMessagesToolChoice,
    ) -> str | dict[str, Any]:
        """Convert Anthropic tool_choice to Responses API tool_choice."""
        tc_type: Final = tool_choice.get("type")
        if tc_type == "any":
            return "required"
        elif tc_type == "tool":
            return {"type": "function", "name": tool_choice.get("name", "")}
        elif tc_type == "none":
            return "none"
        return "auto"

    @staticmethod
    def translate_context_management_to_responses_api(
        context_management: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """
        Convert Anthropic context_management dict to OpenAI Responses API array format.

        Anthropic format: {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}
        OpenAI format:    [{"type": "compaction", "compact_threshold": 150000}]
        """
        if not isinstance(context_management, dict):
            return None

        edits: Final = context_management.get("edits", [])
        if not isinstance(edits, list):
            return None

        result: Final[list[dict[str, Any]]] = []
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            edit_type = edit.get("type", "")
            if edit_type == "compact_20260112":
                entry: dict[str, Any] = {"type": "compaction"}
                trigger = edit.get("trigger")
                if isinstance(trigger, dict) and trigger.get("value") is not None:
                    entry["compact_threshold"] = int(trigger["value"])
                result.append(entry)

        return result if result else None

    @staticmethod
    def translate_thinking_to_reasoning(
        thinking: dict[str, Any],
        output_config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Convert Anthropic thinking param to Responses API reasoning param.

        ``thinking.budget_tokens`` is bucketed via the shared
        ``reasoning_effort_from_thinking_budget`` thresholds. For adaptive
        thinking, uses ``output_config.effort`` if available, otherwise defaults
        to medium.
        """
        if not isinstance(thinking, dict):
            return None

        thinking_type: Final = thinking.get("type")

        if thinking_type == "adaptive":
            # Use output_config.effort if available
            effort = "medium"
            if isinstance(output_config, dict) and output_config.get("effort"):
                effort = output_config["effort"]
        elif thinking_type == "enabled":
            effort = reasoning_effort_from_thinking_budget(thinking.get("budget_tokens", 0))
        else:
            return None

        auto_summary: Final = is_reasoning_auto_summary_enabled()
        result: Final[dict[str, Any]] = {"effort": effort}
        summary: Final = thinking.get("summary")
        if summary:
            result["summary"] = summary
        elif auto_summary:
            result["summary"] = "detailed"
        return result

    def translate_request(
        self,
        anthropic_request: AnthropicMessagesRequest,
    ) -> dict[str, Any]:
        """
        Translate a full Anthropic /v1/messages request dict to
        litellm.responses() / litellm.aresponses() kwargs.
        """
        model: Final[str] = anthropic_request["model"]
        messages_list: Final = cast(
            list[AllAnthropicPassThroughMessageValues],
            anthropic_request["messages"],
        )

        input_items: Final = self.translate_messages_to_responses_input(messages_list)
        system: Final = anthropic_request.get("system")
        developer_parts: Final = (
            self._translate_midturn_system_content_to_responses(system)
            if isinstance(system, list)
            and any(isinstance(block, dict) and block.get("prompt_cache_breakpoint") is not None for block in system)
            else ()
        )
        if developer_parts:
            input_items.insert(
                0,
                {  # mutable-ok: API message payload
                    "type": "message",
                    "role": "developer",
                    "content": developer_parts,
                },
            )

        responses_kwargs: Final[dict[str, Any]] = {
            "model": model,
            "input": input_items,
        }

        if system and not developer_parts:
            if isinstance(system, str):
                responses_kwargs["instructions"] = system
            elif isinstance(system, list):
                responses_kwargs["instructions"] = "\n".join(
                    filter(None, (b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"))
                )

        # max_tokens -> max_output_tokens
        max_tokens: Final = anthropic_request.get("max_tokens")
        if max_tokens:
            responses_kwargs["max_output_tokens"] = max_tokens

        # temperature / top_p passed through
        if "temperature" in anthropic_request:
            responses_kwargs["temperature"] = anthropic_request["temperature"]
        if "top_p" in anthropic_request:
            responses_kwargs["top_p"] = anthropic_request["top_p"]

        # tools
        tools: Final = anthropic_request.get("tools")
        if tools:
            responses_kwargs["tools"] = self.translate_tools_to_responses_api(
                cast(list[AllAnthropicToolsValues], tools)
            )

        # tool_choice
        tool_choice: Final = anthropic_request.get("tool_choice")
        if tool_choice:
            responses_kwargs["tool_choice"] = self.translate_tool_choice_to_responses_api(
                cast(AnthropicMessagesToolChoice, tool_choice)
            )

        # thinking -> reasoning
        thinking: Final = anthropic_request.get("thinking")
        if isinstance(thinking, dict):
            output_config = anthropic_request.get("output_config")
            reasoning: Final = self.translate_thinking_to_reasoning(
                thinking,
                output_config=cast(dict[str, Any] | None, output_config),
            )
            if reasoning:
                responses_kwargs["reasoning"] = reasoning

        # output_format / output_config.format -> text format
        # output_format: {"type": "json_schema", "schema": {...}}
        # output_config: {"format": {"type": "json_schema", "schema": {...}}}
        output_format: Any = anthropic_request.get("output_format")
        output_config = anthropic_request.get("output_config")
        if not isinstance(output_format, dict) and isinstance(output_config, dict):
            output_format = output_config.get("format")
        if isinstance(output_format, dict) and output_format.get("type") == "json_schema":
            schema: Final = output_format.get("schema")
            if schema:
                responses_kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "structured_output",
                        "schema": schema,
                        "strict": output_format.get("strict", False),
                    }
                }

        # context_management: Anthropic dict -> OpenAI array
        context_management: Final = anthropic_request.get("context_management")
        if isinstance(context_management, dict):
            openai_cm: Final = self.translate_context_management_to_responses_api(context_management)
            if openai_cm is not None:
                responses_kwargs["context_management"] = openai_cm

        # metadata user_id -> user and prompt_cache_key
        metadata: Final = anthropic_request.get("metadata")
        if isinstance(metadata, dict) and "user_id" in metadata:
            responses_kwargs["user"] = str(metadata["user_id"])[:64]
            prompt_cache_key: Final = prompt_cache_key_from_user_id(metadata["user_id"])
            if prompt_cache_key is not None:
                responses_kwargs["prompt_cache_key"] = prompt_cache_key

        return responses_kwargs

    # ------------------------------------------------------------------ #
    # Response translation: Responses API -> Anthropic                    #
    # ------------------------------------------------------------------ #

    def translate_response(
        self,
        response: ResponsesAPIResponse,
    ) -> AnthropicMessagesResponse:
        """
        Translate an OpenAI ResponsesAPIResponse to AnthropicMessagesResponse.
        """
        from openai.types.responses import (
            ResponseFunctionToolCall,
            ResponseOutputMessage,
            ResponseReasoningItem,
        )

        content: Final[list[dict[str, Any]]] = []
        stop_reason: AnthropicFinishReason = "end_turn"

        for item in response.output:
            if isinstance(item, ResponseReasoningItem):
                content.extend(self._thinking_blocks_from_reasoning_item(item.summary))

            elif isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if getattr(part, "type", None) == "output_text":
                        content.append(
                            AnthropicResponseContentBlockText(type="text", text=getattr(part, "text", "")).model_dump()
                        )

            elif isinstance(item, ResponseFunctionToolCall):
                try:
                    input_data = json.loads(item.arguments) if item.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    input_data = {}
                content.append(
                    AnthropicResponseContentBlockToolUse(
                        type="tool_use",
                        id=item.call_id or item.id or "",
                        name=item.name,
                        input=input_data,
                    ).model_dump()
                )
                stop_reason = "tool_use"

            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "message":
                    for part in item.get("content", []):
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            content.append(
                                AnthropicResponseContentBlockText(type="text", text=part.get("text", "")).model_dump()
                            )
                elif item_type == "reasoning":
                    content.extend(
                        self._thinking_blocks_from_reasoning_item(
                            cast(Iterable[object], item.get("summary") or ()),  # cast-ok: untyped provider json
                        )
                    )
                elif item_type == "function_call":
                    try:
                        input_data = json.loads(item.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        input_data = {}
                    content.append(
                        AnthropicResponseContentBlockToolUse(
                            type="tool_use",
                            id=item.get("call_id") or item.get("id", ""),
                            name=item.get("name", ""),
                            input=input_data,
                        ).model_dump()
                    )
                    stop_reason = "tool_use"

        # status -> stop_reason override
        if response.status == "incomplete":
            stop_reason = "max_tokens"

        anthropic_usage: Final = self.translate_responses_api_usage_to_anthropic_usage(response.usage)

        return AnthropicMessagesResponse(
            id=response.id,
            type="message",
            role="assistant",
            model=response.model or "unknown-model",
            stop_sequence=None,
            usage=anthropic_usage,
            content=content,
            stop_reason=stop_reason,
        )
