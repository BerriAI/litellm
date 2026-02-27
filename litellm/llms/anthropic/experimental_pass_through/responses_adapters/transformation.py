"""
Transformation layer: Anthropic /v1/messages <-> OpenAI Responses API.

This module owns all format conversions for the direct v1/messages -> Responses API
path used for OpenAI and Azure models.
"""

import json
from typing import Any, Dict, List, Optional, Union, cast

from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthopicMessagesAssistantMessageParam,
    AnthropicFinishReason,
    AnthropicMessagesRequest,
    AnthropicMessagesToolChoice,
    AnthropicMessagesUserMessageParam,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockThinking,
    AnthropicResponseContentBlockToolUse,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from litellm.types.llms.openai import ResponsesAPIResponse


class LiteLLMAnthropicToResponsesAPIAdapter:
    """
    Converts Anthropic /v1/messages requests to OpenAI Responses API format and
    converts Responses API responses back to Anthropic format.
    """

    # ------------------------------------------------------------------ #
    # Request translation: Anthropic -> Responses API                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _translate_anthropic_image_source_to_url(source: dict) -> Optional[str]:
        """Convert Anthropic image source to a URL string."""
        source_type = source.get("type")
        if source_type == "base64":
            media_type = source.get("media_type", "image/jpeg")
            data = source.get("data", "")
            return f"data:{media_type};base64,{data}" if data else None
        elif source_type == "url":
            return source.get("url")
        return None

    def translate_messages_to_responses_input(
        self,
        messages: List[
            Union[
                AnthropicMessagesUserMessageParam,
                AnthopicMessagesAssistantMessageParam,
            ]
        ],
    ) -> List[Dict[str, Any]]:
        """
        Convert Anthropic messages list to Responses API `input` items.

        Mapping:
          user text          -> message(role=user, input_text)
          user image         -> message(role=user, input_image)
          user tool_result   -> function_call_output
          assistant text     -> message(role=assistant, output_text)
          assistant tool_use -> function_call
        """
        input_items: List[Dict[str, Any]] = []

        for m in messages:
            role = m["role"]
            content = m.get("content")

            if role == "user":
                if isinstance(content, str):
                    input_items.append({
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": content}],
                    })
                elif isinstance(content, list):
                    user_parts: List[Dict[str, Any]] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            user_parts.append({"type": "input_text", "text": block.get("text", "")})
                        elif btype == "image":
                            url = self._translate_anthropic_image_source_to_url(block.get("source", {}))
                            if url:
                                user_parts.append({"type": "input_image", "image_url": url})
                        elif btype == "tool_result":
                            tool_use_id = block.get("tool_use_id", "")
                            inner = block.get("content")
                            if inner is None:
                                output_text = ""
                            elif isinstance(inner, str):
                                output_text = inner
                            elif isinstance(inner, list):
                                parts = [
                                    c.get("text", "")
                                    for c in inner
                                    if isinstance(c, dict) and c.get("type") == "text"
                                ]
                                output_text = "\n".join(parts)
                            else:
                                output_text = str(inner)
                            # tool_result is a top-level item, not inside the message
                            input_items.append({
                                "type": "function_call_output",
                                "call_id": tool_use_id,
                                "output": output_text,
                            })
                    if user_parts:
                        input_items.append({
                            "type": "message",
                            "role": "user",
                            "content": user_parts,
                        })

            elif role == "assistant":
                if isinstance(content, str):
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    })
                elif isinstance(content, list):
                    asst_parts: List[Dict[str, Any]] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            asst_parts.append({"type": "output_text", "text": block.get("text", "")})
                        elif btype == "tool_use":
                            # tool_use becomes a top-level function_call item
                            input_items.append({
                                "type": "function_call",
                                "call_id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            })
                        elif btype == "thinking":
                            thinking_text = block.get("thinking", "")
                            if thinking_text:
                                asst_parts.append({"type": "output_text", "text": thinking_text})
                    if asst_parts:
                        input_items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": asst_parts,
                        })

        return input_items

    def translate_tools_to_responses_api(
        self,
        tools: List[AllAnthropicToolsValues],
    ) -> List[Dict[str, Any]]:
        """Convert Anthropic tool definitions to Responses API function tools."""
        result: List[Dict[str, Any]] = []
        for tool in tools:
            tool_dict = cast(Dict[str, Any], tool)
            tool_type = tool_dict.get("type", "")
            tool_name = tool_dict.get("name", "")
            # web_search tool
            if (isinstance(tool_type, str) and tool_type.startswith("web_search")) or tool_name == "web_search":
                result.append({"type": "web_search_preview"})
                continue
            func_tool: Dict[str, Any] = {"type": "function", "name": tool_name}
            if "description" in tool_dict:
                func_tool["description"] = tool_dict["description"]
            if "input_schema" in tool_dict:
                func_tool["parameters"] = tool_dict["input_schema"]
            result.append(func_tool)
        return result

    @staticmethod
    def translate_tool_choice_to_responses_api(
        tool_choice: AnthropicMessagesToolChoice,
    ) -> Dict[str, Any]:
        """Convert Anthropic tool_choice to Responses API tool_choice."""
        tc_type = tool_choice.get("type")
        if tc_type == "any":
            return {"type": "required"}
        elif tc_type == "tool":
            return {"type": "function", "name": tool_choice.get("name", "")}
        return {"type": "auto"}

    @staticmethod
    def translate_context_management_to_responses_api(
        context_management: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Convert Anthropic context_management dict to OpenAI Responses API array format.

        Anthropic format: {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}
        OpenAI format:    [{"type": "compaction", "compact_threshold": 150000}]
        """
        if not isinstance(context_management, dict):
            return None

        edits = context_management.get("edits", [])
        if not isinstance(edits, list):
            return None

        result: List[Dict[str, Any]] = []
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            edit_type = edit.get("type", "")
            if edit_type == "compact_20260112":
                entry: Dict[str, Any] = {"type": "compaction"}
                trigger = edit.get("trigger")
                if isinstance(trigger, dict) and trigger.get("value") is not None:
                    entry["compact_threshold"] = int(trigger["value"])
                result.append(entry)

        return result if result else None

    @staticmethod
    def translate_thinking_to_reasoning(thinking: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert Anthropic thinking param to Responses API reasoning param.

        thinking.budget_tokens maps to reasoning effort:
          >= 10000 -> high, >= 5000 -> medium, >= 2000 -> low, < 2000 -> minimal
        """
        if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
            return None
        budget = thinking.get("budget_tokens", 0)
        if budget >= 10000:
            effort = "high"
        elif budget >= 5000:
            effort = "medium"
        elif budget >= 2000:
            effort = "low"
        else:
            effort = "minimal"
        return {"effort": effort, "summary": "detailed"}

    def translate_request(
        self,
        anthropic_request: AnthropicMessagesRequest,
    ) -> Dict[str, Any]:
        """
        Translate a full Anthropic /v1/messages request dict to
        litellm.responses() / litellm.aresponses() kwargs.
        """
        model: str = anthropic_request["model"]
        messages_list = cast(
            List[Union[AnthropicMessagesUserMessageParam, AnthopicMessagesAssistantMessageParam]],
            anthropic_request["messages"],
        )

        responses_kwargs: Dict[str, Any] = {
            "model": model,
            "input": self.translate_messages_to_responses_input(messages_list),
        }

        # system -> instructions
        system = anthropic_request.get("system")
        if system:
            if isinstance(system, str):
                responses_kwargs["instructions"] = system
            elif isinstance(system, list):
                text_parts = [
                    b.get("text", "")
                    for b in system
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                responses_kwargs["instructions"] = "\n".join(filter(None, text_parts))

        # max_tokens -> max_output_tokens
        max_tokens = anthropic_request.get("max_tokens")
        if max_tokens:
            responses_kwargs["max_output_tokens"] = max_tokens

        # temperature / top_p passed through
        if "temperature" in anthropic_request:
            responses_kwargs["temperature"] = anthropic_request["temperature"]
        if "top_p" in anthropic_request:
            responses_kwargs["top_p"] = anthropic_request["top_p"]

        # tools
        tools = anthropic_request.get("tools")
        if tools:
            responses_kwargs["tools"] = self.translate_tools_to_responses_api(
                cast(List[AllAnthropicToolsValues], tools)
            )

        # tool_choice
        tool_choice = anthropic_request.get("tool_choice")
        if tool_choice:
            responses_kwargs["tool_choice"] = self.translate_tool_choice_to_responses_api(
                cast(AnthropicMessagesToolChoice, tool_choice)
            )

        # thinking -> reasoning
        thinking = anthropic_request.get("thinking")
        if isinstance(thinking, dict):
            reasoning = self.translate_thinking_to_reasoning(thinking)
            if reasoning:
                responses_kwargs["reasoning"] = reasoning

        # output_format / output_config.format -> text format
        # output_format: {"type": "json_schema", "schema": {...}}
        # output_config: {"format": {"type": "json_schema", "schema": {...}}}
        output_format = anthropic_request.get("output_format")
        output_config = anthropic_request.get("output_config")
        if not isinstance(output_format, dict) and isinstance(output_config, dict):
            output_format = output_config.get("format")
        if isinstance(output_format, dict) and output_format.get("type") == "json_schema":
            schema = output_format.get("schema")
            if schema:
                responses_kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "structured_output",
                        "schema": schema,
                        "strict": True,
                    }
                }

        # context_management: Anthropic dict -> OpenAI array
        context_management = anthropic_request.get("context_management")
        if isinstance(context_management, dict):
            openai_cm = self.translate_context_management_to_responses_api(context_management)
            if openai_cm is not None:
                responses_kwargs["context_management"] = openai_cm

        # metadata user_id -> user
        metadata = anthropic_request.get("metadata")
        if isinstance(metadata, dict) and "user_id" in metadata:
            responses_kwargs["user"] = str(metadata["user_id"])[:64]

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

        from litellm.types.llms.openai import ResponseAPIUsage

        content: List[Dict[str, Any]] = []
        stop_reason: AnthropicFinishReason = "end_turn"

        for item in response.output:
            if isinstance(item, ResponseReasoningItem):
                for summary in item.summary:
                    text = getattr(summary, "text", "")
                    if text:
                        content.append(
                            AnthropicResponseContentBlockThinking(
                                type="thinking",
                                thinking=text,
                                signature=None,
                            ).model_dump()
                        )

            elif isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if getattr(part, "type", None) == "output_text":
                        content.append(
                            AnthropicResponseContentBlockText(
                                type="text", text=getattr(part, "text", "")
                            ).model_dump()
                        )

            elif isinstance(item, ResponseFunctionToolCall):
                try:
                    input_data = json.loads(item.arguments) if item.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    input_data = {}
                content.append(
                    AnthropicResponseContentBlockToolUse(
                        type="tool_use",
                        id=item.call_id or item.id,
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
                                AnthropicResponseContentBlockText(
                                    type="text", text=part.get("text", "")
                                ).model_dump()
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

        # usage
        raw_usage: Optional[ResponseAPIUsage] = response.usage
        input_tokens = int(getattr(raw_usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(raw_usage, "output_tokens", 0) or 0)

        anthropic_usage = AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return AnthropicMessagesResponse(
            id=response.id,
            type="message",
            role="assistant",
            model=response.model or "unknown-model",
            stop_sequence=None,
            usage=anthropic_usage,  # type: ignore
            content=content,  # type: ignore
            stop_reason=stop_reason,
        )
