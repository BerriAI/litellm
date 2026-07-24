"""
Common utilities for the Command Code provider.

Wire protocol details were derived from https://github.com/BerriAI/litellm/issues/27582
and the MIT-licensed reference implementation at
https://github.com/patlux/pi-commandcode-provider.
"""

from __future__ import annotations

import json

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.command_code import CommandCodeTool, CommandCodeUsage
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

COMMAND_CODE_API_BASE = "https://api.commandcode.ai"

# Version string the reference implementation currently sends. The API
# requires the x-command-code-version header to be present.
COMMAND_CODE_CLI_VERSION = "0.29.0"

# The generation endpoint rejects larger values; mirrors the reference
# implementation's DEFAULT_GENERATE_MAX_TOKENS.
COMMAND_CODE_DEFAULT_MAX_TOKENS = 65536


class CommandCodeError(BaseLLMException):
    """Exception class for Command Code API errors."""

    pass


def get_command_code_headers(api_key: str) -> dict[str, str]:
    """Build the required request headers for the Command Code API."""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "x-command-code-version": COMMAND_CODE_CLI_VERSION,
        "x-cli-environment": "production",
    }


def map_command_code_finish_reason(finish_reason: str | None) -> str:
    """Map a Command Code finishReason to an OpenAI finish_reason."""
    if finish_reason == "tool-calls":
        return "tool_calls"
    if finish_reason in ("length", "max_tokens", "max-tokens", "max_output_tokens"):
        return "length"
    return "stop"


def parse_stream_event_line(line: str) -> dict | None:
    """Parse one line of the newline-delimited event stream.

    Skips empty lines, comment lines (``:``), ``event:`` lines and
    ``[DONE]`` markers; strips a leading ``data:`` prefix before JSON
    decoding. Returns None for anything that is not a JSON object.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(":") or stripped.startswith("event:"):
        return None
    if stripped.startswith("data:"):
        stripped = stripped[len("data:") :].strip()
    if not stripped or stripped == "[DONE]":
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def parse_tool_call_input(value: object) -> dict:
    """Normalize a tool-call ``input`` field to a dict.

    The API may send tool call arguments as a dict or as a JSON-encoded
    string.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def flatten_system_messages(
    messages: list[AllMessageValues],
) -> tuple[str, list[AllMessageValues]]:
    """Split system messages out of the message list.

    Command Code takes the system prompt as a flat ``params.system`` string
    rather than as messages. Returns (system_text, remaining_messages).
    """
    system_parts: list[str] = []
    remaining: list[AllMessageValues] = []
    for message in messages:
        if message.get("role") in ("system", "developer"):
            content = convert_content_list_to_str(message)
            if content:
                system_parts.append(content)
        else:
            remaining.append(message)
    return "\n\n".join(system_parts), remaining


def _paired_tool_call_ids(messages: list[AllMessageValues]) -> set[str]:
    """Return tool call ids that have both a call and a matching result.

    The API rejects unpaired tool calls, so both orphan tool calls and
    orphan tool results are dropped before sending.
    """
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    call_ids.add(tool_call_id)
        elif message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                result_ids.add(tool_call_id)
    return call_ids & result_ids


def _user_message_content(message: AllMessageValues) -> object:
    """Translate user message content, keeping strings as-is.

    List content is reduced to Command Code text parts; non-text parts
    (e.g. images) are dropped because the generation endpoint does not
    document support for them.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[dict] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
        return parts
    return content if content is not None else ""


def transform_messages_to_command_code(
    messages: list[AllMessageValues],
) -> list[dict]:
    """Translate OpenAI chat messages to Command Code message format."""
    paired_ids = _paired_tool_call_ids(messages)
    out: list[dict] = []

    for message in messages:
        role = message.get("role")
        if role == "user":
            out.append({"role": "user", "content": _user_message_content(message)})
        elif role == "assistant":
            parts: list[dict] = []
            reasoning_content = message.get("reasoning_content")
            if reasoning_content:
                parts.append({"type": "reasoning", "text": reasoning_content})
            content = message.get("content")
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append({"type": "text", "text": part.get("text", "")})
            for tool_call in message.get("tool_calls") or []:
                tool_call_id = tool_call.get("id") or ""
                if tool_call_id not in paired_ids:
                    continue
                function = tool_call.get("function") or {}
                parts.append(
                    {
                        "type": "tool-call",
                        "toolCallId": tool_call_id,
                        "toolName": function.get("name") or "",
                        "input": parse_tool_call_input(function.get("arguments")),
                    }
                )
            if parts:
                out.append({"role": "assistant", "content": parts})
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id or tool_call_id not in paired_ids:
                continue
            out.append(
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": tool_call_id,
                            "toolName": message.get("name") or "",
                            "output": {
                                "type": "text",
                                "value": convert_content_list_to_str(message),
                            },
                        }
                    ],
                }
            )

    return out


def transform_tools_to_command_code(tools: list[dict]) -> list[CommandCodeTool]:
    """Translate OpenAI tool definitions to Command Code format.

    Command Code uses a flat ``input_schema`` field, not OpenAI's nested
    ``function.parameters``.
    """
    out: list[CommandCodeTool] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function = tool.get("function") or {}
        out.append(
            CommandCodeTool(
                type="function",
                name=function.get("name") or "",
                description=function.get("description") or "",
                input_schema=function.get("parameters") or {},
            )
        )
    return out


def usage_from_finish_event(total_usage: CommandCodeUsage) -> Usage:
    """Build a LiteLLM Usage object from a finish event's ``totalUsage``.

    ``inputTokens`` excludes cached tokens (Anthropic-style accounting), so
    cache reads/writes are added into prompt_tokens, mirroring LiteLLM's
    Anthropic transformation.
    """
    input_tokens = total_usage.get("inputTokens") or 0
    output_tokens = total_usage.get("outputTokens") or 0
    details = total_usage.get("inputTokenDetails") or {}
    cache_read_tokens = details.get("cacheReadTokens") or 0
    cache_write_tokens = details.get("cacheWriteTokens") or 0

    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=cache_read_tokens,
            cache_creation_tokens=cache_write_tokens,
            text_tokens=input_tokens,
        ),
        cache_read_input_tokens=cache_read_tokens,
        cache_creation_input_tokens=cache_write_tokens,
    )
