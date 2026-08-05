from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final

from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicUsage
from litellm.types.llms.openai import AllMessageValues


def _anthropic_stream_chunk_events(item: Any) -> list[dict]:
    if isinstance(item, dict):
        return [item]
    if isinstance(item, bytes):
        chunk = item.decode("utf-8", errors="replace")
    elif isinstance(item, str):
        chunk = item
    else:
        return []

    events: Final[list[dict]] = []
    for block in chunk.split("\n\n"):
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                continue
            payload = stripped[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
    return events


def _usage_from_anthropic_stream_chunks(original_response: list[Any]) -> AnthropicUsage | None:
    input_tokens = 0
    output_tokens = 0
    found_usage = False

    for item in original_response:
        for event in _anthropic_stream_chunk_events(item):
            event_type = event.get("type")
            if event_type == "message_start":
                message = event.get("message") or {}
                usage_obj = message.get("usage") or {}
            elif event_type == "message_delta":
                usage_obj = event.get("usage") or {}
            else:
                usage_obj = {}
            if not isinstance(usage_obj, dict):
                continue
            if usage_obj.get("input_tokens") is not None:
                input_tokens = int(usage_obj.get("input_tokens") or 0)
                found_usage = True
            if usage_obj.get("output_tokens") is not None:
                output_tokens = int(usage_obj.get("output_tokens") or 0)
                found_usage = True

    if not found_usage:
        return None
    return AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def blocked_response_usage(original_response: Any | None) -> AnthropicUsage:
    """
    Token usage for a synthetic guardrail-blocked response.

    A post-call block replaces the LLM's response with the violation message,
    but the upstream call already consumed tokens -- report that real usage
    (carried on ``ModifyResponseException.original_response``) rather than
    discarding it. Pre-call blocks never invoked the LLM (no original_response),
    so usage is zero.
    """
    usage_obj: Any = None
    if isinstance(original_response, list):
        stream_usage: Final = _usage_from_anthropic_stream_chunks(original_response)
        if stream_usage is not None:
            return stream_usage
    elif isinstance(original_response, dict):
        usage_obj = original_response.get("usage")
    elif original_response is not None:
        usage_obj = getattr(original_response, "usage", None)

    def _tokens(key: str, fallback_key: str) -> int:
        if isinstance(usage_obj, dict):
            return int(usage_obj.get(key, usage_obj.get(fallback_key, 0)) or 0)
        return int(getattr(usage_obj, key, getattr(usage_obj, fallback_key, 0)) or 0)

    return AnthropicUsage(
        input_tokens=_tokens("input_tokens", "prompt_tokens"),
        output_tokens=_tokens("output_tokens", "completion_tokens"),
    )


def effective_skip_system_message_for_guardrail(guardrail_to_apply: Any) -> bool:
    per: Final = getattr(guardrail_to_apply, "skip_system_message_in_guardrail", None)
    if per is not None:
        return bool(per)
    import litellm

    return bool(getattr(litellm, "skip_system_message_in_guardrail", False))


def effective_skip_tool_message_for_guardrail(guardrail_to_apply: Any) -> bool:
    per: Final = getattr(guardrail_to_apply, "skip_tool_message_in_guardrail", None)
    if per is not None:
        return bool(per)
    import litellm

    return bool(getattr(litellm, "skip_tool_message_in_guardrail", False))


def _message_role(message: AllMessageValues) -> str:
    return str((message or {}).get("role") or "").lower()


def openai_messages_without_system(
    messages: Sequence[AllMessageValues],
) -> tuple[AllMessageValues, ...]:
    return tuple(m for m in messages if _message_role(m) != "system")


def openai_messages_without_tool(
    messages: Sequence[AllMessageValues],
) -> tuple[AllMessageValues, ...]:
    return tuple(m for m in messages if _message_role(m) != "tool")


def openai_messages_only_tool(
    messages: Sequence[AllMessageValues],
) -> tuple[AllMessageValues, ...]:
    return tuple(m for m in messages if _message_role(m) == "tool")


def effective_scan_only_tool_results_for_guardrail(guardrail_to_apply: Any) -> bool:
    return getattr(guardrail_to_apply, "scan_only_tool_results", None) is True


def role_out_of_guardrail_scope(
    role: str,
    *,
    skip_system_message: bool,
    skip_tool_message: bool,
    scan_only_tool_results: bool = False,
) -> bool:
    """Whether a message role falls outside what this guardrail is configured to scan."""
    if skip_system_message and role == "system":
        return True
    if skip_tool_message and role == "tool":
        return True
    return scan_only_tool_results and role != "tool"


def filtered_structured_messages(
    messages: Sequence[AllMessageValues],
    *,
    scan_only_tool_results: bool,
    skip_system: bool,
    skip_tool: bool,
) -> tuple[AllMessageValues, ...]:
    """Narrow the structured messages a guardrail sees, per its skip/scope settings."""
    scoped: Final = openai_messages_only_tool(messages) if scan_only_tool_results else tuple(messages)
    without_system: Final = openai_messages_without_system(scoped) if skip_system else scoped
    return openai_messages_without_tool(without_system) if skip_tool else without_system
