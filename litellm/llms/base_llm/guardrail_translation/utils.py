from __future__ import annotations

import json
from typing import Any, List, Optional

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

    events: list[dict] = []
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


def _usage_from_anthropic_stream_chunks(original_response: list[Any]) -> Optional[AnthropicUsage]:
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


def blocked_response_usage(original_response: Optional[Any]) -> AnthropicUsage:
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
        stream_usage = _usage_from_anthropic_stream_chunks(original_response)
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
    per = getattr(guardrail_to_apply, "skip_system_message_in_guardrail", None)
    if per is not None:
        return bool(per)
    import litellm

    return bool(getattr(litellm, "skip_system_message_in_guardrail", False))


def effective_skip_tool_message_for_guardrail(guardrail_to_apply: Any) -> bool:
    per = getattr(guardrail_to_apply, "skip_tool_message_in_guardrail", None)
    if per is not None:
        return bool(per)
    import litellm

    return bool(getattr(litellm, "skip_tool_message_in_guardrail", False))


def openai_messages_without_system(
    messages: List[AllMessageValues],
) -> List[AllMessageValues]:
    return [m for m in messages if str((m or {}).get("role") or "").lower() != "system"]


def openai_messages_without_tool(
    messages: List[AllMessageValues],
) -> List[AllMessageValues]:
    return [m for m in messages if str((m or {}).get("role") or "").lower() != "tool"]
