from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Final, TypeVar

from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicUsage
from litellm.types.llms.openai import AllMessageValues, ResponseAPIUsage


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


def _blocked_usage_obj(original_response: object) -> object:
    if isinstance(original_response, dict):
        return original_response.get("usage")
    if original_response is not None and not isinstance(original_response, list):
        return getattr(original_response, "usage", None)
    return None


def _usage_tokens(usage_obj: object, key: str, fallback_key: str) -> int:
    if isinstance(usage_obj, dict):
        return int(usage_obj.get(key, usage_obj.get(fallback_key, 0)) or 0)
    return int(getattr(usage_obj, key, getattr(usage_obj, fallback_key, 0)) or 0)


def blocked_response_usage(original_response: Any | None) -> AnthropicUsage:
    """
    Token usage for a synthetic guardrail-blocked response.

    A post-call block replaces the LLM's response with the violation message,
    but the upstream call already consumed tokens -- report that real usage
    (carried on ``ModifyResponseException.original_response``) rather than
    discarding it. Pre-call blocks never invoked the LLM (no original_response),
    so usage is zero.
    """
    if isinstance(original_response, list):
        stream_usage: Final = _usage_from_anthropic_stream_chunks(original_response)
        if stream_usage is not None:
            return stream_usage

    usage_obj: Final = _blocked_usage_obj(original_response)
    return AnthropicUsage(
        input_tokens=_usage_tokens(usage_obj, "input_tokens", "prompt_tokens"),
        output_tokens=_usage_tokens(usage_obj, "output_tokens", "completion_tokens"),
    )


def blocked_responses_api_usage(original_response: object) -> ResponseAPIUsage:
    """
    Token usage for a synthetic guardrail-blocked /v1/responses reply.

    Same contract as ``blocked_response_usage`` in Responses API shape: a
    native ``ResponsesAPIResponse`` usage passes through unchanged, a bridged
    chat ``ModelResponse`` usage maps prompt/completion tokens to input/output
    tokens, and a pre-call block (no original_response) reports zeros.
    """
    usage_obj: Final = _blocked_usage_obj(original_response)
    if isinstance(usage_obj, ResponseAPIUsage):
        return usage_obj

    input_tokens: Final = _usage_tokens(usage_obj, "input_tokens", "prompt_tokens")
    output_tokens: Final = _usage_tokens(usage_obj, "output_tokens", "completion_tokens")
    total_tokens: Final = _usage_tokens(usage_obj, "total_tokens", "total_tokens")
    return ResponseAPIUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or input_tokens + output_tokens,
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


def effective_scan_only_tool_results_for_guardrail(guardrail_to_apply: object) -> bool:
    return getattr(guardrail_to_apply, "scan_only_tool_results", None) is True


def role_out_of_guardrail_scope(
    role: str,
    *,
    skip_system_message: bool,
    skip_tool_message: bool,
    scan_only_tool_results: bool = False,
) -> bool:
    if skip_system_message and role == "system":
        return True
    if skip_tool_message and role == "tool":
        return True
    return scan_only_tool_results and role not in ("tool", "function")


def scoped_structured_message_indices(
    messages: Sequence[AllMessageValues],
    *,
    scan_only_tool_results: bool,
    skip_system: bool,
    skip_tool: bool,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, message in enumerate(messages)
        if not role_out_of_guardrail_scope(
            _message_role(message),
            skip_system_message=skip_system,
            skip_tool_message=skip_tool,
            scan_only_tool_results=scan_only_tool_results,
        )
    )


ToolT = TypeVar("ToolT")


def openai_tool_name(tool: object) -> str | None:
    if not isinstance(tool, dict):
        return None
    function: Final = tool.get("function")
    if isinstance(function, dict):
        function_name: Final = function.get("name")
        return function_name if isinstance(function_name, str) else None
    flat_name: Final = tool.get("name")
    return flat_name if isinstance(flat_name, str) else None


def anthropic_tool_name(tool: object) -> str | None:
    name: Final = tool.get("name") if isinstance(tool, dict) else None
    return name if isinstance(name, str) else None


def merge_returned_tools_into_request_tools(
    request_tools: Sequence[ToolT] | None,
    returned_tools: Sequence[ToolT],
    tool_name: Callable[[ToolT], str | None],
) -> list[ToolT]:
    """Union of the request's tools and guardrail-returned tools, keyed by name.

    Under ``scan_only_tool_results`` the guardrail never saw the request's
    tools, so a returned list can neither replace them (it would drop every
    user-defined function) nor be discarded (it may carry a tool the guardrail
    synthesized and told the model to call, like Compresr's retrieve tool).
    Keep every request tool and append only returned tools whose names aren't
    already taken by a request tool or an earlier returned tool.
    """
    originals: Final = tuple(request_tools or ())
    taken_names: Final = frozenset(name for tool in originals if (name := tool_name(tool)) is not None)
    additions: Final = tuple(
        tool
        for index, tool in enumerate(returned_tools)
        if (name := tool_name(tool)) not in taken_names
        and (name is None or all(tool_name(earlier) != name for earlier in returned_tools[:index]))
    )
    return [*originals, *additions]


def merge_guardrailed_scoped_messages(
    full_messages: Sequence[AllMessageValues],
    scoped_indices: Sequence[int],
    guardrailed_scoped: Sequence[AllMessageValues],
) -> list[AllMessageValues]:
    """Substitute guardrail-returned messages back into the full conversation.

    Guardrails only ever see the scoped subset of messages, so a replacement
    list they hand back describes that subset, not the whole request. Writing
    it over ``data["messages"]`` wholesale would silently drop every
    out-of-scope message (system prompt, prior turns). Instead, swap each
    returned message into the position its scoped original came from; extra
    returned messages land after the last scoped position, and scoped
    originals without a counterpart are treated as removed by the guardrail.
    When nothing was filtered out this degenerates to the returned list
    itself, preserving wholesale-replacement behavior for unscoped guardrails.
    """
    replacements: Final = dict(zip(scoped_indices, guardrailed_scoped))
    removed: Final = frozenset(scoped_indices[len(guardrailed_scoped) :])
    appended: Final = tuple(guardrailed_scoped[len(scoped_indices) :])
    last_scoped_index: Final = scoped_indices[-1] if scoped_indices else None

    def _merged() -> Iterator[AllMessageValues]:
        for index, message in enumerate(full_messages):
            if index in removed:
                continue
            yield replacements.get(index, message)
            if index == last_scoped_index:
                yield from appended

    return list(_merged())
