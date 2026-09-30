"""
Deep Agents harness config: LangChain `deepagents` running in your Python process.

Pure translation only: model kwargs (gateway mode uses `litellm_proxy/<group>` with the same
attribution headers the CLI endpoint adds), permission and tool filtering, and LangGraph
stream chunks to events. `litellm/harness/handlers/deepagents_handler.py` builds the agent,
streams it, answers approvals and counts usage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from litellm.harness.options import DeepAgentsOptions
from litellm.harness.types import (
    Capabilities,
    Event,
    Harness,
    Reasoning,
    Text,
    ToolCall,
    ToolResult,
)
from litellm.llms.base_llm.harness.transformation import BaseHarnessConfig

if TYPE_CHECKING:
    from litellm.harness.context import SessionContext

INSTALL_HINT: Final = "Deep Agents is not installed. Run: pip install deepagents langchain-litellm"
SKILLS_DIR: Final = ".deepagents/skills"
# Graph supersteps per agent turn (model node, tools node, middleware hooks) for recursion_limit.
DEEPAGENTS_STEPS_PER_TURN: Final = 6
DEEPAGENTS_BASE_RECURSION_LIMIT: Final = 25
DEEPAGENTS_DEFAULT_RECURSION_LIMIT: Final = 1000
_MODEL_NODE: Final = "model"
# Only these graph nodes produce new messages; middleware hooks may re-emit history.
_EVENT_NODES: Final = frozenset({"model", "tools"})

NORMALIZED_TO_NATIVE: Final[Mapping[str, str]] = {
    "read": "read_file",
    "write": "write_file",
    "edit": "edit_file",
    "bash": "execute",
}
NATIVE_TO_NORMALIZED: Final[Mapping[str, str]] = {v: k for k, v in NORMALIZED_TO_NATIVE.items()}
BUILTIN_TOOLS: Final = frozenset(
    {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
        "write_todos",
        "task",
    }
)
WRITE_TOOLS: Final = frozenset({"write_file", "edit_file", "delete"})
EXECUTE_TOOLS: Final = frozenset({"execute"})
APPROVAL_TOOLS: Final = WRITE_TOOLS | EXECUTE_TOOLS
_APPROVAL_DECISIONS: Final = ["approve", "reject"]


# ---------------------------------------------------------------------------
# Pure helpers (unit tested directly; kept module-level so they port cleanly)
# ---------------------------------------------------------------------------


def gateway_headers(ctx: SessionContext) -> dict[str, str]:
    """Same attribution headers the session endpoint adds for CLI harnesses."""
    headers = {"x-litellm-tags": f"harness,{ctx.harness.value}"}
    if ctx.metadata:
        headers["x-litellm-spend-logs-metadata"] = json.dumps(dict(ctx.metadata), default=str)
    return headers


def chat_model_kwargs(ctx: SessionContext) -> dict[str, Any]:
    """ChatLiteLLM constructor kwargs for gateway or SDK mode."""
    if not ctx.model:
        raise ValueError("Harness.DEEPAGENTS needs model=")
    if ctx.gateway is not None:
        return {
            "model": f"litellm_proxy/{ctx.model}",
            "api_base": ctx.gateway.api_base,
            "api_key": ctx.gateway.api_key,
            "extra_headers": gateway_headers(ctx),
        }
    return {"model": ctx.model, "api_key": ctx.api_key, "api_base": ctx.api_base}


def native_tool_name(name: str) -> str:
    return NORMALIZED_TO_NATIVE.get(name, name)


def normalized_tool_name(native: str) -> str:
    return NATIVE_TO_NORMALIZED.get(native, native)


def blocked_tools(permissions: str, disable_tools: Sequence[str]) -> frozenset[str]:
    """Native tool names the model must not see or call."""
    blocked = {native_tool_name(name) for name in disable_tools}
    if permissions == "read-only":
        blocked |= WRITE_TOOLS | EXECUTE_TOOLS
    elif permissions == "edit":
        blocked |= EXECUTE_TOOLS
    return frozenset(blocked)


def interrupt_config(permissions: str, blocked: frozenset[str]) -> dict[str, Any] | None:
    """interrupt_on for permissions='ask': approve/reject every mutating built-in."""
    if permissions != "ask":
        return None
    return {name: {"allowed_decisions": list(_APPROVAL_DECISIONS)} for name in sorted(APPROVAL_TOOLS - blocked)}


def recursion_limit(ctx: SessionContext) -> int:
    options = ctx.options if isinstance(ctx.options, DeepAgentsOptions) else None
    if options is not None and options.recursion_limit is not None:
        return options.recursion_limit
    if ctx.max_turns is not None:
        return DEEPAGENTS_BASE_RECURSION_LIMIT + ctx.max_turns * DEEPAGENTS_STEPS_PER_TURN
    return DEEPAGENTS_DEFAULT_RECURSION_LIMIT


def content_text(content: Any) -> str:
    """Plain text of a LangChain message content (str or content blocks)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [block if isinstance(block, str) else block.get("text", "") for block in content if _is_text_block(block)]
    return "".join(parts)


def _is_text_block(block: Any) -> bool:
    return isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")


def reasoning_text(message: Any) -> str:
    """Reasoning deltas from additional_kwargs or reasoning/thinking content blocks."""
    extra = getattr(message, "additional_kwargs", None) or {}
    reasoning = extra.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in ("reasoning", "thinking"):
            parts.append(str(block.get("reasoning") or block.get("thinking") or ""))
    return "".join(parts)


def stream_events(message: Any) -> list[Event]:
    """Text / Reasoning deltas for one streamed message chunk."""
    if getattr(message, "type", None) not in ("AIMessageChunk", "ai"):
        return []
    events: list[Event] = []
    reasoning = reasoning_text(message)
    if reasoning:
        events.append(Reasoning(delta=reasoning))
    text = content_text(getattr(message, "content", ""))
    if text:
        events.append(Text(delta=text))
    return events


def tool_call_event(call: Mapping[str, Any]) -> ToolCall:
    native = str(call.get("name") or "")
    args = call.get("args")
    return ToolCall(
        id=str(call.get("id") or ""),
        name=normalized_tool_name(native),
        native_name=native,
        input=dict(args) if isinstance(args, Mapping) else {"args": args},
        builtin=native in BUILTIN_TOOLS,
    )


def update_events(update: Any, skip_tools: frozenset[str]) -> list[Event]:
    """ToolCall / ToolResult events from one `updates` stream chunk (node -> state delta)."""
    if not isinstance(update, Mapping):
        return []
    events: list[Event] = []
    for node, delta in update.items():
        if node not in _EVENT_NODES or not isinstance(delta, Mapping):
            continue
        messages = delta.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            events += _message_events(message, skip_tools)
    return events


def _message_events(message: Any, skip_tools: frozenset[str]) -> list[Event]:
    kind = getattr(message, "type", None)
    if kind == "ai":
        calls = getattr(message, "tool_calls", None) or []
        return [tool_call_event(call) for call in calls if call.get("name") not in skip_tools]
    if kind == "tool" and getattr(message, "name", None) not in skip_tools:
        return [
            ToolResult(
                id=str(getattr(message, "tool_call_id", "") or ""),
                output=content_text(getattr(message, "content", "")),
                is_error=getattr(message, "status", None) == "error",
            )
        ]
    return []


def interrupts_in(update: Any) -> list[Any]:
    if not isinstance(update, Mapping):
        return []
    found = update.get("__interrupt__")
    return list(found) if isinstance(found, (list, tuple)) else []


def final_ai_text(messages: Sequence[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            text = content_text(getattr(message, "content", ""))
            if text:
                return text
    return ""


def structured_json(value: Any) -> str | None:
    if value is None:
        return None
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        return str(dump())
    return json.dumps(value, default=str)


def approval_requests(interrupt_value: Any) -> list[Mapping[str, Any]]:
    """action_requests of a HumanInTheLoopMiddleware interrupt payload."""
    if not isinstance(interrupt_value, Mapping):
        return []
    requests = interrupt_value.get("action_requests")
    return [r for r in requests if isinstance(r, Mapping)] if isinstance(requests, list) else []


def decision(allowed: bool, reason: str) -> dict[str, Any]:
    if allowed:
        return {"type": "approve"}
    return {"type": "reject", "message": reason or "The user denied this tool call."}


@dataclass
class TurnState:
    """Mutable state across the stream passes of one turn."""

    interrupts: list[Any] = field(default_factory=list)


class DeepAgentsHarnessConfig(BaseHarnessConfig):
    harness = Harness.DEEPAGENTS
    options_type = DeepAgentsOptions
    uses_model_endpoint = False
    capabilities = Capabilities(
        structured_output=True,
        tool_approval=True,
        tool_filtering=True,
        history=True,
        custom_tools=True,
        skills=True,
        resume=True,
        permission_modes=frozenset({"read-only", "ask", "edit", "full"}),
    )

    def validate_environment(self, ctx: SessionContext) -> None:
        self.get_options(ctx)
        if not ctx.model:
            raise ValueError("Harness.DEEPAGENTS needs model=")
