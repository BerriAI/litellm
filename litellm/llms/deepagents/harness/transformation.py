"""
Deep Agents harness config: LangChain `deepagents` running in your Python process.

Pure translation only: model kwargs (gateway mode uses `litellm_proxy/<group>` with the same
attribution headers the CLI endpoint adds), permission and tool filtering, and LangGraph
stream chunks to events. `litellm/harness/handlers/deepagents_handler.py` builds the agent,
streams it, answers approvals and counts usage.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol

from pydantic import TypeAdapter

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
from litellm.llms.base_llm.harness.utils import gateway_headers

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
_TOOL_CALLS: Final = TypeAdapter(Iterable[Mapping[str, object]])

NORMALIZED_TO_NATIVE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "read": "read_file",
        "write": "write_file",
        "edit": "edit_file",
        "bash": "execute",
    }
)
NATIVE_TO_NORMALIZED: Final[Mapping[str, str]] = MappingProxyType({v: k for k, v in NORMALIZED_TO_NATIVE.items()})
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
_APPROVAL_DECISIONS: Final = ("approve", "reject")


def chat_model_kwargs(
    ctx: SessionContext,
) -> dict[str, object]:  # mutable-ok: ChatLiteLLM constructor kwargs, splatted as **kwargs
    """ChatLiteLLM constructor kwargs for gateway or SDK mode."""
    if not ctx.model:
        raise ValueError("Harness.DEEPAGENTS needs model=")
    if ctx.gateway is not None:
        return {  # mutable-ok: ChatLiteLLM constructor kwargs, splatted as **kwargs
            "model": f"litellm_proxy/{ctx.model}",
            "api_base": ctx.gateway.api_base,
            "api_key": ctx.gateway.api_key,
            "extra_headers": gateway_headers(ctx),
        }
    return {"model": ctx.model, "api_key": ctx.api_key, "api_base": ctx.api_base}  # mutable-ok: ChatLiteLLM kwargs


def native_tool_name(name: str) -> str:
    return NORMALIZED_TO_NATIVE.get(name, name)


def normalized_tool_name(native: str) -> str:
    return NATIVE_TO_NORMALIZED.get(native, native)


def blocked_tools(permissions: str, disable_tools: Sequence[str]) -> frozenset[str]:
    """Native tool names the model must not see or call."""
    disabled = frozenset(native_tool_name(name) for name in disable_tools)
    if permissions == "read-only":
        return disabled | WRITE_TOOLS | EXECUTE_TOOLS
    if permissions == "edit":
        return disabled | EXECUTE_TOOLS
    return disabled


def interrupt_config(
    permissions: str, blocked: frozenset[str]
) -> dict[str, dict[str, list[str]]] | None:  # mutable-ok: deepagents create_deep_agent(interrupt_on=) takes a dict
    """interrupt_on for permissions='ask': approve/reject every mutating built-in."""
    if permissions != "ask":
        return None
    return {  # mutable-ok: deepagents interrupt_on config (dict of InterruptOnConfig with list allowed_decisions)
        name: {"allowed_decisions": list(_APPROVAL_DECISIONS)}  # mutable-ok: deepagents InterruptOnConfig shape
        for name in sorted(APPROVAL_TOOLS - blocked)
    }


def recursion_limit(ctx: SessionContext) -> int:
    options = ctx.options if isinstance(ctx.options, DeepAgentsOptions) else None
    if options is not None and options.recursion_limit is not None:
        return options.recursion_limit
    if ctx.max_turns is not None:
        return DEEPAGENTS_BASE_RECURSION_LIMIT + ctx.max_turns * DEEPAGENTS_STEPS_PER_TURN
    return DEEPAGENTS_DEFAULT_RECURSION_LIMIT


def _dict_or_none(value: object) -> Mapping[object, object] | None:
    return value if isinstance(value, dict) else None


def _list_or_none(value: object) -> Sequence[object] | None:
    return value if isinstance(value, list) else None


def _mapping_or_none(value: object) -> Mapping[object, object] | None:
    return value if isinstance(value, Mapping) else None


def content_text(content: object) -> str:
    """Plain text of a LangChain message content (str or content blocks)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block if isinstance(block, str) else block.get("text", "") for block in content if _is_text_block(block)
    )


def _is_text_block(block: object) -> bool:
    if isinstance(block, str):
        return True
    fields: Final = _dict_or_none(block)
    return fields is not None and fields.get("type") == "text"


def reasoning_text(message: object) -> str:
    """Reasoning deltas from additional_kwargs or reasoning/thinking content blocks."""
    extra = getattr(message, "additional_kwargs", None) or MappingProxyType[str, object]({})
    reasoning = extra.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    blocks: Final = _list_or_none(getattr(message, "content", None))
    if blocks is None:
        return ""
    return "".join(
        str(fields.get("reasoning") or fields.get("thinking") or "")
        for fields in (_dict_or_none(block) for block in blocks)
        if fields is not None and fields.get("type") in ("reasoning", "thinking")
    )


def stream_events(
    message: object,
) -> list[Event]:  # mutable-ok: returns a list; existing callers/tests compare it to list literals
    """Text / Reasoning deltas for one streamed message chunk."""
    if getattr(message, "type", None) not in ("AIMessageChunk", "ai"):
        return []  # mutable-ok: returns a list; existing callers/tests compare it to list literals
    reasoning = reasoning_text(message)
    text = content_text(getattr(message, "content", ""))
    reasoning_events: tuple[Event, ...] = (Reasoning(delta=reasoning),) if reasoning else ()
    text_events: tuple[Event, ...] = (Text(delta=text),) if text else ()
    return [  # mutable-ok: returns a list; existing callers/tests compare it to list literals
        *reasoning_events,
        *text_events,
    ]


def tool_call_event(call: Mapping[str, object]) -> ToolCall:
    native = str(call.get("name") or "")
    args = call.get("args")
    return ToolCall(
        id=str(call.get("id") or ""),
        name=normalized_tool_name(native),
        native_name=native,
        input=dict(args) if isinstance(args, Mapping) else {"args": args},  # mutable-ok: ToolCall.input is a dict
        builtin=native in BUILTIN_TOOLS,
    )


def _node_messages(update: Mapping[object, object]) -> Iterator[object]:
    for node, delta in update.items():
        if node not in _EVENT_NODES:
            continue
        fields = _mapping_or_none(delta)
        if fields is None:
            continue
        messages = _list_or_none(fields.get("messages"))
        if messages is not None:
            yield from messages


def update_events(
    update: object, skip_tools: frozenset[str]
) -> list[Event]:  # mutable-ok: returns a list; existing callers/tests compare it to list literals
    """ToolCall / ToolResult events from one `updates` stream chunk (node -> state delta)."""
    nodes: Final = _mapping_or_none(update)
    if nodes is None:
        return []  # mutable-ok: returns a list; existing callers/tests compare it to list literals
    return list(  # mutable-ok: returns a list; existing callers/tests compare it to list literals
        itertools.chain.from_iterable(_message_events(message, skip_tools) for message in _node_messages(nodes))
    )


def _message_events(message: object, skip_tools: frozenset[str]) -> tuple[Event, ...]:
    kind = getattr(message, "type", None)
    if kind == "ai":
        calls: Final = _TOOL_CALLS.validate_python(getattr(message, "tool_calls", None) or ())
        return tuple(tool_call_event(call) for call in calls if call.get("name") not in skip_tools)
    if kind == "tool" and getattr(message, "name", None) not in skip_tools:
        status: Final[object] = getattr(message, "status", None)
        return (
            ToolResult(
                id=str(getattr(message, "tool_call_id", "") or ""),
                output=content_text(getattr(message, "content", "")),
                is_error=status == "error",
            ),
        )
    return ()


class InterruptLike(Protocol):
    @property
    def id(self) -> str: ...


def interrupts_in(
    update: object,
) -> list[InterruptLike]:  # mutable-ok: returns a list; existing callers/tests compare it to list literals
    nodes: Final = _mapping_or_none(update)
    if nodes is None:
        return []  # mutable-ok: returns a list; existing callers/tests compare it to list literals
    found = nodes.get("__interrupt__")
    items = tuple(found) if isinstance(found, (list, tuple)) else ()
    return list(items)  # mutable-ok: list return; callers/tests compare to lists


def final_ai_text(messages: Sequence[object]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            text = content_text(getattr(message, "content", ""))
            if text:
                return text
    return ""


def structured_json(value: object) -> str | None:
    if value is None:
        return None
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        return str(dump())
    return json.dumps(value, default=str)


def approval_requests(
    interrupt_value: object,
) -> list[Mapping[str, Any]]:  # mutable-ok: returns a list; existing callers/tests compare it to list literals
    """action_requests of a HumanInTheLoopMiddleware interrupt payload."""
    payload: Final = _mapping_or_none(interrupt_value)
    if payload is None:
        return []  # mutable-ok: returns a list; existing callers/tests compare it to list literals
    requests: Final = _list_or_none(payload.get("action_requests"))
    kept = tuple(r for r in requests if isinstance(r, Mapping)) if requests is not None else ()
    return list(kept)  # mutable-ok: list return; callers/tests compare to lists


def decision(allowed: bool, reason: str) -> dict[str, str]:  # mutable-ok: LangGraph resume payload (HITL decision dict)
    if allowed:
        return {"type": "approve"}  # mutable-ok: LangGraph resume payload (HITL decision dict)
    return {  # mutable-ok: LangGraph HITL decision
        "type": "reject",
        "message": reason or "The user denied this tool call.",
    }


@dataclass
class TurnState:
    """Mutable state across the stream passes of one turn."""

    interrupts: tuple[InterruptLike, ...] = ()


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
