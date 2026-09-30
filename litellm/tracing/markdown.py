"""
Render a trace as Markdown, for pasting into Claude / Codex.

    GET /v1/traces/{trace_id}?format=md[&span_id=...]

Framework spans (LangGraph middleware, graph nodes) are skipped and their children lifted, so
the document reads as: input -> each decision / tool call / subagent (nested) -> output.
"""

import json
from typing import Final

from litellm.constants import AGENT_TRACING_MARKDOWN_MAX_FIELD_CHARS
from litellm.tracing.types import Span, Trace

_GRAPH_NODES: Final = frozenset({"model", "tools"})


def _hidden(span: Span) -> bool:
    return span["parent_span_id"] is not None and (
        span["type"] == "framework" or (span["type"] == "chain" and span["name"] in _GRAPH_NODES)
    )


def _clip(text: str) -> str:
    if len(text) <= AGENT_TRACING_MARKDOWN_MAX_FIELD_CHARS:
        return text
    return f"{text[:AGENT_TRACING_MARKDOWN_MAX_FIELD_CHARS]}… [{len(text) - AGENT_TRACING_MARKDOWN_MAX_FIELD_CHARS} chars truncated]"


def _first_error_line(error: str) -> str:
    """LangSmith records `repr(exc)` + traceback with no separator; keep just the exception."""
    return error.split("Traceback (most recent call last):", 1)[0].splitlines()[0].strip()


def _messages(raw: str) -> str:
    """LLM/agent IO is a JSON message list (or one message); render it as `role: content` lines."""
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return _clip(raw)
    messages = parsed if isinstance(parsed, list) else [parsed]
    if not all(isinstance(m, dict) and "role" in m for m in messages):
        return _clip(raw)
    lines: list[str] = []
    for m in messages:
        if m.get("content"):
            lines.append(f"**{m['role']}**: {_clip(str(m['content']))}")
        for call in m.get("tool_calls") or []:
            lines.append(f"**{m['role']}** → `{call.get('name')}({json.dumps(call.get('args'), default=str)})`")
    return "\n".join(lines)


def _span_block(span: Span, io: tuple[str, str], depth: int) -> str:
    indent = "  " * depth
    kind = "subagent" if span["type"] == "agent" and depth else span["type"]
    head = f"{indent}- **{kind}** `{span['name']}` · {span['duration_ms'] / 1000:.2f}s"
    if span["type"] == "llm":
        head += f" · {span['model'] or '?'} · {span['input_tokens']}→{span['output_tokens']} tok"
        if span["litellm_request_id"]:
            head += f" · request `{span['litellm_request_id']}`"
    if span["status"] == "error":
        head += " · **FAILED**"
    parts = [head]
    if span["error"]:
        parts.append(f"{indent}  - error: `{_clip(_first_error_line(span['error']))}`")
    span_input, span_output = io
    if span_input:
        parts.append(f"{indent}  - input: {_messages(span_input)}".replace("\n", f"\n{indent}    "))
    if span_output:
        parts.append(f"{indent}  - output: {_messages(span_output)}".replace("\n", f"\n{indent}    "))
    return "\n".join(parts)


def trace_to_markdown(trace: Trace, io: dict[str, tuple[str, str]], span_id: str | None = None) -> str:
    """The whole trace, or the subtree under `span_id`."""
    spans: Final = trace["spans"]
    by_id: Final = {s["span_id"]: s for s in spans}

    def visible_parent(span: Span) -> str | None:
        parent = by_id.get(span["parent_span_id"] or "")
        while parent is not None and _hidden(parent):
            parent = by_id.get(parent["parent_span_id"] or "")
        return parent["span_id"] if parent else None

    children: dict[str | None, list[Span]] = {}
    for s in spans:
        if not _hidden(s):
            children.setdefault(visible_parent(s), []).append(s)
    for siblings in children.values():
        siblings.sort(key=lambda s: s["start_offset_ms"])

    lines: list[str] = []

    def walk(parent: str | None, depth: int) -> None:
        for s in children.get(parent, []):
            lines.append(_span_block(s, io.get(s["span_id"], ("", "")), depth))
            walk(s["span_id"], depth + 1)

    summary: Final = trace["summary"]
    root: Final = by_id.get(span_id) if span_id else next((s for s in spans if s["parent_span_id"] is None), None)
    header: list[str] = [
        f"# Agent trace: {summary['name']}",
        "",
        f"- trace_id: `{summary['trace_id']}` · service: `{summary['service']}` · {summary['start_time']}",
        f"- {summary['duration_ms'] / 1000:.1f}s · {summary['span_count']} spans · "
        f"{summary['agent_count']} agents · {summary['llm_calls']} LLM calls · {summary['tool_calls']} tool calls · "
        f"{summary['error_count']} errors",
        "",
    ]
    if root is None:
        return "\n".join([*header, "_no spans_"])
    root_input, root_output = io.get(root["span_id"], ("", ""))
    if span_id is None:
        header += ["## Input", "", _messages(root_input) or "_empty_", "", "## Steps", ""]
        walk(root["span_id"], 0)
        footer = ["", "## Output", "", _messages(root_output) or "_empty_"]
    else:
        header += [f"## Subtree of `{root['name']}`", ""]
        lines.append(_span_block(root, (root_input, root_output), 0))
        walk(root["span_id"], 1)
        footer = []
    return "\n".join([*header, *lines, *footer]) + "\n"
