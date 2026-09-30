"""
Render a trace as Markdown, for pasting into Claude / Codex.

    GET /v1/traces/{trace_id}?format=md[&span_id=...]

Framework spans (LangGraph middleware, graph nodes) are skipped and their children lifted, so
the document reads as: input -> each decision / tool call / subagent (nested) -> output.
"""

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from litellm.constants import AGENT_TRACING_MARKDOWN_MAX_FIELD_CHARS
from litellm.tracing.types import Span, Trace

_GRAPH_NODES: Final = frozenset({"model", "tools"})
_TRACEBACK_MARKER: Final = "Traceback (most recent call last):"
_NO_IO: Final = ("", "")


def _hidden(span: Span) -> bool:
    is_graph_node: Final = span["type"] == "chain" and span["name"] in _GRAPH_NODES
    return span["parent_span_id"] is not None and (span["type"] == "framework" or is_graph_node)


def _clip(text: str) -> str:
    if len(text) <= AGENT_TRACING_MARKDOWN_MAX_FIELD_CHARS:
        return text
    return f"{text[:AGENT_TRACING_MARKDOWN_MAX_FIELD_CHARS]}… [{len(text) - AGENT_TRACING_MARKDOWN_MAX_FIELD_CHARS} chars truncated]"


def _first_error_line(error: str) -> str:
    """LangSmith records `repr(exc)` + traceback with no separator; keep just the exception."""
    head: Final = error.split(_TRACEBACK_MARKER, 1)[0].strip()
    lines: Final = (head or error).strip().splitlines()
    return lines[0].strip() if lines else ""


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
            args: Final = _clip(json.dumps(call.get("args"), default=str))
            lines.append(f"**{m['role']}** → `{call.get('name')}({args})`")
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


class _SpanTree:
    """Visible-span tree: hidden spans are skipped and their children lifted; cyclic or dangling parents end at a root."""

    def __init__(self, spans: Sequence[Span]) -> None:
        self.by_id: Final = MappingProxyType({s["span_id"]: s for s in spans})
        parent_of: Final = {s["span_id"]: self._visible_parent(s) for s in spans if not _hidden(s)}
        grouped: dict[str | None, list[Span]] = {}
        for span in sorted((s for s in spans if s["span_id"] in parent_of), key=lambda s: s["start_offset_ms"]):
            grouped.setdefault(parent_of[span["span_id"]], []).append(span)
        self.children: Final[Mapping[str | None, tuple[Span, ...]]] = MappingProxyType(
            {parent: tuple(kids) for parent, kids in grouped.items()}
        )

    def _visible_parent(self, span: Span) -> str | None:
        """Nearest non-hidden ancestor present in this trace; None when it is missing or the parent chain loops."""
        seen: set[str] = {span["span_id"]}
        nearest: str | None = None
        parent = self.by_id.get(span["parent_span_id"] or "")
        while parent is not None:
            if parent["span_id"] in seen:
                return None
            seen.add(parent["span_id"])
            if nearest is None and not _hidden(parent):
                nearest = parent["span_id"]
            parent = self.by_id.get(parent["parent_span_id"] or "")
        return nearest

    def visible_children(self, span_id: str) -> tuple[Span, ...]:
        """Children shown under `span_id`; a hidden span's lifted children are its visible descendants."""
        span: Final = self.by_id[span_id]
        if not _hidden(span):
            return self.children.get(span_id, ())
        return tuple(s for s in self.by_id.values() if self._under(s, span_id) and not _hidden(s))

    def _under(self, span: Span, ancestor_id: str) -> bool:
        seen: set[str] = set()
        parent_id = span["parent_span_id"]
        while parent_id and parent_id not in seen:
            if parent_id == ancestor_id:
                return True
            seen.add(parent_id)
            parent = self.by_id.get(parent_id)
            if parent is None or not _hidden(parent):
                return False
            parent_id = parent["parent_span_id"]
        return False


def _render(tree: _SpanTree, io: Mapping[str, tuple[str, str]], spans: Sequence[Span], depth: int) -> list[str]:
    """Depth-first blocks for `spans` and everything under them; each span renders at most once."""
    lines: list[str] = []
    stack: list[tuple[Span, int]] = [(s, depth) for s in reversed(spans)]
    rendered: set[str] = set()
    while stack:
        span, level = stack.pop()
        if span["span_id"] in rendered:
            continue
        rendered.add(span["span_id"])
        lines.append(_span_block(span, io.get(span["span_id"], _NO_IO), level))
        stack.extend((child, level + 1) for child in reversed(tree.children.get(span["span_id"], ())))
    return lines


def _header(trace: Trace) -> list[str]:
    summary: Final = trace["summary"]
    return [
        f"# Agent trace: {summary['name']}",
        "",
        f"- trace_id: `{summary['trace_id']}` · service: `{summary['service']}` · {summary['start_time']}",
        f"- {summary['duration_ms'] / 1000:.1f}s · {summary['span_count']} spans · "
        f"{summary['agent_count']} agents · {summary['llm_calls']} LLM calls · {summary['tool_calls']} tool calls · "
        f"{summary['error_count']} errors",
        "",
    ]


def trace_to_markdown(trace: Trace, io: Mapping[str, tuple[str, str]], span_id: str | None = None) -> str | None:
    """The whole trace, or the subtree under `span_id`; None when `span_id` is not in the trace."""
    tree: Final = _SpanTree(trace["spans"])
    if span_id is not None:
        if span_id not in tree.by_id:
            return None
        selected: Final = tree.by_id[span_id]
        subtree: Final = [
            _span_block(selected, io.get(span_id, _NO_IO), 0),
            *_render(tree, io, tree.visible_children(span_id), 1),
        ]
        return "\n".join([*_header(trace), f"## Subtree of `{selected['name']}`", "", *subtree]) + "\n"
    roots: Final = tree.children.get(None, ())
    if not roots:
        return "\n".join([*_header(trace), "_no spans_"]) + "\n"
    first_input: Final = io.get(roots[0]["span_id"], _NO_IO)[0]
    last_output: Final = io.get(roots[-1]["span_id"], _NO_IO)[1]
    # one root reads as its steps; several (partial or merged traces) each show as their own top-level step
    steps: Final = (
        _render(tree, io, tree.children.get(roots[0]["span_id"], ()), 0)
        if len(roots) == 1
        else _render(tree, io, roots, 0)
    )
    return (
        "\n".join(
            [
                *_header(trace),
                "## Input",
                "",
                _messages(first_input) or "_empty_",
                "",
                "## Steps",
                "",
                *steps,
                "",
                "## Output",
                "",
                _messages(last_output) or "_empty_",
            ]
        )
        + "\n"
    )
