"""
Tests for the Markdown trace export (GET /v1/traces/{id}?format=md).
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.tracing.markdown import trace_to_markdown
from litellm.tracing.types import LiteLLMRequest, Span, Trace, TraceSummary


def _span(span_id, parent, name, type_, start, **extra) -> Span:
    return Span(
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        type=type_,
        agent=extra.pop("agent", "lead"),
        start_offset_ms=start,
        duration_ms=extra.pop("duration_ms", 1000.0),
        status=extra.pop("status", "ok"),
        error=extra.pop("error", None),
        input_preview="",
        model=extra.pop("model", None),
        input_tokens=0,
        output_tokens=0,
        litellm=extra.pop("litellm", None),
    )


REQUEST = LiteLLMRequest(
    request_id="chatcmpl-1",
    model="openai/claude-sonnet-4-5",
    model_group="claude-sonnet-4-5",
    provider="openai",
    key_alias="research-bot",
    team_alias="research-agents",
    spend=0.0087,
    prompt_tokens=3378,
    completion_tokens=514,
    cache_read_tokens=0,
    cache_write_tokens=0,
    latency_ms=8278,
    ttft_ms=None,
    status="success",
)

SPANS = [
    _span("root", None, "lead", "agent", 0),
    _span("mw", "root", "FilesystemMiddleware.wrap_model_call", "framework", 1),
    _span("llm", "mw", "ChatOpenAI", "llm", 2, litellm=REQUEST, model="claude-sonnet-4-5"),
    _span("task", "root", "task", "tool", 3),
    _span("sub", "task", "researcher", "agent", 4, agent="researcher"),
    _span(
        "grep",
        "sub",
        "grep_code",
        "tool",
        5,
        agent="researcher",
        status="error",
        error="TimeoutError('Timeout after 30s')Traceback (most recent call last):\n  File x",
    ),
]
TRACE = Trace(
    summary=TraceSummary(
        trace_id="t1",
        name="lead",
        service="research-agent",
        input_preview="",
        start_time="2026-09-30T00:00:00+00:00",
        duration_ms=10_000,
        status="ok",
        span_count=6,
        agent_count=2,
        agent_invocations=2,
        llm_calls=1,
        tool_calls=2,
        error_count=1,
        input_tokens=3378,
        output_tokens=514,
        spend=0.0087,
        models=["claude-sonnet-4-5"],
    ),
    agents=[],
    spans=SPANS,
)
IO = {
    "root": (
        json.dumps([{"role": "user", "content": "Fix the flaky login test"}]),
        json.dumps({"role": "assistant", "content": "Fixed it."}),
    ),
    "llm": (
        "",
        json.dumps(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "task", "args": {"subagent_type": "researcher"}}],
            }
        ),
    ),
    "grep": ('{"q": "session"}', ""),
}


def test_markdown_reads_input_steps_output():
    md = trace_to_markdown(TRACE, IO)
    assert md.startswith("# Agent trace: lead")
    assert "## Input\n\n**user**: Fix the flaky login test" in md
    assert "## Output\n\n**assistant**: Fixed it." in md
    assert "1 errors" in md


def test_framework_spans_are_skipped_and_children_lifted():
    md = trace_to_markdown(TRACE, IO)
    assert "wrap_model_call" not in md
    # the LLM call is lifted to depth 0 under the root, with LiteLLM cost and the tool call it made
    assert "- **llm** `ChatOpenAI` · 1.00s · claude-sonnet-4-5 · 3378→514 tok · $0.0087" in md
    assert '**assistant** → `task({"subagent_type": "researcher"})`' in md


def test_subagents_nest_and_failures_show_first_error_line():
    md = trace_to_markdown(TRACE, IO)
    assert "  - **subagent** `researcher`" in md
    assert "    - **tool** `grep_code` · 1.00s · **FAILED**" in md
    assert "error: `TimeoutError('Timeout after 30s')`" in md
    assert "Traceback" not in md


def test_span_id_exports_only_that_subtree():
    md = trace_to_markdown(TRACE, IO, span_id="sub")
    assert "## Subtree of `researcher`" in md
    assert "grep_code" in md
    assert "ChatOpenAI" not in md
    assert "## Input" not in md
