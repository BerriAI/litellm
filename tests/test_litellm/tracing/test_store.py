"""
Tests for the pure read-side helpers in litellm/tracing/store.py (no ClickHouse needed).
"""

import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

from litellm.tracing.store import (
    ClickHouseTraceStore,
    agent_nodes,
    decode_cursor,
    encode_cursor,
    span_from_row,
    trace_from_rows,
    trace_summary_from_row,
)
from litellm.tracing.types import TraceScope

T0 = 1_790_742_989_000_000_000  # ns
MS = 1_000_000


def _row(
    span_id: str,
    parent: str,
    name: str,
    type_: str,
    agent: str,
    start_ms: float = 0,
    duration_ms: float = 10,
    status: str = "STATUS_CODE_OK",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "span_id": span_id,
        "parent_span_id": parent,
        "name": name,
        "type": type_,
        "agent": agent,
        "status": status,
        "start_ns": T0 + int(start_ms * MS),
        "duration_ns": int(duration_ms * MS),
        "service": "agent-demo",
        "input_preview": f"input of {name}",
        "model": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "litellm_request_id": "",
        **extra,
    }


def _llm_row(span_id: str, parent: str, agent: str, request_id: str, start_ms: float = 1) -> dict:
    return _row(
        span_id,
        parent,
        "ChatOpenAI",
        "llm",
        agent,
        start_ms=start_ms,
        duration_ms=100,
        model="claude-sonnet-4-5",
        input_tokens=100,
        output_tokens=20,
        litellm_request_id=request_id,
    )


def _deep_agent_rows(researcher_invocations: int = 1) -> list[dict[str, Any]]:
    """root agent -> llm, task tool -> researcher subagent (N times) -> llm + search_docs tool."""
    rows = [
        _row("root", "", "deep_research_agent", "agent", "deep_research_agent", duration_ms=1000),
        _llm_row("llm-root", "root", "deep_research_agent", "chatcmpl-root"),
        _row("task", "root", "task", "tool", "deep_research_agent", start_ms=200, duration_ms=700),
    ]
    for i in range(researcher_invocations):
        rows += [
            _row(f"res-{i}", "task", "researcher", "agent", "researcher", start_ms=201, duration_ms=5),
            _llm_row(f"res-llm-{i}", f"res-{i}", "researcher", f"chatcmpl-res-{i}", start_ms=202),
            _row(f"res-tool-{i}", f"res-{i}", "search_docs", "tool", "researcher", start_ms=203, duration_ms=1),
            _row(f"res-mw-{i}", f"res-{i}", "FilesystemMiddleware.wrap_model_call", "framework", "researcher"),
        ]
    return rows


# ---------------------------------------------------------------- trace_from_rows


def test_empty_rows_is_none():
    assert trace_from_rows("abc", []) is None


def test_llm_response_id_is_preserved_without_spend_enrichment():
    trace = trace_from_rows("t1", _deep_agent_rows())
    assert trace is not None
    spans = {span["span_id"]: span for span in trace["spans"]}
    assert spans["llm-root"]["litellm_request_id"] == "chatcmpl-root"
    assert spans["task"]["litellm_request_id"] is None
    assert "spend" not in trace["summary"]


def test_summary_totals():
    trace = trace_from_rows("t1", _deep_agent_rows())
    assert trace is not None
    summary = trace["summary"]
    assert summary["trace_id"] == "t1"
    assert summary["name"] == "deep_research_agent"
    assert summary["service"] == "agent-demo"
    assert summary["input_preview"] == "input of deep_research_agent"
    assert summary["status"] == "ok"
    assert summary["span_count"] == 7
    assert summary["agent_count"] == 2
    assert summary["llm_calls"] == 2
    assert summary["tool_calls"] == 2
    assert summary["error_count"] == 0
    assert (summary["input_tokens"], summary["output_tokens"]) == (200, 40)
    assert summary["models"] == ("claude-sonnet-4-5",)
    assert summary["duration_ms"] == 1000
    assert summary["start_time"].startswith("2026-09-30T")


def test_error_count_counts_error_spans():
    rows = _deep_agent_rows()
    rows[2]["status"] = "STATUS_CODE_ERROR"
    trace = trace_from_rows("t1", rows)
    assert trace is not None
    assert trace["summary"]["error_count"] == 1
    assert trace["summary"]["status"] == "ok"  # root span status; the UI uses error_count for "failed"
    assert trace["spans"][2]["status"] == "error"


def test_offsets_are_relative_to_trace_start_in_ms():
    trace = trace_from_rows("t1", _deep_agent_rows())
    assert trace is not None
    spans = {s["span_id"]: s for s in trace["spans"]}
    assert spans["root"]["start_offset_ms"] == 0
    assert spans["task"]["start_offset_ms"] == 200
    assert spans["task"]["duration_ms"] == 700
    assert spans["root"]["parent_span_id"] is None
    assert spans["task"]["parent_span_id"] == "root"


def test_span_from_row_optional_fields():
    span = span_from_row(_row("s", "", "x", "chain", "a", status="STATUS_CODE_UNSET"), T0)
    assert (span["model"], span["parent_span_id"], span["status"], span["litellm_request_id"]) == (
        None,
        None,
        "unset",
        None,
    )


def test_agent_nodes_parent_and_per_agent_counts():
    trace = trace_from_rows("t1", _deep_agent_rows())
    assert trace is not None
    assert trace["agents"] == (
        {
            "name": "deep_research_agent",
            "parent_agent": None,
            "invocations": 1,
            "llm_calls": 1,
            "tool_calls": 1,
            "duration_ms": 1000,
        },
        {
            "name": "researcher",
            "parent_agent": "deep_research_agent",
            "invocations": 1,
            "llm_calls": 1,
            "tool_calls": 1,
            "duration_ms": 5,
        },
    )


def test_200_subagent_invocations_aggregate_into_one_node():
    trace = trace_from_rows("t1", _deep_agent_rows(researcher_invocations=200))
    assert trace is not None
    assert [a["name"] for a in trace["agents"]] == ["deep_research_agent", "researcher"]
    researcher = trace["agents"][1]
    assert researcher["parent_agent"] == "deep_research_agent"
    assert researcher["invocations"] == 200
    assert researcher["llm_calls"] == 200
    assert researcher["tool_calls"] == 200
    assert researcher["duration_ms"] == pytest.approx(1000)
    assert trace["summary"]["agent_count"] == 2
    assert trace["summary"]["span_count"] == 3 + 4 * 200


def test_parent_agent_skips_same_name_ancestors():
    """A recursive agent (researcher -> researcher) still reports the nearest *different* agent."""
    rows = [
        _row("root", "", "lead", "agent", "lead"),
        _row("r1", "root", "researcher", "agent", "researcher"),
        _row("r2", "r1", "researcher", "agent", "researcher"),
    ]
    spans = [span_from_row(r, T0) for r in rows]
    nodes = {n["name"]: n for n in agent_nodes(spans)}
    assert nodes["researcher"]["parent_agent"] == "lead"
    assert nodes["researcher"]["invocations"] == 2


def test_parent_agent_stops_at_cyclic_parents():
    rows = [
        _row("self", "self", "researcher", "agent", "researcher"),
        _row("first", "second", "researcher", "agent", "researcher"),
        _row("second", "first", "researcher", "agent", "researcher"),
    ]
    spans = [span_from_row(row, T0) for row in rows]
    assert agent_nodes(spans)[0]["parent_agent"] is None


def test_agent_nodes_ignores_spans_of_unknown_agents():
    spans = [span_from_row(_row("t", "", "tool", "tool", "ghost"), T0)]
    assert agent_nodes(spans) == ()


# ---------------------------------------------------------------- list helpers


def test_cursor_round_trip():
    cursor = encode_cursor(1790742989377, "4bad42b84e9de3ba46fc870185f8f023")
    assert decode_cursor(cursor) == (1790742989377, "4bad42b84e9de3ba46fc870185f8f023")
    assert decode_cursor(None) == (0, "")
    assert decode_cursor("") == (0, "")


@pytest.mark.parametrize("cursor", ["abc", "bm90LWpzb24=", "WzEsIDJd", "WzAsICJ0Il0="])
def test_invalid_cursor_is_rejected(cursor):
    with pytest.raises(ValueError, match="Invalid trace cursor"):
        decode_cursor(cursor)


def test_trace_summary_from_row():
    summary = trace_summary_from_row(
        {
            "trace_id": "t1",
            "name": "deep_research_agent",
            "service": "agent-demo",
            "input_preview": "hi",
            "start_ms": 1790742989377,
            "duration_ms": 51385,
            "status": "STATUS_CODE_OK",
            "span_count": "126",
            "agent_count": "2",
            "llm_calls": "7",
            "tool_calls": "26",
            "error_count": "1",
            "input_tokens": "30175",
            "output_tokens": "2620",
            "models": ["claude-sonnet-4-5"],
        }
    )
    assert summary["status"] == "ok"
    assert (summary["span_count"], summary["error_count"]) == (126, 1)
    assert summary["start_time"] == "2026-09-30T04:36:29.377000+00:00"


@pytest.mark.asyncio
async def test_list_traces_sets_next_cursor_on_full_page():
    client = MagicMock()
    row = {
        "trace_id": "t2",
        "trace_ref": "ref2",
        "name": "a",
        "service": "s",
        "input_preview": "",
        "start_ms": 1000,
        "duration_ms": 1,
        "status": "STATUS_CODE_OK",
        "span_count": 1,
        "agent_count": 1,
        "llm_calls": 0,
        "tool_calls": 0,
        "error_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "models": [],
    }
    client.query = AsyncMock(return_value=[row, {**row, "trace_id": "t1", "trace_ref": "ref1", "start_ms": 900}])
    store = ClickHouseTraceStore(client)
    scope: TraceScope = {"team_ids": ("team-a",), "api_key_hash": ""}

    page = await store.list_traces(scope, 0, 2000, limit=2)
    assert [t["trace_id"] for t in page["data"]] == ["t2", "t1"]
    assert page["next_cursor"] is not None
    assert decode_cursor(page["next_cursor"]) == (900, "ref1")
    params = client.query.call_args.args[1]
    assert params["team_ids"] == ("team-a",) and params["limit"] == 2 and params["cursor_ms"] == 0

    page = await store.list_traces(scope, 0, 2000, cursor=page["next_cursor"], limit=3)
    assert page["next_cursor"] is None
    assert client.query.call_args.args[1]["cursor_trace_id"] == "ref1"


@pytest.mark.asyncio
async def test_get_span_not_found_and_found():
    client = MagicMock()
    client.query = AsyncMock(return_value=[])
    store = ClickHouseTraceStore(client)
    scope: TraceScope = {"team_ids": (), "api_key_hash": ""}
    assert await store.get_span("t", "s", scope) is None
    client.query = AsyncMock(return_value=[{"span_id": "s", "input": "i", "output": "o", "attributes": {"k": "v"}}])
    assert await store.get_span("t", "s", scope) == {
        "span_id": "s",
        "input": "i",
        "output": "o",
        "attributes": {"k": "v"},
    }
