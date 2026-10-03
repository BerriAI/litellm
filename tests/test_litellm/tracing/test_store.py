"""
Tests for the pure read-side helpers in litellm/tracing/store.py (no ClickHouse needed).
"""

from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.rust_bridge.trace_queries import (
    LIST_TRACES,
    TRACE_SPANS,
    SPAN_ERROR,
    SpanErrorParams,
    SpanErrorRow,
    SpendRow,
)
from litellm.tracing.store import (
    TraceStore,
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
        "team_id": "",
        "api_key_hash": "",
        "user_id": "",
        "status_message": "",
        "error_truncated": False,
        **extra,
    }


def _llm_row(span_id: str, parent: str, agent: str, request_id: str, start_ms: float = 1, **extra: Any) -> dict:
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
        **extra,
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


def test_llm_response_id_is_preserved_when_spend_is_unavailable():
    trace = trace_from_rows("t1", _deep_agent_rows())
    assert trace is not None
    spans = {span["span_id"]: span for span in trace["spans"]}
    assert spans["llm-root"]["litellm_request_id"] == "chatcmpl-root"
    assert spans["task"]["litellm_request_id"] is None
    assert trace["summary"]["spend"] is None
    assert spans["llm-root"]["spend"] is None


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
            "spend": None,
        },
        {
            "name": "researcher",
            "parent_agent": "deep_research_agent",
            "invocations": 1,
            "llm_calls": 1,
            "tool_calls": 1,
            "duration_ms": 5,
            "spend": None,
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


def test_trace_groups_normalized_names_and_preserves_span_labels():
    rows = [
        _row("root", "", "invoke_agent research_agent", "agent", "research_agent"),
        _row("r1", "root", "researcher._execute_core", "agent", "researcher"),
        _row("r2", "r1", "invoke_agent researcher", "agent", "researcher"),
        _row("llm", "r2", "chat", "llm", "researcher"),
    ]
    result = trace_from_rows("t1", rows)
    assert result is not None
    assert result["summary"]["agent_names"] == ("research_agent", "researcher")
    assert result["summary"]["name"] == "invoke_agent research_agent"
    agents = {agent["name"]: agent for agent in result["agents"]}
    assert agents["researcher"]["parent_agent"] == "research_agent"
    assert agents["researcher"]["invocations"] == 2
    assert agents["researcher"]["llm_calls"] == 1


def test_trace_frameworks_are_the_sorted_distinct_span_frameworks():
    rows = [
        _row("root", "", "claude_code.interaction", "agent", "claude-code", framework="claude-code"),
        _llm_row("llm", "root", "claude-code", "msg_1", framework="claude-agent-sdk"),
        _row("tool", "root", "Bash", "tool", "claude-code", framework="claude-code"),
        _row("other", "root", "step", "chain", "claude-code", framework=""),
    ]
    validated_rows: Final = TRACE_SPANS.response.validate_python({"data": rows}).data
    trace = trace_from_rows("t1", validated_rows)
    assert trace is not None
    assert trace["summary"]["frameworks"] == ("claude-agent-sdk", "claude-code")
    spans = {span["span_id"]: span for span in trace["spans"]}
    assert (spans["llm"]["framework"], spans["other"]["framework"]) == ("claude-agent-sdk", "")
    assert trace["agents"][0]["llm_calls"] == 1
    assert trace["agents"][0]["tool_calls"] == 1


def test_spans_without_a_framework_column_report_none():
    trace = trace_from_rows("t1", _deep_agent_rows())
    assert trace is not None
    assert trace["summary"]["frameworks"] == ()
    assert {span["framework"] for span in trace["spans"]} == {""}


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
    rows: Final = LIST_TRACES.response.validate_python(
        {
            "data": [
                {
                    "trace_id": "t1",
                    "trace_ref": "ref",
                    "team_id": "team",
                    "api_key_hash": "key",
                    "user_id": "owner",
                    "agent_invocations": 2,
                    "agent_names": ["deep_research_agent"],
                    "request_ids": [],
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
                    "frameworks": ["claude-agent-sdk", "claude-code"],
                }
            ]
        }
    ).data
    summary: Final = trace_summary_from_row(rows[0])
    assert summary["agent_names"] == ("deep_research_agent",)
    assert summary["frameworks"] == ("claude-agent-sdk", "claude-code")
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
    store = TraceStore(client)
    scope: Final[TraceScope] = {"all_teams": 0, "user_id": "", "team_ids": ("team-a",)}

    page = await store.list_traces(scope, 0, 2000, limit=2)
    assert [t["trace_id"] for t in page["data"]] == ["t2", "t1"]
    assert page["next_cursor"] is not None
    assert decode_cursor(page["next_cursor"]) == (900, "ref1")
    params = client.query.call_args.args[1]
    assert params.team_ids == ("team-a",) and params.limit == 2 and params.cursor_ms == 0

    page = await store.list_traces(scope, 0, 2000, cursor=page["next_cursor"], limit=3)
    assert page["next_cursor"] is None
    assert client.query.call_args.args[1].cursor_trace_id == "ref1"


@pytest.mark.asyncio
async def test_get_span_not_found_and_found():
    client = MagicMock()
    client.query = AsyncMock(return_value=[])
    store = TraceStore(client)
    scope: Final[TraceScope] = {"all_teams": 1, "user_id": "", "team_ids": ()}
    assert await store.get_span("t", "s", scope, "ref") is None
    stored_input = '[{"role": "user", "content": "hi"}]'
    client.query = AsyncMock(
        return_value=[{"span_id": "s", "input": stored_input, "output": '{"ok": true}', "attributes": {"k": "v"}}]
    )
    assert await store.get_span("t", "s", scope, "ref") == {
        "span_id": "s",
        "input": stored_input,
        "output": '{"ok": true}',
        "input_ui": {"kind": "messages", "messages": ({"role": "user", "content": "hi"},)},
        "output_ui": {"kind": "fields", "fields": ({"key": "ok", "value": "true"},)},
        "attributes": {"k": "v"},
    }


@pytest.mark.asyncio
async def test_trace_cost_is_scoped_and_counts_repeated_request_once():
    client = MagicMock()
    spans = [
        _row("root", "", "agent", "agent", "agent", team_id="team-a", api_key_hash="key-a"),
        _llm_row("llm-1", "root", "agent", "response-1", team_id="team-a", api_key_hash="key-a"),
        _llm_row("llm-2", "root", "agent", "response-1", team_id="team-a", api_key_hash="key-a"),
    ]
    spend = [
        {
            "request_id": "request-other",
            "response_id": "response-1",
            "team_id": "team-b",
            "api_key": "key-b",
            "spend": 99.0,
            "start_ms": T0 // MS,
        },
        {
            "request_id": "request-1",
            "response_id": "response-1",
            "team_id": "team-a",
            "api_key": "key-a",
            "spend": 0.25,
            "start_ms": T0 // MS,
        },
        {
            "request_id": "request-other-key",
            "response_id": "unrelated-response",
            "team_id": "team-a",
            "api_key": "key-c",
            "spend": 50.0,
            "start_ms": T0 // MS,
        },
    ]
    client.query = AsyncMock(side_effect=[spans, tuple(SpendRow.model_validate({**row, "user": ""}) for row in spend)])
    store = TraceStore(client)
    scope: Final[TraceScope] = {"all_teams": 0, "user_id": "", "team_ids": ("team-a",)}

    trace = await store.get_trace("trace-1", scope, "ref")

    assert trace is not None
    assert trace["summary"]["spend"] == 0.25
    assert trace["agents"][0]["spend"] == 0.25
    assert [span["spend"] for span in trace["spans"]] == [None, 0.25, 0.25]
    assert [call.args[0].name for call in client.query.await_args_list] == ["trace_spans", "spend_by_response_ids"]


@pytest.mark.asyncio
async def test_run_list_uses_matching_spend_and_leaves_missing_cost_unavailable():
    client = MagicMock()
    rows = [
        {
            "trace_id": trace_id,
            "trace_ref": trace_id,
            "team_id": "team-a",
            "api_key_hash": "key-a",
            "request_ids": [request_id],
            "name": "agent",
            "service": "service",
            "input_preview": "",
            "start_ms": 1000,
            "duration_ms": 100,
            "status": "STATUS_CODE_OK",
            "span_count": 1,
            "agent_count": 1,
            "llm_calls": 1,
            "tool_calls": 0,
            "input_tokens": 1,
            "output_tokens": 1,
            "models": [],
        }
        for trace_id, request_id in (("trace-1", "response-1"), ("trace-2", "response-2"))
    ]
    spend = [
        {
            "request_id": "request-1",
            "response_id": "response-1",
            "team_id": "team-a",
            "api_key": "key-a",
            "spend": 0.25,
            "start_ms": 1000,
        }
    ]
    client.query = AsyncMock(side_effect=[rows, tuple(SpendRow.model_validate({**row, "user": ""}) for row in spend)])
    scope: Final[TraceScope] = {"all_teams": 0, "user_id": "", "team_ids": ("team-a",)}

    page = await TraceStore(client).list_traces(scope, 0, 2000)

    assert [run["spend"] for run in page["data"]] == [0.25, None]
    assert [call.args[0].name for call in client.query.await_args_list] == ["list_traces", "spend_by_response_ids"]


@pytest.mark.asyncio
async def test_ambiguous_cache_response_id_keeps_cost_unavailable():
    client = MagicMock()
    span: Final = _llm_row("llm-1", "", "agent", "response-1", team_id="", user_id="user", api_key_hash="key-a")
    spend = [
        {
            "request_id": request_id,
            "response_id": "response-1",
            "team_id": "",
            "api_key": "key-a",
            "spend": cost,
            "start_ms": T0 // MS,
        }
        for request_id, cost in (("response-1", 0.25), ("response-1_cache_hit123", 0.0))
    ]
    client.query = AsyncMock(
        side_effect=[[span], tuple(SpendRow.model_validate({**row, "user": "user"}) for row in spend)]
    )
    store = TraceStore(client)
    scope: Final[TraceScope] = {"all_teams": 0, "user_id": "user", "team_ids": ()}

    trace = await store.get_trace("trace-1", scope, "ref")

    assert trace is not None
    assert trace["summary"]["spend"] is None
    assert trace["spans"][0]["spend"] is None


@pytest.mark.asyncio
async def test_diagnostic_continuation_preserves_content_version_scope_and_unicode_offset():
    from hashlib import sha256

    message = "first 🧪\nlast"
    version = sha256(message.encode()).hexdigest().upper()
    client = MagicMock()
    client.query = AsyncMock(
        side_effect=[
            [SpanErrorRow(span_id="span-1", message="first 🧪", total_chars=len(message), version=version)],
            [SpanErrorRow(span_id="span-1", message="\nlast", total_chars=len(message), version=version)],
        ]
    )
    store = TraceStore(client)
    scope: Final[TraceScope] = {"all_teams": 0, "user_id": "", "team_ids": ("team-a",)}
    first = await store.get_span_error("trace-1", "span-1", scope, "scoped-run")
    assert first is not None and first["next_cursor"] is not None
    last = await store.get_span_error("trace-1", "span-1", scope, "scoped-run", first["next_cursor"])
    assert last is not None
    assert first["message"] + last["message"] == message
    assert last["next_cursor"] is None
    client.query.assert_awaited_with(
        SPAN_ERROR,
        SpanErrorParams(
            **scope,
            trace_id="trace-1",
            span_id="span-1",
            trace_ref="scoped-run",
            error_offset=len(first["message"]),
            error_version=version,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", ["garbage", "e30=", "WzEsMl0="])
async def test_malformed_diagnostic_cursor_never_reaches_storage(cursor):
    client = MagicMock()
    client.query = AsyncMock()
    with pytest.raises(ValueError, match="Invalid diagnostic cursor"):
        await TraceStore(client).get_span_error(
            "trace", "span", {"all_teams": 1, "user_id": "", "team_ids": ()}, cursor=cursor
        )
    client.query.assert_not_awaited()


@pytest.mark.parametrize(
    ("trace_team", "trace_user", "trace_key", "spend_team", "spend_user", "spend_key", "known"),
    (
        ("team", "", "export", "team", "", "request", False),
        ("team", "", "export", "team", "", "export", True),
        ("", "user", "export", "", "user", "request", True),
        ("", "", "key", "", "", "key", True),
        ("team", "user", "key", "other-team", "user", "key", False),
        ("", "user", "export", "", "other-user", "request", False),
        ("", "", "export", "", "", "request", False),
        ("", "", "", "", "", "", False),
        ("", "", "master", "", "", "", False),
    ),
)
def test_cost_attribution_requires_shared_ownership_after_visibility(
    trace_team: str,
    trace_user: str,
    trace_key: str,
    spend_team: str,
    spend_user: str,
    spend_key: str,
    known: bool,
) -> None:
    rows: Final = (
        _row("agent", "", "agent", "agent", "agent", team_id=trace_team, user_id=trace_user, api_key_hash=trace_key),
        _llm_row("llm", "agent", "agent", "response", team_id=trace_team, user_id=trace_user, api_key_hash=trace_key),
    )
    spend: Final = SpendRow(
        request_id="request",
        response_id="response",
        team_id=spend_team,
        user=spend_user,
        api_key=spend_key,
        spend=0.25,
        start_ms=T0 // MS,
    )
    trace: Final = trace_from_rows("trace", rows, "visible-reference", (spend,))
    assert trace is not None
    expected: Final = spend.spend if known else None
    assert trace["summary"]["spend"] == expected
    assert trace["agents"][0]["spend"] == expected
    assert trace["spans"][1]["spend"] == expected
    summary: Final = trace_summary_from_row(
        {
            "trace_id": "trace",
            "team_id": trace_team,
            "user_id": trace_user,
            "api_key_hash": trace_key,
            "request_ids": ("response",),
            "name": "agent",
            "service": "service",
            "input_preview": "",
            "start_ms": T0 // MS,
            "duration_ms": 10,
            "status": "STATUS_CODE_OK",
            "span_count": 2,
            "agent_count": 1,
            "llm_calls": 1,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "models": (),
        },
        (spend,),
    )
    assert summary["spend"] == expected


@pytest.mark.parametrize("failure", ("missing_id", "missing_spend", "duplicate_spend"))
def test_incomplete_llm_cost_never_becomes_a_partial_trace_or_agent_total(failure: str) -> None:
    second_id: Final = "" if failure == "missing_id" else "second"
    rows: Final = (
        _row("agent", "", "agent", "agent", "agent", team_id="team", api_key_hash="export"),
        _llm_row("first", "agent", "agent", "first", team_id="team", api_key_hash="export"),
        _llm_row("second", "agent", "agent", second_id, team_id="team", api_key_hash="export"),
    )
    first: Final = SpendRow(
        request_id="first",
        response_id="first",
        team_id="team",
        user="",
        api_key="export",
        spend=0.25,
        start_ms=0,
    )
    second: Final = first.model_copy(update={"request_id": "second", "response_id": "second"})
    spend: Final = (
        (first, second, second.model_copy(update={"request_id": "duplicate"}))
        if failure == "duplicate_spend"
        else (first,)
    )
    trace: Final = trace_from_rows("trace", rows, "ref", spend)
    assert trace is not None
    assert trace["spans"][1]["spend"] == first.spend
    assert trace["spans"][2]["spend"] is None
    assert trace["summary"]["spend"] is None
    assert trace["agents"][0]["spend"] is None


@pytest.mark.asyncio
async def test_trace_id_collision_requires_a_visible_reference_before_reading_content() -> None:
    from litellm.rust_bridge.trace_queries import TraceIdentityRow
    from litellm.tracing.store import AmbiguousTraceError

    storage: Final = MagicMock()
    storage.query = AsyncMock(return_value=(TraceIdentityRow(trace_ref="first"), TraceIdentityRow(trace_ref="second")))
    store: Final = TraceStore(storage)
    scope: Final[TraceScope] = {"all_teams": 1, "user_id": "", "team_ids": ()}
    with pytest.raises(AmbiguousTraceError, match="provide trace_ref"):
        await store.get_trace("shared-id", scope)
    with pytest.raises(AmbiguousTraceError, match="provide trace_ref"):
        await store.get_span("shared-id", "span", scope)
    with pytest.raises(AmbiguousTraceError, match="provide trace_ref"):
        await store.get_span_error("shared-id", "span", scope)
