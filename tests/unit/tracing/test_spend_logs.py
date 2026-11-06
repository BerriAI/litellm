import json
import math
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.tracing.generated.models import TraceAgentsParams
from litellm.tracing.generated.types import TraceScope
from litellm.tracing.spend_logs import SpendLogRow, SpendLogTraceStorage, build_trace

ALL: Final = TraceScope(all_teams=1, user_id="", team_ids=())
OWNED: Final = TraceScope(all_teams=0, user_id="u1", team_ids=("t1",))
SDK_UA: Final = "claude-cli/2.1.318 (external, sdk-py, agent-sdk/0.2.178)"


def row(
    request_id: str,
    start: str,
    end: str,
    *,
    trace_key: str = "s1",
    tags: Sequence[str] = (),
    user_agent: str = "",
    messages: object = (),
    response: Mapping[str, object] | None = None,
    status: str = "success",
    error: str | None = None,
    spend: float = 0.01,
    agent_id: str | None = None,
    request: Mapping[str, object] | None = None,
) -> SpendLogRow:
    metadata: Final = {
        "user_agent": user_agent,
        **({"error_information": {"error_message": error}} if error else {}),
    }
    return SpendLogRow(
        request_id=request_id,
        trace_key=trace_key,
        call_type="acompletion",
        model="gpt-4o-mini",
        status=status,
        spend=spend,
        prompt_tokens=10,
        completion_tokens=5,
        start_time=f"2026-11-06 05:51:{start}",
        end_time=f"2026-11-06 05:51:{end}",
        agent_id=agent_id,
        request_tags=json.dumps(list(tags)),
        metadata=json.dumps(metadata),
        messages="{}",
        proxy_server_request=json.dumps(request or {"messages": messages, "tags": list(tags)}),
        response=json.dumps(response or {}),
    )


class FakeDatabase:
    def __init__(self, *responses: Sequence[Mapping[str, object]]) -> None:
        self.responses: Final = responses
        self.calls: tuple[tuple[str, tuple[object, ...]], ...] = ()

    async def fetch(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        self.calls = (*self.calls, (query, args))
        return self.responses[len(self.calls) - 1]


def test_tagged_calls_become_an_agent_tree_with_tool_spans() -> None:
    tool_response: Final = {
        "choices": [{"message": {"tool_calls": [{"function": {"name": "classify_ticket", "arguments": '{"t":1}'}}]}}]
    }
    trace: Final = build_trace(
        (
            row(
                "r1",
                "00.000",
                "01.000",
                tags=("agent:triage",),
                response=tool_response,
                messages=[{"role": "user", "content": "refund me"}],
            ),
            row("r2", "01.500", "02.000", tags=("agent:classifier", "parent_agent:triage"), spend=0.02),
            row("r3", "03.000", "04.000", tags=("agent:triage",), status="failure", error="boom"),
        )
    )

    assert {(agent["name"], agent["parent_agent"], agent["llm_calls"]) for agent in trace["agents"]} == {
        ("triage", None, 2),
        ("classifier", "triage", 1),
    }
    spans: Final = {span["span_id"]: span for span in trace["spans"]}
    assert spans["agent:classifier"]["parent_span_id"] == "agent:triage"
    assert spans["r2"]["parent_span_id"] == "agent:classifier"
    assert spans["r1:tool:0"]["parent_span_id"] == "r1"
    assert spans["r1:tool:0"]["name"] == "classify_ticket"
    assert math.isclose(spans["r1:tool:0"]["duration_ms"], 2000)
    assert spans["r3"]["status"] == "error" and spans["r3"]["error"] == "boom"
    summary: Final = trace["summary"]
    assert summary["name"] == "triage"
    assert summary["llm_calls"] == 3 and summary["tool_calls"] == 1 and summary["error_count"] == 1
    assert summary["status"] == "error"
    assert math.isclose(summary.get("spend") or 0, 0.04)
    assert summary["input_tokens"] == 30 and summary["output_tokens"] == 15
    assert summary["duration_ms"] == 4000
    assert summary["input_preview"] == "refund me"


def test_claude_agent_sdk_subagent_is_inferred_from_its_task_prompt() -> None:
    task: Final = {
        "content": [
            {"type": "tool_use", "name": "Agent", "input": {"subagent_type": "researcher", "prompt": "find MVCC facts"}}
        ]
    }
    reminder: Final = {"type": "text", "text": "<system-reminder>skills</system-reminder>"}
    trace: Final = build_trace(
        (
            row(
                "r1",
                "00.000",
                "01.000",
                user_agent=SDK_UA,
                response=task,
                messages=[{"role": "user", "content": [reminder, {"type": "text", "text": "research MVCC"}]}],
            ),
            row(
                "r2",
                "01.100",
                "02.000",
                user_agent=SDK_UA,
                messages=[{"role": "user", "content": [reminder, {"type": "text", "text": "find MVCC facts"}]}],
            ),
            row("r3", "02.100", "03.000", user_agent=SDK_UA, messages=[{"role": "user", "content": "research MVCC"}]),
        )
    )

    assert {(agent["name"], agent["parent_agent"]) for agent in trace["agents"]} == {
        ("claude-agent-sdk", None),
        ("researcher", "claude-agent-sdk"),
    }
    agents_by_span: Final = {span["span_id"]: span["agent"] for span in trace["spans"]}
    assert (agents_by_span["r1"], agents_by_span["r2"], agents_by_span["r3"]) == (
        "claude-agent-sdk",
        "researcher",
        "claude-agent-sdk",
    )
    assert trace["summary"].get("frameworks") == ("claude-agent-sdk",)
    assert trace["summary"]["input_preview"] == "research MVCC"


def test_claude_agent_sdk_subagent_wins_over_a_tag_inherited_from_its_parent() -> None:
    task: Final = {
        "content": [{"type": "tool_use", "name": "Task", "input": {"subagent_type": "researcher", "prompt": "dig"}}]
    }
    tags: Final = ("agent:support",)
    trace: Final = build_trace(
        (
            row(
                "r1",
                "00.000",
                "01.000",
                tags=tags,
                user_agent=SDK_UA,
                response=task,
                messages=[{"role": "user", "content": "help"}],
            ),
            row("r2", "01.100", "02.000", tags=tags, user_agent=SDK_UA, messages=[{"role": "user", "content": "dig"}]),
            row(
                "r3",
                "02.100",
                "03.000",
                tags=("agent:billing",),
                user_agent=SDK_UA,
                messages=[{"role": "user", "content": "dig"}],
            ),
        )
    )

    assert {(agent["name"], agent["parent_agent"]) for agent in trace["agents"]} == {
        ("support", None),
        ("researcher", "support"),
        ("billing", None),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "scope_args"),
    (pytest.param(ALL, (), id="admin"), pytest.param(OWNED, ("u1", ["t1"]), id="owned")),
)
async def test_list_traces_pages_by_last_activity_inside_the_reader_scope(
    scope: TraceScope, scope_args: tuple[object, ...]
) -> None:
    keys: Final = (
        {"trace_key": "s2", "last_activity": "2026-11-06 05:52:00"},
        {"trace_key": "s1", "last_activity": "2026-11-06 05:51:00"},
    )
    db: Final = FakeDatabase(
        keys, (row("r9", "00.000", "01.000", trace_key="s2"),), keys[1:], (row("r1", "00.000", "01.000"),)
    )
    storage: Final = SpendLogTraceStorage(lambda: db)

    first: Final = await storage.list_traces(scope, 0, 99, limit=1)
    second: Final = await storage.list_traces(scope, 0, 99, cursor=first["next_cursor"], limit=1)

    assert [summary["trace_id"] for summary in first["data"]] == ["s2"]
    assert first["next_cursor"] is not None
    assert [summary["trace_id"] for summary in second["data"]] == ["s1"]
    assert second["next_cursor"] is None
    assert db.calls[0][1] == (0, 99, *scope_args)
    assert db.calls[1][1] == (["s2"], *scope_args)
    assert db.calls[2][1] == (0, 99, *scope_args, "2026-11-06 05:52:00", "s2")
    assert ("team_id = ANY" in db.calls[0][0]) == (scope["all_teams"] == 0)


@pytest.mark.asyncio
async def test_span_reads_return_the_stored_request_and_error() -> None:
    rows: Final = (
        row(
            "r1",
            "00.000",
            "01.000",
            tags=("agent:triage",),
            messages=[{"role": "user", "content": "hi"}],
            response={"choices": [{"message": {"content": "hello"}}]},
        ),
        row("r2", "02.000", "03.000", tags=("agent:triage",), status="failure", error="rate limited"),
    )
    db: Final = FakeDatabase(rows, rows, rows)
    storage: Final = SpendLogTraceStorage(lambda: db)

    detail: Final = await storage.get_span("s1", "r1", ALL)
    error: Final = await storage.get_span_error("s1", "r2", ALL)
    missing: Final = await storage.get_span_error("s1", "r1", ALL)

    assert detail is not None and detail["output"] == "hello"
    assert detail["input_ui"] == {
        "kind": "messages",
        "messages": ({"role": "user", "content": "hi", "tool_calls": ()},),
    }
    assert error is not None and error["message"] == "rate limited" and error["total_chars"] == len("rate limited")
    assert missing is None


@pytest.mark.asyncio
async def test_agents_list_counts_runs_and_failures_per_agent() -> None:
    keys: Final = (
        {"trace_key": "s1", "last_activity": "2026-11-06 05:51:03"},
        {"trace_key": "s2", "last_activity": "2026-11-06 05:51:01"},
    )
    rows: Final = (
        row("r1", "00.000", "01.000", tags=("agent:triage",)),
        row("r2", "01.000", "02.000", tags=("agent:writer", "parent_agent:triage"), status="failure", error="x"),
        row("r3", "00.000", "01.000", trace_key="s2", tags=("agent:triage",)),
    )
    db: Final = FakeDatabase(keys, rows)
    storage: Final = SpendLogTraceStorage(lambda: db)

    agents: Final = await storage.trace_agents(
        TraceAgentsParams(all_teams=1, user_id="", team_ids=(), start_ms=0, end_ms=99, limit=10)
    )

    assert {(agent.agent_name, agent.runs, agent.failed_runs) for agent in agents} == {
        ("triage", 2, 0),
        ("writer", 1, 1),
    }


@pytest.mark.asyncio
async def test_no_database_reads_as_no_traces() -> None:
    page: Final = await SpendLogTraceStorage(lambda: None).list_traces(ALL, 0, 99)

    assert page == {"data": (), "next_cursor": None}


@pytest.mark.asyncio
async def test_trace_pages_include_every_call_and_keep_complete_totals() -> None:
    rows: Final = tuple(row(f"r{i:04d}", "00.000", "01.000", tags=("agent:triage",)) for i in range(601))
    storage: Final = SpendLogTraceStorage(lambda: FakeDatabase(rows))
    first: Final = await storage.get_trace("s1", ALL, page_size=500)
    assert first is not None and first["next_cursor"]
    second: Final = await storage.get_trace("s1", ALL, cursor=first["next_cursor"], page_size=500)
    assert second is not None and second["next_cursor"] is None
    assert len(first["spans"]) == 500
    assert {span["span_id"] for span in (*first["spans"], *second["spans"])} == {
        "agent:triage",
        *(item["request_id"] for item in rows),
    }
    assert first["summary"]["llm_calls"] == second["summary"]["llm_calls"] == 601
    assert math.isclose(first["summary"]["spend"] or 0, 6.01)


@pytest.mark.asyncio
async def test_cursor_rejects_a_changed_run_and_invalid_offsets() -> None:
    from litellm.tracing.errors import TraceChanged

    rows: Final = (row("r1", "00.000", "01.000"), row("r2", "02.000", "03.000"))
    db: Final = FakeDatabase(rows, (*rows, row("r3", "04.000", "05.000")))
    storage: Final = SpendLogTraceStorage(lambda: db)
    first: Final = await storage.get_trace("s1", ALL, page_size=1)
    assert first is not None and first["next_cursor"]
    with pytest.raises(TraceChanged):
        await storage.get_trace("s1", ALL, cursor=first["next_cursor"], page_size=1)


@pytest.mark.asyncio
async def test_oversized_reads_fail_instead_of_returning_incomplete_runs() -> None:
    from litellm.tracing.spend_logs import TRACE_ROW_LIMIT

    rows: Final = (row("r1", "00.000", "01.000"),) * (TRACE_ROW_LIMIT + 1)
    storage: Final = SpendLogTraceStorage(lambda: FakeDatabase(rows))
    with pytest.raises(OverflowError):
        await storage.get_trace("s1", ALL)


@pytest.mark.parametrize(
    "body",
    [
        {"input": "Find the invoice"},
        {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Find the invoice"}]}]},
    ],
)
@pytest.mark.asyncio
async def test_responses_inputs_are_visible_in_the_run_and_step(body: Mapping[str, object]) -> None:
    item: Final = SpendLogRow(**{**row("r1", "00.000", "01.000"), "proxy_server_request": json.dumps(body)})
    storage: Final = SpendLogTraceStorage(lambda: FakeDatabase((item,)))
    assert build_trace((item,))["summary"]["input_preview"] == "Find the invoice"
    detail: Final = await storage.get_span("s1", "r1", ALL)
    assert detail is not None
    assert "Find the invoice" in detail["input"]


@pytest.mark.parametrize(
    ("response", "body"),
    [
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            },
            {"messages": [{"role": "tool", "tool_call_id": "call-1", "content": "Invoice paid"}]},
        ),
        (
            {"content": [{"type": "tool_use", "id": "call-1", "name": "lookup", "input": {}}]},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "Invoice paid"}],
                    }
                ]
            },
        ),
        (
            {"output": [{"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"}]},
            {"input": [{"type": "function_call_output", "call_id": "call-1", "output": "Invoice paid"}]},
        ),
    ],
)
@pytest.mark.asyncio
async def test_tool_details_use_the_result_recorded_in_the_next_request(
    response: Mapping[str, object], body: Mapping[str, object]
) -> None:
    rows: Final = (
        row("r1", "00.000", "01.000", response=response),
        SpendLogRow(**{**row("r2", "02.000", "03.000"), "proxy_server_request": json.dumps(body)}),
    )
    storage: Final = SpendLogTraceStorage(lambda: FakeDatabase(rows))
    detail: Final = await storage.get_span("s1", "r1:tool:0", ALL)
    assert detail is not None and detail["output"] == "Invoice paid"


def test_registered_agent_names_are_readable_and_request_tags_override_them() -> None:
    item: Final = SpendLogRow(**{**row("r1", "00.000", "01.000", agent_id="agent-uuid"), "agent_name": "Support"})
    assert build_trace((item,))["summary"]["name"] == "Support"
    tagged: Final = SpendLogRow(**{**item, "request_tags": '["agent:Research"]'})
    assert build_trace((tagged,))["summary"]["name"] == "Research"


def test_offset_timestamps_preserve_the_actual_instant() -> None:
    item: Final = SpendLogRow(
        **{
            **row("r1", "00.000", "01.000"),
            "start_time": "2026-11-06T13:00:00+08:00",
            "end_time": "2026-11-06T13:00:01+08:00",
        }
    )
    assert build_trace((item,))["summary"]["start_time"] == "2026-11-06T05:00:00Z"


def test_request_agent_name_groups_calls_without_tags_or_registered_identity() -> None:
    named: Final = tuple(
        SpendLogRow(**{**row(f"r{i}", "00.000", "01.000"), "metadata": '{"agent_name": "Support agent"}'})
        for i in range(2)
    )
    trace: Final = build_trace(named)
    assert trace["summary"]["name"] == "Support agent"
    assert trace["summary"]["agent_names"] == ("Support agent",)
    assert trace["summary"]["llm_calls"] == 2
    assert {span["agent"] for span in trace["spans"]} == {"Support agent"}


def test_responses_api_input_is_read_from_the_stored_request() -> None:
    trace: Final = build_trace(
        (row("r1", "00.000", "01.000", tags=("agent:triage",), request={"model": "gpt-5", "input": "refund me"}),)
    )

    assert trace["summary"]["input_preview"] == "refund me"
