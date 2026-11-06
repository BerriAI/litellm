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
        proxy_server_request=json.dumps({"messages": messages, "tags": list(tags)}),
        response=json.dumps(response or {}),
    )


class FakeDatabase:
    def __init__(self, *responses: Sequence[Mapping[str, object]]) -> None:
        self.responses: Final = responses
        self.calls: tuple[tuple[str, tuple[object, ...]], ...] = ()

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
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
