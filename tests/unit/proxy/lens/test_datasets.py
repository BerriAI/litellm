import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.constants import LENS_DATASET_MAX_CASE_CHARS, LENS_DATASET_MAX_CASES
from litellm.proxy.lens.datasets import build_cases, case_id, export_jsonl, revision_problem
from litellm.proxy.lens.models import (
    BuildRequest,
    BuildResult,
    CaseSource,
    DatasetCase,
    DatasetMessage,
    DatasetToolCall,
    Evidence,
    Finding,
    FindingSource,
    SkippedCase,
    TextSource,
    TraceSource,
)
from litellm.proxy.lens.sources import execution_id
from litellm.rust_bridge.trace.generated.types import (
    Span,
    SpanDetail,
    SpanType,
    Trace,
    TraceSummary,
    UIContent,
    UIField,
    UIFields,
    UIMessage,
    UIMessages,
    UIText,
)

NOW: Final = datetime(2026, 1, 15, tzinfo=timezone.utc)


def messages(*items: UIMessage) -> UIMessages:
    return UIMessages(kind="messages", messages=items)


def detail(
    span_id: str,
    question: str,
    answer: str = "Done",
    attributes: Mapping[str, str] | None = None,
    input_ui: UIContent | None = None,
) -> SpanDetail:
    return SpanDetail(
        span_id=span_id,
        input_ui=input_ui
        or messages(UIMessage(role="system", content="Be terse"), UIMessage(role="user", content=question)),
        output_ui=messages(
            UIMessage(
                role="assistant",
                content=answer,
                tool_calls=({"name": "search", "arguments": json.dumps({"q": question})},),
            )
        ),
        input=question,
        output=answer,
        attributes=attributes or {},
    )


def span(span_id: str, kind: SpanType, offset: float) -> Span:
    return Span(
        span_id=span_id,
        parent_span_id=None,
        name=span_id,
        type=kind,
        agent="support",
        framework="",
        start_offset_ms=offset,
        duration_ms=1,
        status="ok",
        error=None,
        error_truncated=False,
        input_preview="",
        model=None,
        input_tokens=0,
        output_tokens=0,
        litellm_request_id=None,
        spend=None,
    )


def trace(*spans: Span) -> Trace:
    summary: Final = TraceSummary(
        trace_id="t1",
        name="run",
        service="svc",
        input_preview="",
        start_time="",
        duration_ms=1,
        status="ok",
        span_count=len(spans),
        agent_count=1,
        agent_invocations=1,
        llm_calls=len(spans),
        tool_calls=0,
        error_count=0,
        input_tokens=0,
        output_tokens=0,
        models=(),
        spend=None,
    )
    return Trace(summary=summary, agents=(), spans=spans)


def stored_finding(finding_id: str, *evidence: Evidence) -> Finding:
    return Finding(
        id=finding_id,
        title="Repeated failed searches",
        description="The agent repeats the same failed search",
        check_id="retries",
        evidence=evidence,
        first_seen=NOW,
        last_seen=NOW,
        revision=1,
    )


class FakeReader:
    def __init__(
        self,
        spans: Mapping[tuple[str, str], SpanDetail],
        traces: Mapping[str, Trace] | None = None,
        findings: tuple[Finding, ...] = (),
    ) -> None:
        self.spans: Final = spans
        self.traces: Final = traces or {}
        self.stored: Final = findings

    async def trace(self, trace_id: str, trace_ref: str) -> Trace | None:
        return self.traces.get(trace_id)

    async def span(self, trace_id: str, span_id: str, trace_ref: str) -> SpanDetail | None:
        return self.spans.get((trace_id, span_id))

    async def findings(self, lens_id: str, ids: tuple[str, ...]) -> tuple[Finding, ...]:
        return tuple(f for f in self.stored if f.id in ids)


async def build(reader: FakeReader, *sources: TraceSource | FindingSource | TextSource) -> BuildResult:
    return await build_cases(BuildRequest(sources=sources), reader, ())


@pytest.mark.asyncio
async def test_a_span_becomes_a_case_with_its_conversation_reply_and_tool_calls() -> None:
    reader: Final = FakeReader({("t1", "s1"): detail("s1", "refund?", "Refunded", {"agent.version": "v7"})})
    result: Final = await build(reader, TraceSource(trace_id="t1", trace_ref="ref", span_id="s1"))

    case: Final = result.cases[0]
    assert case.messages == (
        DatasetMessage(role="system", content="Be terse"),
        DatasetMessage(role="user", content="refund?"),
    )
    assert case.reply == "Refunded"
    assert case.tool_calls == (DatasetToolCall(name="search", arguments=json.dumps({"q": "refund?"})),)
    assert case.source == CaseSource(trace_id="t1", trace_ref="ref", span_id="s1")
    assert case.agent_version == "v7"
    assert case.id == case_id(case.messages, case.reply, case.tool_calls)


def history(tool: str) -> UIMessages:
    return messages(
        UIMessage(role="user", content="refund A1"),
        UIMessage(role="assistant", content="", tool_calls=({"name": tool, "arguments": '{"id":"A1"}'},)),
        UIMessage(role="tool", content="found"),
    )


@pytest.mark.asyncio
async def test_tool_calls_in_the_conversation_history_are_kept_and_change_the_case_id() -> None:
    reader: Final = FakeReader(
        {
            ("t1", "s1"): detail("s1", "refund A1", input_ui=history("lookup_order")),
            ("t1", "s2"): detail("s2", "refund A1", input_ui=history("cancel_order")),
        }
    )
    result: Final = await build(
        reader, TraceSource(trace_id="t1", span_id="s1"), TraceSource(trace_id="t1", span_id="s2")
    )

    assert result.cases[0].messages[1].tool_calls == (DatasetToolCall(name="lookup_order", arguments='{"id":"A1"}'),)
    assert result.cases[1].messages[1].tool_calls == (DatasetToolCall(name="cancel_order", arguments='{"id":"A1"}'),)
    assert result.skipped == ()


@pytest.mark.asyncio
async def test_agent_version_is_empty_when_the_span_does_not_report_one() -> None:
    reader: Final = FakeReader({("t1", "s1"): detail("s1", "refund?")})
    result: Final = await build(reader, TraceSource(trace_id="t1", span_id="s1"))
    assert result.cases[0].agent_version == ""


@pytest.mark.asyncio
async def test_whole_trace_uses_the_last_llm_span_that_has_a_conversation() -> None:
    reader: Final = FakeReader(
        {
            ("t1", "early"): detail("early", "first"),
            ("t1", "late"): detail("late", "second"),
            ("t1", "tool"): detail("tool", "not an llm"),
            ("t1", "text"): detail("text", "raw", input_ui=UIText(kind="text", text="raw")),
        },
        traces={
            "t1": trace(
                span("early", "llm", 1), span("late", "llm", 5), span("tool", "tool", 9), span("text", "llm", 7)
            )
        },
    )
    result: Final = await build(reader, TraceSource(trace_id="t1"))
    assert tuple(c.source.span_id for c in result.cases) == ("late",)
    assert result.cases[0].messages[-1].content == "second"


@pytest.mark.asyncio
async def test_finding_yields_one_case_per_distinct_evidence_span_located_by_its_execution() -> None:
    first: Final = execution_id("traces", "alpha", "t1", "ref1")
    second: Final = execution_id("traces", "alpha", "t2")
    reader: Final = FakeReader(
        {("t1", "s1"): detail("s1", "one"), ("t2", "s2"): detail("s2", "two")},
        findings=(
            stored_finding("f1", Evidence(execution_id=first, span_id="s1", quote="a")),
            stored_finding(
                "f2",
                Evidence(execution_id=first, span_id="s1", quote="b"),
                Evidence(execution_id=second, span_id="s2", quote="c"),
            ),
        ),
    )
    result: Final = await build(reader, FindingSource(lens_id="lens", finding_ids=("f1", "f2")))

    assert tuple(c.source for c in result.cases) == (
        CaseSource(trace_id="t1", trace_ref="ref1", span_id="s1", finding_id="f1", lens_id="lens"),
        CaseSource(trace_id="t2", trace_ref="", span_id="s2", finding_id="f2", lens_id="lens"),
    )
    assert result.skipped == ()


@pytest.mark.asyncio
async def test_text_jsonl_lines_become_cases_and_plain_text_becomes_one_user_message() -> None:
    lines: Final = "\n".join(
        (
            json.dumps({"messages": [{"role": "user", "content": "hi"}], "reply": "hello", "expected": "greet"}),
            "",
            json.dumps({"messages": [{"role": "user", "content": "bye"}]}),
        )
    )
    jsonl: Final = await build(FakeReader({}), TextSource(text=lines))
    plain: Final = await build(FakeReader({}), TextSource(text="Cancel my order\nplease"))

    assert tuple((c.messages[0].content, c.reply, c.expected) for c in jsonl.cases) == (
        ("hi", "hello", "greet"),
        ("bye", "", ""),
    )
    assert plain.cases[0].messages == (DatasetMessage(role="user", content="Cancel my order\nplease"),)


@pytest.mark.asyncio
async def test_building_again_from_the_same_sources_adds_nothing() -> None:
    reader: Final = FakeReader({("t1", "s1"): detail("s1", "one"), ("t1", "s2"): detail("s2", "two")})
    request: Final = BuildRequest(
        sources=(TraceSource(trace_id="t1", span_id="s1"), TraceSource(trace_id="t1", span_id="s2"))
    )
    first: Final = await build_cases(request, reader, ())
    second: Final = await build_cases(request, reader, first.cases)

    assert len(first.cases) == 2
    assert second.cases == ()
    assert tuple(s.reason for s in second.skipped) == ("duplicate", "duplicate")


@pytest.mark.asyncio
async def test_each_skip_reason_is_reported_against_its_source() -> None:
    huge: Final = "x" * (LENS_DATASET_MAX_CASE_CHARS + 1)
    reader: Final = FakeReader(
        {("t1", "s1"): detail("s1", "same"), ("t1", "big"): detail("big", huge), ("t1", "s3"): detail("s3", "new")}
    )
    full: Final = tuple(
        DatasetCase(id=str(i), messages=(DatasetMessage(role="user", content=str(i)),), source=CaseSource())
        for i in range(LENS_DATASET_MAX_CASES - 1)
    )
    result: Final = await build_cases(
        BuildRequest(
            sources=(
                TraceSource(trace_id="t1", span_id="missing"),
                TraceSource(trace_id="t1", span_id="big"),
                TraceSource(trace_id="t1", span_id="s1"),
                TraceSource(trace_id="t1", span_id="s1"),
                TraceSource(trace_id="t1", span_id="s3"),
            )
        ),
        reader,
        full,
    )

    assert tuple(c.source.span_id for c in result.cases) == ("s1",)
    assert tuple((s.source.span_id, s.reason) for s in result.skipped) == (
        ("missing", "no_content"),
        ("big", "too_large"),
        ("s1", "over_limit"),
        ("s3", "over_limit"),
    )


class CountingReader(FakeReader):
    def __init__(self, spans: Mapping[tuple[str, str], SpanDetail]) -> None:
        super().__init__(spans)
        self.reads: list[str] = []  # mutable-ok: records which spans the build actually fetched

    async def span(self, trace_id: str, span_id: str, trace_ref: str) -> SpanDetail | None:
        self.reads.append(span_id)
        return await super().span(trace_id, span_id, trace_ref)


@pytest.mark.asyncio
async def test_sources_past_the_case_limit_are_not_read() -> None:
    reader: Final = CountingReader({("t1", "s1"): detail("s1", "a"), ("t1", "s2"): detail("s2", "b")})
    full: Final = tuple(
        DatasetCase(id=str(i), messages=(DatasetMessage(role="user", content=str(i)),), source=CaseSource())
        for i in range(LENS_DATASET_MAX_CASES - 1)
    )
    result: Final = await build_cases(
        BuildRequest(sources=(TraceSource(trace_id="t1", span_id="s1"), TraceSource(trace_id="t1", span_id="s2"))),
        reader,
        full,
    )

    assert reader.reads == ["s1"]
    assert result.skipped == (SkippedCase(source=CaseSource(trace_id="t1", span_id="s2"), reason="over_limit"),)


@pytest.mark.asyncio
async def test_a_malformed_jsonl_line_is_skipped_without_losing_the_valid_lines() -> None:
    good: Final = json.dumps({"messages": [{"role": "user", "content": "refund?"}], "reply": "No"})
    result: Final = await build(FakeReader({}), TextSource(text=f'{good}\n{{not json\n{{"reply": "no messages"}}'))

    assert tuple(c.reply for c in result.cases) == ("No",)
    assert tuple(s.reason for s in result.skipped) == ("invalid", "invalid")


@pytest.mark.asyncio
async def test_an_exported_case_rebuilds_with_its_source_and_agent_version() -> None:
    original: Final = DatasetCase(
        id="",
        messages=(DatasetMessage(role="user", content="refund?"),),
        reply="No",
        expected="Decline politely",
        source=CaseSource(trace_id="t1", span_id="s1"),
        agent_version="v7",
    )
    result: Final = await build(FakeReader({}), TextSource(text=export_jsonl((original,))))

    rebuilt: Final = result.cases[0]
    assert (rebuilt.source, rebuilt.agent_version, rebuilt.expected) == (original.source, "v7", "Decline politely")


@pytest.mark.parametrize(
    "case",
    (
        DatasetCase(
            id="",
            messages=(DatasetMessage(role="user", content="q"),),
            expected="x" * LENS_DATASET_MAX_CASE_CHARS,
            source=CaseSource(),
        ),
        DatasetCase(
            id="",
            messages=(DatasetMessage(role="user", content="q", name="x" * LENS_DATASET_MAX_CASE_CHARS),),
            source=CaseSource(),
        ),
    ),
)
def test_expected_and_message_names_count_toward_the_case_size_limit(case: DatasetCase) -> None:
    assert revision_problem((case,)) is not None


def raw_span(span_id: str, input_ui: UIContent, output_ui: UIContent, raw_input: str, raw_output: str) -> SpanDetail:
    return SpanDetail(
        span_id=span_id, input_ui=input_ui, output_ui=output_ui, input=raw_input, output=raw_output, attributes={}
    )


@pytest.mark.asyncio
async def test_spans_without_chat_messages_fall_back_to_their_text_or_raw_input_and_output() -> None:
    fields: Final = UIFields(kind="fields", fields=(UIField(key="q", value="v"),))
    reader: Final = FakeReader(
        {
            ("t1", "text"): raw_span(
                "text", UIText(kind="text", text="shown"), UIText(kind="text", text="answer"), "asked", "raw answer"
            ),
            ("t1", "fields"): raw_span("fields", fields, fields, '{"q":"v"}', '{"ok":true}'),
        }
    )
    result: Final = await build(
        reader, TraceSource(trace_id="t1", span_id="text"), TraceSource(trace_id="t1", span_id="fields")
    )

    assert tuple((c.messages, c.reply) for c in result.cases) == (
        ((DatasetMessage(role="user", content="asked"),), "answer"),
        ((DatasetMessage(role="user", content='{"q":"v"}'),), '{"ok":true}'),
    )


@pytest.mark.asyncio
async def test_a_span_with_blank_input_and_output_is_skipped_as_no_content() -> None:
    blank: Final = UIText(kind="text", text="  ")
    reader: Final = FakeReader({("t1", "s1"): raw_span("s1", blank, blank, "  ", "")})
    result: Final = await build(reader, TraceSource(trace_id="t1", span_id="s1"))

    assert result.cases == ()
    assert result.skipped == (SkippedCase(source=CaseSource(trace_id="t1", span_id="s1"), reason="no_content"),)


@pytest.mark.asyncio
async def test_a_whole_trace_without_a_conversation_or_that_is_missing_is_skipped_as_no_content() -> None:
    reader: Final = FakeReader(
        {("t1", "text"): detail("text", "raw", input_ui=UIText(kind="text", text="raw"))},
        traces={"t1": trace(span("text", "llm", 1), span("gone", "llm", 2))},
    )
    result: Final = await build(reader, TraceSource(trace_id="t1"), TraceSource(trace_id="missing"))

    assert result.cases == ()
    assert result.skipped == (
        SkippedCase(source=CaseSource(trace_id="t1"), reason="no_content"),
        SkippedCase(source=CaseSource(trace_id="missing"), reason="no_content"),
    )


@pytest.mark.asyncio
async def test_finding_evidence_with_an_undecodable_execution_or_a_missing_span_is_skipped() -> None:
    located: Final = execution_id("traces", "alpha", "t1", "ref1")
    reader: Final = FakeReader(
        {},
        findings=(
            stored_finding(
                "f1",
                Evidence(execution_id="not-an-execution", span_id="s1", quote="a"),
                Evidence(execution_id=located, span_id="gone", quote="b"),
            ),
        ),
    )
    result: Final = await build(reader, FindingSource(lens_id="lens", finding_ids=("f1",)))

    assert result.cases == ()
    assert result.skipped == (
        SkippedCase(source=CaseSource(span_id="s1", finding_id="f1", lens_id="lens"), reason="no_content"),
        SkippedCase(
            source=CaseSource(trace_id="t1", trace_ref="ref1", span_id="gone", finding_id="f1", lens_id="lens"),
            reason="no_content",
        ),
    )


def test_export_writes_only_included_cases_one_per_line() -> None:
    kept: Final = DatasetCase(id="a", messages=(DatasetMessage(role="user", content="keep"),), source=CaseSource())
    dropped: Final = kept.model_copy(update={"id": "b", "included": False})
    lines: Final = export_jsonl((kept, dropped)).splitlines()
    assert tuple(DatasetCase.model_validate_json(line) for line in lines) == (kept,)
