from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.lens.models import (
    ActivitySelection,
    Evidence,
    Execution,
    Job,
    Scope,
    execution_id,
    parse_execution,
)
from litellm.proxy.lens.sources import BUDGET, OMITTED, SourceReader, lens_access, selected_count
from litellm.rust_bridge.trace.generated.types import (
    QueryScope,
    RunOrder,
    Span,
    SpanText,
    Trace,
    TracePage,
    TraceSummary,
)
from litellm.rust_bridge.trace.storage import BY_REFERENCE, NEWEST, SpanPart
from tests.unit.proxy.lens.test_state import NOW

REF: Final = "A" * 64


def summary(trace_ref: str, trace_id: str = "trace", span_count: int = 1) -> TraceSummary:
    return TraceSummary(
        trace_id=trace_id,
        trace_ref=trace_ref,
        name="run",
        service="svc",
        input_preview="",
        start_time="2026-01-01T00:00:00Z",
        duration_ms=1.0,
        status="ok",
        span_count=span_count,
        agent_count=0,
        agent_invocations=0,
        llm_calls=0,
        tool_calls=0,
        error_count=0,
        input_tokens=0,
        output_tokens=0,
        models=(),
        spend=None,
    )


def span(span_id: str, parent: str | None = "root", status: str = "ok") -> Span:
    return Span(
        span_id=span_id,
        parent_span_id=parent,
        name=f"name-{span_id}",
        type="tool",
        agent="",
        framework="",
        start_offset_ms=0.0,
        duration_ms=1.0,
        status=status,
        error=None,
        error_truncated=False,
        input_preview="",
        model=None,
        input_tokens=0,
        output_tokens=0,
        litellm_request_id=None,
        spend=None,
    )


@dataclass
class FakeStorage:
    """Answers the general trace reads over in-memory runs, spans and texts."""

    runs: Sequence[TraceSummary] = ()
    spans: Sequence[Span] = ()
    texts: dict[tuple[str, SpanPart], str] = field(default_factory=dict)
    text_reads: list[tuple[tuple[str, ...], SpanPart, int, int | None, bool]] = field(default_factory=list)
    listed: list[tuple[QueryScope, str, str | None, int, RunOrder, tuple[str, ...]]] = field(default_factory=list)

    def _matching(self, q: str, trace_refs: Sequence[str]) -> list[TraceSummary]:
        return [run for run in self.runs if (not trace_refs or run.get("trace_ref") in trace_refs) and q in run["name"]]

    async def list_traces(
        self,
        scope: QueryScope,
        start_ms: int,
        end_ms: int,
        q: str = "",
        cursor: str | None = None,
        limit: int = 50,
        order: RunOrder = NEWEST,
        trace_refs: Sequence[str] = (),
    ) -> TracePage:
        self.listed.append((scope, q, cursor, limit, order, tuple(trace_refs)))
        ordered: Final = sorted(self._matching(q, trace_refs), key=lambda run: run.get("trace_ref", ""))
        after: Final = [run for run in ordered if cursor is None or run.get("trace_ref", "") > cursor][:limit]
        return TracePage(
            data=tuple(after),
            next_cursor=after[-1].get("trace_ref") if len(after) == limit else None,
        )

    async def count_traces(
        self, scope: QueryScope, start_ms: int, end_ms: int, q: str = "", trace_refs: Sequence[str] = ()
    ) -> int:
        return len(self._matching(q, trace_refs))

    async def get_trace(
        self,
        trace_id: str,
        scope: QueryScope,
        trace_ref: str = "",
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Trace | None:
        if not self.spans:
            return None
        return Trace(summary=summary(trace_ref, trace_id), agents=(), spans=tuple(self.spans))

    async def span_text(
        self,
        trace_id: str,
        trace_ref: str,
        span_ids: Sequence[str],
        part: SpanPart,
        scope: QueryScope,
        offset: int = 0,
        max_chars: int | None = None,
        tail: bool = False,
        contains: str | None = None,
    ) -> tuple[SpanText, ...]:
        self.text_reads.append((tuple(span_ids), part, offset, max_chars, tail))

        def read(text: str) -> str:
            if tail:
                return text[len(text) - (max_chars or 0) :]
            return text[offset : None if max_chars is None else offset + max_chars]

        return tuple(
            SpanText(
                span_id=span_id,
                text=read(self.texts[(span_id, part)]),
                total_chars=len(self.texts[(span_id, part)]),
                version="0" * 64,
                contains=contains is not None and contains in self.texts[(span_id, part)],
            )
            for span_id in span_ids
            if (span_id, part) in self.texts
        )


def refs(count: int) -> tuple[str, ...]:
    return tuple(f"{index:064X}" for index in range(count))


@pytest.mark.parametrize(
    ("percent", "cap", "eligible", "selected"),
    (
        (100, None, 7, 7),
        (10, None, 7, 1),
        (100, 3, 7, 3),
        (50, 10, 7, 4),
        (50, 10, 0, 0),
    ),
)
def test_selection_keeps_the_rounded_up_share_within_the_cap(
    percent: float, cap: int | None, eligible: int, selected: int
) -> None:
    selection: Final = ActivitySelection(sample_percent=percent, sample_size=cap)
    assert selected_count(selection, eligible) == selected


@pytest.mark.asyncio
@pytest.mark.parametrize(("preview", "expected"), ((False, 3), (True, 7)))
async def test_sample_pages_runs_in_reference_order_up_to_its_bound(preview: bool, expected: int) -> None:
    runs: Final = tuple(summary(ref) for ref in reversed(refs(7)))
    storage: Final = FakeStorage(runs=runs)
    reader: Final = SourceReader(storage)
    selection: Final = ActivitySelection(sample_size=3)
    pages: list[tuple[str, ...]] = []  # mutable-ok: collects the pages a cursor walk returns
    cursor = ""  # rebind-ok: follows the sample cursor until it ends
    while True:
        page = await reader.sample(Scope(all_teams=True), selection, 1, 2, cursor=cursor, page_size=2, preview=preview)
        pages.append(tuple(e.trace_ref for e in page.executions))
        assert (page.eligible, page.selected) == (7, 3)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    walked: Final = tuple(ref for page in pages for ref in page)  # comprehension-ok: flatten pages
    assert walked == refs(7)[:expected]
    assert all(order == BY_REFERENCE for *_, order, _ in storage.listed)


@pytest.mark.asyncio
async def test_picked_executions_restrict_the_sample_to_their_runs() -> None:
    storage: Final = FakeStorage(runs=tuple(summary(ref) for ref in refs(4)))
    picked: Final = (execution_id(refs(4)[2], "trace"), execution_id(refs(4)[0], "trace"))
    sample: Final = await SourceReader(storage).sample(
        Scope(team_id="alpha"), ActivitySelection(execution_ids=picked), 1, 2
    )
    assert {e.id for e in sample.executions} == set(picked)
    assert storage.listed[0][0] == {"kind": "owned", "user_id": "", "team_ids": ("alpha",)}
    assert storage.listed[0][-1] == (refs(4)[2], refs(4)[0])


@pytest.mark.asyncio
async def test_a_span_within_budget_reads_whole_and_a_long_one_keeps_both_ends() -> None:
    long_output: Final = "".join(f"{index:05d}" for index in range(2_000))
    storage: Final = FakeStorage(
        spans=(span("root", None), span("short"), span("long", status="error")),
        texts={
            ("root", "input"): "task",
            ("short", "input"): "question",
            ("short", "output"): "answer",
            ("long", "output"): long_output,
            ("long", "error"): "boom",
        },
    )
    execution: Final = Execution(id=execution_id(REF, "trace"), trace_id="trace", trace_ref=REF)
    content: Final = await SourceReader(storage).content(Scope(all_teams=True), execution)
    by_id: Final = {part.span_id: part for part in content.parts}
    assert by_id["short"].content == "Input: question\nOutput: answer\nStatus: ok "
    assert not by_id["short"].truncated
    long_part: Final = by_id["long"]
    assert long_part.truncated
    assert long_part.content == (
        "Input: \nOutput: "
        + long_output[: 5_000 // 3]
        + OMITTED
        + long_output[-(5_000 - 5_000 // 3) :]
        + "\nStatus: error boom"
    )
    assert content.partial
    tails: Final = [read for read in storage.text_reads if read[-1]]
    assert {read[0] for read in tails} == {("long",)}


@pytest.mark.asyncio
async def test_a_later_offset_reads_the_budget_window_of_the_full_text() -> None:
    output: Final = "x" * BUDGET + "TAIL" + "y" * 100
    storage: Final = FakeStorage(spans=(span("root", None),), texts={("root", "output"): output})
    execution: Final = Execution(id=execution_id(REF, "trace"), trace_id="trace", trace_ref=REF)
    labelled: Final = "Input: \nOutput: " + output + "\nStatus: ok "
    offset: Final = len("Input: \nOutput: ") + BUDGET - 2
    content: Final = await SourceReader(storage).content(Scope(all_teams=True), execution, offset=offset)
    assert content.parts[0].content == labelled[offset : offset + BUDGET]
    assert content.parts[0].content.startswith("xxTAIL")
    assert not content.parts[0].truncated
    assert not content.partial


@pytest.mark.asyncio
async def test_content_pages_forty_spans_at_a_time_in_trace_order() -> None:
    spans: Final = (span("root", None), *(span(f"s{index:02d}") for index in range(45)))
    storage: Final = FakeStorage(spans=spans)
    execution: Final = Execution(id=execution_id(REF, "trace"), trace_id="trace", trace_ref=REF)
    reader: Final = SourceReader(storage)
    first: Final = await reader.content(Scope(all_teams=True), execution)
    second: Final = await reader.content(Scope(all_teams=True), execution, cursor=first.next_cursor or "")
    assert [p.span_id for p in (*first.parts, *second.parts)] == [s["span_id"] for s in spans]
    assert second.next_cursor is None
    assert [(len(ids), part, tail) for ids, part, _, _, tail in storage.text_reads] == [
        (40, "input", False),
        (40, "output", False),
        (40, "error", False),
        (6, "input", False),
        (6, "output", False),
        (6, "error", False),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("quote", "found"), (("time", True), ("boom", True), ("absent", False)))
async def test_evidence_must_appear_in_the_span_input_output_or_error(quote: str, found: bool) -> None:
    storage: Final = FakeStorage(texts={("span", "output"): "timeout", ("span", "error"): "boom"})
    execution: Final = Execution(id=execution_id(REF, "trace"), trace_id="trace", trace_ref=REF)
    evidence: Final = Evidence(execution_id=execution.id, span_id="span", quote=quote)
    assert await SourceReader(storage).verify_evidence(Scope(all_teams=True), execution, evidence) is found


@pytest.mark.parametrize(
    ("scope", "access"),
    (
        (Scope(all_teams=True), {"kind": "all"}),
        (Scope(team_id="alpha", api_key_hash="key"), {"kind": "owned", "user_id": "", "team_ids": ("alpha",)}),
        (Scope(), {"kind": "owned", "user_id": "", "team_ids": ()}),
    ),
)
def test_lens_scope_reads_with_the_same_access_as_traces(scope: Scope, access: QueryScope) -> None:
    assert lens_access(scope) == access


def test_execution_ids_carry_reference_and_trace_id() -> None:
    assert parse_execution(execution_id(REF, "trace:with:colons")) == (REF, "trace:with:colons")
    with pytest.raises(ValueError, match=r"^Not an execution ID$"):
        parse_execution("short:trace")


def test_current_selection_preserves_search_and_execution_ids() -> None:
    selection: Final = ActivitySelection.model_validate(
        {
            "q": 'agent:"research agent" service:billing team:alpha attr.tenant.tier:gold',
            "execution_ids": [execution_id(REF, "trace")],
            "sample_percent": 50,
        }
    )
    assert selection.q == 'agent:"research agent" service:billing team:alpha attr.tenant.tier:gold'
    assert selection.execution_ids == (execution_id(REF, "trace"),)
    assert selection.sample_percent == 50


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source", "both"),
        ("agent_name", "research agent"),
        ("service", "billing"),
        ("team_id", "alpha"),
        ("filters", ({"key": "tenant.tier", "value": "gold"},)),
    ),
)
def test_selection_rejects_obsolete_fields(field: str, value: str | tuple[Mapping[str, str], ...]) -> None:
    with pytest.raises(ValidationError, match=field):
        ActivitySelection.model_validate({"q": "agent:research", field: value})


def test_saved_job_preserves_its_current_sample() -> None:
    job: Final = Job.model_validate(
        {
            "id": "job",
            "created_at": NOW,
            "start": NOW,
            "end": NOW,
            "settings": {"name": "Research", "model": "analysis", "context": "Find failures", "q": "agent:a"},
            "revision": 1,
            "sample": {
                "executions": [{"id": execution_id(REF, "trace"), "trace_id": "trace", "trace_ref": REF}],
                "eligible": 1,
            },
        }
    )
    assert job.sample is not None
    assert job.sample.executions == (Execution(id=execution_id(REF, "trace"), trace_id="trace", trace_ref=REF),)
    assert job.sample.eligible == 1
    assert job.settings.q == "agent:a"
    assert Job.model_validate_json(job.model_dump_json()) == job


def test_saved_job_rejects_an_obsolete_execution_shape() -> None:
    with pytest.raises(ValidationError, match="source"):
        Job.model_validate(
            {
                "id": "job",
                "created_at": NOW,
                "start": NOW,
                "end": NOW,
                "settings": {"name": "Research", "model": "analysis", "context": "Find failures"},
                "revision": 1,
                "sample": {"executions": [{"id": "x", "source": "traces"}], "eligible": 1},
            }
        )
