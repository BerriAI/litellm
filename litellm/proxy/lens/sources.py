import base64
import math
import time
from collections.abc import Awaitable, Mapping, Sequence
from itertools import accumulate, chain
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.proxy.lens.models import (
    ActivityAvailability,
    ActivitySelection,
    Evidence,
    Execution,
    ExecutionContent,
    Sample,
    Scope,
    TracePart,
    execution_id,
    parse_execution,
)
from litellm.rust_bridge.trace.generated.types import (
    AllQueryScope,
    OwnedQueryScope,
    QueryScope,
    RunOrder,
    Span,
    SpanText,
    Trace,
    TracePage,
    TraceSummary,
)
from litellm.rust_bridge.trace.storage import BY_REFERENCE, NEWEST, SpanPart


class Storage(Protocol):
    def list_traces(
        self,
        scope: QueryScope,
        start_ms: int,
        end_ms: int,
        q: str = "",
        cursor: str | None = None,
        limit: int = 50,
        order: RunOrder = NEWEST,
        trace_refs: Sequence[str] = (),
    ) -> Awaitable[TracePage]: ...

    def count_traces(
        self, scope: QueryScope, start_ms: int, end_ms: int, q: str = "", trace_refs: Sequence[str] = ()
    ) -> Awaitable[int]: ...

    def get_trace(
        self,
        trace_id: str,
        scope: QueryScope,
        trace_ref: str = "",
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Awaitable[Trace | None]: ...

    def span_text(
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
    ) -> Awaitable[tuple[SpanText, ...]]: ...


PAGE_SPANS: Final = 40
BUDGET: Final = 8_000
OMITTED: Final = "\n[... content omitted ...]\n"
PARTS: Final[tuple[tuple[SpanPart, str, int], ...]] = (
    ("input", "Input: ", 2_000),
    ("output", "\nOutput: ", 5_000),
    ("error", "\nStatus: ", 500),
)
Texts = Mapping[tuple[str, SpanPart], SpanText]


def lens_access(scope: Scope) -> QueryScope:
    if scope.all_teams:
        return AllQueryScope(kind="all")
    return OwnedQueryScope(kind="owned", user_id="", team_ids=(scope.team_id,) if scope.team_id else ())


def execution_of(run: TraceSummary) -> Execution:
    trace_ref: Final = run["id"]
    return Execution(
        id=execution_id(trace_ref, run["trace_id"]), trace_id=run["trace_id"], trace_ref=trace_ref, summary=run
    )


class SamplePosition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    cursor: str | None
    offset: int


_POSITION: Final = TypeAdapter(SamplePosition)


def _encode(position: SamplePosition) -> str:
    return base64.urlsafe_b64encode(position.model_dump_json().encode()).decode()


def _decode(cursor: str) -> SamplePosition:
    if not cursor:
        return SamplePosition(cursor=None, offset=0)
    try:
        return _POSITION.validate_json(base64.urlsafe_b64decode(cursor))
    except (ValueError, ValidationError) as error:
        raise ValueError("Invalid sample cursor") from error


def selected_count(selection: ActivitySelection, eligible: int) -> int:
    share: Final = math.ceil(eligible * selection.sample_percent / 100)
    return min(share, selection.sample_size) if selection.sample_size else share


def _status(span: Span) -> str:
    return f"{span['status']} "


def _label(span: Span, part: SpanPart, label: str) -> str:
    return label + _status(span) if part == "error" else label


def _pieces(span: Span, texts: Texts) -> tuple[tuple[SpanPart, str, SpanText | None], ...]:
    return tuple((part, _label(span, part, label), texts.get((span["span_id"], part))) for part, label, _ in PARTS)


def _total(pieces: tuple[tuple[SpanPart, str, SpanText | None], ...]) -> int:
    return sum(len(label) + (text["total_chars"] if text else 0) for _, label, text in pieces)


class SourceReader:
    def __init__(self, storage: Storage) -> None:
        self.storage: Final = storage

    async def availability(self, scope: Scope) -> ActivityAvailability:
        found: Final = await self.storage.count_traces(lens_access(scope), 0, int(time.time() * 1000) + 1)
        return ActivityAvailability(traces=found > 0)

    async def sample(
        self,
        scope: Scope,
        selection: ActivitySelection,
        start: int,
        end: int,
        cursor: str = "",
        page_size: int = 100,
        preview: bool = False,
    ) -> Sample:
        """The selected runs in reference order: a stable order unrelated to time, so a prefix is a fair sample."""
        access: Final = lens_access(scope)
        refs: Final = tuple(parse_execution(identity)[0] for identity in selection.execution_ids)
        eligible: Final = await self.storage.count_traces(access, start, end, selection.q, refs)
        selected: Final = selected_count(selection, eligible)
        bound: Final = eligible if preview else selected
        position: Final = _decode(cursor)
        limit: Final = min(page_size, bound - position.offset)
        if limit <= 0:
            return Sample(executions=(), eligible=eligible, selected=selected)
        page: Final = await self.storage.list_traces(
            access, start, end, selection.q, position.cursor, limit, BY_REFERENCE, refs
        )
        executions: Final = tuple(execution_of(run) for run in page["data"])
        offset: Final = position.offset + len(executions)
        next_page: Final = page["next_cursor"]
        return Sample(
            executions=executions,
            eligible=eligible,
            selected=selected,
            next_cursor=(
                _encode(SamplePosition(cursor=next_page, offset=offset)) if next_page and offset < bound else None
            ),
        )

    async def _texts(
        self, access: QueryScope, execution: Execution, span_ids: Sequence[str], max_chars: int, tail: bool = False
    ) -> Mapping[tuple[str, SpanPart], SpanText]:
        if not span_ids:
            return {}
        reads: Final[tuple[tuple[SpanPart, tuple[SpanText, ...]], ...]] = tuple(
            [
                (
                    part,
                    await self.storage.span_text(
                        execution.trace_id,
                        execution.trace_ref,
                        span_ids,
                        part,
                        access,
                        max_chars=max_chars if not tail or max_chars else budget - budget // 3,
                        tail=tail,
                    ),
                )
                for part, _, budget in PARTS
            ]
        )
        return {
            (text["span_id"], part): text for part, texts in reads for text in texts
        }  # comprehension-ok: flatten one read per part

    async def content(self, scope: Scope, execution: Execution, cursor: str = "", offset: int = 0) -> ExecutionContent:
        """Each span as `Input: … Output: … Status: …`. A first read keeps the start and end of parts over budget;
        a later `offset` reads the budget's worth of the full text from there."""
        access: Final = lens_access(scope)
        trace: Final = await self.storage.get_trace(execution.trace_id, access, execution.trace_ref)
        if trace is None:
            return ExecutionContent(execution=execution, parts=(), partial=True)
        spans: Final = trace["spans"]
        ids: Final = tuple(span["span_id"] for span in spans)
        start: Final = ids.index(cursor) + 1 if cursor in ids else 0 if not cursor else len(ids)
        page: Final = spans[start : start + PAGE_SPANS]
        page_ids: Final = tuple(span["span_id"] for span in page)
        heads: Final = await self._texts(access, execution, page_ids, BUDGET)
        long: Final = tuple(span["span_id"] for span in page if offset == 0 and _total(_pieces(span, heads)) > BUDGET)
        tails: Final = await self._texts(access, execution, long, 0, tail=True)
        parts: Final = tuple([await self._part(access, execution, span, heads, tails, offset) for span in page])
        root_seen: Final = any(span.get("parent_span_id") is None for span in spans)
        return ExecutionContent(
            execution=execution,
            parts=parts,
            next_cursor=page_ids[-1] if page_ids and start + len(page) < len(spans) else None,
            partial=not root_seen or any(part.truncated for part in parts),
        )

    async def _part(
        self, access: QueryScope, execution: Execution, span: Span, heads: Texts, tails: Texts, offset: int
    ) -> TracePart:
        pieces: Final = _pieces(span, heads)
        total: Final = _total(pieces)
        content, truncated = (
            (_excerpt(span, pieces, tails), total > BUDGET)
            if offset == 0
            else (await self._window(access, execution, span, pieces, offset), offset + BUDGET < total)
        )
        return TracePart(
            execution_id=execution.id,
            span_id=span["span_id"],
            parent_span_id=span.get("parent_span_id") or "",
            name=span["name"],
            kind=span["type"],
            content=content,
            truncated=truncated,
        )

    async def _window(
        self,
        access: QueryScope,
        execution: Execution,
        span: Span,
        pieces: tuple[tuple[SpanPart, str, SpanText | None], ...],
        offset: int,
    ) -> str:
        starts: Final = tuple(
            accumulate((len(label) + (text["total_chars"] if text else 0) for _, label, text in pieces), initial=0)
        )
        chunks: Final = [
            await self._window_piece(access, execution, span, piece, start, offset)
            for piece, start in zip(pieces, starts)
        ]
        return "".join(chunks)

    async def _window_piece(
        self,
        access: QueryScope,
        execution: Execution,
        span: Span,
        piece: tuple[SpanPart, str, SpanText | None],
        start: int,
        offset: int,
    ) -> str:
        part, label, text = piece
        end: Final = offset + BUDGET
        shown_label: Final = label[max(offset - start, 0) : max(end - start, 0)]
        text_start: Final = start + len(label)
        if text is None:
            return shown_label
        low, high = max(offset, text_start), min(end, text_start + text["total_chars"])
        if low >= high:
            return shown_label
        if high - text_start <= len(text["text"]):
            return shown_label + text["text"][low - text_start : high - text_start]
        read: Final = await self.storage.span_text(
            execution.trace_id,
            execution.trace_ref,
            (span["span_id"],),
            part,
            access,
            offset=low - text_start,
            max_chars=high - low,
        )
        return shown_label + (read[0]["text"] if read else "")

    async def verify_evidence(self, scope: Scope, execution: Execution, evidence: Evidence) -> bool:
        access: Final = lens_access(scope)
        found: Final = [
            await self.storage.span_text(
                execution.trace_id,
                execution.trace_ref,
                (evidence.span_id,),
                part,
                access,
                max_chars=0,
                contains=evidence.quote,
            )
            for part, _, _ in PARTS
        ]
        if any(text["contains"] for text in chain.from_iterable(found)):
            return True
        if len(evidence.quote) > BUDGET:
            return False
        trace: Final = await self.storage.get_trace(execution.trace_id, access, execution.trace_ref)
        if trace is None:
            return False
        span: Final = next((span for span in trace["spans"] if span["span_id"] == evidence.span_id), None)
        if span is None:
            return False
        heads: Final = await self._texts(access, execution, (evidence.span_id,), BUDGET)
        pieces: Final = _pieces(span, heads)
        tails: Final = await self._texts(
            access, execution, (evidence.span_id,) if _total(pieces) > BUDGET else (), BUDGET, tail=True
        )
        if evidence.quote in _excerpt(span, pieces, tails):
            return True
        if _total(pieces) <= BUDGET:
            return False
        prefixes: Final = tuple(label + (text["text"] if text else "") for _, label, text in pieces)
        endings: Final = tuple(
            (label if text is None or text["total_chars"] <= BUDGET else "")
            + (tail["text"] if (tail := tails.get((evidence.span_id, part))) else "")
            for part, label, text in pieces
        )
        boundaries: Final = tuple(end + prefix for end, prefix in zip(endings, prefixes[1:]))
        middle: Final = pieces[1][2]
        joined: Final = (
            endings[0] + prefixes[1] + prefixes[2] if middle is None or middle["total_chars"] <= BUDGET else ""
        )
        return evidence.quote in joined or any(evidence.quote in boundary for boundary in boundaries)


def _excerpt(span: Span, pieces: tuple[tuple[SpanPart, str, SpanText | None], ...], tails: Texts) -> str:
    if _total(pieces) <= BUDGET:
        return "".join(label + (text["text"] if text else "") for _, label, text in pieces)
    return "".join(
        label + _shortened(text, budget, tails.get((span["span_id"], part)))
        for (part, label, text), (_, _, budget) in zip(pieces, PARTS)
    )


def _shortened(text: SpanText | None, budget: int, tail: SpanText | None) -> str:
    if text is None:
        return ""
    if text["total_chars"] <= budget:
        return text["text"]
    return text["text"][: budget // 3] + OMITTED + (tail["text"][-(budget - budget // 3) :] if tail else "")
