import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Final, Literal

from pydantic import Field

from .analysis import ReadContent
from .models import Evidence, Execution, ExecutionContent, Record, Sample, TracePart
from .python_tool import PythonInputError


class EvidenceReadError(ValueError):
    pass


class SessionContent(Record):
    execution: Execution
    parts: tuple[TracePart, ...] = ()
    partial: bool


class SessionSummary(Record):
    characters: int | None
    span_count: int
    partial: bool


class EvidenceRequest(Record):
    action: Literal["catalog", "read", "search", "review_catalog", "read_reviews", "search_reviews", "history"]
    execution_id: str | None = None
    span_ids: tuple[str, ...] = ()
    query: str = ""
    char_start: int = Field(default=0, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    review_phase: Literal["initial", "revisited"] | None = None
    turn_start: int = Field(default=0, ge=0)
    turn_end: int | None = Field(default=None, ge=0)
    include_initial: bool = False


class PythonRequest(Record):
    action: Literal["python"]
    code: str = Field(min_length=1)
    execution_ids: tuple[str, ...] = ()
    span_ids: tuple[str, ...] = ()


class CatalogEntry(Record):
    execution: Execution
    spans: tuple[tuple[str, str, str, str, int | None], ...]
    partial: bool
    characters: int | None


class ReviewRecord(Record):
    execution_id: str
    phase: Literal["initial", "revisited"]
    content: str


class ReviewIndex(Record):
    execution_id: str
    phase: Literal["initial", "revisited"]
    characters: int


class EvidenceReply(Record):
    request: EvidenceRequest
    catalog: tuple[CatalogEntry, ...] = ()
    parts: tuple[TracePart, ...] = ()
    error: str = ""
    review_catalog: tuple[ReviewIndex, ...] = ()
    reviews: tuple[ReviewRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class SourcePart:
    execution: Execution
    cursor: str
    part: TracePart


@dataclass(frozen=True, slots=True)
class EvidenceWorkspace:
    sessions: tuple[SessionContent, ...] = ()
    reviews: tuple[ReviewRecord, ...] = ()
    read: ReadContent | None = None
    partial_sessions: set[str] = field(  # mutable-ok: retain source-reported incompleteness across concurrent reads
        default_factory=set
    )
    read_errors: set[str] = field(  # mutable-ok: preserve source diagnostics when concurrent agents recover
        default_factory=set
    )
    verified_parts: dict[Evidence, TracePart] = field(  # mutable-ok: retain verified quote metadata for review previews
        default_factory=dict
    )

    def with_reviews(self, records: tuple[ReviewRecord, ...]) -> "EvidenceWorkspace":
        return replace(self, reviews=records)

    def _content_error(self, execution: Execution, message: str) -> EvidenceReadError:
        detail: Final = f"{message} (execution {execution.id}, trace {execution.trace_id})"
        self.partial_sessions.add(execution.id)
        self.read_errors.add(detail)
        return EvidenceReadError(detail)

    async def summary(self, execution_id: str) -> SessionSummary:
        session: Final = next(session for session in self.sessions if session.execution.id == execution_id)
        return SessionSummary(
            characters=None if self.read is not None else sum(len(part.content) for part in session.parts),
            span_count=session.execution.span_count if self.read is not None else len(session.parts),
            partial=session.partial or execution_id in self.partial_sessions,
        )

    async def _page(self, execution: Execution, cursor: str, offset: int) -> ExecutionContent:
        assert self.read is not None
        page: Final = await self.read(execution.id, cursor, offset)
        if page.partial and not any(part.truncated for part in page.parts):
            self.partial_sessions.add(execution.id)
        return page

    async def _sources(
        self, session: SessionContent, span_ids: tuple[str, ...] = ()
    ) -> AsyncGenerator[SourcePart, None]:
        if self.read is None:
            for part in session.parts:
                if not span_ids or part.span_id in span_ids:
                    yield SourcePart(session.execution, "", part)
            return
        cursor = ""  # rebind-ok: advance the gateway's source cursor without retaining content pages
        seen: frozenset[str] = frozenset(("",))  # rebind-ok: detect broken cursor cycles without a scan quota
        missing = frozenset(span_ids)  # rebind-ok: stop targeted reads when every requested span is found
        while True:
            page: ExecutionContent = await self._page(session.execution, cursor, 1)
            for part in page.parts:
                if not span_ids or part.span_id in span_ids:
                    yield SourcePart(session.execution, cursor, part)
                    missing = missing - frozenset((part.span_id,))
            if page.next_cursor is None or (span_ids and not missing):
                return
            if page.next_cursor in seen:
                raise self._content_error(
                    session.execution, "Original trace content repeated a pagination cursor before completion"
                )
            cursor = page.next_cursor
            seen = seen | frozenset((cursor,))

    async def _chunks(self, source: SourcePart, start: int = 0) -> AsyncGenerator[TracePart, None]:
        if self.read is None:
            yield source.part.model_copy(
                update=MappingProxyType({"content": source.part.content[start:], "truncated": False})
            )
            return
        initial: Final = await self._page(source.execution, source.cursor, start + 1) if start else None
        first: Final = (
            next((part for part in initial.parts if part.span_id == source.part.span_id), None)
            if initial is not None
            else source.part
        )
        if first is None:
            raise self._content_error(
                source.execution, "Original trace span disappeared while reading its character range"
            )
        yield first
        pending = first.truncated  # rebind-ok: follow complete character pages for this span
        offset = start + 8001  # rebind-ok: gateway character offsets are one-based
        while pending:
            page: ExecutionContent = await self._page(source.execution, source.cursor, offset)
            if (
                part := next((part for part in page.parts if part.span_id == source.part.span_id), None)
            ) is None or not part.content:
                raise self._content_error(
                    source.execution, "Original trace content ended before all truncated spans were read"
                )
            yield part
            pending = part.truncated
            offset += 8000

    async def _complete(self, source: SourcePart) -> TracePart:
        chunks: Final = tuple([chunk.content async for chunk in self._chunks(source)])
        return source.part.model_copy(update=MappingProxyType({"content": "".join(chunks), "truncated": False}))

    async def _ranged(self, source: SourcePart, request: EvidenceRequest) -> TracePart:
        chunks: tuple[str, ...] = ()  # rebind-ok: retain only the explicitly requested character range
        offset = request.char_start  # rebind-ok: track source position without assembling the full span
        beyond = False  # rebind-ok: distinguish an exact complete read from a range ending before source EOF
        async for piece in self._chunks(source, request.char_start):
            chunk: str = piece.content
            left: int = max(0, request.char_start - offset)
            right: int = len(chunk) if request.char_end is None else max(0, request.char_end - offset)
            if fragment := chunk[left:right]:
                chunks = (*chunks, fragment)
            offset += len(chunk)
            if request.char_end is not None and offset >= request.char_end:
                beyond = offset > request.char_end or piece.truncated
                break
        return source.part.model_copy(
            update=MappingProxyType(
                {
                    "content": "".join(chunks),
                    "truncated": request.char_start > 0 or beyond,
                }
            )
        )

    async def _contains(self, source: SourcePart, query: str, *, literal_quote: bool = False) -> bool:
        if not query:
            return True
        needle: Final = query if literal_quote else query.casefold()
        marker: Final = "\n[... content omitted ...]\n"
        delay: Final = len(marker) - 1 if literal_quote else 0
        retained: Final = len(needle) - 1 + delay
        tail = ""  # rebind-ok: retain only enough text to match across source chunks
        async for piece in self._chunks(source):
            chunk: str = piece.content
            segments: tuple[str, ...] = (
                tuple((tail + chunk).split(marker)) if literal_quote else (tail + chunk.casefold(),)
            )
            if any(needle in segment for segment in segments[:-1]):
                return True
            if needle in (segments[-1][:-delay] if delay else segments[-1]):
                return True
            tail = segments[-1][-retained:] if retained else ""
        return needle in tail

    async def get_parts(
        self, execution_ids: tuple[str, ...] = (), span_ids: tuple[str, ...] = ()
    ) -> tuple[TracePart, ...]:
        parts: tuple[TracePart, ...] = ()  # rebind-ok: explicit reads return every selected original span
        for session in self.sessions:
            if execution_ids and session.execution.id not in execution_ids:
                continue
            async for source in self._sources(session, span_ids):
                parts = (*parts, await self._complete(source))
        return parts

    def cited_parts(self, evidence: tuple[Evidence, ...]) -> tuple[TracePart, ...]:
        parts: tuple[TracePart, ...] = ()  # rebind-ok: retain only cited execution/span pairs
        for session in self.sessions:
            spans: tuple[str, ...] = tuple(
                dict.fromkeys(quote.span_id for quote in evidence if quote.execution_id == session.execution.id)
            )
            for span in spans:
                verified: tuple[TracePart, ...] = tuple(
                    self.verified_parts[quote]
                    for quote in evidence
                    if quote.execution_id == session.execution.id and quote.span_id == span
                )
                parts = (
                    *parts,
                    verified[0].model_copy(
                        update=MappingProxyType(
                            {
                                "content": "\n[... content omitted ...]\n".join(
                                    dict.fromkeys(p.content for p in verified)
                                )
                            }
                        )
                    ),
                )
        return parts

    async def valid(self, evidence: Evidence) -> bool:
        for session in self.sessions:
            if session.execution.id != evidence.execution_id:
                continue
            async for source in self._sources(session, (evidence.span_id,)):
                if await self._contains(source, evidence.quote, literal_quote=True):
                    self.verified_parts[evidence] = source.part.model_copy(
                        update=MappingProxyType({"content": evidence.quote, "truncated": True})
                    )
                    return True
        return False

    def python_data(self, request: PythonRequest) -> AsyncGenerator[str, None] | str:
        missing: Final = frozenset(request.execution_ids) - frozenset(session.execution.id for session in self.sessions)
        if missing:
            return "Unknown execution IDs: " + ", ".join(sorted(missing))
        return self._python_chunks(request)

    async def _python_chunks(self, request: PythonRequest) -> AsyncGenerator[str, None]:
        yield '{"sessions":['
        separator = ""  # rebind-ok: JSON array separators require no materialized selected corpus
        missing = frozenset(request.span_ids)  # rebind-ok: validate span selectors before finishing the input document
        for session in self.sessions:
            if request.execution_ids and session.execution.id not in request.execution_ids:
                continue
            yield separator + '{"execution":' + session.execution.model_dump_json() + ',"parts":['
            separator = ","
            part_separator = ""
            async for source in self._sources(session, request.span_ids):
                metadata: str = source.part.model_copy(update=MappingProxyType({"truncated": False})).model_dump_json(
                    exclude={"content"}
                )
                yield part_separator + metadata[:-1] + ',"content":"'
                part_separator = ","
                async for chunk in self._chunks(source):
                    yield json.dumps(chunk.content, ensure_ascii=False)[1:-1]
                yield '"}'
                missing = missing - frozenset((source.part.span_id,))
            yield '],"partial":' + json.dumps((await self.summary(session.execution.id)).partial) + "}"
        if missing:
            raise PythonInputError("Unknown span IDs: " + ", ".join(sorted(missing)))
        yield '],"reviews":['
        review_separator = ""  # rebind-ok: stream reviewer records in their original order
        for review in self.reviews:
            if not request.execution_ids or review.execution_id in request.execution_ids:
                yield review_separator + review.model_dump_json()
                review_separator = ","
        yield "]}"

    def review_reply(self, request: EvidenceRequest) -> EvidenceReply:
        records: Final = tuple(
            review
            for review in self.reviews
            if request.execution_id in (None, review.execution_id) and request.review_phase in (None, review.phase)
        )
        if request.action == "review_catalog":
            return EvidenceReply(
                request=request,
                review_catalog=tuple(
                    ReviewIndex(execution_id=record.execution_id, phase=record.phase, characters=len(record.content))
                    for record in records
                ),
            )
        if request.action == "search_reviews" and not request.query:
            return EvidenceReply(request=request, error="Review search requires a nonempty literal text query.")
        selected: Final = tuple(
            record
            for record in records
            if request.action != "search_reviews" or request.query.casefold() in record.content.casefold()
        )
        return EvidenceReply(
            request=request,
            reviews=tuple(
                record.model_copy(
                    update=MappingProxyType({"content": record.content[request.char_start : request.char_end]})
                )
                for record in selected
            ),
        )

    async def respond(self, request: EvidenceRequest) -> EvidenceReply:
        if request.char_end is not None and request.char_end < request.char_start:
            return EvidenceReply(request=request, error="char_end must be at least char_start.")
        if request.action in ("review_catalog", "read_reviews", "search_reviews"):
            return self.review_reply(request)
        if request.action == "history":
            return EvidenceReply(request=request, error="History is available through the agent runtime.")
        sessions: Final = tuple(
            session for session in self.sessions if request.execution_id in (None, session.execution.id)
        )
        if request.execution_id is not None and not sessions:
            return EvidenceReply(request=request, error="Unknown execution_id. Use the supplied catalog.")
        if request.action == "search" and not request.query:
            return EvidenceReply(request=request, error="Search requires a nonempty literal text query.")
        catalog: tuple[CatalogEntry, ...] = ()  # rebind-ok: explicit catalog requests retain metadata only
        parts: tuple[TracePart, ...] = ()  # rebind-ok: preserve unrestricted explicit read/search results
        missing = frozenset(request.span_ids)  # rebind-ok: report unknown selectors after traversing selected sessions
        for session in sessions:
            if request.action == "catalog":
                metadata: tuple[tuple[str, str, str, str, int | None], ...] = (
                    tuple(
                        [
                            (
                                source.part.span_id,
                                source.part.parent_span_id,
                                source.part.name,
                                source.part.kind,
                                None if source.part.truncated else len(source.part.content),
                            )
                            async for source in self._sources(session)
                        ]
                    )
                    if request.execution_id is not None
                    else ()
                )
                summary: SessionSummary = await self.summary(session.execution.id)
                catalog = (
                    *catalog,
                    CatalogEntry(
                        execution=session.execution,
                        spans=metadata,
                        partial=summary.partial,
                        characters=summary.characters,
                    ),
                )
                continue
            async for source in self._sources(session, request.span_ids):
                missing = missing - frozenset((source.part.span_id,))
                if request.action == "search" and not await self._contains(source, request.query):
                    continue
                parts = (*parts, await self._ranged(source, request))
        return EvidenceReply(
            request=request,
            catalog=catalog,
            parts=parts,
            error="Unknown span IDs: " + ", ".join(sorted(missing)) if missing and request.action != "catalog" else "",
        )


async def load_workspace(sample: Sample, read: ReadContent, _concurrency: int) -> EvidenceWorkspace:
    return EvidenceWorkspace(
        sessions=tuple(
            SessionContent(execution=execution, partial=not execution.root_seen) for execution in sample.executions
        ),
        read=read,
    )
