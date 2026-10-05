import re
from contextlib import aclosing
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal

from pydantic import Field

from .analysis import ReadContent, concurrent_results, evidence_valid
from .models import Evidence, Execution, ExecutionContent, Record, Sample, TracePart


class SessionContent(Record):
    execution: Execution
    parts: tuple[TracePart, ...]
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


class PythonData(Record):
    sessions: tuple[SessionContent, ...]
    reviews: tuple["ReviewRecord", ...]


class CatalogEntry(Record):
    execution: Execution
    spans: tuple[tuple[str, str, str, str, int], ...]
    partial: bool
    characters: int


class ReviewRecord(Record):
    execution_id: str
    phase: Literal["initial", "revisited"]
    content: str


class ReviewIndex(Record):
    execution_id: str
    phase: Literal["initial", "revisited"]
    characters: int


class SearchMatch(Record):
    execution_id: str
    span_id: str
    parent_span_id: str
    name: str
    kind: str
    char_start: int
    char_end: int
    characters: int


class EvidenceReply(Record):
    request: EvidenceRequest
    catalog: tuple[CatalogEntry, ...] = ()
    parts: tuple[TracePart, ...] = ()
    error: str = ""
    review_catalog: tuple[ReviewIndex, ...] = ()
    reviews: tuple[ReviewRecord, ...] = ()
    matches: tuple[SearchMatch, ...] = ()


def search_matches(part: TracePart, pattern: re.Pattern[str], request: EvidenceRequest) -> tuple[SearchMatch, ...]:
    return tuple(
        SearchMatch(
            execution_id=part.execution_id,
            span_id=part.span_id,
            parent_span_id=part.parent_span_id,
            name=part.name,
            kind=part.kind,
            char_start=match.start(),
            char_end=match.end(),
            characters=len(part.content),
        )
        for match in pattern.finditer(
            part.content, request.char_start, request.char_end if request.char_end is not None else len(part.content)
        )
    )


class EvidenceWorkspace(Record):
    sessions: tuple[SessionContent, ...] = Field(default=())
    reviews: tuple[ReviewRecord, ...] = ()

    @property
    def parts(self) -> tuple[TracePart, ...]:
        return tuple(chain.from_iterable(session.parts for session in self.sessions))

    @property
    def catalog(self) -> tuple[CatalogEntry, ...]:
        return tuple(
            CatalogEntry(
                execution=session.execution,
                spans=tuple((p.span_id, p.parent_span_id, p.name, p.kind, len(p.content)) for p in session.parts),
                partial=session.partial,
                characters=sum(len(part.content) for part in session.parts),
            )
            for session in self.sessions
        )

    def python_data(self, request: PythonRequest) -> PythonData | str:
        sessions: Final = tuple(
            session
            for session in self.sessions
            if not request.execution_ids or session.execution.id in request.execution_ids
        )
        missing_sessions: Final = frozenset(request.execution_ids) - frozenset(s.execution.id for s in sessions)
        if missing_sessions:
            return "Unknown execution IDs: " + ", ".join(sorted(missing_sessions))
        selected: Final = tuple(
            session.model_copy(
                update=MappingProxyType(
                    {
                        "parts": tuple(
                            part for part in session.parts if not request.span_ids or part.span_id in request.span_ids
                        )
                    }
                )
            )
            for session in sessions
        )
        known_spans: Final = frozenset(part.span_id for part in chain.from_iterable(s.parts for s in selected))
        missing_spans: Final = frozenset(request.span_ids) - known_spans
        if missing_spans:
            return "Unknown span IDs: " + ", ".join(sorted(missing_spans))
        return PythonData(
            sessions=selected,
            reviews=tuple(
                review
                for review in self.reviews
                if not request.execution_ids or review.execution_id in request.execution_ids
            ),
        )

    def valid(self, evidence: Evidence) -> bool:
        return evidence_valid(evidence, self.parts)

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

    def respond(self, request: EvidenceRequest) -> EvidenceReply:
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
        if request.action == "catalog":
            catalog: Final = EvidenceWorkspace(sessions=sessions).catalog
            return EvidenceReply(
                request=request,
                catalog=tuple(
                    entry.model_copy(update=MappingProxyType({"spans": ()})) if request.execution_id is None else entry
                    for entry in catalog
                ),
            )
        if request.action == "search" and not request.query:
            return EvidenceReply(request=request, error="Search requires a nonempty literal text query.")
        parts: Final = tuple(chain.from_iterable(session.parts for session in sessions))
        selected: Final = tuple(p for p in parts if not request.span_ids or p.span_id in request.span_ids)
        missing: Final = frozenset(request.span_ids) - frozenset(p.span_id for p in selected)
        error: Final = "Unknown span IDs: " + ", ".join(sorted(missing)) if missing else ""
        if request.action == "search":
            pattern: Final = re.compile(re.escape(request.query), re.IGNORECASE)
            return EvidenceReply(
                request=request,
                matches=tuple(chain.from_iterable(search_matches(part, pattern, request) for part in selected)),
                error=error,
            )
        return EvidenceReply(
            request=request,
            parts=tuple(
                part.model_copy(
                    update=MappingProxyType(
                        {
                            "content": part.content[request.char_start : request.char_end],
                            "truncated": request.char_start > 0
                            or (request.char_end is not None and request.char_end < len(part.content)),
                        }
                    )
                )
                for part in selected
            ),
            error=error,
        )


async def complete_page(execution: Execution, cursor: str, read: ReadContent) -> ExecutionContent:
    initial: Final = await read(execution.id, cursor, 1)
    pages: tuple[ExecutionContent, ...] = (initial,)  # rebind-ok: retain every source chunk until complete
    offset = 8001  # rebind-ok: source pages use one-based character offsets
    pending = frozenset(p.span_id for p in initial.parts if p.truncated)  # rebind-ok: track unfinished source spans
    while pending:
        page: ExecutionContent = await read(execution.id, cursor, offset)
        received: tuple[TracePart, ...] = tuple(p for p in page.parts if p.span_id in pending and p.content)
        if frozenset(p.span_id for p in received) != pending:
            raise ValueError("Original trace content ended before all truncated spans were read")
        pages = (*pages, page.model_copy(update=MappingProxyType({"parts": received})))
        pending = frozenset(p.span_id for p in received if p.truncated)
        offset += 8000
    chunks: Final = tuple(chain.from_iterable(page.parts for page in pages))
    assembled: Final = tuple(
        part.model_copy(
            update=MappingProxyType(
                {"content": "".join(p.content for p in chunks if p.span_id == part.span_id), "truncated": False}
            )
        )
        for part in initial.parts
    )
    partial: Final = not execution.root_seen or any(
        page.partial and not any(part.truncated for part in page.parts) for page in pages
    )
    return initial.model_copy(update=MappingProxyType({"parts": assembled, "partial": partial}))


async def load_session(execution: Execution, read: ReadContent) -> SessionContent:
    cursor = ""  # rebind-ok: source cursor advances through every span page
    pages: tuple[ExecutionContent, ...] = ()  # rebind-ok: preserve source pages without discarding content
    seen: frozenset[str] = frozenset(("",))  # rebind-ok: detect a broken source cursor without imposing a read quota
    while True:
        page: ExecutionContent = await complete_page(execution, cursor, read)
        pages = (*pages, page)
        if page.next_cursor is None:
            return SessionContent(
                execution=execution,
                parts=tuple(chain.from_iterable(p.parts for p in pages)),
                partial=any(p.partial for p in pages),
            )
        if page.next_cursor in seen:
            raise ValueError("Original trace content repeated a pagination cursor before completion")
        cursor = page.next_cursor
        seen = seen | frozenset((cursor,))


async def load_workspace(sample: Sample, read: ReadContent, concurrency: int) -> EvidenceWorkspace:
    async def load(execution: Execution) -> SessionContent:
        return await load_session(execution, read)

    async with aclosing(concurrent_results(sample.executions, load, concurrency)) as results:
        sessions: Final = tuple([session async for session in results])
    indexed: Final = MappingProxyType({session.execution.id: session for session in sessions})
    return EvidenceWorkspace(sessions=tuple(indexed[execution.id] for execution in sample.executions))
