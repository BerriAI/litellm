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
    action: Literal["catalog", "read", "search"]
    execution_id: str | None = None
    span_ids: tuple[str, ...] = ()
    query: str = ""
    char_start: int = Field(default=0, ge=0)
    char_end: int | None = Field(default=None, ge=0)


class CatalogEntry(Record):
    execution: Execution
    spans: tuple[tuple[str, str, str, str, int], ...]
    partial: bool


class EvidenceReply(Record):
    request: EvidenceRequest
    catalog: tuple[CatalogEntry, ...] = ()
    parts: tuple[TracePart, ...] = ()
    error: str = ""


class EvidenceWorkspace(Record):
    sessions: tuple[SessionContent, ...] = Field(default=())

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
            )
            for session in self.sessions
        )

    def valid(self, evidence: Evidence) -> bool:
        return evidence_valid(evidence, self.parts)

    def respond(self, request: EvidenceRequest) -> EvidenceReply:
        sessions: Final = tuple(
            session for session in self.sessions if request.execution_id in (None, session.execution.id)
        )
        if request.execution_id is not None and not sessions:
            return EvidenceReply(request=request, error="Unknown execution_id. Use the supplied catalog.")
        if request.action == "catalog":
            return EvidenceReply(request=request, catalog=EvidenceWorkspace(sessions=sessions).catalog)
        if request.action == "search" and not request.query:
            return EvidenceReply(request=request, error="Search requires a nonempty literal text query.")
        if request.char_end is not None and request.char_end < request.char_start:
            return EvidenceReply(request=request, error="char_end must be at least char_start.")
        parts: Final = tuple(chain.from_iterable(session.parts for session in sessions))
        selected: Final = tuple(p for p in parts if not request.span_ids or p.span_id in request.span_ids)
        matches: Final = (
            tuple(p for p in selected if request.query.casefold() in p.content.casefold())
            if request.action == "search"
            else selected
        )
        missing: Final = frozenset(request.span_ids) - frozenset(p.span_id for p in selected)
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
                for part in matches
            ),
            error="Unknown span IDs: " + ", ".join(sorted(missing)) if missing else "",
        )


async def complete_page(execution: Execution, cursor: str, read: ReadContent) -> ExecutionContent:
    initial: Final = await read(execution.id, cursor, 1)
    pages: tuple[ExecutionContent, ...] = (initial,)  # rebind-ok: retain every source chunk until complete
    offset = 8001  # rebind-ok: source pages use one-based character offsets
    pending = frozenset(p.span_id for p in initial.parts if p.truncated)  # rebind-ok: track unfinished source spans
    while pending:
        page: ExecutionContent = await read(execution.id, cursor, offset)  # rebind-ok: fetch the next source chunk
        received: tuple[TracePart, ...] = tuple(  # rebind-ok: select the current unfinished spans
            p for p in page.parts if p.span_id in pending and p.content
        )
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
    return initial.model_copy(update=MappingProxyType({"parts": assembled}))


async def load_session(execution: Execution, read: ReadContent) -> SessionContent:
    cursor = ""  # rebind-ok: source cursor advances through every span page
    pages: tuple[ExecutionContent, ...] = ()  # rebind-ok: preserve source pages without discarding content
    seen: frozenset[str] = frozenset(("",))  # rebind-ok: detect a broken source cursor without imposing a read quota
    while True:
        page: ExecutionContent = await complete_page(execution, cursor, read)  # rebind-ok: advance source pages
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
