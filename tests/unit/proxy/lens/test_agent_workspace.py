from types import MappingProxyType
from typing import Final

import pytest

from litellm.proxy.lens.agent_workspace import (
    EvidenceReadError,
    EvidenceRequest,
    EvidenceWorkspace,
    PythonRequest,
    ReviewRecord,
    SessionContent,
    load_workspace,
)
from litellm.proxy.lens.models import Evidence, Execution, ExecutionContent, Record, Sample, TracePart
from litellm.proxy.lens.python_tool import PythonInputError


class PythonData(Record):
    sessions: tuple[SessionContent, ...]
    reviews: tuple[ReviewRecord, ...]


async def python_data(workspace: EvidenceWorkspace, request: PythonRequest) -> PythonData:
    source: Final = workspace.python_data(request)
    assert not isinstance(source, str), source
    return PythonData.model_validate_json("".join([chunk async for chunk in source]))


def execution(identity: str, count: int = 1) -> Execution:
    return Execution(
        id=identity, source="traces", trace_id=identity, team_id="", name=identity, start_time="", span_count=count
    )


@pytest.mark.asyncio
async def test_original_content_is_reassembled_across_character_and_span_pages() -> None:
    run: Final = execution("run", 3)
    original: Final = "before " + "x" * 7991 + "split boundary" + "y" * 10000 + " final result"
    root: Final = TracePart(
        execution_id=run.id,
        span_id="a",
        name="root",
        kind="agent",
        content=original,
        start_time="2026-10-03 10:00:00.123456789",
        end_time="2026-10-03 10:00:01.123456789",
    )
    child: Final = TracePart(
        execution_id=run.id, span_id="b", parent_span_id="a", name="child", kind="agent", content="subagent evidence"
    )
    last: Final = TracePart(
        execution_id=run.id, span_id="c", parent_span_id="b", name="tool", kind="tool", content="child tool result"
    )

    async def read(identity: str, cursor: str, offset: int) -> ExecutionContent:
        assert identity == run.id
        assert offset > 0
        selected: Final = (last,) if cursor == "b" else (root, child)
        return ExecutionContent(
            execution=run,
            parts=tuple(
                p.model_copy(
                    update=MappingProxyType(
                        {
                            "content": p.content[offset - 1 : offset - 1 + 8000],
                            "truncated": len(p.content) > offset - 1 + 8000,
                        }
                    )
                )
                for p in selected
            ),
            next_cursor=None if cursor == "b" else "b",
        )

    workspace: Final = await load_workspace(Sample(executions=(run,), eligible=1), read, 1)
    assert all(not session.parts for session in workspace.sessions)
    assert await workspace.get_parts() == (root, child, last)
    assert await workspace.valid(Evidence(execution_id=run.id, span_id="a", quote="split boundary"))
    assert (await workspace.respond(EvidenceRequest(action="read", execution_id=run.id, span_ids=("c",)))).parts == (
        last,
    )
    assert (await workspace.respond(EvidenceRequest(action="search", query="SUBAGENT"))).parts == (child,)


@pytest.mark.asyncio
async def test_broken_pagination_fails_explicitly_instead_of_losing_evidence() -> None:
    run: Final = execution("run").model_copy(update=MappingProxyType({"root_seen": True}))

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=run, parts=(), next_cursor="repeat")

    workspace: Final = await load_workspace(Sample(executions=(run,), eligible=1), read, 1)
    with pytest.raises(EvidenceReadError, match="repeated a pagination cursor"):
        await workspace.get_parts()
    assert (await workspace.summary(run.id)).partial


@pytest.mark.asyncio
async def test_python_scopes_sessions_spans_and_reviewer_records_without_changing_original_evidence() -> None:
    first: Final = SessionContent(
        execution=execution("one"),
        partial=False,
        parts=(
            TracePart(execution_id="one", span_id="shared", name="tool", kind="tool", content="first"),
            TracePart(execution_id="one", span_id="extra", name="tool", kind="tool", content="other part"),
        ),
    )
    second: Final = SessionContent(
        execution=execution("two"),
        partial=False,
        parts=(TracePart(execution_id="two", span_id="shared", name="tool", kind="tool", content="second"),),
    )
    review: Final = ReviewRecord(execution_id="one", phase="initial", content="first findings")
    workspace: Final = EvidenceWorkspace(
        sessions=(first, second),
        reviews=(
            review,
            ReviewRecord(execution_id="two", phase="initial", content="second findings"),
        ),
    )
    selected: Final = await python_data(
        workspace,
        PythonRequest(
            action="python",
            code="print(data)",
            execution_ids=("one",),
            span_ids=("shared",),
        ),
    )
    assert selected == PythonData(
        sessions=(first.model_copy(update={"parts": (first.parts[0],)}),),
        reviews=(review,),
    )
    assert await workspace.get_parts() == (*first.parts, *second.parts)
    assert await python_data(workspace, PythonRequest(action="python", code="print(data)")) == PythonData(
        sessions=workspace.sessions,
        reviews=workspace.reviews,
    )
    assert (
        workspace.python_data(
            PythonRequest(
                action="python",
                code="print(data)",
                execution_ids=("missing",),
            )
        )
        == "Unknown execution IDs: missing"
    )
    with pytest.raises(PythonInputError, match="Unknown span IDs: extra"):
        await python_data(
            workspace, PythonRequest(action="python", code="print(data)", execution_ids=("two",), span_ids=("extra",))
        )


@pytest.mark.asyncio
async def test_metadata_and_global_catalog_do_not_fetch_any_sampled_trace() -> None:
    runs: Final = tuple(execution(str(index), 10000) for index in range(2500))

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        raise AssertionError("Metadata inspection fetched trace bodies")

    workspace: Final = await load_workspace(Sample(executions=runs, eligible=len(runs)), read, 8)
    assert len(workspace.sessions) == len(runs)
    assert all(not session.parts for session in workspace.sessions)
    summary: Final = await workspace.summary(runs[0].id)
    assert summary.characters is None and summary.span_count == runs[0].span_count
    catalog: Final = await workspace.respond(EvidenceRequest(action="catalog"))
    assert len(catalog.catalog) == len(runs)
    assert all(entry.characters is None and not entry.spans for entry in catalog.catalog)


@pytest.mark.asyncio
async def test_small_distant_range_does_not_collect_or_fetch_the_rest_of_a_large_span() -> None:
    from queue import SimpleQueue

    run: Final = execution("large")
    offsets: Final = SimpleQueue[int]()
    size: Final = 16000000

    async def read(_identity: str, _cursor: str, offset: int) -> ExecutionContent:
        offsets.put(offset)
        return ExecutionContent(
            execution=run,
            parts=(
                TracePart(
                    execution_id=run.id,
                    span_id="huge",
                    parent_span_id="subagent",
                    name="output",
                    kind="tool",
                    content="x" * min(8000, max(0, size - offset + 1)),
                    truncated=offset - 1 + 8000 < size,
                ),
            ),
        )

    workspace: Final = await load_workspace(Sample(executions=(run,), eligible=1), read, 1)
    reply: Final = await workspace.respond(
        EvidenceRequest(action="read", span_ids=("huge",), char_start=15000000, char_end=15001000)
    )
    assert reply.parts[0].content == "x" * 1000 and reply.parts[0].truncated
    assert reply.parts[0].parent_span_id == "subagent"
    assert tuple(offsets.get_nowait() for _ in range(offsets.qsize())) == (1, 15000001)


@pytest.mark.asyncio
async def test_python_evidence_stream_is_lazy_and_preserves_escaped_chunk_boundaries() -> None:
    from queue import SimpleQueue

    run: Final = execution("selected")
    calls: Final = SimpleQueue[int]()
    content: Final = "x" * 7999 + '"\\\ntracé' + "z" * 9000

    async def read(identity: str, _cursor: str, offset: int) -> ExecutionContent:
        assert identity == run.id
        calls.put(offset)
        return ExecutionContent(
            execution=run,
            parts=(
                TracePart(
                    execution_id=run.id,
                    span_id="nested",
                    parent_span_id="parent",
                    name="tool",
                    kind="tool",
                    content=content[offset - 1 : offset - 1 + 8000],
                    truncated=offset - 1 + 8000 < len(content),
                ),
            ),
        )

    workspace: Final = await load_workspace(Sample(executions=(run, execution("unselected")), eligible=2), read, 2)
    stream: Final = workspace.python_data(PythonRequest(action="python", code="print(data)", execution_ids=(run.id,)))
    assert not isinstance(stream, str)
    first: Final = await anext(stream)
    assert calls.empty()
    fragments: Final = (first, *tuple([chunk async for chunk in stream]))
    assert max(map(len, fragments)) < 16000
    parsed: Final = PythonData.model_validate_json("".join(fragments))
    assert len(parsed.sessions) == 1 and parsed.sessions[0].parts[0].content == content
    assert parsed.sessions[0].parts[0].parent_span_id == "parent"
    assert calls.qsize() == 3


@pytest.mark.asyncio
async def test_quotes_cross_chunks_but_cannot_cross_missing_content_markers() -> None:
    run: Final = execution("one")
    text: Final = "x" * 7997 + "exact quote" + "\n[... content omitted ...]\n" + "after"

    async def read(_identity: str, _cursor: str, offset: int) -> ExecutionContent:
        return ExecutionContent(
            execution=run,
            parts=(
                TracePart(
                    execution_id=run.id,
                    span_id="span",
                    name="tool",
                    kind="tool",
                    content=text[offset - 1 : offset - 1 + 8000],
                    truncated=offset - 1 + 8000 < len(text),
                ),
            ),
        )

    workspace: Final = await load_workspace(Sample(executions=(run,), eligible=1), read, 1)
    assert await workspace.valid(Evidence(execution_id=run.id, span_id="span", quote="exact quote"))
    assert not await workspace.valid(Evidence(execution_id=run.id, span_id="span", quote="content omitted"))
    assert not await workspace.valid(
        Evidence(execution_id=run.id, span_id="span", quote="quote\n[... content omitted ...]\nafter")
    )


@pytest.mark.asyncio
async def test_range_ending_at_source_page_boundary_does_not_fetch_the_next_page() -> None:
    run: Final = execution("one")

    async def read(_identity: str, _cursor: str, offset: int) -> ExecutionContent:
        assert offset == 1, "The complete requested range was already delivered"
        return ExecutionContent(
            execution=run,
            parts=(
                TracePart(
                    execution_id=run.id,
                    span_id="span",
                    name="tool",
                    kind="tool",
                    content="x" * 8000,
                    truncated=True,
                ),
            ),
        )

    workspace: Final = await load_workspace(Sample(executions=(run,), eligible=1), read, 1)
    reply: Final = await workspace.respond(EvidenceRequest(action="read", char_end=8000))
    assert reply.parts[0].content == "x" * 8000 and reply.parts[0].truncated
