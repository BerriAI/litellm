from types import MappingProxyType
from typing import Final

import pytest

from litellm.proxy.lens.agent_workspace import (
    EvidenceRequest,
    EvidenceWorkspace,
    PythonData,
    PythonRequest,
    ReviewRecord,
    SearchMatch,
    SessionContent,
    load_workspace,
)
from litellm.proxy.lens.models import Evidence, Execution, ExecutionContent, Sample, TracePart


def execution(identity: str, count: int = 1) -> Execution:
    return Execution(
        id=identity, source="traces", trace_id=identity, team_id="", name=identity, start_time="", span_count=count
    )


@pytest.mark.asyncio
async def test_original_content_is_reassembled_across_character_and_span_pages() -> None:
    run: Final = execution("run", 3)
    original: Final = "before " + "x" * 7991 + "split boundary" + "y" * 10000 + " final result"
    root: Final = TracePart(execution_id=run.id, span_id="a", name="root", kind="agent", content=original)
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
    assert workspace.parts == (root, child, last)
    assert workspace.valid(Evidence(execution_id=run.id, span_id="a", quote="split boundary"))
    assert workspace.respond(EvidenceRequest(action="read", execution_id=run.id, span_ids=("c",))).parts == (last,)
    assert workspace.respond(EvidenceRequest(action="search", query="SUBAGENT")).matches == (
        SearchMatch(
            execution_id=run.id,
            span_id="b",
            parent_span_id="a",
            name="child",
            kind="agent",
            char_start=0,
            char_end=8,
            characters=len(child.content),
        ),
    )


def test_search_returns_literal_unicode_match_offsets_within_the_requested_scope() -> None:
    content: Final = "🧪 ÉrRoR[1].* before éRrOr[1].* after ÉRROR[1].*"
    part: Final = TracePart(
        execution_id="one", span_id="child", parent_span_id="root", name="nested tool", kind="tool", content=content
    )
    unrelated: Final = part.model_copy(update=MappingProxyType({"span_id": "unrelated"}))
    other_session: Final = part.model_copy(update=MappingProxyType({"execution_id": "two"}))
    workspace: Final = EvidenceWorkspace(
        sessions=(
            SessionContent(execution=execution("one", 2), parts=(part, unrelated), partial=False),
            SessionContent(execution=execution("two"), parts=(other_session,), partial=False),
        )
    )
    reply: Final = workspace.respond(
        EvidenceRequest(
            action="search",
            execution_id="one",
            span_ids=("child",),
            query="érRoR[1].*",
            char_start=content.index("before"),
            char_end=content.index(" after"),
        )
    )
    assert reply.parts == ()
    assert reply.matches == (
        SearchMatch(
            execution_id="one",
            span_id="child",
            parent_span_id="root",
            name="nested tool",
            kind="tool",
            char_start=content.index("éRrOr"),
            char_end=content.index(" after"),
            characters=len(content),
        ),
    )
    hit: Final = reply.matches[0]
    retrieved: Final = workspace.respond(
        EvidenceRequest(
            action="read",
            execution_id=hit.execution_id,
            span_ids=(hit.span_id,),
            char_start=hit.char_start,
            char_end=hit.char_end,
        )
    )
    assert retrieved.parts[0].content == "éRrOr[1].*"
    assert workspace.valid(
        Evidence(execution_id=hit.execution_id, span_id=hit.span_id, quote=retrieved.parts[0].content)
    )
    assert (
        workspace.respond(
            EvidenceRequest(
                action="search",
                execution_id="one",
                span_ids=("child",),
                query="érRoR[1].*",
                char_start=hit.char_start,
                char_end=hit.char_end - 1,
            )
        ).matches
        == ()
    )


def test_search_retains_every_late_match_without_returning_the_large_span() -> None:
    content: Final = "unrelated content " * 10000 + "needle " * 300
    part: Final = TracePart(
        execution_id="run", span_id="span", parent_span_id="parent", name="result", kind="tool", content=content
    )
    workspace: Final = EvidenceWorkspace(
        sessions=(
            SessionContent(
                execution=execution("run"),
                parts=(part,),
                partial=False,
            ),
        )
    )
    reply: Final = workspace.respond(EvidenceRequest(action="search", query="NEEDLE"))
    assert reply.parts == ()
    assert tuple((hit.char_start, hit.char_end) for hit in reply.matches) == tuple(
        (offset, offset + len("needle")) for offset in range(content.index("needle"), len(content), len("needle "))
    )
    assert all(hit.characters == len(content) and hit.parent_span_id == "parent" for hit in reply.matches)
    assert len(reply.model_dump_json()) < len(content)


def test_search_uses_nonoverlapping_literal_occurrences() -> None:
    part: Final = TracePart(execution_id="run", span_id="span", name="tool", kind="tool", content="aaaaa")
    workspace: Final = EvidenceWorkspace(
        sessions=(
            SessionContent(
                execution=execution("run"),
                parts=(part,),
                partial=False,
            ),
        )
    )
    reply: Final = workspace.respond(EvidenceRequest(action="search", query="AA"))
    assert tuple((hit.char_start, hit.char_end) for hit in reply.matches) == ((0, 2), (2, 4))


@pytest.mark.asyncio
async def test_broken_pagination_fails_explicitly_instead_of_losing_evidence() -> None:
    run: Final = execution("run")

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=run, parts=(), next_cursor="repeat")

    with pytest.raises(ValueError, match="repeated a pagination cursor"):
        await load_workspace(Sample(executions=(run,), eligible=1), read, 1)


def test_python_scopes_sessions_spans_and_reviewer_records_without_changing_original_evidence() -> None:
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
    selected: Final = workspace.python_data(
        PythonRequest(
            action="python",
            code="print(data)",
            execution_ids=("one",),
            span_ids=("shared",),
        )
    )
    assert selected == PythonData(
        sessions=(first.model_copy(update={"parts": (first.parts[0],)}),),
        reviews=(review,),
    )
    assert workspace.parts == (*first.parts, *second.parts)
    assert workspace.python_data(PythonRequest(action="python", code="print(data)")) == PythonData(
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
    assert (
        workspace.python_data(
            PythonRequest(
                action="python",
                code="print(data)",
                execution_ids=("two",),
                span_ids=("extra",),
            )
        )
        == "Unknown span IDs: extra"
    )
