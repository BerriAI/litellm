from types import MappingProxyType
from typing import Final

import pytest

from litellm.proxy.lens.agent_workspace import EvidenceRequest, load_workspace
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
    assert workspace.respond(EvidenceRequest(action="search", query="SUBAGENT")).parts == (child,)


@pytest.mark.asyncio
async def test_broken_pagination_fails_explicitly_instead_of_losing_evidence() -> None:
    run: Final = execution("run")

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=run, parts=(), next_cursor="repeat")

    with pytest.raises(ValueError, match="repeated a pagination cursor"):
        await load_workspace(Sample(executions=(run,), eligible=1), read, 1)


def test_reviewer_catalog_search_and_ranges_keep_both_review_rounds_accessible() -> None:
    from litellm.proxy.lens.agent_workspace import EvidenceWorkspace, ReviewRecord

    records: Final = (
        ReviewRecord(execution_id="one", phase="initial", content="First interpretation with a shared clue"),
        ReviewRecord(execution_id="one", phase="revisited", content="Revised interpretation with a shared clue"),
        ReviewRecord(execution_id="two", phase="revisited", content="A different explanation"),
    )
    workspace: Final = EvidenceWorkspace(reviews=records)
    catalog: Final = workspace.respond(EvidenceRequest(action="review_catalog")).review_catalog
    assert tuple((item.execution_id, item.phase, item.characters) for item in catalog) == tuple(
        (record.execution_id, record.phase, len(record.content)) for record in records
    )
    assert workspace.respond(EvidenceRequest(action="search_reviews", query="SHARED CLUE")).reviews == records[:2]
    assert workspace.respond(EvidenceRequest(action="read_reviews", review_phase="revisited")).reviews == records[1:]
    selected: Final = workspace.respond(
        EvidenceRequest(
            action="read_reviews",
            execution_id="one",
            review_phase="initial",
            char_start=6,
            char_end=20,
        )
    ).reviews
    assert len(selected) == 1
    assert selected[0].content == records[0].content[6:20]
    assert workspace.respond(EvidenceRequest(action="read_reviews")).reviews == records


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "root_seen,explicit_partial,expected_partial",
    ((True, False, False), (False, False, True), (True, True, True)),
)
async def test_complete_character_reassembly_preserves_only_missing_evidence_partial_signals(
    root_seen: bool, explicit_partial: bool, expected_partial: bool
) -> None:
    run: Final = execution("long-session", 2).model_copy(update=MappingProxyType({"root_seen": root_seen}))
    original: Final = "Beginning " + "x" * 16001 + " complete ending"
    part: Final = TracePart(
        execution_id=run.id,
        span_id="root",
        parent_span_id="" if root_seen else "missing-root",
        name="coordinator",
        kind="agent",
        content=original,
    )
    child: Final = TracePart(
        execution_id=run.id, span_id="child", parent_span_id="root", name="tool", kind="tool", content="Short result"
    )

    async def read(_identity: str, _cursor: str, offset: int) -> ExecutionContent:
        start: Final = max(offset - 1, 0)
        truncated: Final = len(original) > start + 8000
        return ExecutionContent(
            execution=run,
            parts=(
                part.model_copy(
                    update=MappingProxyType(
                        {
                            "content": original[start : start + 8000],
                            "truncated": truncated,
                        }
                    )
                ),
                child.model_copy(update=MappingProxyType({"content": child.content[start : start + 8000]})),
            ),
            partial=not root_seen or truncated or explicit_partial,
        )

    workspace: Final = await load_workspace(Sample(executions=(run,), eligible=1), read, 1)
    assert workspace.parts == (part, child)
    assert workspace.sessions[0].partial is expected_partial
    assert (
        workspace.respond(EvidenceRequest(action="catalog", execution_id=run.id)).catalog[0].partial is expected_partial
    )
