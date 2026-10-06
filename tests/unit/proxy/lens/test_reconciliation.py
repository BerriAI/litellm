from typing import Final

import pytest

from litellm.proxy.lens.models import ModelRequest, ModelResult
from litellm.proxy.lens.reconciliation import FindingGroup, FindingGroups, reconcile_findings
from litellm.proxy.lens.state import merge_finding
from tests.unit.proxy.lens.test_state import NOW, finding, lens


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ("missing", "representative", "kind", "feedback"))
async def test_invalid_semantic_merges_fail_without_discarding_evidence_or_feedback(problem: str) -> None:
    from litellm.proxy.lens.analysis import AnalysisResponseError

    first: Final = merge_finding(lens(), finding("first"), 1, NOW, "first-run")
    second: Final = merge_finding(lens(), finding("second"), 1, NOW, "second-run").model_copy(
        update={"id": "second-id", "status": "dismissed", "reason": "Expected recovery"}
    )
    incoming: Final = finding("new").model_copy(update={"kind": "pattern" if problem == "kind" else "issue"})
    references: Final = ("new:0", f"saved:{first.id}", f"saved:{second.id}")
    invalid: Final = FindingGroups(
        groups=(
            FindingGroup(
                members=references[:1] if problem == "missing" else references,
                representative="invented" if problem == "representative" else "new:0",
            ),
        )
    )

    async def model(_request: ModelRequest) -> ModelResult:
        return ModelResult(content=invalid.model_dump_json(), cost=0)

    expected: Final = {
        "missing": "Partition every input",
        "representative": "representative must be a member",
        "kind": "Issues and positive patterns",
        "feedback": "conflicting user feedback",
    }
    with pytest.raises(AnalysisResponseError, match=expected[problem]):
        await reconcile_findings((incoming,), (first, second), model)


@pytest.mark.asyncio
async def test_reconciliation_unions_checks_and_evidence_and_reuses_prior_issue() -> None:
    saved: Final = merge_finding(lens(), finding("old-trace"), 1, NOW, "earlier-run")
    one: Final = finding("new-trace").model_copy(update={"title": "Failed lookup blocks the task"})
    two: Final = finding("another-trace").model_copy(
        update={"title": "The same lookup remains unavailable", "check_id": "blocked"}
    )

    async def model(request: ModelRequest) -> ModelResult:
        assert saved.id in request.prompt
        return ModelResult(
            content=FindingGroups(
                groups=(
                    FindingGroup(
                        members=("new:0", "new:1", f"saved:{saved.id}"),
                        representative="new:0",
                    ),
                )
            ).model_dump_json(),
            cost=0,
        )

    result: Final = await reconcile_findings((one, two), (saved,), model)
    assert len(result) == 1
    assert result[0].existing_finding_id == saved.id
    assert result[0].check_ids == ("blocked", "retries")
    assert result[0].evidence == (*one.evidence, *two.evidence)


@pytest.mark.asyncio
async def test_separate_semantic_groups_with_the_same_title_keep_independent_feedback() -> None:
    from litellm.proxy.lens.endpoints import merge_results
    from litellm.proxy.lens.models import Coverage, Result

    saved: Final = merge_finding(lens(), finding("old-trace"), 1, NOW, "earlier-run").model_copy(
        update={"status": "dismissed", "reason": "Expected recovery"}
    )
    incoming: Final = finding("new-trace")

    async def model(_request: ModelRequest) -> ModelResult:
        return ModelResult(
            content=FindingGroups(
                groups=(
                    FindingGroup(members=("new:0",), representative="new:0"),
                    FindingGroup(members=(f"saved:{saved.id}",), representative=f"saved:{saved.id}"),
                )
            ).model_dump_json(),
            cost=0,
        )

    drafts: Final = await reconcile_findings((incoming,), (saved,), model)
    updated: Final = merge_results(
        lens().model_copy(update={"findings": (saved,)}),
        Result(coverage=Coverage(), findings=drafts),
        1,
        NOW,
        "new-run",
    )
    assert len(updated.findings) == 2
    assert saved in updated.findings
    fresh: Final = next(item for item in updated.findings if item.id != saved.id)
    assert fresh.status == "open" and fresh.reason == ""
    assert fresh.occurrences == ("new-trace",)
