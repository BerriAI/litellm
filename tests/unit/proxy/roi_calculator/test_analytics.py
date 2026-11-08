from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from litellm.proxy.roi_calculator.analytics import match_identity, normalize_email, summarize
from litellm.types.roi_calculator import (
    ROIPullRecord,
    ROIReport,
    ROISummaryMetrics,
    ROITrendDay,
)

EMPTY_IDENTITY_MAP: Final[Mapping[str, str]] = MappingProxyType({})


def _pull(
    number: int = 42,
    emails: tuple[str, ...] | None = None,
    estimate_status: Literal["estimated", "needs_review", "error"] = "estimated",
    hours: float | None = 4.0,
) -> ROIPullRecord:
    pull: Final[ROIPullRecord] = {
        "repo": "org/repo",
        "number": number,
        "title": "Fix timezone conversion",
        "url": f"https://github.com/org/repo/pull/{number}",
        "login": "alice",
        "emails": emails if emails is not None else ("alice@example.com",),
        "profile_email": "alice@example.com",
        "merged_at": "2026-09-12T12:00:00Z",
        "head_sha": "abcdef",
        "additions": 1,
        "deletions": 1,
        "changed_files": 1,
        "commit_count": 1,
        "incomplete_metadata": False,
        "estimate": {
            "status": estimate_status,
            "hours": hours,
            "reasoning": "Timezone conversion and regression verification.",
        },
        "cache_key": f"cache-{number}",
    }
    return pull


def _report(pulls: tuple[ROIPullRecord, ...] | None = None) -> ROIReport:
    report: Final[ROIReport] = {
        "mode": "live",
        "start": "2026-09-01",
        "end": "2026-09-30",
        "synced_at": "2026-09-30T12:00:00Z",
        "repos": ("org/repo",),
        "estimator_model": "test-estimator",
        "estimator_prompt": "Estimate effort.",
        "effort_basis": "without_ai",
        "spend": (
            {"date": "2026-09-12", "email": " Alice@Example.com ", "user_id": "u1", "spend": 12, "requests": 2},
            {"date": "2026-09-12", "email": "bob@example.com", "user_id": "u2", "spend": 8, "requests": 1},
            {"date": "2026-09-12", "email": "", "user_id": "shared", "spend": 5, "requests": 3},
        ),
        "pulls": pulls if pulls is not None else (_pull(),),
        "settings_fingerprint": "fingerprint",
    }
    return report


def test_summary_uses_matched_cohort_for_ratio_and_reports_coverage_and_excluded_spend() -> None:
    summary: Final = summarize(
        _report((_pull(), _pull(number=43, emails=("unknown@example.test",)))),
        EMPTY_IDENTITY_MAP,
    )

    expected_metrics: Final[ROISummaryMetrics] = {
        "matched_spend": 12,
        "output_hours": 4,
        "total_spend": 25,
        "total_output_hours": 8,
        "excluded_spend": 13,
        "cost_per_hour": 3,
        "hours_per_dollar": 1 / 3,
        "merged_prs": 2,
        "estimated_prs": 2,
        "matched_prs": 1,
        "cohort_people": 1,
        "people_with_prs": 2,
        "pending_prs": 0,
    }
    expected_trend: Final[ROITrendDay] = {
        "date": "2026-09-12",
        "spend": 12,
        "hours": 4,
        "prs": 1,
    }
    assert summary["metrics"] == expected_metrics
    assert summary["trend"] == (expected_trend,)
    assert summary["metrics"]["matched_prs"] / summary["metrics"]["merged_prs"] == 0.5


def test_manual_login_mapping_overrides_ambiguous_email_candidates() -> None:
    pull: Final = _pull(emails=("alice@example.com", "bob@example.com"))

    assert match_identity(
        pull,
        frozenset({"alice@example.com", "bob@example.com"}),
        EMPTY_IDENTITY_MAP,
    ) == (
        "",
        "ambiguous emails",
    )
    manual_map: Final[Mapping[str, str]] = MappingProxyType({"alice": "bob@example.com"})
    assert match_identity(
        pull,
        frozenset({"alice@example.com", "bob@example.com"}),
        manual_map,
    ) == ("bob@example.com", "manual")


def test_manual_mapping_recomputes_a_pull_without_email_evidence() -> None:
    report: Final = _report((_pull(emails=()),))

    before: Final = summarize(report, EMPTY_IDENTITY_MAP)
    manual_map: Final[Mapping[str, str]] = MappingProxyType({"alice": "alice@example.com"})
    after: Final = summarize(report, manual_map)

    assert before["metrics"]["output_hours"] == 0
    assert before["people"][0]["spend"] is None
    assert after["metrics"]["cost_per_hour"] == 3
    assert after["pulls"][0]["match_method"] == "manual"


def test_pending_estimates_exclude_the_person_from_the_ratio() -> None:
    report: Final = _report((_pull(), _pull(number=43, estimate_status="error", hours=None)))

    summary: Final = summarize(report, EMPTY_IDENTITY_MAP)

    assert summary["metrics"]["cost_per_hour"] is None
    assert summary["metrics"]["matched_spend"] == 0
    assert summary["metrics"]["total_output_hours"] == 4
    assert summary["metrics"]["pending_prs"] == 1


def test_email_normalization_rejects_private_or_unusable_addresses() -> None:
    assert normalize_email(" Alice+work@Example.com ") == "alice+work@example.com"
    assert normalize_email("123+alice@users.noreply.github.com") == ""
    assert normalize_email("alice") == ""
    assert normalize_email("") == ""


def test_branch_costs_are_independent_of_identity_and_never_count_reused_branches_twice() -> None:
    from litellm.types.roi_calculator import ROIBranchSpend

    base: Final = _pull(emails=())
    pulls: Final[tuple[ROIPullRecord, ...]] = (
        {**base, "number": 1, "source_repo": "gitlab.com/group/repo", "source_branch": "feature"},
        {**base, "number": 2, "source_repo": "gitlab.com/group/repo", "source_branch": "reused"},
        {**base, "number": 3, "source_repo": "gitlab.com/group/repo", "source_branch": "reused"},
        {**base, "number": 4, "source_repo": "gitlab.com/group/repo", "source_branch": "missing"},
        {**base, "number": 5, "source_repo": "gitlab.com/group/repo", "source_branch": "free"},
        {
            **_pull(emails=(), estimate_status="error", hours=None),
            "number": 6,
            "source_repo": "gitlab.com/group/repo",
            "source_branch": "pending",
        },
    )
    report: Final[ROIReport] = {
        **_report(pulls),
        "branch_spend": (
            ROIBranchSpend(repo="gitlab.com/group/repo", branch="feature", spend=12, requests=2),
            ROIBranchSpend(repo="gitlab.com/group/repo", branch="reused", spend=7, requests=1),
            ROIBranchSpend(repo="gitlab.com/group/repo", branch="free", spend=0, requests=1),
            ROIBranchSpend(repo="gitlab.com/group/repo", branch="pending", spend=9, requests=1),
        ),
    }
    result: Final = summarize(report, EMPTY_IDENTITY_MAP)
    costs: Final = {pull["number"]: pull["branch_cost"] for pull in result["pulls"]}
    assert costs[1].spend == 12
    assert costs[2].status == costs[3].status == "ambiguous"
    assert costs[2].spend is None
    assert costs[4].spend is None and costs[4].status == "unattributed"
    assert costs[5].spend == 0 and costs[5].status == "matched"
    assert result["branch_metrics"].cost_per_hour == 12 / 8
    assert result["branch_metrics"].unlinked_spend == 16
    assert result["branch_metrics"].matched_pulls == 3
    assert result["branch_metrics"].spend == 12
    assert result["metrics"]["matched_spend"] == 0
    incomplete: Final = summarize({**report, "unavailable_repos": ("other/repo",)}, EMPTY_IDENTITY_MAP)
    assert incomplete["branch_metrics"].cost_per_hour is None
