from datetime import date, datetime, timezone
from typing import Final

import pytest

from litellm.proxy.roi_calculator.github import SourceError
from litellm.proxy.roi_calculator.observed_workspace import (
    combine_observed,
    scoped_data,
    source_details,
    summarize_workspace,
)
from litellm.proxy.roi_calculator.settings import StoredConnection
from litellm.types.roi_calculator import ROIBranchSpend, ROISettings
from litellm.types.roi_observed import ObservedData, ObservedIssue, ObservedPeriodData, ObservedPull, ObservedWindow


def _source(settings: ROISettings, issues: tuple[ObservedIssue, ...] | None = ()) -> ObservedData:
    host: Final = "github.com" if settings.source_provider == "github" else "gitlab.com"
    period: Final = ObservedPeriodData(
        window=ObservedWindow(start=date(2026, 9, 1), end=date(2026, 9, 28)),
        pulls=tuple(
            ObservedPull(
                repo=repo,
                number=1,
                title="Change",
                url=f"https://{host}/{repo}/pull/1",
                author="ari",
                created_at=datetime(2026, 9, 10, 0, 0, 0, tzinfo=timezone.utc),
                merged_at=datetime(2026, 9, 10, 0, 0, 30, tzinfo=timezone.utc),
                source_repo=f"{host}/{repo}",
                source_branch="feature/one",
            )
            for repo in settings.repos
        ),
        issues=issues,
        spend=({"date": "2026-09-10", "user_id": "ari", "email": "ari@example.test", "spend": 60.0, "requests": 3},),
        branch_spend=tuple(
            ROIBranchSpend(repo=f"{host}/{repo}", branch="feature/one", spend=2, requests=1) for repo in settings.repos
        ),
    )
    data: Final = ObservedData(
        source_provider=settings.source_provider,
        source_api_url=settings.source_api_url,
        repos=settings.repos,
        captured_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
        gateway_emails=("ari@example.test", "bea@example.test"),
        current=period,
        previous=period,
        last_year=period,
    )
    return scoped_data(data, source_details(settings))


@pytest.mark.parametrize("same_person", (True, False))
def test_multiple_repos_and_providers_scope_usernames_and_count_spend_once(same_person: bool) -> None:
    github: Final = ROISettings(repos=("org/service", "org/docs"))
    gitlab: Final = ROISettings(source_provider="gitlab", repos=("org/service",))
    combined: Final = combine_observed(
        (_source(github), _source(gitlab)), ("github.com/org/service", "github.com/org/docs", "gitlab.com/org/service")
    )
    report: Final = summarize_workspace(
        combined,
        (
            StoredConnection(
                source_provider="github", api_url=github.source_api_url, identity_map={"ari": "ari@example.test"}
            ),
            StoredConnection(
                source_provider="gitlab",
                api_url=gitlab.source_api_url,
                identity_map={"ari": "ari@example.test" if same_person else "bea@example.test"},
            ),
        ),
    )
    person: Final = next(person for person in report.people if person.email == "ari@example.test")
    assert report.source_provider == "mixed"
    assert report.periods.current.merged_prs == 3
    assert report.periods.current.matched_users_recorded_spend == 60
    assert person.periods.current.merged_prs == (3 if same_person else 2)
    assert person.periods.current.gateway_recorded_spend == 60
    assert person.periods.current.recorded_spend_per_attributed_pr == (20 if same_person else 30)
    assert len(person.accounts) == (2 if same_person else 1)
    assert all(pull.branch_cost.spend == 2 for pull in report.pulls.current)
    assert report.unlinked_branches == ()


def test_empty_repository_is_a_successful_zero_activity_report() -> None:
    settings: Final = ROISettings(repos=())
    data: Final = combine_observed((_source(settings),), ("org/empty",))
    report: Final = summarize_workspace(data, ())
    assert report.periods.current.merged_prs == 0
    assert report.periods.current.new_bug_labeled_issues == 0
    assert report.periods.current.median_merge_hours is None
    assert report.people == () and report.pulls.current == ()


@pytest.mark.parametrize(
    ("issues", "expected"),
    (
        (None, None),
        ((), 0),
        (
            (
                ObservedIssue(
                    repo="org/service",
                    number=1,
                    created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
                    labels=("bug", "regression"),
                ),
            ),
            1,
        ),
    ),
)
def test_disabled_tracking_does_not_hide_other_connections_quality_counts(
    issues: tuple[ObservedIssue, ...] | None, expected: int | None
) -> None:
    github: Final = _source(ROISettings(repos=("org/docs",)), issues=None)
    gitlab: Final = _source(ROISettings(source_provider="gitlab", repos=("org/service",)), issues=issues)
    report: Final = summarize_workspace(combine_observed((github, gitlab), ("org/docs", "org/service")), ())
    assert tuple(
        (period.new_bug_labeled_issues, period.new_regression_labeled_issues)
        for period in (report.periods.current, report.periods.previous, report.periods.last_year)
    ) == ((expected, expected),) * 3
    assert report.periods.current.merged_prs == 2


def test_duplicate_connection_cannot_double_count_a_merged_change() -> None:
    data: Final = _source(ROISettings(repos=("org/service",)))
    with pytest.raises(SourceError, match="more than one connection"):
        combine_observed((data, data), data.repos)


def test_github_repository_selection_deduplicates_case_variants() -> None:
    settings: Final = ROISettings(repos=("Org/Service", "org/service", "org/docs", "org/docs.git"))
    assert settings.repos == ("Org/Service", "org/docs")
