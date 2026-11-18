from datetime import date, datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.roi_calculator.observed_analytics import (
    declared_requester,
    merge_hours,
    reporting_windows,
    summarize_observed,
)
from litellm.types.roi_observed import ObservedData, ObservedIssue, ObservedPeriodData, ObservedPull, ObservedWindow

_NOW: Final = datetime(2026, 10, 3, tzinfo=timezone.utc)
_WINDOW: Final = ObservedWindow(start=date(2026, 9, 5), end=date(2026, 10, 2))


def _pull(login: str, number: int = 1, repo: str = "org/service", **fields: object) -> ObservedPull:
    return ObservedPull.model_validate(
        {
            "repo": repo,
            "number": number,
            "title": "Ship change",
            "url": f"https://github.com/{repo}/pull/{number}",
            "author": login,
            "created_at": "2026-09-10T00:00:00Z",
            "merged_at": "2026-09-10T00:01:19Z",
            **fields,
        }
    )


def _data(current: ObservedPeriodData, previous: ObservedPeriodData | None = None) -> ObservedData:
    empty: Final = ObservedPeriodData(window=_WINDOW, pulls=(), issues=(), spend=())
    return ObservedData(
        source_provider="github",
        source_api_url="https://api.github.com",
        repos=("org/service",),
        captured_at=_NOW,
        gateway_emails=("ari@example.test", "bea@example.test"),
        current=current,
        previous=previous or empty,
        last_year=empty,
    )


def test_multiple_accounts_share_one_cost_denominator_and_pr_numbers_are_scoped_to_repositories() -> None:
    direct: Final = _pull("ari", profile_email="ari@example.test")
    alternate: Final = _pull("old-ari", repo="org/other")
    agent: Final = _pull("devin-ai[bot]", 3, agent=True, requester="old-ari")
    unowned: Final = _pull("devin-ai[bot]", 4, agent=True)
    period: Final = ObservedPeriodData(
        window=_WINDOW,
        pulls=(direct, alternate, agent, unowned),
        issues=(),
        spend=({"email": "ari@example.test", "spend": 90.0, "date": "2026-09-10", "user_id": "ari", "requests": 1},),
    )
    report: Final = summarize_observed(_data(period), {"old-ari": "ari@example.test"})
    assert len(report.people) == 1
    person: Final = report.people[0]
    assert person.logins == ("ari", "old-ari")
    assert person.periods.current.pr_urls == (direct.url, alternate.url, agent.url)
    assert (person.periods.current.direct_authored, person.periods.current.declared_agent_owned) == (2, 1)
    assert person.periods.current.recorded_spend_per_attributed_pr == 30.0
    assert person.periods.current.prs_per_week == 0.75
    assert (report.periods.current.merged_prs, report.periods.current.matched_internal_prs) == (4, 3)
    assert report.periods.current.agents_without_requester == 1


def test_missing_spend_stays_unknown_and_a_recorded_zero_stays_zero() -> None:
    period: Final = ObservedPeriodData(
        window=_WINDOW,
        pulls=(_pull("ari"), _pull("bea", 2)),
        issues=None,
        spend=({"email": "bea@example.test", "spend": 0.0, "date": "2026-09-10", "user_id": "bea", "requests": 1},),
    )
    report: Final = summarize_observed(_data(period), {"ari": "ari@example.test", "bea": "bea@example.test"})
    assert tuple(person.periods.current.recorded_spend_per_attributed_pr for person in report.people) == (None, 0.0)
    assert tuple(person.periods.current.spend_observation for person in report.people) == (
        "no_records",
        "records_present",
    )
    assert report.periods.current.new_bug_labeled_issues is None
    assert report.periods.previous.new_bug_labeled_issues == 0
    assert report.people[0].periods.previous.recorded_spend_per_attributed_pr is None


def test_manual_links_override_automatic_matches_and_removal_suppresses_rematching() -> None:
    period: Final = ObservedPeriodData(
        window=_WINDOW, pulls=(_pull("ari", profile_email="ari@example.test"),), issues=(), spend=()
    )
    data: Final = _data(period)
    assert summarize_observed(data, {}).people[0].email == "ari@example.test"
    assert summarize_observed(data, {"ari": "bea@example.test"}).people[0].email == "bea@example.test"
    removed: Final = summarize_observed(data, {}, ("ari",))
    assert removed.people == ()
    assert removed.unmatched_logins == ("ari",)
    assert summarize_observed(data, {"ari": "bea@example.test"}, ("ari",)).people[0].email == "bea@example.test"


def test_conflicting_public_emails_do_not_silently_choose_an_owner() -> None:
    current: Final = ObservedPeriodData(
        window=_WINDOW, pulls=(_pull("ari", profile_email="ari@example.test"),), issues=(), spend=()
    )
    previous: Final = current.model_copy(update={"pulls": (_pull("ari", profile_email="bea@example.test"),)})
    report: Final = summarize_observed(_data(current, previous), {})
    assert report.people == ()
    assert report.unmatched_logins == ("ari",)


def test_quality_counts_labelled_issues_once_and_does_not_infer_bugs_from_pr_titles() -> None:
    period: Final = ObservedPeriodData(
        window=_WINDOW,
        pulls=(_pull("ari", title="fix: critical bug"), _pull("ari", 2, title='Revert "change"')),
        issues=tuple(
            ObservedIssue(repo="org/service", number=index, created_at=_NOW, labels=labels)
            for index, labels in enumerate(
                (
                    ("BUG", "kind:bug"),
                    ("type::bug", "type::regression"),
                    ("debug",),
                )
            )
        ),
        spend=(),
    )
    report: Final = summarize_observed(_data(period), {})
    assert (
        report.periods.current.new_bug_labeled_issues,
        report.periods.current.new_regression_labeled_issues,
        report.periods.current.explicitly_titled_revert_prs,
    ) == (2, 1, 1)


@pytest.mark.parametrize(
    "created,merged,expected",
    (
        (None, "2026-09-10T00:00:16Z", None),
        ("2026-09-10T00:00:00Z", "2026-09-10T00:00:16Z", 16 / 3600),
        ("2026-09-10T00:00:00Z", "2026-09-10T00:00:00Z", 0),
        ("2026-09-10T00:00:01Z", "2026-09-10T00:00:00Z", None),
        ("2026-09-10T00:00:00", "2026-09-10T00:00:16Z", None),
    ),
)
def test_merge_duration_preserves_seconds_and_rejects_invalid_intervals(
    created: str | None, merged: str, expected: float | None
) -> None:
    assert merge_hours(_pull("ari", created_at=created, merged_at=merged)) == expected


@pytest.mark.parametrize(
    "now", (datetime(2024, 3, 1, tzinfo=timezone.utc), _NOW, _NOW.replace(tzinfo=timezone(timedelta(hours=14))))
)
@pytest.mark.parametrize("days", (1, 7, 28, 90, 366))
def test_reporting_windows_have_equal_lengths_and_exclude_today(now: datetime, days: int) -> None:
    current, previous, yearly = reporting_windows(now, days)
    assert all((window.end - window.start).days + 1 == days for window in (current, previous, yearly))
    assert current.end == now.astimezone(timezone.utc).date() - timedelta(days=1)
    assert previous.end == current.start - timedelta(days=1)
    assert yearly.end.year == current.end.year - 1
    assert yearly.end.month == current.end.month


@pytest.mark.parametrize(
    "author,body,expected",
    (
        ("devin-ai[bot]", "Requested by: @Ari", "ari"),
        ("devin-ai-integration", "Requested by: @Ari", "ari"),
        ("devin-ai", "Requested by: @Ari", "ari"),
        ("human", "Requested by: @ari", ""),
        ("devin-ai[bot]", "Requested by: @ari\nRequested by: @bea", ""),
        ("devin-ai[bot]", "Mentions @ari", ""),
    ),
)
def test_agent_ownership_requires_one_explicit_requester(author: str, body: str, expected: str) -> None:
    assert declared_requester(author, body) == expected
