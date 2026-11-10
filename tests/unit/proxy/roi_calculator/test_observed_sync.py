from datetime import date, datetime, timezone
from typing import Final, Literal

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from litellm.proxy.roi_calculator.observed_analytics import summarize_observed
from litellm.proxy.roi_calculator.observed_sync import collect_observed
from litellm.proxy.roi_calculator.source import repository_tag
from litellm.types.roi_calculator import ROIBranchSpend, ROISettings, ROISpendRecord


class _Variables(BaseModel):
    q: str


class _Query(BaseModel):
    variables: _Variables


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("github", "gitlab"))
@pytest.mark.parametrize("days", (7, 28, 90))
@pytest.mark.parametrize("include_disabled_repo", (False, True))
async def test_live_provider_metadata_reaches_people_quality_durations_and_branch_spend_without_an_estimator(
    provider: Literal["github", "gitlab"],
    days: int,
    include_disabled_repo: bool,
) -> None:
    settings: Final = ROISettings(
        source_provider=provider,
        repos=("org/repo", "org/disabled") if include_disabled_repo else ("org/repo",),
        github_token=SecretStr("source-test-token"),
        gitlab_token=SecretStr("source-test-token"),
        identity_map={"old-ari": "ari@example.test"},
    )
    tag: Final = repository_tag(settings, "org/repo")

    async def spend(start: date, end: date) -> tuple[ROISpendRecord, ...]:
        return ({"date": str(start), "user_id": "ari", "email": "ari@example.test", "spend": 30.0, "requests": 10},)

    async def users() -> frozenset[str]:
        return frozenset(("ari@example.test",))

    async def branches(start: date, end: date, repos: tuple[str, ...]) -> tuple[ROIBranchSpend, ...]:
        assert repos == tuple(sorted(repository_tag(settings, repo) for repo in settings.repos))
        return (ROIBranchSpend(repo=tag, branch="fix/parser", spend=5.0, requests=2),)

    def respond(request: httpx.Request) -> httpx.Response:
        path: Final = request.url.path
        if path == "/graphql":
            query: Final = _Query.model_validate_json(request.content).variables.q
            if "repo:org/disabled " in query:
                return httpx.Response(
                    200, json={"data": {"search": {"issueCount": 0, "nodes": [], "pageInfo": {"hasNextPage": False}}}}
                )
            kind: Final = "pull" if "is:pr" in query else "issue"
            start: Final = query.split("merged:" if kind == "pull" else "created:")[1][:10]
            node: Final = {
                "number": 1,
                "url": "https://github.com/org/repo/pull/1",
                "title": "Change",
                "createdAt": f"{start}T12:00:00Z",
                "updatedAt": f"{start}T12:00:16Z",
                "mergedAt": f"{start}T12:00:16Z",
                "author": {"login": "devin-ai-integration", "__typename": "Bot"},
                "body": "Requested by: @old-ari",
                "headRefName": "fix/parser",
                "headRepository": {"nameWithOwner": "org/repo"},
                "labels": {"nodes": [{"name": "bug"}], "pageInfo": {"hasNextPage": False}},
            }
            return httpx.Response(
                200, json={"data": {"search": {"issueCount": 1, "nodes": [node], "pageInfo": {"hasNextPage": False}}}}
            )
        if path == "/repos/org/repo":
            return httpx.Response(200, json={"has_issues": True})
        if path == "/repos/org/disabled":
            return httpx.Response(200, json={"has_issues": False})
        if path.endswith("/projects/org/disabled"):
            return httpx.Response(200, json={"id": 2, "path_with_namespace": "org/disabled", "issues_enabled": False})
        if path.endswith("/projects/2/merge_requests"):
            return httpx.Response(200, json=[])
        if path.endswith("/projects/org/repo"):
            return httpx.Response(200, json={"id": 1, "path_with_namespace": "org/repo"})
        if path.endswith("/merge_requests"):
            start: Final = request.url.params["merged_after"][:10]
            return httpx.Response(
                200,
                json=[
                    {
                        "iid": 1,
                        "web_url": "https://gitlab.com/org/repo/-/merge_requests/1",
                        "title": "Change",
                        "author": {"username": "old-ari"},
                        "created_at": f"{start}T12:00:00Z",
                        "updated_at": f"{start}T12:00:16Z",
                        "merged_at": f"{start}T12:00:16Z",
                        "source_branch": "fix/parser",
                        "source_project_id": 1,
                    }
                ],
            )
        if path.endswith("/issues"):
            start: Final = request.url.params["created_after"][:10]
            return httpx.Response(200, json=[{"iid": 1, "created_at": f"{start}T12:00:00Z", "labels": ["bug"]}])
        raise AssertionError(f"Unexpected API request: {request.method} {path}")

    data: Final = await collect_observed(
        settings,
        spend,
        users,
        branches,
        datetime(2026, 10, 3, tzinfo=timezone.utc),
        lambda stage, done, total: None,
        httpx.MockTransport(respond),
        days=days,
    )
    assert all(
        (period.window.end - period.window.start).days + 1 == days
        for period in (data.current, data.previous, data.last_year)
    )
    report: Final = summarize_observed(data, settings.identity_map)
    person: Final = report.people[0].periods.current
    assert (person.merged_prs, person.gateway_recorded_spend, person.recorded_spend_per_attributed_pr) == (
        1,
        30.0,
        30.0,
    )
    assert person.median_merge_hours == 16 / 3600
    assert person.declared_agent_owned == (1 if provider == "github" else 0)
    assert tuple(
        window.merged_prs for window in (report.periods.current, report.periods.previous, report.periods.last_year)
    ) == (1, 1, 1)
    assert tuple(
        period.new_bug_labeled_issues
        for period in (report.periods.current, report.periods.previous, report.periods.last_year)
    ) == (1, 1, 1)
    assert report.pulls.current[0].branch_cost.spend == 5.0
    assert report.unlinked_branches == ()
