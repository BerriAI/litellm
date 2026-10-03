import asyncio
from datetime import date
from typing import Final

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy.roi_calculator.estimator import metadata_evidence
from litellm.proxy.roi_calculator.github import GitHubPullListItem, SourceError
from litellm.proxy.roi_calculator.gitlab import GitLab
from litellm.types.roi_calculator import ROISettings


@pytest.mark.asyncio
async def test_fork_lookups_overlap_with_a_bounded_number_of_requests() -> None:
    started: Final[asyncio.Queue[int]] = asyncio.Queue()
    release: Final = tuple(asyncio.Event() for _ in range(9))
    source_ids: Final = (*range(2, 11), 3)

    async def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/projects/group/repo"):
            return httpx.Response(200, json={"id": 1, "path_with_namespace": "group/repo"})
        if request.url.path.endswith("/merge_requests"):
            return httpx.Response(
                200,
                json=[
                    {
                        "iid": index,
                        "title": "Fix parser",
                        "web_url": f"https://gitlab.com/group/repo/-/merge_requests/{index}",
                        "author": {"username": "dev"},
                        "merged_at": "2026-09-30T12:00:00Z",
                        "updated_at": "2026-09-30T12:00:00Z",
                        "source_branch": f"fix/{index}",
                        "source_project_id": source_id,
                    }
                    for index, source_id in enumerate(source_ids)
                ],
            )
        project_id: Final = int(request.url.path.rsplit("/", 1)[1])
        started.put_nowait(project_id)
        await release[project_id - 2].wait()
        if project_id == 3:
            return httpx.Response(404)
        return httpx.Response(200, json={"id": project_id, "path_with_namespace": f"fork-{project_id}/repo"})

    source: Final = GitLab(ROISettings(source_provider="gitlab"), httpx.MockTransport(respond))
    pending: Final = asyncio.create_task(source.pulls("group/repo", date(2026, 9, 1), date(2026, 9, 30)))
    try:
        first_wave: Final = tuple([await asyncio.wait_for(started.get(), timeout=1) for _ in range(8)])
        assert len(set(first_wave)) == 8
        assert started.empty()
        release[first_wave[0] - 2].set()
        next_id: Final = await asyncio.wait_for(started.get(), timeout=1)
        assert next_id not in first_wave
        for event in release:
            event.set()
        pulls: Final = await asyncio.wait_for(pending, timeout=1)
        assert tuple(pull.head.repo.full_name if pull.head and pull.head.repo else None for pull in pulls) == tuple(
            None if source_id == 3 else f"fork-{source_id}/repo" for source_id in source_ids
        )
        assert started.empty()
    finally:
        for event in release:
            event.set()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await source.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_fork,source_id", ((False, 2), (True, 2), (False, None)))
async def test_gitlab_paginates_nested_projects_and_keeps_source_code_out_of_estimates(
    missing_fork: bool, source_id: int | None
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["PRIVATE-TOKEN"] == "test-only-token"
        assert request.url.host == "git.example.test"
        path: Final = request.url.path
        detail: Final = {
            "iid": 8,
            "title": "Fix parser",
            "description": "Handle empty input",
            "web_url": "https://git.example.test/g/sub/p/-/merge_requests/8",
            "author": {"username": "dev.name"},
            "merged_at": "2026-09-30T23:59:59Z",
            "updated_at": "2026-10-01T00:00:00Z",
            "sha": "sha",
            "source_branch": "fix/parser",
            "source_project_id": source_id,
            "changes_count": "1",
        }
        if path.endswith("/projects/g/sub/p"):
            assert "%2F" in str(request.url)
            return httpx.Response(200, json={"id": 1, "path_with_namespace": "g/sub/p"})
        if path.endswith("/projects/2"):
            return (
                httpx.Response(404)
                if missing_fork
                else httpx.Response(200, json={"id": 2, "path_with_namespace": "dev/fork"})
            )
        if path.endswith("/merge_requests"):
            assert request.url.params["scope"] == "all"
            if request.url.params["page"] == "1":
                return httpx.Response(
                    200, json=[{**detail, "iid": 7, "merged_at": "2026-10-01T00:00:00Z"}], headers={"x-next-page": "2"}
                )
            return httpx.Response(200, json=[detail])
        if path.endswith("/merge_requests/8"):
            return httpx.Response(200, json=detail)
        if path.endswith("/diffs"):
            return httpx.Response(
                200,
                json=[
                    {
                        "new_path": "parser.py",
                        "old_path": "parser.py",
                        "diff": "@@ -1 +1 @@\n---old-code\n+++private-code",
                    }
                ],
            )
        if path.endswith("/commits"):
            return httpx.Response(
                200, json=[{"id": "sha", "message": "Fix empty input", "author_email": "untrusted@example.test"}]
            )
        if path.endswith("/users"):
            return httpx.Response(200, json=[{"username": "dev.name", "public_email": "dev@example.test"}])
        raise AssertionError(path)

    settings: Final = ROISettings(
        source_provider="gitlab",
        gitlab_api_url="https://git.example.test/api/v4",
        gitlab_token=SecretStr("test-only-token"),
        repos=("g/sub/p",),
    )
    client: Final = GitLab(settings, httpx.MockTransport(respond))
    try:
        pulls: Final = await client.pulls("g/sub/p", date(2026, 9, 1), date(2026, 9, 30))
        assert tuple(pull.number for pull in pulls) == (8,)
        evidence: Final = await client.evidence("g/sub/p", pulls[0])
        assert evidence["source_repo"] == ("" if missing_fork or source_id is None else "git.example.test/dev/fork")
        assert evidence["source_branch"] == "fix/parser"
        assert evidence["emails"] == ("dev@example.test",)
        assert evidence["commit_emails"] == ()
        assert (evidence["additions"], evidence["deletions"]) == (1, 1)
        assert not evidence["incomplete_metadata"]
        assert "private-code" not in metadata_evidence(evidence).model_dump_json()
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (301, 401, 403, 404))
async def test_gitlab_errors_do_not_follow_redirects_or_disclose_upstream_content(status: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "gitlab.com"
        return httpx.Response(status, text="secret-upstream-response", headers={"location": "https://untrusted.test/"})

    source: Final = GitLab(ROISettings(source_provider="gitlab"), httpx.MockTransport(respond))
    try:
        with pytest.raises(SourceError, match=f"HTTP {status}") as error:
            await source.test_repositories(("group/project",))
        assert "secret-upstream-response" not in str(error.value)
    finally:
        await source.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ("", "test-token"))
async def test_gitlab_repository_browser_preserves_visibility_pagination_and_membership(token: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search"] == "gateway"
        assert request.url.params["page"] == "2"
        assert (request.url.params.get("membership") == "true") == bool(token)
        return httpx.Response(
            200,
            json=[
                {
                    "id": 1,
                    "path_with_namespace": "group/sub/gateway",
                    "visibility": "internal",
                    "archived": True,
                }
            ],
            headers={"link": '<https://gitlab.com/api/v4/projects?page=3>; rel="next"'},
        )

    source: Final = GitLab(
        ROISettings(source_provider="gitlab", gitlab_token=SecretStr(token)), httpx.MockTransport(respond)
    )
    try:
        assert await source.repositories("gateway", 2) == ((("group/sub/gateway", "internal", True),), True)
    finally:
        await source.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resource,message",
    (
        ("projects", "page of results"),
        ("projects/group/repo", "project details"),
        ("projects/1/merge_requests/8", "merge request details"),
    ),
)
async def test_gitlab_rejects_malformed_responses(resource: str, message: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/" + resource):
            return httpx.Response(200, json={"private-error": "must not be disclosed"})
        return httpx.Response(200, json={"id": 1, "path_with_namespace": "group/repo"})

    source: Final = GitLab(ROISettings(source_provider="gitlab"), httpx.MockTransport(respond))
    operation: Final = (
        source.repositories()
        if resource == "projects"
        else source.test_repositories(("group/repo",))
        if resource == "projects/group/repo"
        else source.evidence("group/repo", GitHubPullListItem(number=8, title="Fix", updated_at="2026-09-30"))
    )
    try:
        with pytest.raises(SourceError, match=message):
            await operation
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_gitlab_connection_failure_is_sanitized_and_profile_uses_fallback() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private host detail", request=request)

    source: Final = GitLab(ROISettings(source_provider="gitlab"), httpx.MockTransport(respond))
    try:
        with pytest.raises(SourceError, match="Could not reach GitLab") as error:
            await source.repositories()
        assert "private host detail" not in str(error.value)
        assert await source.profile_email("alice", fallback="known@example.test") == "known@example.test"
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_gitlab_stops_an_endless_pagination_response() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/projects/group/repo"):
            return httpx.Response(200, json={"id": 1, "path_with_namespace": "group/repo"})
        assert int(request.url.params["page"]) <= 100
        return httpx.Response(200, json=[], headers={"x-next-page": "101"})

    source: Final = GitLab(ROISettings(source_provider="gitlab"), httpx.MockTransport(respond))
    try:
        with pytest.raises(SourceError, match="pagination limit"):
            await source.pulls("group/repo", date(2026, 9, 1), date(2026, 9, 30))
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_gitlab_retries_transient_errors_and_checks_merge_request_access() -> None:
    statuses: Final = iter((429, 503, 200))
    reads: Final = iter(("/api/v4/projects/group/repo", "/api/v4/projects/1/merge_requests"))

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/projects/group/repo"):
            status: Final = next(statuses)
            if status != 200:
                return httpx.Response(status)
            assert request.url.path == next(reads)
            return httpx.Response(200, json={"id": 1, "path_with_namespace": "group/repo"})
        assert request.url.path == next(reads)
        assert request.url.params["state"] == "merged"
        return httpx.Response(200, json=[])

    source: Final = GitLab(ROISettings(source_provider="gitlab"), httpx.MockTransport(respond))
    try:
        await source.test_repositories(("group/repo",))
        assert next(reads, None) is None
        assert next(statuses, None) is None
    finally:
        await source.close()
