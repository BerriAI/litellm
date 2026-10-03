from datetime import date
from typing import Final

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy.roi_calculator.estimator import metadata_evidence
from litellm.proxy.roi_calculator.gitlab import GitLab
from litellm.proxy.roi_calculator.github import SourceError
from litellm.types.roi_calculator import ROISettings


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_fork", (False, True))
async def test_gitlab_paginates_nested_projects_and_keeps_source_code_out_of_estimates(missing_fork: bool) -> None:
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
            "source_project_id": 2,
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
        assert evidence["source_repo"] == ("" if missing_fork else "git.example.test/dev/fork")
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
