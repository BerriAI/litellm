from datetime import date
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy.roi_calculator.github import GitHub, SourceError
from litellm.types.roi_calculator import ROISettings

_NEXT_PAGE_HEADERS: Final = MappingProxyType({"link": '<https://api.github.com/next>; rel="next"'})
_PULLS_PAGE_ONE_JSON: Final = """[
  {
    "number": 1,
    "title": "At end of range",
    "merged_at": "2026-09-30T23:59:59Z",
    "updated_at": "2026-10-01T00:00:00Z",
    "head": {"sha": "one"},
    "user": {"login": "alice"}
  },
  {
    "number": 2,
    "title": "Unmerged",
    "merged_at": null,
    "updated_at": "2026-09-15T00:00:00Z",
    "head": {"sha": "two"},
    "user": {"login": "alice"}
  }
]"""
_PULLS_PAGE_TWO_JSON: Final = """[
  {
    "number": 3,
    "title": "At start of range",
    "merged_at": "2026-09-01T00:00:00Z",
    "updated_at": "2026-09-01T00:00:00Z",
    "head": {"sha": "three"},
    "user": {"login": "alice"}
  },
  {
    "number": 4,
    "title": "Outside range",
    "merged_at": "2026-08-31T23:59:59Z",
    "updated_at": "2026-08-31T23:59:59Z",
    "head": {"sha": "four"},
    "user": {"login": "alice"}
  }
]"""
_REPOSITORIES_JSON: Final = """[
  {"full_name": "org/backend", "visibility": "private", "archived": false},
  {"full_name": "other/frontend", "visibility": "public", "archived": true}
]"""


def _settings() -> ROISettings:
    return ROISettings(
        github_token=SecretStr("test-github-token"),
        repos=("org/repo",),
    )


def _github(transport: httpx.MockTransport) -> GitHub:
    client: Final = httpx.AsyncClient(transport=transport, timeout=45, follow_redirects=False)
    return GitHub(_settings(), client=client)


@pytest.mark.parametrize("repo", ("../user", "org/.."))
def test_github_rejects_repository_path_segments(repo: str) -> None:
    with pytest.raises(ValueError, match="owner/repo format"):
        ROISettings(repos=(repo,))


@pytest.mark.asyncio
async def test_github_paginates_and_filters_merged_pull_requests_to_the_requested_window() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        page: Final = request.url.params["page"]
        if page == "1":
            return httpx.Response(
                200,
                headers=_NEXT_PAGE_HEADERS,
                content=_PULLS_PAGE_ONE_JSON,
            )
        return httpx.Response(200, content=_PULLS_PAGE_TWO_JSON)

    github: Final = _github(httpx.MockTransport(respond))
    try:
        pulls: Final = await github.pulls("org/repo", date(2026, 9, 1), date(2026, 9, 30))
    finally:
        await github.close()

    assert tuple(pull.number for pull in pulls) == (1, 3)


@pytest.mark.asyncio
async def test_github_maps_upstream_errors_without_returning_response_secrets() -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="private token response")

    github: Final = _github(httpx.MockTransport(respond))
    try:
        with pytest.raises(SourceError) as error:
            await github.repositories()
    finally:
        await github.close()

    assert "Authentication failed" in str(error.value)
    assert "private token response" not in str(error.value)
    assert "test-github-token" not in str(error.value)


@pytest.mark.asyncio
async def test_github_repository_search_starts_page_two_at_github_page_eleven() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.params["page"] == "11"
        assert request.url.params["affiliation"] == "owner,collaborator,organization_member"
        assert request.headers["authorization"] == "Bearer test-github-token"
        return httpx.Response(200, content=_REPOSITORIES_JSON)

    github: Final = _github(httpx.MockTransport(respond))
    try:
        repositories, has_more = await github.repositories(query="BACK", page=2)
    finally:
        await github.close()

    assert repositories == (("org/backend", "private", False),)
    assert not has_more


@pytest.mark.asyncio
async def test_github_repository_search_scans_until_a_later_page_match() -> None:
    expected_pages: Final = iter(("1", "2", "3"))

    def respond(request: httpx.Request) -> httpx.Response:
        page: Final = request.url.params["page"]
        assert page == next(expected_pages)
        if page == "3":
            return httpx.Response(
                200,
                content='[{"full_name":"org/target-repo","visibility":"private","archived":false}]',
            )
        return httpx.Response(200, headers=_NEXT_PAGE_HEADERS, content=_REPOSITORIES_JSON)

    github: Final = _github(httpx.MockTransport(respond))
    try:
        repositories, has_more = await github.repositories(query="TARGET", page=1)
    finally:
        await github.close()

    assert repositories == (("org/target-repo", "private", False),)
    assert not has_more
    assert next(expected_pages, None) is None


@pytest.mark.asyncio
async def test_github_repository_search_pages_ten_github_pages_per_search_page() -> None:
    expected_pages: Final = iter(tuple(str(page) for page in range(1, 21)))

    def respond(request: httpx.Request) -> httpx.Response:
        page: Final = request.url.params["page"]
        assert page == next(expected_pages)
        return httpx.Response(200, headers=_NEXT_PAGE_HEADERS, content="[]")

    github: Final = _github(httpx.MockTransport(respond))
    try:
        first_repositories, first_has_more = await github.repositories(query="missing", page=1)
        second_repositories, second_has_more = await github.repositories(query="missing", page=2)
    finally:
        await github.close()

    assert first_repositories == ()
    assert first_has_more
    assert second_repositories == ()
    assert second_has_more
    assert next(expected_pages, None) is None
