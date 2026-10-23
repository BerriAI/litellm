import asyncio
import re
from datetime import date, datetime, timedelta, timezone
from typing import Final
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from litellm.proxy.roi_calculator.github import SourceError
from litellm.proxy.roi_calculator.github_observed import GitHubObserved
from litellm.types.roi_calculator import ROISettings


class _Variables(BaseModel):
    q: str
    after: str | None


class _Query(BaseModel):
    variables: _Variables


def _node(number: int, merged: datetime) -> dict[str, object]:
    return {
        "number": number,
        "url": f"https://github.com/org/repo/pull/{number}",
        "title": "Change",
        "createdAt": (merged - timedelta(seconds=16)).isoformat(),
        "updatedAt": merged.isoformat(),
        "mergedAt": merged.isoformat(),
        "author": {"login": "ari", "__typename": "User"},
    }


def _page(nodes: tuple[dict[str, object], ...], count: int, cursor: str | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": {
                "search": {
                    "issueCount": count,
                    "nodes": nodes,
                    "pageInfo": {"hasNextPage": cursor is not None, "endCursor": cursor},
                }
            }
        },
    )


@pytest.mark.asyncio
async def test_large_history_splits_the_search_limit_without_losing_midnight_or_split_boundaries() -> None:
    start: Final = datetime(2026, 9, 1, tzinfo=timezone.utc)
    timestamps: Final = tuple(start + timedelta(seconds=index * 60) for index in range(1001))

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-only-token"
        assert request.url.path == "/graphql"
        query: Final = _Query.model_validate_json(request.content).variables
        bounds: Final = re.search(r"merged:([^ ]+)\.\.([^ ]+)", query.q)
        assert bounds is not None
        lower, upper = (datetime.fromisoformat(value.replace("Z", "+00:00")) for value in bounds.groups())
        assert query.q.count("merged:") == 1
        matching: Final = tuple(
            _node(index, timestamp) for index, timestamp in enumerate(timestamps) if lower <= timestamp <= upper
        )
        offset: Final = int(query.after or 0)
        next_cursor: Final = str(offset + 100) if offset + 100 < len(matching) else None
        return _page(matching[offset : offset + 100], len(matching), next_cursor)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source: Final = GitHubObserved(ROISettings(github_token=SecretStr("test-only-token")), client)
        pulls: Final = await source.pulls("org/repo", start.date(), start.date())
    assert tuple(pull.number for pull in pulls) == tuple(range(1001))
    assert all(
        pull.created_at and (datetime.fromisoformat(pull.merged_at or "") - pull.created_at).total_seconds() == 16
        for pull in pulls
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("count", "duplicate", "cursor", "partial"))
async def test_incomplete_source_results_fail_instead_of_publishing_understated_counts(failure: str) -> None:
    node: Final = _node(1, datetime(2026, 9, 1, tzinfo=timezone.utc))

    def respond(request: httpx.Request) -> httpx.Response:
        if failure == "partial":
            return httpx.Response(200, json={"data": None, "errors": [{"message": "permission denied"}]})
        if failure == "duplicate":
            return _page((node, node), 2)
        if failure == "cursor":
            return _page((node,), 2, "repeated")
        return _page((node,), 2)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source: Final = GitHubObserved(ROISettings(), client)
        with pytest.raises(SourceError):
            await source.pulls("org/repo", date(2026, 9, 1), date(2026, 9, 1))


@pytest.mark.asyncio
async def test_disabled_issue_tracking_is_unknown_instead_of_zero_bugs() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200, json={"has_issues": False})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source: Final = GitHubObserved(ROISettings(), client)
        assert await source.issues("org/repo", date(2026, 9, 1), date(2026, 9, 1)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("timeout", "unavailable", "rate_limit"))
async def test_read_queries_recover_from_temporary_provider_failures(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    responses: Final = iter((False, False, True))
    node: Final = _node(1, datetime(2026, 9, 1, tzinfo=timezone.utc))

    def respond(request: httpx.Request) -> httpx.Response:
        if next(responses):
            return _page((node,), 1)
        if failure == "timeout":
            raise httpx.ReadTimeout("scripted timeout", request=request)
        return httpx.Response(429 if failure == "rate_limit" else 502)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source: Final = GitHubObserved(ROISettings(), client)
        pulls: Final = await source.pulls("org/repo", date(2026, 9, 1), date(2026, 9, 1))
    assert tuple(pull.number for pull in pulls) == (1,)


@pytest.mark.asyncio
async def test_read_retries_stop_after_three_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source: Final = GitHubObserved(ROISettings(), client)
        with pytest.raises(SourceError, match="HTTP 503"):
            await source.pulls("org/repo", date(2026, 9, 1), date(2026, 9, 1))
    assert requests.qsize() == 3
