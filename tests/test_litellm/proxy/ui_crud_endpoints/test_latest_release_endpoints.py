import json
import time
from typing import Final

import httpx
import pytest
from fastapi.testclient import TestClient

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app
from litellm.proxy.ui_crud_endpoints.latest_release_endpoints import (
    LATEST_RELEASE_CACHE_KEY,
    LATEST_RELEASE_CACHE_TTL_SECONDS,
    LATEST_RELEASE_UNAVAILABLE_CACHE_TTL_SECONDS,
    LATEST_RELEASE_URL,
    LatestReleaseInfo,
    LatestReleaseUnavailable,
    _default_cache,
    _default_client,
    count_release_bullets,
    get_latest_release_info,
)

SAMPLE_BODY: Final = """## What's Changed
* feat(proxy): add upgrade banner by @kerry in https://github.com/BerriAI/litellm/pull/1
* fix(azure): retry on 429 by @a in https://github.com/BerriAI/litellm/pull/2
* fix: handle empty body by @b in https://github.com/BerriAI/litellm/pull/3
* Feat(ui)!: drop legacy theme by @c in https://github.com/BerriAI/litellm/pull/4
* chore(deps): bump httpx by @d in https://github.com/BerriAI/litellm/pull/5
* docs: fix typo by @e in https://github.com/BerriAI/litellm/pull/6
* Litellm dev 09 08 2026 by @f in https://github.com/BerriAI/litellm/pull/7
* refactor(router)  : spaced colon does not match by @g in https://github.com/BerriAI/litellm/pull/8

## New Contributors
* @kerry made their first contribution in https://github.com/BerriAI/litellm/pull/1

**Full Changelog**: https://github.com/BerriAI/litellm/compare/v1.101.0...v1.102.0
"""

SAMPLE_RELEASE: Final = {
    "tag_name": "v1.102.0",
    "html_url": "https://github.com/BerriAI/litellm/releases/tag/v1.102.0",
    "body": SAMPLE_BODY,
}
EXPECTED_INFO: Final = {
    "version": "1.102.0",
    "new_features": 2,
    "bug_fixes": 2,
    "other_updates": 2,
    "release_url": SAMPLE_RELEASE["html_url"],
}


class _RecordingClient:
    def __init__(self, outcomes: list[httpx.Response | Exception]) -> None:
        self._outcomes = outcomes
        self.calls: list[tuple[str, float | None]] = []

    async def get(self, url: str, *, timeout: float | None = None) -> httpx.Response:
        self.calls.append((url, timeout))
        outcome = self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _github_response(status: int = 200, payload: object = SAMPLE_RELEASE) -> httpx.Response:
    return httpx.Response(status, content=json.dumps(payload).encode())


def _fresh_cache() -> InMemoryCache:
    return InMemoryCache(max_size_in_memory=1, default_ttl=LATEST_RELEASE_CACHE_TTL_SECONDS)


def _override_dependencies(client: _RecordingClient, cache: InMemoryCache, role: LitellmUserRoles) -> None:
    async def auth() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(api_key="sk-test", user_id="test-user", user_role=role)

    app.dependency_overrides[user_api_key_auth] = auth
    app.dependency_overrides[_default_client] = lambda: client
    app.dependency_overrides[_default_cache] = lambda: cache


@pytest.fixture
def http_client():
    yield TestClient(app)
    app.dependency_overrides.pop(user_api_key_auth, None)
    app.dependency_overrides.pop(_default_client, None)
    app.dependency_overrides.pop(_default_cache, None)


class TestCountReleaseBullets:
    def test_buckets_by_conventional_commit_type(self):
        counts = count_release_bullets(SAMPLE_BODY)
        assert counts["new_features"] == 2
        assert counts["bug_fixes"] == 2
        assert counts["other_updates"] == 2

    def test_ignores_non_bullet_lines_and_contributor_entries(self):
        assert (
            sum(count_release_bullets("## What's Changed\n\n* @x made their first contribution in url\n").values()) == 0
        )

    def test_empty_body_yields_zero_counts(self):
        counts = count_release_bullets("")
        assert (counts["new_features"], counts["bug_fixes"], counts["other_updates"]) == (0, 0, 0)


class TestGetLatestReleaseInfo:
    @pytest.mark.asyncio
    async def test_fetches_and_parses_github_release(self):
        client = _RecordingClient([_github_response()])
        result = await get_latest_release_info(client=client, cache=_fresh_cache())
        assert isinstance(result, LatestReleaseInfo)
        assert result.model_dump() == EXPECTED_INFO
        assert client.calls == [(LATEST_RELEASE_URL, 5)]

    @pytest.mark.asyncio
    async def test_second_call_within_ttl_does_not_refetch(self):
        client = _RecordingClient([_github_response()])
        cache = _fresh_cache()
        first = await get_latest_release_info(client=client, cache=cache)
        second = await get_latest_release_info(client=client, cache=cache)
        assert first == second
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_success_is_cached_for_the_full_ttl(self):
        cache = _fresh_cache()
        await get_latest_release_info(client=_RecordingClient([_github_response()]), cache=cache)
        remaining = await cache.async_get_ttl(LATEST_RELEASE_CACHE_KEY) - time.time()
        assert LATEST_RELEASE_CACHE_TTL_SECONDS - 5 < remaining <= LATEST_RELEASE_CACHE_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_failure_is_cached_briefly_so_github_is_not_hammered(self):
        client = _RecordingClient([httpx.ConnectError("boom")])
        cache = _fresh_cache()
        first = await get_latest_release_info(client=client, cache=cache)
        second = await get_latest_release_info(client=client, cache=cache)
        assert isinstance(first, LatestReleaseUnavailable)
        assert first == second
        assert len(client.calls) == 1
        remaining = await cache.async_get_ttl(LATEST_RELEASE_CACHE_KEY) - time.time()
        assert (
            LATEST_RELEASE_UNAVAILABLE_CACHE_TTL_SECONDS - 5 < remaining <= LATEST_RELEASE_UNAVAILABLE_CACHE_TTL_SECONDS
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [
            _github_response(status=403, payload={"message": "rate limited"}),
            _github_response(status=500, payload={}),
            _github_response(payload={"tag_name": "v1.0.0"}),
            httpx.Response(200, content=b"not json"),
        ],
        ids=["rate_limited", "server_error", "missing_fields", "not_json"],
    )
    async def test_bad_github_responses_are_unavailable(self, response: httpx.Response):
        result = await get_latest_release_info(client=_RecordingClient([response]), cache=_fresh_cache())
        assert isinstance(result, LatestReleaseUnavailable)


class TestLatestReleaseInfoEndpoint:
    def test_returns_release_stats_for_authenticated_user(self, http_client):
        _override_dependencies(_RecordingClient([_github_response()]), _fresh_cache(), LitellmUserRoles.INTERNAL_USER)
        response = http_client.get("/get/latest_release_info")
        assert response.status_code == 200
        assert response.json() == EXPECTED_INFO

    def test_returns_null_when_github_is_unreachable(self, http_client):
        _override_dependencies(
            _RecordingClient([httpx.ConnectError("boom")]), _fresh_cache(), LitellmUserRoles.PROXY_ADMIN
        )
        response = http_client.get("/get/latest_release_info")
        assert response.status_code == 200
        assert response.json() is None

    def test_repeated_requests_reuse_cache(self, http_client):
        client = _RecordingClient([_github_response()])
        _override_dependencies(client, _fresh_cache(), LitellmUserRoles.PROXY_ADMIN)
        assert http_client.get("/get/latest_release_info").json() == EXPECTED_INFO
        assert http_client.get("/get/latest_release_info").json() == EXPECTED_INFO
        assert len(client.calls) == 1

    def test_rejects_unauthenticated_requests(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-1234")
        response = TestClient(app).get("/get/latest_release_info")
        assert response.status_code in (401, 403)
