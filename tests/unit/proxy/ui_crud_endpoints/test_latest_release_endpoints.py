import asyncio
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
    CHANGELOG_FEED_URL,
    CHANGELOG_URL,
    LATEST_RELEASE_CACHE_KEY,
    LATEST_RELEASE_CACHE_TTL_SECONDS,
    LATEST_RELEASE_UNAVAILABLE_CACHE_TTL_SECONDS,
    RELEASE_BY_TAG_URL,
    LatestReleaseInfo,
    LatestReleaseUnavailable,
    _default_cache,
    _default_client,
    _default_fetch_lock,
    count_release_bullets,
    earlier_patch_tags,
    get_latest_release_info,
)
from tests._master_key import MASTER_KEY

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


def _changelog_page(folder: str) -> str:
    return f"{CHANGELOG_URL}{folder}/{folder.removeprefix('v').replace('.', '-')}"


def _feed(*folders: str) -> httpx.Response:
    items: Final = "".join(
        f"<item><title>{folder}</title><link>{_changelog_page(folder)}</link></item>" for folder in folders
    )
    return httpx.Response(200, content=f'<?xml version="1.0"?><rss><channel>{items}</channel></rss>'.encode())


EXPECTED_INFO: Final = {
    "version": "1.102.0",
    "new_features": 2,
    "bug_fixes": 2,
    "other_updates": 4,
    "release_url": _changelog_page("v1.102.0"),
}


def _github_response(status: int = 200, payload: object = SAMPLE_RELEASE) -> httpx.Response:
    return httpx.Response(status, content=json.dumps(payload).encode())


def _tag_url(tag: str) -> str:
    return RELEASE_BY_TAG_URL.format(tag=tag)


class _ByUrl:
    """Answers each URL from a fixed map; unknown URLs are 404, like a tag with no release."""

    def __init__(self, routes: dict[str, httpx.Response | Exception]) -> None:
        self._routes = routes
        self.calls: list[tuple[str, float | None]] = []

    async def get(self, url: str, *, timeout: float | None = None) -> httpx.Response:
        self.calls.append((url, timeout))
        outcome = self._routes.get(url, httpx.Response(404, content=b"{}"))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


def _minor_release_routes(**overrides: httpx.Response | Exception) -> dict[str, httpx.Response | Exception]:
    return {CHANGELOG_FEED_URL: _feed("v1.102.0"), _tag_url("v1.102.0"): _github_response(), **overrides}


def _minor_release(**overrides: httpx.Response | Exception) -> _ByUrl:
    return _ByUrl(_minor_release_routes(**overrides))


def _fresh_cache() -> InMemoryCache:
    return InMemoryCache(max_size_in_memory=1, default_ttl=LATEST_RELEASE_CACHE_TTL_SECONDS)


def _override_dependencies(client: _ByUrl, cache: InMemoryCache, role: LitellmUserRoles) -> None:
    async def auth() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(api_key="sk-test", user_id="test-user", user_role=role)

    app.dependency_overrides[user_api_key_auth] = auth
    app.dependency_overrides[_default_client] = lambda: client
    app.dependency_overrides[_default_cache] = lambda: cache
    app.dependency_overrides[_default_fetch_lock] = lambda: asyncio.Lock()


@pytest.fixture
def http_client():
    yield TestClient(app)
    app.dependency_overrides.pop(user_api_key_auth, None)
    app.dependency_overrides.pop(_default_client, None)
    app.dependency_overrides.pop(_default_cache, None)
    app.dependency_overrides.pop(_default_fetch_lock, None)


class TestCountReleaseBullets:
    def test_buckets_by_conventional_commit_type(self):
        counts = count_release_bullets(SAMPLE_BODY)
        assert counts["new_features"] == 2
        assert counts["bug_fixes"] == 2
        assert counts["other_updates"] == 4

    def test_unprefixed_bullets_count_as_other_updates(self):
        counts = count_release_bullets("* Litellm dev 09 08 2026 by @f in https://x/pull/7\n")
        assert (counts["new_features"], counts["bug_fixes"], counts["other_updates"]) == (0, 0, 1)

    def test_ignores_non_bullet_lines_and_contributor_entries(self):
        assert (
            sum(
                count_release_bullets(
                    "## What's Changed\n\n* @x made their first contribution in url\n"
                    "\n**Full Changelog**: https://github.com/BerriAI/litellm/compare/v1...v2\n"
                ).values()
            )
            == 0
        )

    def test_empty_body_yields_zero_counts(self):
        counts = count_release_bullets("")
        assert (counts["new_features"], counts["bug_fixes"], counts["other_updates"]) == (0, 0, 0)


class TestGetLatestReleaseInfo:
    @pytest.mark.asyncio
    async def test_announces_the_changelog_release_and_links_its_changelog_page(self):
        client = _minor_release()
        result = await get_latest_release_info(client=client, cache=_fresh_cache(), fetch_lock=asyncio.Lock())
        assert isinstance(result, LatestReleaseInfo)
        assert result.model_dump() == EXPECTED_INFO
        assert client.calls == [(CHANGELOG_FEED_URL, 5), (_tag_url("v1.102.0"), 5)]

    @pytest.mark.asyncio
    async def test_github_release_missing_from_the_changelog_is_not_announced(self):
        automated: Final = _github_response(payload=_release("v1.103.0", "* feat(proxy): automated release\n"))
        result = await _info(_minor_release(**{_tag_url("v1.103.0"): automated}))
        assert isinstance(result, LatestReleaseInfo)
        assert result.model_dump() == EXPECTED_INFO

    @pytest.mark.asyncio
    async def test_highest_stable_version_wins_regardless_of_feed_order(self):
        client = _minor_release(**{CHANGELOG_FEED_URL: _feed("v1.101.6", "v1.103.0rc1", "v1.102.0", "v1.100.9")})
        result = await _info(client)
        assert isinstance(result, LatestReleaseInfo)
        assert result.model_dump() == EXPECTED_INFO

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "feed",
        [
            httpx.Response(503, content=_feed("v1.102.0").content),
            httpx.Response(200, content=b"<html>not a feed"),
            _feed("v1.103.0rc1", "v1.103.0-dev.1"),
            httpx.Response(
                200,
                content=f"<rss><channel><item><link>https://example.com/?{_changelog_page('v9.0.0')}</link>"
                "</item><item><link>https://example.com/release_notes/v9.0.0/x</link></item></channel></rss>".encode(),
            ),
        ],
        ids=["server_error", "not_xml", "only_prereleases", "foreign_links"],
    )
    async def test_bad_changelog_feeds_are_unavailable(self, feed: httpx.Response):
        client = _minor_release(**{CHANGELOG_FEED_URL: feed})
        result = await _info(client)
        assert isinstance(result, LatestReleaseUnavailable)
        assert client.urls == [CHANGELOG_FEED_URL]

    @pytest.mark.asyncio
    async def test_second_call_within_ttl_does_not_refetch(self):
        client = _minor_release()
        cache = _fresh_cache()
        fetch_lock = asyncio.Lock()
        first = await get_latest_release_info(client=client, cache=cache, fetch_lock=fetch_lock)
        second = await get_latest_release_info(client=client, cache=cache, fetch_lock=fetch_lock)
        assert first == second
        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_success_is_cached_for_the_full_ttl(self):
        cache = _fresh_cache()
        await get_latest_release_info(client=_minor_release(), cache=cache, fetch_lock=asyncio.Lock())
        remaining = await cache.async_get_ttl(LATEST_RELEASE_CACHE_KEY) - time.time()
        assert LATEST_RELEASE_CACHE_TTL_SECONDS - 5 < remaining <= LATEST_RELEASE_CACHE_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_failure_is_cached_briefly_so_github_is_not_hammered(self):
        client = _ByUrl({CHANGELOG_FEED_URL: httpx.ConnectError("boom")})
        cache = _fresh_cache()
        fetch_lock = asyncio.Lock()
        first = await get_latest_release_info(client=client, cache=cache, fetch_lock=fetch_lock)
        second = await get_latest_release_info(client=client, cache=cache, fetch_lock=fetch_lock)
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
            _github_response(status=404, payload={"message": "Not Found"}),
            _github_response(status=403, payload={"message": "rate limited"}),
            _github_response(status=500, payload={}),
            _github_response(payload={"tag_name": "v1.0.0"}),
            httpx.Response(200, content=b"not json"),
            httpx.ConnectError("boom"),
        ],
        ids=["no_release_for_tag", "rate_limited", "server_error", "missing_fields", "not_json", "unreachable"],
    )
    async def test_bad_github_responses_are_unavailable(self, response: httpx.Response | Exception):
        result = await _info(_minor_release(**{_tag_url("v1.102.0"): response}))
        assert isinstance(result, LatestReleaseUnavailable)

    @pytest.mark.asyncio
    async def test_concurrent_misses_share_one_fetch(self):
        event = asyncio.Event()

        class _BlockingClient(_ByUrl):
            async def get(self, url: str, *, timeout: float | None = None) -> httpx.Response:
                await event.wait()
                return await super().get(url, timeout=timeout)

        client = _BlockingClient(_minor_release_routes())
        cache = _fresh_cache()
        fetch_lock = asyncio.Lock()
        tasks = [
            asyncio.create_task(get_latest_release_info(client=client, cache=cache, fetch_lock=fetch_lock))
            for _ in range(5)
        ]
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        event.set()
        results = await asyncio.gather(*tasks)
        expected: Final = LatestReleaseInfo.model_validate(EXPECTED_INFO)
        assert results == [expected] * 5
        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_failure_under_lock_is_also_coalesced(self):
        event = asyncio.Event()

        class _FailingBlockingClient(_ByUrl):
            async def get(self, url: str, *, timeout: float | None = None) -> httpx.Response:
                self.calls.append((url, timeout))
                await event.wait()
                raise httpx.ConnectError("boom")

        client = _FailingBlockingClient({})
        cache = _fresh_cache()
        fetch_lock = asyncio.Lock()
        tasks = [
            asyncio.create_task(get_latest_release_info(client=client, cache=cache, fetch_lock=fetch_lock))
            for _ in range(5)
        ]
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        event.set()
        results = await asyncio.gather(*tasks)
        assert all(isinstance(result, LatestReleaseUnavailable) for result in results)
        assert len(client.calls) == 1


def _release(tag: str, body: str | None, *, prerelease: bool = False) -> dict[str, object]:
    return {
        "tag_name": tag,
        "html_url": f"https://github.com/BerriAI/litellm/releases/tag/{tag}",
        "body": body,
        "prerelease": prerelease,
        "draft": False,
    }


PATCH_RELEASE: Final = _release("v1.102.2", "* fix(proxy): second backport by @k in https://x/pull/10\n")
FIRST_PATCH: Final = _release("v1.102.1", "* fix(proxy): first backport by @k in https://x/pull/9\n")


def _minor_line(**overrides: httpx.Response | Exception) -> _ByUrl:
    return _ByUrl(
        {
            CHANGELOG_FEED_URL: _feed("v1.102.2", "v1.101.6"),
            _tag_url("v1.102.2"): _github_response(payload=PATCH_RELEASE),
            _tag_url("v1.102.0"): _github_response(payload=SAMPLE_RELEASE),
            _tag_url("v1.102.1"): _github_response(payload=FIRST_PATCH),
            **{_tag_url(tag): outcome for tag, outcome in overrides.items()},
        }
    )


async def _info(client: _ByUrl) -> LatestReleaseInfo | LatestReleaseUnavailable:
    return await get_latest_release_info(client=client, cache=_fresh_cache(), fetch_lock=asyncio.Lock())


class TestPatchReleaseAggregatesMinorLine:
    def test_earlier_patch_tags_cover_the_whole_line(self):
        assert earlier_patch_tags((1, 105, 2)) == ("v1.105.0", "v1.105.1")
        assert earlier_patch_tags((1, 105, 0)) == ()

    @pytest.mark.asyncio
    async def test_patch_release_counts_include_every_earlier_patch_on_its_line(self):
        client = _minor_line()
        result = await _info(client)
        assert isinstance(result, LatestReleaseInfo)
        assert result.model_dump() == {
            "version": "1.102.2",
            "new_features": 2,
            "bug_fixes": 4,
            "other_updates": 4,
            "release_url": _changelog_page("v1.102.2"),
        }
        assert sorted(client.urls) == sorted(
            [CHANGELOG_FEED_URL, _tag_url("v1.102.2"), _tag_url("v1.102.0"), _tag_url("v1.102.1")]
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "skipped",
        [
            httpx.Response(404, content=b"{}"),
            _github_response(payload=_release("v1.102.1", "* feat: rc only\n", prerelease=True)),
            _github_response(payload=_release("v1.102.1", None)),
        ],
        ids=["never_published", "prerelease", "null_body"],
    )
    async def test_missing_prerelease_or_empty_patch_adds_nothing(self, skipped: httpx.Response):
        result = await _info(_minor_line(**{"v1.102.1": skipped}))
        assert isinstance(result, LatestReleaseInfo)
        assert (result.new_features, result.bug_fixes, result.other_updates) == (2, 3, 4)

    @pytest.mark.asyncio
    async def test_latest_release_with_null_body_still_counts_its_line(self):
        result = await _info(_minor_line(**{"v1.102.2": _github_response(payload=_release("v1.102.2", None))}))
        assert isinstance(result, LatestReleaseInfo)
        assert (result.new_features, result.bug_fixes, result.other_updates) == (2, 3, 4)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "outcome",
        [
            httpx.ConnectError("boom"),
            _github_response(status=403, payload={"message": "rate limited"}),
            httpx.Response(200, content=b"not json"),
        ],
        ids=["unreachable", "rate_limited", "not_json"],
    )
    async def test_earlier_patch_failure_is_unavailable(self, outcome: httpx.Response | Exception):
        result = await _info(_minor_line(**{"v1.102.0": outcome}))
        assert isinstance(result, LatestReleaseUnavailable)


class TestLatestReleaseInfoEndpoint:
    def test_returns_release_stats_for_authenticated_user(self, http_client):
        _override_dependencies(_minor_release(), _fresh_cache(), LitellmUserRoles.INTERNAL_USER)
        response = http_client.get("/get/latest_release_info")
        assert response.status_code == 200
        assert response.json() == EXPECTED_INFO

    def test_returns_null_when_the_changelog_is_unreachable(self, http_client):
        _override_dependencies(
            _ByUrl({CHANGELOG_FEED_URL: httpx.ConnectError("boom")}), _fresh_cache(), LitellmUserRoles.PROXY_ADMIN
        )
        response = http_client.get("/get/latest_release_info")
        assert response.status_code == 200
        assert response.json() is None

    @pytest.mark.parametrize(
        ("flag", "expected_body", "expected_urls"),
        [
            ("true", None, []),
            (" TRUE ", None, []),
            ("false", EXPECTED_INFO, [CHANGELOG_FEED_URL, _tag_url("v1.102.0")]),
        ],
    )
    def test_disable_flag_skips_every_outbound_request(
        self, http_client, monkeypatch, flag, expected_body, expected_urls
    ):
        monkeypatch.setenv("LITELLM_DISABLE_LATEST_RELEASE_CHECK", flag)
        client = _minor_release()
        _override_dependencies(client, _fresh_cache(), LitellmUserRoles.PROXY_ADMIN)
        response = http_client.get("/get/latest_release_info")
        assert response.status_code == 200
        assert response.json() == expected_body
        assert client.urls == expected_urls

    def test_repeated_requests_reuse_cache(self, http_client):
        client = _minor_release()
        _override_dependencies(client, _fresh_cache(), LitellmUserRoles.PROXY_ADMIN)
        assert http_client.get("/get/latest_release_info").json() == EXPECTED_INFO
        assert http_client.get("/get/latest_release_info").json() == EXPECTED_INFO
        assert len(client.calls) == 2

    def test_rejects_unauthenticated_requests(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", MASTER_KEY)
        response = TestClient(app).get("/get/latest_release_info")
        assert response.status_code in (401, 403)
