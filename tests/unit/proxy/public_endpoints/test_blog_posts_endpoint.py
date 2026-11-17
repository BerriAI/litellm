"""Tests for the /public/litellm_blog_posts endpoint."""

import asyncio
import time
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from litellm.litellm_core_utils.get_blog_posts import BLOG_POSTS_TTL_SECONDS, BlogPost, GetBlogPosts
from litellm.proxy.public_endpoints.public_endpoints import get_litellm_blog_posts

SAMPLE_POSTS = (
    BlogPost(
        title="Test Post",
        description="A test post.",
        date="2026-01-01",
        url="https://www.litellm.ai/blog/test",
        tags=("anthropic", "day 0 support"),
    ),
)


@pytest.fixture
def client():
    """Create a TestClient with just the public_endpoints router."""
    from fastapi import FastAPI

    from litellm.proxy.public_endpoints.public_endpoints import router

    app = FastAPI()
    app.include_router(router)
    GetBlogPosts._cached_posts = None
    try:
        yield TestClient(app)
    finally:
        GetBlogPosts._cached_posts = None


def test_get_blog_posts_returns_response_shape(client):
    GetBlogPosts._cached_posts = SAMPLE_POSTS
    GetBlogPosts._last_fetch_time = time.time()
    response = client.get("/public/litellm_blog_posts")

    assert response.status_code == 200
    assert response.json() == {
        "posts": [
            {
                "title": "Test Post",
                "description": "A test post.",
                "date": "2026-01-01",
                "url": "https://www.litellm.ai/blog/test",
                "tags": ["anthropic", "day 0 support"],
            }
        ]
    }


def test_get_blog_posts_returns_every_fetched_post(client):
    """The parser owns the cap (BLOG_POSTS_MAX), so the endpoint passes every fetched post through."""
    many_posts = tuple(
        BlogPost(title=f"Post {i}", description="desc", date="2026-01-01", url=f"https://www.litellm.ai/blog/{i}")
        for i in range(10)
    )

    GetBlogPosts._cached_posts = many_posts
    GetBlogPosts._last_fetch_time = time.time()
    response = client.get("/public/litellm_blog_posts")

    assert response.status_code == 200
    assert [p["title"] for p in response.json()["posts"]] == [f"Post {i}" for i in range(10)]


def test_get_blog_posts_returns_local_backup_on_failure(client):
    """Endpoint returns the local backup when the fetcher raises."""
    with patch(
        "litellm.proxy.public_endpoints.public_endpoints._fetch_remote_bytes",
        side_effect=Exception("fetch failed"),
    ):
        response = client.get("/public/litellm_blog_posts")

    assert response.status_code == 200
    backup = [post.model_dump(mode="json") for post in GetBlogPosts.load_local_blog_posts()]
    assert response.json()["posts"] == backup


@pytest.mark.asyncio
async def test_blog_fetch_allows_concurrent_coroutine_progress() -> None:
    progress: Final = asyncio.Event()

    async def fetch(*args: object, **kwargs: object) -> httpx.Response:
        await asyncio.sleep(0)
        await progress.wait()
        return httpx.Response(503, request=httpx.Request("GET", "https://example.test/rss"))

    async def other_request() -> None:
        await asyncio.sleep(0)
        progress.set()

    with patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as client_factory:
        client_factory.return_value.get = AsyncMock(side_effect=fetch)
        with patch("litellm.litellm_core_utils.get_blog_posts.httpx.get", side_effect=httpx.ConnectTimeout("timeout")):
            GetBlogPosts._cached_posts = None
            try:
                task: Final = asyncio.create_task(other_request())
                response: Final = await get_litellm_blog_posts()
                assert task.done()
                assert response.posts == GetBlogPosts.load_local_blog_posts()
            finally:
                await task
                GetBlogPosts._cached_posts = None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [httpx.ConnectTimeout("timeout"), "not xml", "<rss><channel /></rss>"])
async def test_blog_fetch_failure_is_cached(failure: Exception | str) -> None:
    with patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as client_factory:
        client_factory.return_value.get = AsyncMock(
            side_effect=failure if isinstance(failure, Exception) else None,
            return_value=httpx.Response(
                200,
                text=failure if isinstance(failure, str) else "",
                request=httpx.Request("GET", "https://example.test"),
            ),
        )
        with patch(
            "litellm.litellm_core_utils.get_blog_posts.httpx.get", side_effect=httpx.ConnectTimeout("timeout")
        ) as sync:
            GetBlogPosts._cached_posts = None
            try:
                first: Final = await get_litellm_blog_posts()
                second: Final = await get_litellm_blog_posts()
                assert first.posts == second.posts == GetBlogPosts.load_local_blog_posts()
                assert client_factory.return_value.get.call_count + sync.call_count == 1
                with patch(
                    "litellm.litellm_core_utils.get_blog_posts.time.time",
                    return_value=GetBlogPosts._last_fetch_time + BLOG_POSTS_TTL_SECONDS,
                ):
                    expired: Final = await get_litellm_blog_posts()
                assert expired.posts == first.posts
                assert client_factory.return_value.get.call_count == 2
            finally:
                GetBlogPosts._cached_posts = None


@pytest.mark.asyncio
async def test_blog_fetch_success_is_cached() -> None:
    rss: Final = (
        "<rss><channel><item><title>Async post</title><link>https://example.test/post</link></item></channel></rss>"
    )
    with patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as client_factory:
        client_factory.return_value.get = AsyncMock(
            return_value=httpx.Response(200, text=rss, request=httpx.Request("GET", "https://example.test/rss"))
        )
        GetBlogPosts._cached_posts = None
        try:
            first: Final = await get_litellm_blog_posts()
            second: Final = await get_litellm_blog_posts()
            assert (
                first.posts
                == second.posts
                == (BlogPost(title="Async post", description="", date="", url="https://example.test/post"),)
            )
            client_factory.return_value.get.assert_awaited_once()
        finally:
            GetBlogPosts._cached_posts = None


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_concurrent_blog_refresh_is_shared(expired: bool, fails: bool) -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    rss: Final = (
        b"<rss><channel><item><title>Shared</title><link>https://example.test/post</link></item></channel></rss>"
    )

    async def fetch(url: str) -> bytes:
        started.set()
        await release.wait()
        if fails:
            raise httpx.ConnectTimeout("timeout")
        return rss

    fetcher: Final = AsyncMock(side_effect=fetch)
    GetBlogPosts._cached_posts = SAMPLE_POSTS if expired else None
    GetBlogPosts._last_fetch_time = 0.0
    tasks: Final = tuple(
        asyncio.create_task(GetBlogPosts.aget_blog_posts("https://example.test/rss", fetcher)) for _ in range(32)
    )
    try:
        await started.wait()
        await asyncio.sleep(0)
        release.set()
        results: Final = await asyncio.gather(*tasks)
        expected: Final = GetBlogPosts.load_local_blog_posts() if fails else GetBlogPosts.parse_rss_to_posts(rss)
        assert all(posts == expected for posts in results)
        assert fetcher.await_count == 1
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        GetBlogPosts._cached_posts = None


@pytest.mark.asyncio
async def test_cancelled_blog_refresh_does_not_wedge_waiters() -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def fetch(url: str) -> bytes:
        started.set()
        await release.wait()
        return (
            b"<rss><channel><item><title>Recovered</title><link>https://example.test/post</link></item></channel></rss>"
        )

    GetBlogPosts._cached_posts = None
    first: Final = asyncio.create_task(GetBlogPosts.aget_blog_posts("https://example.test/rss", fetch))
    await started.wait()
    second: Final = asyncio.create_task(GetBlogPosts.aget_blog_posts("https://example.test/rss", fetch))
    try:
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        posts: Final = await asyncio.wait_for(second, timeout=1)
        assert posts == (BlogPost(title="Recovered", description="", date="", url="https://example.test/post"),)
    finally:
        first.cancel()
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        GetBlogPosts._cached_posts = None
