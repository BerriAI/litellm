"""Tests for the /public/litellm_blog_posts endpoint."""

from typing import Final
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from litellm.litellm_core_utils.get_blog_posts import BlogPost, GetBlogPosts

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
    return TestClient(app)


def test_get_blog_posts_returns_response_shape(client):
    with patch(
        "litellm.proxy.public_endpoints.public_endpoints.get_blog_posts",
        return_value=SAMPLE_POSTS,
    ):
        response = client.get("/public/litellm_blog_posts")

    assert response.status_code == 200
    assert response.json() == {
        "hide_updates": False,
        "posts": [
            {
                "title": "Test Post",
                "description": "A test post.",
                "date": "2026-01-01",
                "url": "https://www.litellm.ai/blog/test",
                "tags": ["anthropic", "day 0 support"],
            }
        ],
    }


def test_get_blog_posts_returns_every_fetched_post(client):
    """The parser owns the cap (BLOG_POSTS_MAX), so the endpoint passes every fetched post through."""
    many_posts = tuple(
        BlogPost(title=f"Post {i}", description="desc", date="2026-01-01", url=f"https://www.litellm.ai/blog/{i}")
        for i in range(10)
    )

    with patch(
        "litellm.proxy.public_endpoints.public_endpoints.get_blog_posts",
        return_value=many_posts,
    ):
        response = client.get("/public/litellm_blog_posts")

    assert response.status_code == 200
    assert [p["title"] for p in response.json()["posts"]] == [f"Post {i}" for i in range(10)]


def test_get_blog_posts_returns_local_backup_on_failure(client):
    """Endpoint returns the local backup when the fetcher raises."""
    with patch(
        "litellm.proxy.public_endpoints.public_endpoints.get_blog_posts",
        side_effect=Exception("fetch failed"),
    ):
        response = client.get("/public/litellm_blog_posts")

    assert response.status_code == 200
    backup = [post.model_dump(mode="json") for post in GetBlogPosts.load_local_blog_posts()]
    assert response.json()["posts"] == backup


@pytest.mark.parametrize("hide_updates", ("true", "TRUE", "false", None))
@pytest.mark.parametrize("local_posts", ("true", "false"))
def test_hide_updates_overrides_blog_source(
    client: TestClient, monkeypatch: MonkeyPatch, hide_updates: str | None, local_posts: str
) -> None:
    if hide_updates is None:
        monkeypatch.delenv("LITELLM_HIDE_UPDATES", raising=False)
    else:
        monkeypatch.setenv("LITELLM_HIDE_UPDATES", hide_updates)
    monkeypatch.setenv("LITELLM_LOCAL_BLOG_POSTS", local_posts)
    monkeypatch.setenv("LITELLM_OFFLINE_MODE", "true")

    response: Final = client.get("/public/litellm_blog_posts")

    assert response.status_code == 200
    hidden: Final = hide_updates is not None and hide_updates.lower() == "true"
    assert response.json() == {
        "posts": [] if hidden else [post.model_dump(mode="json") for post in GetBlogPosts.load_local_blog_posts()],
        "hide_updates": hidden,
    }


def test_hidden_updates_do_not_load_remote_or_bundled_posts(client: TestClient, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_HIDE_UPDATES", "true")
    with (
        patch(
            "litellm.proxy.public_endpoints.public_endpoints.get_blog_posts",
            side_effect=AssertionError("Remote source must not be consulted"),
        ) as remote_source,
        patch("litellm.proxy.public_endpoints.public_endpoints.GetBlogPosts", spec=GetBlogPosts) as local_source,
    ):
        local_source.load_local_blog_posts.side_effect = AssertionError("Bundled source must not be consulted")
        response: Final = client.get("/public/litellm_blog_posts")

    assert response.status_code == 200
    assert response.json() == {"posts": [], "hide_updates": True}
    remote_source.assert_not_called()
    local_source.load_local_blog_posts.assert_not_called()
