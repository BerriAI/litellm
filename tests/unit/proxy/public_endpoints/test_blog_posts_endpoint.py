"""Tests for the /public/litellm_blog_posts endpoint."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

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

    with patch(
        "litellm.proxy.public_endpoints.public_endpoints.get_blog_posts",
        return_value=many_posts,
    ):
        response = client.get("/public/litellm_blog_posts")

    assert response.status_code == 200
    assert [p["title"] for p in response.json()["posts"]] == [f"Post {i}" for i in range(10)]


def test_get_blog_posts_returns_local_backup_on_failure(client):
    """Endpoint returns the local backup, with empty tags, when the fetcher raises."""
    with patch(
        "litellm.proxy.public_endpoints.public_endpoints.get_blog_posts",
        side_effect=Exception("fetch failed"),
    ):
        response = client.get("/public/litellm_blog_posts")

    assert response.status_code == 200
    backup = [post.model_dump(mode="json") for post in GetBlogPosts.load_local_blog_posts()]
    assert response.json()["posts"] == backup
    assert all(post["tags"] == [] for post in response.json()["posts"])
