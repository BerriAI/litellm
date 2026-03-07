"""Tests for the /public/litellm_blog_posts endpoint."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

SAMPLE_POSTS = [
    {
        "title": "Test Post",
        "description": "A test post.",
        "date": "2026-01-01",
        "url": "https://www.litellm.ai/blog/test",
    }
]


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
    data = response.json()
    assert "posts" in data
    assert len(data["posts"]) == 1
    post = data["posts"][0]
    assert post["title"] == "Test Post"
    assert post["description"] == "A test post."
    assert post["date"] == "2026-01-01"
    assert post["url"] == "https://www.litellm.ai/blog/test"


def test_get_blog_posts_limits_to_five(client):
    """Endpoint returns at most 5 posts."""
    many_posts = [
        {
            "title": f"Post {i}",
            "description": "desc",
            "date": "2026-01-01",
            "url": f"https://www.litellm.ai/blog/{i}",
        }
        for i in range(10)
    ]

    with patch(
        "litellm.proxy.public_endpoints.public_endpoints.get_blog_posts",
        return_value=many_posts,
    ):
        response = client.get("/public/litellm_blog_posts")

    assert response.status_code == 200
    assert len(response.json()["posts"]) == 5


def test_get_blog_posts_returns_local_backup_on_failure(client):
    """Endpoint returns local backup (non-empty list) when fetcher fails."""
    with patch(
        "litellm.proxy.public_endpoints.public_endpoints.get_blog_posts",
        side_effect=Exception("fetch failed"),
    ):
        response = client.get("/public/litellm_blog_posts")

    # Should not 500 — returns local backup
    assert response.status_code == 200
    assert "posts" in response.json()
    assert len(response.json()["posts"]) > 0
