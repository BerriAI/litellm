"""Tests for GetBlogPosts utility class."""
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from litellm.litellm_core_utils.get_blog_posts import (
    BLOG_POSTS_GITHUB_URL,
    BlogPost,
    BlogPostsResponse,
    GetBlogPosts,
    get_blog_posts,
)

SAMPLE_RESPONSE = {
    "posts": [
        {
            "title": "Test Post",
            "description": "A test post.",
            "date": "2026-01-01",
            "url": "https://www.litellm.ai/blog/test",
        }
    ]
}


def test_load_local_blog_posts_returns_list():
    posts = GetBlogPosts.load_local_blog_posts()
    assert isinstance(posts, list)
    assert len(posts) > 0
    first = posts[0]
    assert "title" in first
    assert "description" in first
    assert "date" in first
    assert "url" in first


def test_validate_blog_posts_valid():
    assert GetBlogPosts.validate_blog_posts(SAMPLE_RESPONSE) is True


def test_validate_blog_posts_missing_posts_key():
    assert GetBlogPosts.validate_blog_posts({"other": []}) is False


def test_validate_blog_posts_empty_list():
    assert GetBlogPosts.validate_blog_posts({"posts": []}) is False


def test_validate_blog_posts_not_dict():
    assert GetBlogPosts.validate_blog_posts("not a dict") is False


def test_get_blog_posts_success(monkeypatch):
    """Fetches from remote on first call."""
    # Reset class cache state
    GetBlogPosts._cached_posts = None
    GetBlogPosts._last_fetch_time = 0.0

    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch("litellm.litellm_core_utils.get_blog_posts.httpx.get", return_value=mock_response):
        posts = get_blog_posts()

    assert len(posts) == 1
    assert posts[0]["title"] == "Test Post"


def test_get_blog_posts_network_error_falls_back_to_local(monkeypatch):
    """Falls back to local backup on network error."""
    GetBlogPosts._cached_posts = None
    GetBlogPosts._last_fetch_time = 0.0

    with patch(
        "litellm.litellm_core_utils.get_blog_posts.httpx.get",
        side_effect=Exception("Network error"),
    ):
        posts = get_blog_posts()

    assert isinstance(posts, list)
    assert len(posts) > 0


def test_get_blog_posts_invalid_json_falls_back_to_local(monkeypatch):
    """Falls back when remote returns non-dict."""
    GetBlogPosts._cached_posts = None
    GetBlogPosts._last_fetch_time = 0.0

    mock_response = MagicMock()
    mock_response.json.return_value = "not a dict"
    mock_response.raise_for_status = MagicMock()

    with patch("litellm.litellm_core_utils.get_blog_posts.httpx.get", return_value=mock_response):
        posts = get_blog_posts()

    assert isinstance(posts, list)
    assert len(posts) > 0


def test_get_blog_posts_ttl_cache_not_refetched(monkeypatch):
    """Within TTL window, does not re-fetch."""
    GetBlogPosts._cached_posts = SAMPLE_RESPONSE["posts"]
    GetBlogPosts._last_fetch_time = time.time()  # just now

    call_count = 0

    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        m = MagicMock()
        m.json.return_value = SAMPLE_RESPONSE
        m.raise_for_status = MagicMock()
        return m

    with patch("litellm.litellm_core_utils.get_blog_posts.httpx.get", side_effect=mock_get):
        posts = get_blog_posts()

    assert call_count == 0  # cache hit, no fetch
    assert len(posts) == 1


def test_get_blog_posts_ttl_expired_refetches(monkeypatch):
    """After TTL window, re-fetches from remote."""
    GetBlogPosts._cached_posts = SAMPLE_RESPONSE["posts"]
    GetBlogPosts._last_fetch_time = time.time() - 7200  # 2 hours ago

    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch(
        "litellm.litellm_core_utils.get_blog_posts.httpx.get", return_value=mock_response
    ) as mock_get:
        posts = get_blog_posts()

    mock_get.assert_called_once()
    assert len(posts) == 1


def test_blog_post_pydantic_model():
    post = BlogPost(
        title="T",
        description="D",
        date="2026-01-01",
        url="https://example.com",
    )
    assert post.title == "T"


def test_blog_posts_response_pydantic_model():
    resp = BlogPostsResponse(
        posts=[BlogPost(title="T", description="D", date="2026-01-01", url="https://x.com")]
    )
    assert len(resp.posts) == 1
