"""Tests for GetBlogPosts utility class."""
import time
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.litellm_core_utils.get_blog_posts import (
    BlogPost,
    BlogPostsResponse,
    GetBlogPosts,
    get_blog_posts,
)

SAMPLE_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>LiteLLM Blog</title>
    <item>
      <title>Test Post</title>
      <link>https://docs.litellm.ai/blog/test</link>
      <description>A test post.</description>
      <pubDate>Wed, 01 Jan 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Second Post</title>
      <link>https://docs.litellm.ai/blog/second</link>
      <description>Another post.</description>
      <pubDate>Tue, 31 Dec 2025 10:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


@pytest.fixture(autouse=True)
def reset_blog_posts_cache():
    GetBlogPosts._cached_posts = None
    GetBlogPosts._last_fetch_time = 0.0
    yield
    GetBlogPosts._cached_posts = None
    GetBlogPosts._last_fetch_time = 0.0


def test_load_local_blog_posts_returns_list():
    posts = GetBlogPosts.load_local_blog_posts()
    assert isinstance(posts, list)
    assert len(posts) > 0
    first = posts[0]
    assert "title" in first
    assert "description" in first
    assert "date" in first
    assert "url" in first


def test_parse_rss_to_posts():
    posts = GetBlogPosts.parse_rss_to_posts(SAMPLE_RSS, max_posts=1)
    assert len(posts) == 1
    assert posts[0]["title"] == "Test Post"
    assert posts[0]["url"] == "https://docs.litellm.ai/blog/test"
    assert posts[0]["description"] == "A test post."
    assert posts[0]["date"] == "2026-01-01"


def test_parse_rss_to_posts_multiple():
    posts = GetBlogPosts.parse_rss_to_posts(SAMPLE_RSS, max_posts=5)
    assert len(posts) == 2
    assert posts[1]["title"] == "Second Post"


def test_parse_rss_to_posts_invalid_xml():
    with pytest.raises(Exception):
        GetBlogPosts.parse_rss_to_posts("not xml")


def test_parse_rss_to_posts_missing_channel():
    with pytest.raises(ValueError, match="missing <channel>"):
        GetBlogPosts.parse_rss_to_posts("<rss></rss>")


def test_validate_blog_posts_valid():
    posts = [{"title": "T", "description": "D", "date": "2026-01-01", "url": "https://x.com"}]
    assert GetBlogPosts.validate_blog_posts(posts) is True


def test_validate_blog_posts_empty_list():
    assert GetBlogPosts.validate_blog_posts([]) is False


def test_validate_blog_posts_not_list():
    assert GetBlogPosts.validate_blog_posts("not a list") is False


def test_get_blog_posts_success():
    """Fetches from RSS on first call."""
    mock_response = MagicMock()
    mock_response.text = SAMPLE_RSS
    mock_response.raise_for_status = MagicMock()

    with patch("litellm.litellm_core_utils.get_blog_posts.httpx.get", return_value=mock_response):
        posts = get_blog_posts(url=litellm.blog_posts_url)

    assert len(posts) == 1
    assert posts[0]["title"] == "Test Post"


def test_get_blog_posts_network_error_falls_back_to_local():
    """Falls back to local backup on network error."""
    with patch(
        "litellm.litellm_core_utils.get_blog_posts.httpx.get",
        side_effect=Exception("Network error"),
    ):
        posts = get_blog_posts(url=litellm.blog_posts_url)

    assert isinstance(posts, list)
    assert len(posts) > 0


def test_get_blog_posts_invalid_xml_falls_back_to_local():
    """Falls back when remote returns invalid XML."""
    mock_response = MagicMock()
    mock_response.text = "not valid xml"
    mock_response.raise_for_status = MagicMock()

    with patch("litellm.litellm_core_utils.get_blog_posts.httpx.get", return_value=mock_response):
        posts = get_blog_posts(url=litellm.blog_posts_url)

    assert isinstance(posts, list)
    assert len(posts) > 0


def test_get_blog_posts_ttl_cache_not_refetched():
    """Within TTL window, does not re-fetch."""
    cached = [{"title": "Cached", "description": "D", "date": "2026-01-01", "url": "https://x.com"}]
    GetBlogPosts._cached_posts = cached
    GetBlogPosts._last_fetch_time = time.time()  # just now

    call_count = 0

    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        m = MagicMock()
        m.text = SAMPLE_RSS
        m.raise_for_status = MagicMock()
        return m

    with patch("litellm.litellm_core_utils.get_blog_posts.httpx.get", side_effect=mock_get):
        posts = get_blog_posts(url=litellm.blog_posts_url)

    assert call_count == 0  # cache hit, no fetch
    assert len(posts) == 1


def test_get_blog_posts_ttl_expired_refetches():
    """After TTL window, re-fetches from remote."""
    cached = [{"title": "Cached", "description": "D", "date": "2026-01-01", "url": "https://x.com"}]
    GetBlogPosts._cached_posts = cached
    GetBlogPosts._last_fetch_time = time.time() - 7200  # 2 hours ago

    mock_response = MagicMock()
    mock_response.text = SAMPLE_RSS
    mock_response.raise_for_status = MagicMock()

    with patch(
        "litellm.litellm_core_utils.get_blog_posts.httpx.get", return_value=mock_response
    ) as mock_get:
        posts = get_blog_posts(url=litellm.blog_posts_url)

    mock_get.assert_called_once()
    assert len(posts) == 1


def test_get_blog_posts_local_env_var_skips_remote(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_BLOG_POSTS", "true")
    with patch("litellm.litellm_core_utils.get_blog_posts.httpx.get") as mock_get:
        posts = get_blog_posts(url=litellm.blog_posts_url)
    mock_get.assert_not_called()
    assert isinstance(posts, list)
    assert len(posts) > 0


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
