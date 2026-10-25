"""Tests for GetBlogPosts utility class."""

import time
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree

import pytest

import litellm
from litellm.litellm_core_utils.get_blog_posts import (
    BLOG_POSTS_MAX,
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
      <category>anthropic</category>
      <category>day 0 support</category>
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

CACHED_POST = BlogPost(title="Cached", description="D", date="2026-01-01", url="https://x.com")


@pytest.fixture(autouse=True)
def reset_blog_posts_cache():
    GetBlogPosts._cached_posts = None
    GetBlogPosts._last_fetch_time = 0.0
    yield
    GetBlogPosts._cached_posts = None
    GetBlogPosts._last_fetch_time = 0.0


def test_load_local_blog_posts_returns_posts_without_tags():
    posts = GetBlogPosts.load_local_blog_posts()
    assert len(posts) > 0
    assert all(isinstance(post, BlogPost) for post in posts)
    assert posts[0].tags == ()


def test_parse_rss_to_posts():
    posts = GetBlogPosts.parse_rss_to_posts(SAMPLE_RSS, max_posts=1)
    assert len(posts) == 1
    assert posts[0].title == "Test Post"
    assert posts[0].url == "https://docs.litellm.ai/blog/test"
    assert posts[0].description == "A test post."
    assert posts[0].date == "2026-01-01"


def test_parse_rss_to_posts_reads_categories_as_tags():
    posts = GetBlogPosts.parse_rss_to_posts(SAMPLE_RSS)
    assert posts[0].tags == ("anthropic", "day 0 support")
    assert posts[1].tags == ()


def test_parse_rss_to_posts_multiple():
    posts = GetBlogPosts.parse_rss_to_posts(SAMPLE_RSS, max_posts=5)
    assert len(posts) == 2
    assert posts[1].title == "Second Post"


def test_parse_rss_to_posts_default_cap():
    items = "".join(
        f"<item><title>Post {i}</title><link>https://docs.litellm.ai/blog/{i}</link>"
        f"<description>D</description><pubDate>Thu, 0{1 + i % 9} Jan 2026 00:00:00 GMT</pubDate></item>"
        for i in range(BLOG_POSTS_MAX + 1)
    )
    posts = GetBlogPosts.parse_rss_to_posts(f"<rss><channel>{items}</channel></rss>")
    assert len(posts) == BLOG_POSTS_MAX
    assert posts[0].title == "Post 0"
    assert posts[-1].title == f"Post {BLOG_POSTS_MAX - 1}"


def test_parse_rss_to_posts_skips_items_without_title_or_link_before_capping():
    items = "<item><description>no title</description></item>" + "".join(
        f"<item><title>Post {i}</title><link>https://docs.litellm.ai/blog/{i}</link></item>" for i in range(2)
    )
    posts = GetBlogPosts.parse_rss_to_posts(f"<rss><channel>{items}</channel></rss>", max_posts=2)
    assert [post.title for post in posts] == ["Post 0", "Post 1"]


def test_parse_rss_to_posts_invalid_xml():
    with pytest.raises(ElementTree.ParseError):
        GetBlogPosts.parse_rss_to_posts("not xml")


def test_parse_rss_to_posts_missing_channel():
    with pytest.raises(ValueError, match="missing <channel>"):
        GetBlogPosts.parse_rss_to_posts("<rss></rss>")


def test_validate_blog_posts_valid():
    assert GetBlogPosts.validate_blog_posts((CACHED_POST,)) is True


def test_validate_blog_posts_empty():
    assert GetBlogPosts.validate_blog_posts(()) is False


def test_get_blog_posts_success():
    """Fetches from RSS on first call."""
    mock_response = MagicMock()
    mock_response.text = SAMPLE_RSS
    mock_response.raise_for_status = MagicMock()

    with patch(
        "litellm.litellm_core_utils.get_blog_posts.httpx.get",
        return_value=mock_response,
    ):
        posts = get_blog_posts(url=litellm.blog_posts_url)

    assert len(posts) == 2
    assert posts[0].title == "Test Post"
    assert posts[0].tags == ("anthropic", "day 0 support")


def test_get_blog_posts_network_error_falls_back_to_local():
    """Falls back to local backup on network error."""
    with patch(
        "litellm.litellm_core_utils.get_blog_posts.httpx.get",
        side_effect=Exception("Network error"),
    ):
        posts = get_blog_posts(url=litellm.blog_posts_url)

    assert posts == GetBlogPosts.load_local_blog_posts()


def test_get_blog_posts_invalid_xml_falls_back_to_local():
    """Falls back when remote returns invalid XML."""
    mock_response = MagicMock()
    mock_response.text = "not valid xml"
    mock_response.raise_for_status = MagicMock()

    with patch(
        "litellm.litellm_core_utils.get_blog_posts.httpx.get",
        return_value=mock_response,
    ):
        posts = get_blog_posts(url=litellm.blog_posts_url)

    assert posts == GetBlogPosts.load_local_blog_posts()


def test_get_blog_posts_ttl_cache_not_refetched():
    """Within TTL window, does not re-fetch."""
    GetBlogPosts._cached_posts = (CACHED_POST,)
    GetBlogPosts._last_fetch_time = time.time()

    with patch("litellm.litellm_core_utils.get_blog_posts.httpx.get") as mock_get:
        posts = get_blog_posts(url=litellm.blog_posts_url)

    mock_get.assert_not_called()
    assert posts == (CACHED_POST,)


def test_get_blog_posts_ttl_expired_refetches():
    """After TTL window, re-fetches from remote."""
    GetBlogPosts._cached_posts = (CACHED_POST,)
    GetBlogPosts._last_fetch_time = time.time() - 7200

    mock_response = MagicMock()
    mock_response.text = SAMPLE_RSS
    mock_response.raise_for_status = MagicMock()

    with patch(
        "litellm.litellm_core_utils.get_blog_posts.httpx.get",
        return_value=mock_response,
    ) as mock_get:
        posts = get_blog_posts(url=litellm.blog_posts_url)

    mock_get.assert_called_once()
    assert len(posts) == 2


@pytest.mark.parametrize("flag", ["LITELLM_LOCAL_BLOG_POSTS", "LITELLM_OFFLINE_MODE"])
def test_get_blog_posts_local_env_var_skips_remote(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    with patch("litellm.litellm_core_utils.get_blog_posts.httpx.get") as mock_get:
        posts = get_blog_posts(url=litellm.blog_posts_url)
    mock_get.assert_not_called()
    assert posts == GetBlogPosts.load_local_blog_posts()


def test_blog_posts_response_serializes_tags_as_a_list():
    resp = BlogPostsResponse(posts=(CACHED_POST.model_copy(update={"tags": ("day 0 support",)}),))
    assert resp.model_dump(mode="json")["posts"][0]["tags"] == ["day 0 support"]
