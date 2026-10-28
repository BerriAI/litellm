"""
Pulls the latest LiteLLM blog posts from the docs RSS feed.

Falls back to the bundled local backup on any failure.
RSS URL is configured via litellm.blog_posts_url (or LITELLM_BLOG_POSTS_URL env var).

Disable remote fetching entirely:
    export LITELLM_LOCAL_BLOG_POSTS=True
"""

import os
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from importlib.resources import files
from typing import Final

import httpx
from pydantic import ConfigDict

from litellm import verbose_logger
from litellm.types.llms.base import LiteLLMBaseModel

BLOG_POSTS_TTL_SECONDS: Final[int] = 3600  # 1 hour
BLOG_POSTS_MAX: Final[int] = 50


class BlogPost(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    description: str
    date: str
    url: str
    tags: tuple[str, ...] = ()


class BlogPostsResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    posts: tuple[BlogPost, ...]


def _pub_date(pub_date_el: ET.Element | None) -> str:
    if pub_date_el is None or not pub_date_el.text:
        return ""
    try:
        return parsedate_to_datetime(pub_date_el.text).strftime("%Y-%m-%d")
    except Exception:
        return pub_date_el.text


def _parse_item(item: ET.Element) -> BlogPost | None:
    title_el: Final = item.find("title")
    link_el: Final = item.find("link")
    if title_el is None or link_el is None:
        return None
    desc_el: Final = item.find("description")
    return BlogPost(
        title=title_el.text or "",
        description=(desc_el.text or "") if desc_el is not None else "",
        date=_pub_date(item.find("pubDate")),
        url=link_el.text or "",
        tags=tuple(category.text for category in item.findall("category") if category.text),
    )


class GetBlogPosts:
    """
    Fetches, validates, and caches LiteLLM blog posts.

    - Fetches RSS feed from docs site with a 5-second timeout
    - Parses the XML and extracts up to BLOG_POSTS_MAX blog posts, with their RSS categories as tags
    - Caches the result in-process for BLOG_POSTS_TTL_SECONDS (1 hour)
    - Falls back to the bundled local backup on any failure
    """

    _cached_posts: tuple[BlogPost, ...] | None = None
    _last_fetch_time: float = 0.0

    @staticmethod
    def load_local_blog_posts() -> tuple[BlogPost, ...]:
        """Load the bundled local backup blog posts."""
        return BlogPostsResponse.model_validate_json(
            files("litellm").joinpath("blog_posts.json").read_text(encoding="utf-8")
        ).posts

    @staticmethod
    def fetch_rss_feed(url: str, timeout: int = 5) -> str:
        """
        Fetch RSS XML from a remote URL.

        Returns the raw XML text. Raises on network errors.
        """
        response: Final = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text

    @staticmethod
    def parse_rss_to_posts(xml_text: str, max_posts: int = BLOG_POSTS_MAX) -> tuple[BlogPost, ...]:
        """
        Parse RSS XML into blog posts.

        Extracts title, description, date (YYYY-MM-DD), url and the <category> tags from each <item>.
        """
        root: Final = ET.fromstring(xml_text)
        channel: Final = root.find("channel")
        if channel is None:
            raise ValueError("RSS feed missing <channel> element")

        parsed: Final = (_parse_item(item) for item in channel.findall("item"))
        return tuple(post for post in parsed if post is not None)[:max_posts]

    @staticmethod
    def validate_blog_posts(posts: tuple[BlogPost, ...]) -> bool:
        """Return True if there is at least one post."""
        if len(posts) == 0:
            verbose_logger.warning(
                "LiteLLM: Parsed RSS feed has no valid posts. Falling back to local backup.",
            )
            return False
        return True

    @classmethod
    def get_blog_posts(cls, url: str) -> tuple[BlogPost, ...]:
        """
        Return the blog posts.

        Uses the in-process cache if within BLOG_POSTS_TTL_SECONDS.
        Fetches from ``url`` otherwise, falling back to local backup on failure.
        """
        if os.getenv("LITELLM_LOCAL_BLOG_POSTS", "").lower() == "true":
            return cls.load_local_blog_posts()

        now: Final = time.time()
        cached: Final = cls._cached_posts
        if cached is not None and (now - cls._last_fetch_time) < BLOG_POSTS_TTL_SECONDS:
            return cached

        try:
            xml_text: Final = cls.fetch_rss_feed(url)
            posts: Final = cls.parse_rss_to_posts(xml_text)
        except Exception as e:
            verbose_logger.warning(
                "LiteLLM: Failed to fetch blog posts from %s: %s. Falling back to local backup.",
                url,
                str(e),
            )
            return cls.load_local_blog_posts()

        if not cls.validate_blog_posts(posts):
            return cls.load_local_blog_posts()

        cls._cached_posts = posts
        cls._last_fetch_time = now
        return posts


def get_blog_posts(url: str) -> tuple[BlogPost, ...]:
    """Public entry point, returns the blog posts."""
    return GetBlogPosts.get_blog_posts(url=url)
