"""
Pulls the latest LiteLLM blog posts from the docs RSS feed.

Falls back to the bundled local backup on any failure.
RSS URL is configured via litellm.blog_posts_url (or LITELLM_BLOG_POSTS_URL env var).

Disable remote fetching entirely:
    export LITELLM_LOCAL_BLOG_POSTS=True
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from importlib.resources import files
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

from litellm import verbose_logger

BLOG_POSTS_TTL_SECONDS: int = 3600  # 1 hour


class BlogPost(BaseModel):
    title: str
    description: str
    date: str
    url: str


class BlogPostsResponse(BaseModel):
    posts: List[BlogPost]


class GetBlogPosts:
    """
    Fetches, validates, and caches LiteLLM blog posts.

    - Fetches RSS feed from docs site with a 5-second timeout
    - Parses the XML and extracts the latest blog post
    - Caches the result in-process for BLOG_POSTS_TTL_SECONDS (1 hour)
    - Falls back to the bundled local backup on any failure
    """

    _cached_posts: Optional[List[Dict[str, str]]] = None
    _last_fetch_time: float = 0.0

    @staticmethod
    def load_local_blog_posts() -> List[Dict[str, str]]:
        """Load the bundled local backup blog posts."""
        content = json.loads(
            files("litellm").joinpath("blog_posts.json").read_text(encoding="utf-8")
        )
        return content.get("posts", [])

    @staticmethod
    def fetch_rss_feed(url: str, timeout: int = 5) -> str:
        """
        Fetch RSS XML from a remote URL.

        Returns the raw XML text. Raises on network errors.
        """
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text

    @staticmethod
    def parse_rss_to_posts(xml_text: str, max_posts: int = 1) -> List[Dict[str, str]]:
        """
        Parse RSS XML and return a list of blog post dicts.

        Extracts title, description, date (YYYY-MM-DD), and url from each <item>.
        """
        root = ET.fromstring(xml_text)
        channel = root.find("channel")
        if channel is None:
            raise ValueError("RSS feed missing <channel> element")

        posts: List[Dict[str, str]] = []
        for item in channel.findall("item"):
            if len(posts) >= max_posts:
                break

            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            pub_date_el = item.find("pubDate")

            if title_el is None or link_el is None:
                continue

            # Parse RFC 2822 date to YYYY-MM-DD
            date_str = ""
            if pub_date_el is not None and pub_date_el.text:
                try:
                    dt = parsedate_to_datetime(pub_date_el.text)
                    date_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    date_str = pub_date_el.text

            posts.append(
                {
                    "title": title_el.text or "",
                    "description": desc_el.text or "" if desc_el is not None else "",
                    "date": date_str,
                    "url": link_el.text or "",
                }
            )

        return posts

    @staticmethod
    def validate_blog_posts(posts: List[Dict[str, str]]) -> bool:
        """Return True if posts is a non-empty list."""
        if not isinstance(posts, list) or len(posts) == 0:
            verbose_logger.warning(
                "LiteLLM: Parsed RSS feed has no valid posts. "
                "Falling back to local backup.",
            )
            return False
        return True

    @classmethod
    def get_blog_posts(cls, url: str) -> List[Dict[str, str]]:
        """
        Return the blog posts list.

        Uses the in-process cache if within BLOG_POSTS_TTL_SECONDS.
        Fetches from ``url`` otherwise, falling back to local backup on failure.
        """
        if os.getenv("LITELLM_LOCAL_BLOG_POSTS", "").lower() == "true":
            return cls.load_local_blog_posts()

        now = time.time()
        cached = cls._cached_posts
        if cached is not None and (now - cls._last_fetch_time) < BLOG_POSTS_TTL_SECONDS:
            return cached

        try:
            xml_text = cls.fetch_rss_feed(url)
            posts = cls.parse_rss_to_posts(xml_text)
        except Exception as e:
            verbose_logger.warning(
                "LiteLLM: Failed to fetch blog posts from %s: %s. "
                "Falling back to local backup.",
                url,
                str(e),
            )
            return cls.load_local_blog_posts()

        if not cls.validate_blog_posts(posts):
            return cls.load_local_blog_posts()

        cls._cached_posts = posts
        cls._last_fetch_time = now
        return posts


def get_blog_posts(url: str) -> List[Dict[str, str]]:
    """Public entry point — returns the blog posts list."""
    return GetBlogPosts.get_blog_posts(url=url)
