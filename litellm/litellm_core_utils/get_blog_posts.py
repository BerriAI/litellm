"""
Pulls the latest LiteLLM blog posts from GitHub.

Falls back to the bundled local backup on any failure.
GitHub JSON URL is configured via litellm.blog_posts_url (or LITELLM_BLOG_POSTS_URL env var).

Disable remote fetching entirely:
    export LITELLM_LOCAL_BLOG_POSTS=True
"""

import json
import os
import time
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

    Mirrors the structure of GetModelCostMap:
    - Fetches from GitHub with a 5-second timeout
    - Validates the response has a non-empty ``posts`` list
    - Caches the result in-process for BLOG_POSTS_TTL_SECONDS (1 hour)
    - Falls back to the bundled local backup on any failure
    """

    _cached_posts: Optional[List[Dict[str, str]]] = None
    _last_fetch_time: float = 0.0

    @staticmethod
    def load_local_blog_posts() -> List[Dict[str, str]]:
        """Load the bundled local backup blog posts."""
        content = json.loads(
            files("litellm")
            .joinpath("blog_posts.json")
            .read_text(encoding="utf-8")
        )
        return content.get("posts", [])

    @staticmethod
    def fetch_remote_blog_posts(url: str, timeout: int = 5) -> dict:
        """
        Fetch blog posts JSON from a remote URL.

        Returns the parsed response. Raises on network/parse errors.
        """
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def validate_blog_posts(data: Any) -> bool:
        """Return True if data is a dict with a non-empty ``posts`` list."""
        if not isinstance(data, dict):
            verbose_logger.warning(
                "LiteLLM: Blog posts response is not a dict (type=%s). "
                "Falling back to local backup.",
                type(data).__name__,
            )
            return False
        posts = data.get("posts")
        if not isinstance(posts, list) or len(posts) == 0:
            verbose_logger.warning(
                "LiteLLM: Blog posts response has no valid 'posts' list. "
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
            data = cls.fetch_remote_blog_posts(url)
        except Exception as e:
            verbose_logger.warning(
                "LiteLLM: Failed to fetch blog posts from %s: %s. "
                "Falling back to local backup.",
                url,
                str(e),
            )
            return cls.load_local_blog_posts()

        if not cls.validate_blog_posts(data):
            return cls.load_local_blog_posts()

        posts = data["posts"]
        cls._cached_posts = posts
        cls._last_fetch_time = now
        return posts


def get_blog_posts(url: str) -> List[Dict[str, str]]:
    """Public entry point — returns the blog posts list."""
    return GetBlogPosts.get_blog_posts(url=url)
