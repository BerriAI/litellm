"""
Search providers module.
"""

from litellm.search.webz import async_search as webz_async_search, search as webz_search

__all__ = ["webz_async_search", "webz_search"]
