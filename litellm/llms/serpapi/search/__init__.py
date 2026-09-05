"""SerpApi Search API module."""

from litellm.llms.serpapi.search.transformation import SerpApiSearchConfig

__all__ = ["SerpApiSearchConfig"]  # mutable-ok: matches neighboring search provider export modules
