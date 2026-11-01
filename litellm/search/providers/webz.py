"""
Webz.io Search Provider.
Re-exported from litellm.search.webz for standard path compatibility.
"""

from litellm.search.webz import (
    WEBZ_API_URL,
    _build_webz_payload,
    _parse_webz_response,
    async_search,
    search,
)

__all__ = [
    "WEBZ_API_URL",
    "_build_webz_payload",
    "_parse_webz_response",
    "async_search",
    "search",
]
