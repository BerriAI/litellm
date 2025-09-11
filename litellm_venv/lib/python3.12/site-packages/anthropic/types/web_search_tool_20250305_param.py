# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List, Optional
from typing_extensions import Literal, Required, TypedDict

from .cache_control_ephemeral_param import CacheControlEphemeralParam

__all__ = ["WebSearchTool20250305Param", "UserLocation"]


class UserLocation(TypedDict, total=False):
    type: Required[Literal["approximate"]]

    city: Optional[str]
    """The city of the user."""

    country: Optional[str]
    """
    The two letter
    [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the
    user.
    """

    region: Optional[str]
    """The region of the user."""

    timezone: Optional[str]
    """The [IANA timezone](https://nodatime.org/TimeZones) of the user."""


class WebSearchTool20250305Param(TypedDict, total=False):
    name: Required[Literal["web_search"]]
    """Name of the tool.

    This is how the tool will be called by the model and in `tool_use` blocks.
    """

    type: Required[Literal["web_search_20250305"]]

    allowed_domains: Optional[List[str]]
    """If provided, only these domains will be included in results.

    Cannot be used alongside `blocked_domains`.
    """

    blocked_domains: Optional[List[str]]
    """If provided, these domains will never appear in results.

    Cannot be used alongside `allowed_domains`.
    """

    cache_control: Optional[CacheControlEphemeralParam]
    """Create a cache control breakpoint at this content block."""

    max_uses: Optional[int]
    """Maximum number of times the tool can be used in the API request."""

    user_location: Optional[UserLocation]
    """Parameters for the user's location.

    Used to provide more relevant search results.
    """
