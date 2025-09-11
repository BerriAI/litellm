# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, Required, TypedDict

from .cache_control_ephemeral_param import CacheControlEphemeralParam

__all__ = ["ServerToolUseBlockParam"]


class ServerToolUseBlockParam(TypedDict, total=False):
    id: Required[str]

    input: Required[object]

    name: Required[Literal["web_search"]]

    type: Required[Literal["server_tool_use"]]

    cache_control: Optional[CacheControlEphemeralParam]
    """Create a cache control breakpoint at this content block."""
