# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, Required, TypedDict

from .beta_cache_control_ephemeral_param import BetaCacheControlEphemeralParam

__all__ = ["BetaServerToolUseBlockParam"]


class BetaServerToolUseBlockParam(TypedDict, total=False):
    id: Required[str]

    input: Required[object]

    name: Required[Literal["web_search", "code_execution"]]

    type: Required[Literal["server_tool_use"]]

    cache_control: Optional[BetaCacheControlEphemeralParam]
    """Create a cache control breakpoint at this content block."""
