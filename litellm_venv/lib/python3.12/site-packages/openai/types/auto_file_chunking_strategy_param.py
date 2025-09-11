# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["AutoFileChunkingStrategyParam"]


class AutoFileChunkingStrategyParam(TypedDict, total=False):
    type: Required[Literal["auto"]]
    """Always `auto`."""
