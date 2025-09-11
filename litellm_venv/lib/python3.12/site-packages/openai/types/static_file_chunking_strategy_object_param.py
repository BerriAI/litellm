# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .static_file_chunking_strategy_param import StaticFileChunkingStrategyParam

__all__ = ["StaticFileChunkingStrategyObjectParam"]


class StaticFileChunkingStrategyObjectParam(TypedDict, total=False):
    static: Required[StaticFileChunkingStrategyParam]

    type: Required[Literal["static"]]
    """Always `static`."""
