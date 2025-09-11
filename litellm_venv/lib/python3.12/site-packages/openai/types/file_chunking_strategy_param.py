# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union
from typing_extensions import TypeAlias

from .auto_file_chunking_strategy_param import AutoFileChunkingStrategyParam
from .static_file_chunking_strategy_object_param import (
    StaticFileChunkingStrategyObjectParam,
)

__all__ = ["FileChunkingStrategyParam"]

FileChunkingStrategyParam: TypeAlias = Union[
    AutoFileChunkingStrategyParam, StaticFileChunkingStrategyObjectParam
]
