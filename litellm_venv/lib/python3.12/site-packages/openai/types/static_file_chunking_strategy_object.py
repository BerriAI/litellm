# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from .._models import BaseModel
from .static_file_chunking_strategy import StaticFileChunkingStrategy

__all__ = ["StaticFileChunkingStrategyObject"]


class StaticFileChunkingStrategyObject(BaseModel):
    static: StaticFileChunkingStrategy

    type: Literal["static"]
    """Always `static`."""
