# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated, TypeAlias

from .._utils import PropertyInfo
from .other_file_chunking_strategy_object import OtherFileChunkingStrategyObject
from .static_file_chunking_strategy_object import StaticFileChunkingStrategyObject

__all__ = ["FileChunkingStrategy"]

FileChunkingStrategy: TypeAlias = Annotated[
    Union[StaticFileChunkingStrategyObject, OtherFileChunkingStrategyObject],
    PropertyInfo(discriminator="type"),
]
