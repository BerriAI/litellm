import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

from typing_extensions import (
    Protocol,
    Required,
    Self,
    TypeGuard,
    get_origin,
    override,
    runtime_checkable,
)


class InfinityRerankResult(TypedDict):
    index: int
    relevance_score: float
    document: Optional[str]
