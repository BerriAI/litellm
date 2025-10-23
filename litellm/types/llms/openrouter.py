import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from typing_extensions import TypedDict


class OpenRouterErrorMessage(TypedDict):
    message: str
    code: int
    metadata: Dict
