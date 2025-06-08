import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union


class OpenRouterErrorMessage(TypedDict):
    message: str
    code: int
    metadata: Dict
