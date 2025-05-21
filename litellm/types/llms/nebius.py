import json
from enum import Enum
from typing import Any, List, Optional, TypedDict, Union

from pydantic import BaseModel


class NebiusAPIParams(TypedDict):
    project_id: str
    folder_id: Optional[str]
    region_name: Optional[str]


class NebiusCredentials(TypedDict):
    api_key: str
    api_base: str
    token: Optional[str]


class NebiusAIEndpoint(str, Enum):
    TEXT_GENERATION = "/text/completions"
    CHAT = "/chat/completions"
    EMBEDDINGS = "/embeddings" 