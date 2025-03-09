import os
from datetime import datetime as dt
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, TypedDict


class ArgillaItem(TypedDict):
    fields: Dict[str, Any]


class ArgillaPayload(TypedDict):
    items: List[ArgillaItem]


class ArgillaCredentialsObject(TypedDict):
    ARGILLA_API_KEY: str
    ARGILLA_DATASET_NAME: str
    ARGILLA_BASE_URL: str


SUPPORTED_PAYLOAD_FIELDS = ["messages", "response"]
