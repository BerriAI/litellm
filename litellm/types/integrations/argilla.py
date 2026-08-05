from typing import Any, Dict, Final, List

from typing_extensions import TypedDict


class ArgillaItem(TypedDict):
    fields: Dict[str, Any]


class ArgillaPayload(TypedDict):
    items: List[ArgillaItem]


class ArgillaCredentialsObject(TypedDict):
    ARGILLA_API_KEY: str
    ARGILLA_DATASET_NAME: str
    ARGILLA_BASE_URL: str


SUPPORTED_PAYLOAD_FIELDS: Final = ["messages", "response"]
