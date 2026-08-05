from typing import Dict

from typing_extensions import TypedDict


class OpenRouterErrorMessage(TypedDict):
    message: str
    code: int
    metadata: Dict
