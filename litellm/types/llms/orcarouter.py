from typing import Dict

from typing_extensions import TypedDict


class OrcaRouterErrorMessage(TypedDict):
    message: str
    code: int
    metadata: Dict
