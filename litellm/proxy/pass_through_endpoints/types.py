from enum import Enum
from typing import Optional, TypedDict


class EndpointType(str, Enum):
    VERTEX_AI = "vertex-ai"
    GENERIC = "generic"


class PassthroughStandardLoggingObject(TypedDict, total=False):
    url: str
    request_body: Optional[dict]
    response_body: Optional[dict]
